"""Chạy 9 cấu hình đã chốt của pipeline gốc trên một bộ dữ liệu bất kỳ (một "nhánh" của mục 12).

Một nhánh là một ArmData: ảnh train/val/test (uint8, 256×256×3) và, nếu chấm trên tập test chung,
các phiên bản của bộ kiểm tra khó. Cấu hình và siêu tham số giữ nguyên như bước 8:
4 cấu hình truyền thống (results/select_traditional/selected.json) và 5 CNN
(a_lenet; a_dropout có tăng cường; a_vgg16 lr 1e-3; b_mobilenetv3 fine-tune lr 1e-4; b_resnet50 fine-tune conv5 lr 1e-5).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import CACHE_DIR, CKPT_DIR, RESULTS_DIR
from .features import compute
from .metrics import save_result
from .report_utils import save_json
from .stress import STRESS, STRESS_LABEL
from .trad_config import _params, build_clf

TRAD = ["trad_color", "trad_edge", "trad_hog", "trad_fusion"]
CNN = ["a_lenet", "a_dropout", "a_vgg16", "b_mobilenetv3", "b_resnet50"]
GROUP = {"trad_color": "Truyền thống (màu)", "trad_edge": "Truyền thống (cạnh)", "trad_hog": "Truyền thống (HOG)",
         "trad_fusion": "Truyền thống (ghép)", "a_lenet": "CNN hướng A", "a_dropout": "CNN hướng A", "a_vgg16": "CNN hướng A",
         "b_mobilenetv3": "CNN hướng B", "b_resnet50": "CNN hướng B"}
CNN_CFG = {  # kích thước ảnh, lr, số epoch tối đa, patience (giống biến thể đã chốt ở bước 4–5)
    "a_lenet": (64, 1e-3, 60, 10), "a_dropout": (128, 1e-3, 40, 40), "a_vgg16": (224, 1e-3, 40, 8),
    "b_mobilenetv3": (224, 1e-4, 20, 5), "b_resnet50": (224, 1e-5, 20, 5)}


@dataclass
class ArmData:
    name: str                      # tên ngắn dùng cho thư mục checkpoint và cache, ví dụ "T-kr"
    out: Path                      # thư mục kết quả
    Xtr: np.ndarray; ytr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray
    stress: dict = field(default_factory=dict)   # {tên phép: ảnh test đã biến đổi}; rỗng nếu không chấm trên tập test chung
    pretrained: dict | None = None               # nhánh G: {mô hình: tên run đã huấn luyện ở bước 4–5}
    note: str = ""


def _feats(cfg, imgs, cache: Path | None = None):
    if cache is not None and cache.exists():
        z = np.load(cache); return z["X"], z["dims"].tolist()
    Xs = [compute(f, imgs, n_jobs=4, **_params(fp)) for f, fp in cfg["parts"]]
    X, dims = np.concatenate(Xs, axis=1), [x.shape[1] for x in Xs]
    if cache is not None:
        np.savez(cache, X=X, dims=np.array(dims))
    return X, dims


def _scores(y, p):
    return {"macro_f1": float(f1_score(y, p, average="macro", labels=range(4), zero_division=0)), "accuracy": float(accuracy_score(y, p))}


def _resize(X, size):
    from .cnn.train import resize_batch
    return resize_batch(X, size)


def run_trad(arm: ArmData, key: str, cfg: dict):
    c = lambda part: CACHE_DIR / f"arm_{arm.name}_{key}_{part}.npz"
    Ftr, dims = _feats(cfg, arm.Xtr, c("tr")); Fva, _ = _feats(cfg, arm.Xva, c("va")); Fte, _ = _feats(cfg, arm.Xte, c("te"))
    val_pred = build_clf(cfg, dims).fit(Ftr, arm.ytr).predict(Fva)
    clf = build_clf(cfg, dims).fit(np.concatenate([Ftr, Fva]), np.concatenate([arm.ytr, arm.yva]))   # như bước 8: học lại trên train+val
    preds = {"test": clf.predict(Fte)}
    for s, Xs in arm.stress.items():
        preds[s] = clf.predict(_feats(cfg, Xs)[0])
    return val_pred, preds


def _trained(run: str) -> bool:
    """Run đã huấn luyện xong và còn trọng số trong checkpoints/ (khi đó không cần chuẩn bị dữ liệu train)."""
    st = CKPT_DIR / run / "state.json"
    return st.exists() and bool(json.loads(st.read_text()).get("done")) and (CKPT_DIR / run / "best.keras").exists()


def run_cnn(arm: ArmData, model: str):
    """Huấn luyện một CNN (với nhánh G: nạp mô hình của bước 4–5) và dự đoán trên val, test và bộ kiểm tra khó.
    Run đã có trọng số trong checkpoints/ thì không huấn luyện lại, chỉ nạp trọng số và dự đoán."""
    import keras
    from .cnn import models as M
    from .cnn.train import fit_loop, setup
    size, lr, epochs, patience = CNN_CFG[model]
    f32 = lambda X: _resize(X, size).astype(np.float32)
    if arm.pretrained:                                    # nhánh G: dùng mô hình đã huấn luyện ở bước 4–5
        from .cnn.predict import Predictor
        P = Predictor(arm.pretrained[model])
        predict = lambda X: P(X).argmax(1)
    else:
        setup()
        run = f"{arm.name}__{model}"
        if model in ("a_lenet", "a_dropout"):
            net = M.a_lenet(size) if model == "a_lenet" else M.a_dropout(size, augment=True)
            net, _ = fit_loop(net, _resize(arm.Xtr, size), arm.ytr, f32(arm.Xva), arm.yva, run, lr, epochs, patience, resume=True)
            predict = lambda X: net.predict(f32(X), batch_size=64, verbose=0).argmax(1)
        elif model == "a_vgg16":
            base = M.vgg16_features(size)
            def feat(X, tag):
                f = CACHE_DIR / f"arm_{arm.name}_vgg16_{tag}.npy"
                if tag and f.exists():
                    return np.load(f)
                out = np.concatenate([base.predict(f32(X[a:a + 512]), batch_size=32, verbose=0) for a in range(0, len(X), 512)])
                if tag:
                    np.save(f, out)
                return out
            Ftr = None if _trained(run) else feat(arm.Xtr, "tr")
            head, _ = fit_loop(M.a_vgg16_head(), Ftr, arm.ytr, feat(arm.Xva, "va"), arm.yva, run, lr, epochs, patience, resume=True)
            predict = lambda X, tag="": head.predict(feat(X, tag), batch_size=64, verbose=0).argmax(1)
        else:                                             # hướng B: giai đoạn 1 (đóng băng) rồi giai đoạn 2 (fine-tune)
            full, base = M.b_model(model, size, augment=True)
            emb = keras.Model(full.input, full.get_layer("embedding").output)
            E = lambda X: np.concatenate([emb.predict(f32(X[a:a + 512]), batch_size=32, verbose=0) for a in range(0, len(X), 512)])
            Etr = None if _trained(run + "_s1") else E(arm.Xtr)
            head, _ = fit_loop(M.b_head(base.output_shape[-1]), Etr, arm.ytr, E(arm.Xva), arm.yva, run + "_s1", 1e-3, 60, 10, resume=True)
            full.get_layer("softmax").set_weights(head.get_layer("softmax").get_weights())
            M.unfreeze(base, model)
            full, _ = fit_loop(full, _resize(arm.Xtr, size), arm.ytr, f32(arm.Xva), arm.yva, run, lr, epochs, patience, resume=True)
            predict = lambda X: full.predict(f32(X), batch_size=64, verbose=0).argmax(1)
    val_pred = predict(arm.Xva)
    preds = {"test": predict(arm.Xte)}
    for s, Xs in arm.stress.items():
        preds[s] = predict(Xs)
    return val_pred, preds


def run_arm(arm: ArmData, models=None) -> list[dict]:
    """Chạy các cấu hình trên nhánh `arm`; ghi results/<nhánh>/<mô hình>/ và results/<nhánh>/table.md.
    Mặc định tính lại mọi cấu hình (CNN dùng lại trọng số đã có trong checkpoints/). Đặt biến môi trường SKIP_DONE=1
    để bỏ qua cấu hình đã có results/<nhánh>/<mô hình>/metrics.json, dùng khi chạy tiếp một lần chạy bị ngắt."""
    skip_done = os.environ.get("SKIP_DONE", "0") == "1"
    from .final_eval import bootstrap_ci
    from .report_utils import md_table
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    arm.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in (models or TRAD + CNN):
        done = arm.out / key / "metrics.json"
        if skip_done and done.exists():
            m = json.loads(done.read_text())
        else:
            name = sel[key]["name"] if key in TRAD else key
            val_pred, preds = run_trad(arm, key, sel[key]) if key in TRAD else run_cnn(arm, key)
            ci_f, _ = bootstrap_ci(arm.yte, preds["test"])
            st = {s: _scores(arm.yte, preds[s]) for s in arm.stress}
            extra = {"arm": arm.name, "config": name, "group": GROUP[key], "n_train": int(len(arm.ytr)), "n_test": int(len(arm.yte)),
                     "val": _scores(arm.yva, val_pred), "macro_f1_ci95": ci_f, "n_errors": int((preds["test"] != arm.yte).sum()),
                     "stress": st, "robust_score": float(np.mean([v["macro_f1"] for v in st.values()])) if st else None}
            m = save_result(arm.out / key, arm.yte, preds["test"], extra=extra, title=f"{arm.name} — {name} — test")
            np.savez(arm.out / key / "predictions.npz", y=arm.yte, **preds)
        row = {"Nhóm": m["group"], "Mô hình": m["config"], "Macro-F1 val": f"{m['val']['macro_f1']:.3f}", "Macro-F1 test": f"{m['macro_f1']:.3f}",
               "KTC 95%": f"[{m['macro_f1_ci95'][0]:.3f}; {m['macro_f1_ci95'][1]:.3f}]", "Accuracy test": f"{m['accuracy']:.3f}",
               f"Số ảnh sai / {m['n_test']}": m["n_errors"]}
        if m["stress"]:
            row.update({STRESS_LABEL[s]: f"{m['stress'][s]['macro_f1']:.3f}" for s in STRESS if s in m["stress"]})
            row["Điểm bền"] = f"{m['robust_score']:.3f}"
        rows.append(row); print(arm.name, row, flush=True)
    title = f"Nhánh {arm.name}: 9 cấu hình của pipeline gốc ({arm.note}; {len(arm.ytr)} ảnh train, chấm trên {len(arm.yte)} ảnh test)"
    (arm.out / "table.md").write_text(f"### {title}\n\n" + md_table(rows))
    save_json(arm.out / "summary.json", {"arm": arm.name, "note": arm.note, "n_train": int(len(arm.ytr)), "n_val": int(len(arm.yva)), "n_test": int(len(arm.yte))})
    return rows


def base_split():
    """Ảnh và nhãn train/val/test của pipeline gốc."""
    from .data import load_all_images, load_split
    imgs = load_all_images(256); out = {}
    for s in ("train", "val", "test"):
        df = load_split(s)
        out[s] = (np.stack([imgs[i] for i in df["idx"].values]), df["label_id"].values, df["idx"].values)
    return out


def arm_g() -> ArmData:
    """Nhánh G: pipeline gốc; CNN lấy từ các run đã huấn luyện ở bước 4–5."""
    from .exp_cnn import final_run
    from .stress import stress_sets
    b = base_split()
    return ArmData("G", RESULTS_DIR / "so_sanh_nhanh" / "G", b["train"][0], b["train"][1], b["val"][0], b["val"][1], b["test"][0], b["test"][1],
                   stress=stress_sets("test"), pretrained={m: final_run(m) for m in CNN}, note="pipeline gốc, ảnh gốc resize 256×256")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Chấm lại nhánh G (pipeline gốc) trên test và bộ kiểm tra khó: python -m src.pipeline")
    ap.parse_args()
    run_arm(arm_g())
