"""Bước 8: đánh giá cuối trên tập test cho các cấu hình đã chốt, và phân tích lỗi.

Nhóm truyền thống: 4 cấu hình trong results/select_traditional/selected.json, học trên train+val.
CNN: 5 mô hình (biến thể được chọn theo val), học trên train. Tập test (371 ảnh) chỉ được dùng ở đây.

    python -m src.final_eval
Kết quả: results/final/<tên>/ (metrics.json, classification_report.txt, confusion_matrix.png),
results/final/table.md, results/final/errors.csv, hình results/figures/final_*.png.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import CACHE_DIR, CLASS_NAMES, FIG_DIR, RESULTS_DIR, SEED
from .data import dev_and_test, list_images, load_all_images
from .metrics import save_result
from .report_utils import exp_dir, save_json, save_table
from .trad_config import build_clf, features_all

GROUP = {"trad_color": "Truyền thống (màu)", "trad_edge": "Truyền thống (cạnh)", "trad_hog": "Truyền thống (HOG)", "trad_fusion": "Truyền thống (ghép)"}


def bootstrap_ci(y, p, n: int = 2000, seed: int = SEED):
    """Khoảng tin cậy 95% của macro-F1 và accuracy bằng bootstrap trên ảnh test."""
    rng = np.random.default_rng(seed); f, a = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f.append(f1_score(y[i], p[i], average="macro", labels=range(4), zero_division=0)); a.append(accuracy_score(y[i], p[i]))
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from .exp_cnn import A_RUNS, B_RUNS, final_run, meta
    from .cnn.predict import Predictor
    dev, te = dev_and_test()
    idev, ydev, ite, yte = dev["idx"].values, dev["label_id"].values, te["idx"].values, te["label_id"].values
    out = exp_dir("final")
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    preds, probs, rows, srows = {}, {}, [], []

    def record(key, name, group, p, dev_score, dev_label, extra=None, proba=None):
        ci_f, ci_a = bootstrap_ci(yte, p)
        m = save_result(out / key, yte, p, extra=dict({"config": name, "group": group, "eval_set": "test (371 ảnh)", "macro_f1_ci95": ci_f, "accuracy_ci95": ci_a,
                                                        "n_errors": int((p != yte).sum()), "dev_score": dev_score, "dev_score_kind": dev_label}, **(extra or {})),
                        title=f"{name} — test")
        preds[key] = p
        if proba is not None:
            probs[key] = proba
        rc = [m["per_class"][c]["recall"] for c in CLASS_NAMES]
        rows.append({"Nhóm": group, "Mô hình": name, "Macro-F1 test": f"{m['macro_f1']:.3f}", "KTC 95% (bootstrap)": f"[{ci_f[0]:.3f}; {ci_f[1]:.3f}]",
                     "Accuracy test": f"{m['accuracy']:.3f}", "Số ảnh sai / 371": int((p != yte).sum()),
                     "Recall thấp nhất": f"{min(rc):.3f} ({CLASS_NAMES[int(np.argmin(rc))].replace(' Kola', '')})", dev_label: dev_score, "_key": key})
        srows.append({"experiment": "final", "group": group, "config": name, "condition": "clean", "eval_set": "test", "macro_f1": round(m["macro_f1"], 4),
                      "macro_f1_std": None, "accuracy": round(m["accuracy"], 4), "macro_f1_ci95_low": round(ci_f[0], 4), "macro_f1_ci95_high": round(ci_f[1], 4)})
        print(rows[-1], flush=True)

    for key in ("trad_color", "trad_edge", "trad_hog", "trad_fusion"):
        cfg = sel[key]; X, dims = features_all(cfg)
        t0 = time.perf_counter(); clf = build_clf(cfg, dims).fit(X[idev], ydev); t_fit = time.perf_counter() - t0
        p = clf.predict(X[ite])
        proba = clf.predict_proba(X[ite]) if hasattr(clf, "predict_proba") else None
        record(key, cfg["name"], GROUP[key], p, f"{cfg['cv_macro_f1']:.3f} ± {cfg['cv_macro_f1_std']:.3f}", "Macro-F1 lúc chọn (5-fold CV hoặc val)",
               {"fit_time_s_train_plus_val": t_fit, "n_train": int(len(idev))}, proba)
    imgs_te = np.stack([load_all_images(256)[i] for i in ite])
    for model in A_RUNS + B_RUNS:
        run = final_run(model); m = meta(run)
        proba = Predictor(run)(imgs_te)
        record(model, f"{model} (biến thể: {run})", "CNN hướng A" if model.startswith("a_") else "CNN hướng B", proba.argmax(1), f"{m['macro_f1']:.3f}",
               "Macro-F1 lúc chọn (5-fold CV hoặc val)", {"run": run, "n_train": m["n_train"]}, proba)
    np.savez(out / "test_predictions.npz", idx=ite, y=yte, **{k: v for k, v in preds.items()})
    keys = [r.pop("_key") for r in rows]
    register_rows = srows
    from .exp_common import register
    register("final", register_rows)
    print(save_table("final", rows, title="Đánh giá cuối trên tập test (371 ảnh; mỗi ảnh sai tương ứng 0,27% accuracy)"))

    # --- so sánh từng cặp: số ảnh chỉ mô hình này sai / chỉ mô hình kia sai
    best = {g: max([k for k in keys if (k.startswith(pref))], key=lambda k: (f1_score(yte, preds[k], average="macro"), -keys.index(k)))
            for g, pref in [("truyền thống", "trad_"), ("hướng A", "a_"), ("hướng B", "b_")]}
    pair_rows = []
    bk = list(best.values())
    for i in range(len(bk)):
        for j in range(i + 1, len(bk)):
            a, b = preds[bk[i]] != yte, preds[bk[j]] != yte
            pair_rows.append({"Mô hình 1": bk[i], "Mô hình 2": bk[j], "Cả hai sai": int((a & b).sum()), "Chỉ mô hình 1 sai": int((a & ~b).sum()),
                              "Chỉ mô hình 2 sai": int((~a & b).sum()), "Cả hai đúng": int((~a & ~b).sum())})
    print(save_table("final", pair_rows, "table_pairs", title="So sánh từng cặp giữa mô hình tốt nhất của ba nhóm trên tập test"))

    # --- ảnh gần trùng: accuracy trên ảnh test có / không có ảnh gần trùng trong train+val
    D = np.load(CACHE_DIR / "phash_dist.npy") if (CACHE_DIR / "phash_dist.npy").exists() else None
    if D is None:
        from .split_data import pairwise_hamming, phash_all
        D = pairwise_hamming(phash_all(list_images())).astype(np.uint8); np.save(CACHE_DIR / "phash_dist.npy", D)
    dup_rows = []
    for th in (2, 4):
        near = (D[np.ix_(ite, idev)] <= th).any(1)
        for k in bk:
            ok = preds[k] == yte
            dup_rows.append({"Ngưỡng Hamming": f"≤ {th}", "Mô hình": k, "Ảnh test có ảnh gần trùng trong train+val": int(near.sum()),
                             "Accuracy trên nhóm đó": f"{ok[near].mean():.3f}" if near.any() else "—",
                             "Ảnh test không có": int((~near).sum()), "Accuracy trên nhóm không có": f"{ok[~near].mean():.3f}"})
    print(save_table("final", dup_rows, "table_near_duplicates", title="Accuracy trên test tách theo việc có ảnh gần trùng (pHash) trong tập huấn luyện"))

    # --- phân tích lỗi
    df = list_images().set_index("idx")
    err = []
    for k in keys:
        for j in np.where(preds[k] != yte)[0]:
            err.append({"model": k, "path": df.loc[ite[j], "path"], "true": CLASS_NAMES[yte[j]], "pred": CLASS_NAMES[preds[k][j]],
                        "prob_pred": float(probs[k][j].max()) if k in probs else None, "prob_true": float(probs[k][j][yte[j]]) if k in probs else None})
    err = pd.DataFrame(err); err.to_csv(out / "errors.csv", index=False)
    n_wrong = np.sum([preds[k] != yte for k in keys], axis=0)
    hard = [{"Ảnh": df.loc[ite[j], "path"], "Lớp đúng": CLASS_NAMES[yte[j]], "Số mô hình sai / %d" % len(keys): int(n_wrong[j]),
             "Các mô hình sai → lớp đoán": "; ".join(f"{k}→{CLASS_NAMES[preds[k][j]].replace(' Kola', '')}" for k in keys if preds[k][j] != yte[j])}
            for j in np.argsort(-n_wrong) if n_wrong[j] >= 2]
    save_table("final", hard, "table_hard_images", title="Ảnh test bị từ 2 mô hình trở lên đoán sai")
    conf = {}
    for k in bk:
        cm = np.array(json.loads((out / k / "metrics.json").read_text())["confusion_matrix"])
        pairs = sorted([(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(4) for j in range(4) if i != j and cm[i, j] > 0], reverse=True)
        conf[k] = [{"Mô hình": k, "Thực tế": a, "Đoán thành": b, "Số ảnh": int(n)} for n, a, b in pairs]
    save_table("final", [r for k in bk for r in conf[k]] or [{"Mô hình": "—"}], "table_confusions", title="Các cặp lớp bị nhầm của mô hình tốt nhất mỗi nhóm (test)")
    save_json(out / "summary.json", {"best_per_group": best, "n_test": int(len(yte)), "n_images_wrong_by_at_least_one_model": int((n_wrong > 0).sum()),
                                     "n_images_wrong_by_all_models": int((n_wrong == len(keys)).sum()), "models": keys})

    # hình: ảnh đoán sai của mô hình tốt nhất mỗi nhóm
    for g, k in best.items():
        w = np.where(preds[k] != yte)[0][:12]
        if len(w) == 0:
            continue
        cols = min(6, len(w)); rws = int(np.ceil(len(w) / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(2.3 * cols, 2.7 * rws), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, j in zip(axes.ravel(), w):
            ax.imshow(imgs_te[j])
            pr = f" ({probs[k][j].max():.2f})" if k in probs else ""
            ax.set_title(f"đúng: {CLASS_NAMES[yte[j]].replace(' Kola', '')}\nđoán: {CLASS_NAMES[preds[k][j]].replace(' Kola', '')}{pr}", fontsize=8)
        fig.suptitle(f"Ảnh test bị đoán sai — {k} ({int((preds[k] != yte).sum())} ảnh sai / 371)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG_DIR / f"final_errors_{k}.png", dpi=110); plt.close(fig)
    # hình: macro-F1 test kèm khoảng tin cậy
    fig, ax = plt.subplots(figsize=(9, 4))
    ms = [json.loads((out / k / "metrics.json").read_text()) for k in keys]
    f = [m["macro_f1"] for m in ms]; lo = [m["macro_f1"] - m["macro_f1_ci95"][0] for m in ms]; hi = [m["macro_f1_ci95"][1] - m["macro_f1"] for m in ms]
    col = ["tab:green" if k.startswith("trad") else "tab:blue" if k.startswith("a_") else "tab:orange" for k in keys]
    ax.bar(range(len(keys)), f, yerr=[lo, hi], capsize=4, color=col)
    for i, v in enumerate(f):
        ax.text(i, v + hi[i] + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=20, fontsize=8); ax.set_ylim(min(f) - 0.06, 1.02)
    ax.set_ylabel("Macro-F1 trên test"); ax.set_title("Đánh giá cuối trên test (xanh lá: truyền thống, xanh dương: hướng A, cam: hướng B; thanh: KTC 95%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_test_macro_f1.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
