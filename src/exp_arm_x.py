"""Nhánh X (mục 12.2): xử lý dữ liệu do nhóm thiết kế (làm sạch, tiền xử lý, tăng cường) rồi chạy lại
9 cấu hình của pipeline gốc trên dữ liệu đã xử lý.

    python -m src.exp_arm_x --only X1     # thống kê làm sạch, kiểm tra "một quả chụp nhiều mặt"
    python -m src.exp_arm_x --only X2     # chọn tổ hợp tiền xử lý × tăng cường trên val (X2-X4)
    python -m src.exp_arm_x --only X3     # kiểm tra: cân bằng trắng theo nền có đưa thông tin nền vào màu quả không
    python -m src.exp_arm_x --only RUN    # chạy 9 cấu hình với tổ hợp được chọn
    python -m src.exp_arm_x               # tất cả
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .augment import MODE_NAME, make_augmented
from .classical import cv_eval, folds, make_clf
from .config import CACHE_DIR, CLASS_NAMES, FIG_DIR, RESULTS_DIR, ROOT, SEED
from .data import list_images, load_all_images, load_split
from .pipeline import ArmData, _feats, base_split, run_arm
from .preprocess import LEVEL_NAME, LEVELS, process_batch
from .report_utils import exp_dir, pm, save_json, save_table
from .stress import STRESS, STRESS_LABEL, stress_sets
from .trad_config import build_clf

EXP = "E1_xu_ly"
OUT = RESULTS_DIR / EXP
TESTS = ["clean"] + STRESS


# ------------------------------------------------------------------ X1: làm sạch
def near_dup_drop_list():
    """Chỉ số các ảnh train bị loại: gần trùng (pHash, Hamming ≤ 2) với một ảnh val/test, hoặc là bản lặp
    của một ảnh train khác (giữ ảnh có chỉ số nhỏ nhất trong mỗi nhóm)."""
    D = np.load(CACHE_DIR / "phash_dist.npy")
    tr, va, te = (load_split(s)["idx"].values for s in ("train", "val", "test"))
    held = np.r_[va, te]
    cross = tr[(D[np.ix_(tr, held)] <= 2).any(1)]
    keep = np.setdiff1d(tr, cross)
    sub = D[np.ix_(keep, keep)] <= 2
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(csr_matrix(sub), directed=False)
    first = {}
    within = []
    for i, g in zip(keep, lab):
        if g in first:
            within.append(i)
        else:
            first[g] = i
    return np.array(sorted(cross)), np.array(sorted(within))


def file_number(path: str) -> int:
    return int(re.search(r"_(\d+)\.[A-Za-z]+$", path).group(1))


def x1():
    df = list_images(); imgs = load_all_images(256)
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    # 1) kiểm tra file
    md5 = [hashlib.md5((ROOT / p).read_bytes()).hexdigest() for p in df["path"]]
    sizes = [cv2.imread(str(ROOT / p)).shape[:2] for p in df["path"]]
    blur = np.array([cv2.Laplacian(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for im in imgs])
    cross, within = near_dup_drop_list()
    # 2) mask của bước S1 so với mask mới
    from .exp_segment import all_masks, all_masks_v2
    m1, m2 = all_masks(), all_masks_v2()
    a1, a2 = (m1 > 0).reshape(len(m1), -1).mean(1), (m2 > 0).reshape(len(m2), -1).mean(1)
    spill = (a1 > 1.5 * a2)
    rows = [
        {"Kiểm tra": "File không đọc được", "Kết quả": 0 if all(s is not None for s in sizes) else "có"},
        {"Kiểm tra": "Ảnh không phải 256×256 (được resize khi đọc)", "Kết quả": int(sum(s != (256, 256) for s in sizes))},
        {"Kiểm tra": "File trùng tuyệt đối (md5)", "Kết quả": int(len(md5) - len(set(md5)))},
        {"Kiểm tra": "Ảnh train gần trùng (pHash, Hamming ≤ 2) với ảnh val hoặc test → loại khỏi train", "Kết quả": int(len(cross))},
        {"Kiểm tra": "Bản lặp gần trùng trong nội bộ train → giữ 1 ảnh mỗi nhóm, loại", "Kết quả": int(len(within))},
        {"Kiểm tra": "Số ảnh train sau khi làm sạch", "Kết quả": f"{len(tr) - len(cross) - len(within)} / {len(tr)}"},
        {"Kiểm tra": "Độ nét (phương sai Laplacian): phân vị 5% / trung vị / phân vị 95%", "Kết quả": f"{np.percentile(blur, 5):.0f} / {np.median(blur):.0f} / {np.percentile(blur, 95):.0f}"},
        {"Kiểm tra": "Ảnh có độ nét dưới 1/4 trung vị (nghi mờ; giữ lại, chỉ liệt kê)", "Kết quả": int((blur < np.median(blur) / 4).sum())},
        {"Kiểm tra": "Mask của bước S1 lớn hơn mask mới từ 1.5 lần (mask cũ tràn ra nền)", "Kết quả": f"{int(spill.sum())} / {len(df)} ({100 * spill.mean():.1f}%)"},
    ]
    print(save_table(EXP, rows, "table_cleaning", title="X1: Kiểm tra và làm sạch dữ liệu"))
    pd.DataFrame({"idx": np.r_[cross, within], "path": df["path"].values[np.r_[cross, within]],
                  "ly_do": ["gần trùng với val/test"] * len(cross) + ["bản lặp trong train"] * len(within)}).to_csv(OUT / "train_dropped.csv", index=False)
    pd.DataFrame({"path": df["path"].values[blur < np.median(blur) / 4], "do_net": blur[blur < np.median(blur) / 4].round(1)}).to_csv(OUT / "low_sharpness.csv", index=False)

    # 3) ảnh nghi sai nhãn: bị cả hai mô hình (truyền thống tốt nhất, embedding MobileNetV3 + LogReg) đoán sai khi không nằm trong phần học
    from .cnn.train import backbone_features
    from .exp_common import dev_set
    from .trad_config import features_all
    ids, y = dev_set()
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    Xf, dims = features_all(sel["trad_fusion"])
    p1 = cv_eval(Xf[ids], y, lambda: build_clf(sel["trad_fusion"], dims))["oof_pred"]
    E = np.asarray(backbone_features("b_mobilenetv3"))
    p2 = cv_eval(E[ids], y, lambda: make_clf("logreg"))["oof_pred"]
    both = (p1 != y) & (p2 != y)
    sus = [{"Ảnh": df["path"].values[ids[k]], "Nhãn": CLASS_NAMES[y[k]], "Truyền thống đoán": CLASS_NAMES[p1[k]], "Embedding CNN đoán": CLASS_NAMES[p2[k]]}
           for k in np.where(both)[0]]
    save_table(EXP, sus or [{"Ảnh": "(không có)"}], "table_suspect_labels",
               title=f"X1: Ảnh train+val bị cả hai mô hình đoán sai ngoài phần học ({len(sus)} ảnh; cần người kiểm tra nhãn, đồ án không tự sửa)")

    # 4) một quả chụp nhiều mặt: ảnh có số thứ tự liền kề có giống nhau bất thường không?
    num = np.array([file_number(p) for p in df["path"]]); lab = df["label_id"].values
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    S_ = En @ En.T; np.fill_diagonal(S_, -1)
    S_[lab[:, None] != lab[None, :]] = -1               # chỉ xét láng giềng cùng lớp
    nn = S_.argmax(1); gap = np.abs(num[nn] - num)
    n_c = np.bincount(lab)[lab]
    chance = np.minimum(6, n_c - 1) / (n_c - 1)          # xác suất ngẫu nhiên để |Δ số thứ tự| ≤ 3
    adj = [{"Chỉ số": "Ảnh có láng giềng gần nhất cùng lớp (embedding MobileNetV3) mang số thứ tự cách ≤ 3", "Đo được": f"{100 * (gap <= 3).mean():.1f}%",
            "Nếu ngẫu nhiên": f"{100 * chance.mean():.1f}%"},
           {"Chỉ số": "… cách đúng 1", "Đo được": f"{100 * (gap == 1).mean():.1f}%", "Nếu ngẫu nhiên": f"{100 * (2 / (n_c - 1)).mean():.1f}%"}]
    print(save_table(EXP, adj, "table_adjacent", title="X1: Ảnh mang số thứ tự liền kề giống nhau tới mức nào (toàn bộ 2471 ảnh)"))
    # cross-validation chia ngẫu nhiên so với chia theo khối ảnh liên tiếp (5 khối theo số thứ tự trong mỗi lớp)
    nd = num[ids]; block = np.zeros(len(ids), int)
    for c in range(4):
        m = np.where(y == c)[0]; order = m[np.argsort(nd[m])]
        block[order] = (np.arange(len(order)) * 5) // len(order)
    blocked = [(np.where(block != k)[0], np.where(block == k)[0]) for k in range(5)]
    cmp_rows = []
    for name, X, fac in [("Hist HSV (8,8,8) + LogReg C=0.1", features_all(sel["trad_color"])[0][ids], lambda: build_clf(sel["trad_color"], [512])),
                         ("HOG 64×128 + LogReg C=0.1", features_all(sel["trad_hog"])[0][ids], lambda: build_clf(sel["trad_hog"], [3780])),
                         ("Hist H-S + HOG + LogReg C=100", Xf[ids], lambda: build_clf(sel["trad_fusion"], dims)),
                         ("Embedding MobileNetV3 (ImageNet) + LogReg", E[ids], lambda: make_clf("logreg"))]:
        r = cv_eval(X, y, fac)
        f1b = []
        for tr_i, te_i in blocked:
            f1b.append(f1_score(y[te_i], fac().fit(X[tr_i], y[tr_i]).predict(X[te_i]), average="macro"))
        cmp_rows.append({"Mô hình": name, "5-fold ngẫu nhiên phân tầng": pm(r["macro_f1"], r["macro_f1_std"]),
                         "5-fold theo khối ảnh liên tiếp": pm(np.mean(f1b), np.std(f1b)), "Chênh": f"{r['macro_f1'] - np.mean(f1b):+.3f}"})
    print(save_table(EXP, cmp_rows, "table_blocked_cv", title="X1: Macro-F1 trên train+val: chia ngẫu nhiên so với chia theo khối ảnh có số thứ tự liên tiếp"))
    save_json(OUT / "x1_metrics.json", {"cleaning": rows, "adjacent": adj, "blocked_cv": cmp_rows, "n_suspect_labels": len(sus),
                                        "dropped_cross": cross.tolist(), "dropped_within": within.tolist()})


# ------------------------------------------------------------------ X2-X4: chọn tổ hợp trên val
def cleaned_train():
    b = base_split(); Xtr, ytr, itr = b["train"]
    cross, within = near_dup_drop_list()
    keep = ~np.isin(itr, np.r_[cross, within])
    return Xtr[keep], ytr[keep], itr[keep], b


def combos():
    return [(lv, ag) for lv in LEVELS for ag in (["none", "geo", "geo+bg"] if lv in ("none", "wb") else ["none", "geo"])]


def build_sets(level: str, aug: str):
    """Tập train (đã làm sạch, đã tiền xử lý, kèm bản tăng cường) của một tổ hợp."""
    from .segmentation import banana_mask_v2
    Xtr, ytr, itr, b = cleaned_train()
    masks = None
    if aug == "geo+bg":
        from .exp_segment import all_masks_v2
        masks = all_masks_v2()[itr]
    Xa, ya = make_augmented(Xtr, ytr, aug, 2, masks)          # tăng cường trên ảnh thô, rồi mới tiền xử lý như mọi ảnh khác
    X = process_batch(np.concatenate([Xtr, Xa]) if len(Xa) else Xtr, level)
    return X, np.concatenate([ytr, ya]) if len(Xa) else ytr, b


def x2():
    from .cnn.train import backbone_net, resize_batch
    net = backbone_net("b_mobilenetv3")
    emb = lambda X: np.concatenate([net.predict(resize_batch(X[a:a + 512], 224).astype(np.float32), batch_size=32, verbose=0) for a in range(0, len(X), 512)])
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text()); cfg = sel["trad_fusion"]
    va = load_split("val"); yva = va["label_id"].values
    raw_val = {"clean": np.stack([load_all_images(256)[i] for i in va["idx"].values]), **stress_sets("val")}
    raw, rows, preds = {}, [], {}
    val_cache = {}
    for level, aug in combos():
        if level not in val_cache:                      # ảnh val (gốc + 6 phép thử) sau tiền xử lý của mức này
            P = {t: process_batch(raw_val[t], level) for t in TESTS}
            val_cache[level] = ({t: _feats(cfg, P[t])[0] for t in TESTS}, {t: emb(P[t]) for t in TESTS})
        Fv, Ev = val_cache[level]
        X, y, _ = build_sets(level, aug)
        Ft, dims = _feats(cfg, X); clf_t = build_clf(cfg, dims).fit(Ft, y)
        clf_e = make_clf("logreg").fit(emb(X), y)
        pt = {t: clf_t.predict(Fv[t]) for t in TESTS}; pe = {t: clf_e.predict(Ev[t]) for t in TESTS}
        preds[f"{level}|{aug}"] = np.stack([pt[t] for t in TESTS] + [pe[t] for t in TESTS])      # (14, số ảnh val)
        sc_t = {t: float(f1_score(yva, pt[t], average="macro")) for t in TESTS}
        sc_e = {t: float(f1_score(yva, pe[t], average="macro")) for t in TESTS}
        mean_t, mean_e = float(np.mean(list(sc_t.values()))), float(np.mean(list(sc_e.values())))
        raw[f"{level}|{aug}"] = {"trad": sc_t, "cnn_embedding": sc_e, "mean_trad": mean_t, "mean_cnn": mean_e, "score": (mean_t + mean_e) / 2, "n_train": int(len(y))}
        row = {"Tiền xử lý": LEVEL_NAME[level], "Tăng cường (train)": MODE_NAME[aug], "Số ảnh train": len(y)}
        row.update({f"{STRESS_LABEL[t]} (TT / CNN)": f"{sc_t[t]:.3f} / {sc_e[t]:.3f}" for t in TESTS})
        row.update({"TB 7 phép thử (TT)": f"{mean_t:.3f}", "TB 7 phép thử (CNN)": f"{mean_e:.3f}", "Điểm chọn": f"{(mean_t + mean_e) / 2:.3f}"})
        rows.append(row); print(row, flush=True)
    best = max(raw, key=lambda k: raw[k]["score"])
    level, aug = best.split("|")
    # Sai số của điểm chọn: bootstrap 1000 lần trên ảnh val (lấy mẫu lại cùng chỉ số ảnh cho mọi tổ hợp, nên so được từng cặp)
    rng = np.random.default_rng(SEED); n = len(yva)

    def score(P, idx):
        return float(np.mean([f1_score(yva[idx], P[j][idx], average="macro", labels=range(4), zero_division=0) for j in range(P.shape[0])]))

    boots = {k: [] for k in preds}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        for k in preds:
            boots[k].append(score(preds[k], idx))
    boots = {k: np.array(v) for k, v in boots.items()}
    for k, row in zip(raw, rows):
        d = boots[best] - boots[k]
        raw[k]["score_boot_std"] = float(boots[k].std()); raw[k]["diff_to_top"] = float(raw[best]["score"] - raw[k]["score"])
        raw[k]["diff_to_top_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        raw[k]["tied_with_top"] = bool(np.percentile(d, 2.5) <= 0)
        row["Điểm chọn"] = f"{raw[k]['score']:.3f} ± {raw[k]['score_boot_std']:.3f}"
        row["Kém tổ hợp đứng đầu (KTC 95%)"] = "-" if k == best else f"{raw[k]['diff_to_top']:.3f} [{raw[k]['diff_to_top_ci95'][0]:+.3f}; {raw[k]['diff_to_top_ci95'][1]:+.3f}]"
        row["Phân biệt được với tổ hợp đứng đầu"] = "-" if k == best else ("không" if raw[k]["tied_with_top"] else "có")
    # Hai biến thể vào chạy đủ 9 cấu hình:
    #   X-a: tổ hợp có điểm chọn cao nhất;
    #   X-b: tổ hợp có điểm chọn cao nhất trong các tổ hợp không xử lý ảnh lúc suy luận (mức "none"), tức chỉ can thiệp vào tập train.
    best_b = max([k for k in raw if k.startswith("none|")], key=lambda k: raw[k]["score"])
    lb, ab = best_b.split("|")
    save_json(OUT / "selection.json", {"selected": {"level": level, "aug": aug}, "selected_b": {"level": lb, "aug": ab}, "combos": raw,
                                        "rule": "điểm chọn = trung bình macro-F1 trên val của 7 phép thử (ảnh gốc + 6 phép của bộ kiểm tra khó), lấy trung bình hai mô hình đại diện; "
                                                "X-a = điểm cao nhất; X-b = điểm cao nhất trong các tổ hợp không xử lý ảnh lúc suy luận"})
    print(save_table(EXP, rows, "table_selection", title="X2-X4: Chọn tổ hợp tiền xử lý × tăng cường trên val (macro-F1; TT = Hist H-S + HOG + LogReg, CNN = embedding MobileNetV3 đóng băng + LogReg; ± là độ lệch chuẩn bootstrap)"))
    print("X-a:", LEVEL_NAME[level], "+", MODE_NAME[aug], "| X-b:", LEVEL_NAME[lb], "+", MODE_NAME[ab])
    # hình
    keys = list(raw); M = np.array([[(raw[k]["trad"][t] + raw[k]["cnn_embedding"][t]) / 2 for t in TESTS] + [raw[k]["score"]] for k in keys])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.2, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold" if keys[i] == best else "normal")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([STRESS_LABEL[t] for t in TESTS] + ["Điểm chọn"], fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{LEVEL_NAME[k.split('|')[0]]} | {MODE_NAME[k.split('|')[1]]}" for k in keys], fontsize=7); ax.grid(False)
    ax.set_title("X2-X4: Macro-F1 trên val theo tổ hợp xử lý dữ liệu (trung bình hai mô hình đại diện; dòng in đậm: tổ hợp được chọn)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig(FIG_DIR / "E1_selection.png", dpi=130); plt.close(fig)


def x3():
    """Kiểm tra: cân bằng trắng theo nền có chuyển thông tin của nền vào màu quả chuối không?
    Histogram HSV + LogReg học trên train đã tiền xử lý, chấm trên val gốc và val nền hoán đổi, với bốn mức tiền xử lý.
    Kèm mức thay đổi màu của chính các điểm ảnh quả chuối (theo mask) giữa ảnh gốc và ảnh nền hoán đổi sau tiền xử lý."""
    from .exp_segment import all_masks_v2
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text()); cfg = sel["trad_color"]
    va = load_split("val"); yva, iva = va["label_id"].values, va["idx"].values
    raw_clean = np.stack([load_all_images(256)[i] for i in iva]); raw_swap = stress_sets("val")["swapbg"]
    masks = all_masks_v2()[iva]
    core = [cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0 for m in masks]     # lõi quả chuối, tránh viền ghép
    rows, raw = [], {}
    for level in ("none", "bg", "wb", "wb+bg"):
        X, y, _ = build_sets(level, "none")
        Ft, dims = _feats(cfg, X); clf = build_clf(cfg, dims).fit(Ft, y)
        Pc, Ps = process_batch(raw_clean, level), process_batch(raw_swap, level)
        f_c = float(f1_score(yva, clf.predict(_feats(cfg, Pc)[0]), average="macro")); f_s = float(f1_score(yva, clf.predict(_feats(cfg, Ps)[0]), average="macro"))
        lab = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
        d = float(np.mean([np.linalg.norm(lab(a)[c] - lab(b)[c], axis=1).mean() for a, b, c in zip(Pc, Ps, core) if c.any()]))
        raw[level] = {"val_clean_macro_f1": f_c, "val_swapbg_macro_f1": f_s, "banana_color_shift_lab": d}
        rows.append({"Tiền xử lý": LEVEL_NAME[level], "Hist HSV + LogReg: val gốc": f"{f_c:.3f}", "Hist HSV + LogReg: val nền hoán đổi": f"{f_s:.3f}",
                     "Sụt": f"{f_c - f_s:+.3f}", "Màu quả chuối lệch đi khi đổi nền (khoảng cách Lab trung bình)": f"{d:.1f}"})
    save_json(OUT / "x3_wb_leak.json", raw)
    print(save_table(EXP, rows, "table_wb_leak", title="X3: Cân bằng trắng theo nền có đưa thông tin của nền vào màu quả chuối không (371 ảnh val)"))


def fig_examples(level, aug):
    """Hình minh họa: ảnh gốc, ảnh sau từng mức tiền xử lý, và bản tăng cường."""
    from .augment import photo_geo
    b = base_split(); X, y, _ = b["train"]
    pick = [int(np.where(y == c)[0][k]) for c in range(4) for k in (4, 60)]
    rows = [("Ảnh gốc", X[pick])] + [(LEVEL_NAME[lv], process_batch(X[pick], lv, n_jobs=1)) for lv in LEVELS[1:]]
    rows.append((MODE_NAME["geo"], np.stack([photo_geo(X[i], np.random.default_rng([SEED, i])) for i in pick])))
    fig, axes = plt.subplots(len(rows), len(pick), figsize=(1.7 * len(pick), 1.75 * len(rows)))
    for r, (name, ims) in enumerate(rows):
        for c in range(len(pick)):
            axes[r, c].imshow(ims[c]); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name.replace(" + ", "\n+ ").replace(" theo ", "\ntheo "), fontsize=7)
    fig.suptitle("Nhánh X: các mức tiền xử lý và tăng cường dữ liệu", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "E1_preprocess_examples.png", dpi=110); plt.close(fig)


def arm_x(variant: str = "a") -> ArmData:
    """variant 'a': tổ hợp có điểm chọn cao nhất; 'b': tổ hợp tốt nhất không xử lý ảnh lúc suy luận."""
    selj = json.loads((OUT / "selection.json").read_text())["selected" if variant == "a" else "selected_b"]; level, aug = selj["level"], selj["aug"]
    X, y, b = build_sets(level, aug)
    st = {k: process_batch(v, level) for k, v in stress_sets("test").items()}
    name, sub = ("X", "pipeline") if variant == "a" else ("Xb", "pipeline_b")
    return ArmData(name, OUT / sub, X, y, process_batch(b["val"][0], level), b["val"][1], process_batch(b["test"][0], level), b["test"][1],
                   stress=st, note=f"làm sạch; tiền xử lý: {LEVEL_NAME[level].lower()}; tăng cường train: {MODE_NAME[aug].lower()}")


def run():
    selj = json.loads((OUT / "selection.json").read_text())
    fig_examples(selj["selected"]["level"], selj["selected"]["aug"])
    run_arm(arm_x("a"))
    # X-b là biến thể đối chiếu: mặc định chỉ chạy 4 cấu hình truyền thống và hai CNN đại diện (a_vgg16, b_mobilenetv3)
    # để giới hạn thời gian trên CPU; chạy đủ 9 cấu hình: python -m src.exp_arm_x --only RUN --xb-all
    if "selected_b" in selj and selj["selected_b"] != selj["selected"]:
        from .pipeline import TRAD
        run_arm(arm_x("b"), None if XB_ALL else TRAD + ["a_vgg16", "b_mobilenetv3"])


XB_ALL = False
EXPS = {"X1": x1, "X2": x2, "X3": x3, "RUN": run}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(EXPS))
    ap.add_argument("--xb-all", action="store_true", help="chạy đủ 9 cấu hình cho biến thể X-b")
    a = ap.parse_args()
    global XB_ALL
    XB_ALL = a.xb_all
    exp_dir(EXP)
    for k in (a.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
