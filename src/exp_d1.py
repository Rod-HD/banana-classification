"""Thí nghiệm D1: rò rỉ dữ liệu khi tăng cường ảnh trước lúc chia tập.

Chỉ dùng tập phát triển (train+val), không chạm vào tập test chính thức. Với mỗi lần lặp:
  (a) tăng cường rồi chia: gộp ảnh gốc + bản lật + bản xoay, chia ngẫu nhiên 80/20;
  (b) chia rồi tăng cường: chia ảnh gốc 80/20, chỉ tăng cường phần huấn luyện, chấm trên ảnh gốc;
  (c) chia theo ảnh gốc rồi tăng cường cả hai phía: phần chấm có cùng thành phần với (a)
      nhưng không có bản sao của ảnh huấn luyện -> (a) − (c) là mức thổi phồng thuần do rò rỉ.
Phần phụ D1b: ảnh gần trùng tự nhiên (perceptual hash) — so StratifiedKFold với
StratifiedGroupKFold giữ các ảnh gần trùng trong cùng một fold.

    python -m src.exp_d1
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .classical import cv_eval, make_clf
from .config import FIG_DIR, RESULTS_DIR, SEED
from .data import dev_and_test, load_all_images
from .features import extract
from .report_utils import exp_dir, pm, save_json, save_table
from .transforms import hflip, rotate

MODELS = [("hsv_hist", {}, "knn", {"n_neighbors": 1}), ("hsv_hist", {}, "logreg", {}),
          ("hog", {}, "knn", {"n_neighbors": 1}), ("hog", {}, "logreg", {}),
          ("edge", {}, "knn", {"n_neighbors": 1})]


def augmented_sets(imgs, ids):
    """Hai bản tăng cường cho mỗi ảnh: lật ngang, xoay góc ngẫu nhiên trong ±[10°, 30°]."""
    flip = np.stack([hflip(np.asarray(imgs[i])) for i in ids])
    rot = np.empty_like(flip)
    for k, i in enumerate(ids):
        rng = np.random.default_rng([SEED, 7, int(i)])
        ang = rng.uniform(10, 30) * rng.choice([-1, 1])
        rot[k] = rotate(np.asarray(imgs[i]), angle=float(ang))
    return flip, rot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    imgs = load_all_images(256)
    dev, _ = dev_and_test()
    ids = dev["idx"].values; y = dev["label_id"].values; n = len(ids)
    dev_imgs = np.stack([imgs[i] for i in ids])
    flip, rot = augmented_sets(imgs, ids)

    rows, raw = [], {}
    for feat, fp, kind, cp in MODELS:
        X0 = extract(feat, dev_imgs, "dev", **fp)
        X1 = extract(feat, flip, "dev_flip", **fp)
        X2 = extract(feat, rot, "dev_rot", **fp)
        Xall = np.concatenate([X0, X1, X2]); yall = np.tile(y, 3); orig = np.tile(np.arange(n), 3)
        res = {k: {"acc": [], "f1": []} for k in "abc"}
        leak_frac = []
        for r in range(args.repeats):
            seed = SEED + r
            # (a) tăng cường rồi chia ngẫu nhiên
            tr, te = train_test_split(np.arange(3 * n), test_size=0.2, stratify=yall, random_state=seed)
            clf = make_clf(kind, **cp).fit(Xall[tr], yall[tr]); p = clf.predict(Xall[te])
            res["a"]["acc"].append(accuracy_score(yall[te], p)); res["a"]["f1"].append(f1_score(yall[te], p, average="macro"))
            leak_frac.append(np.isin(orig[te], orig[tr]).mean())
            # (b), (c): chia ảnh gốc trước
            otr, ote = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=seed)
            trm = np.isin(orig, otr)
            clf = make_clf(kind, **cp).fit(Xall[trm], yall[trm])
            p = clf.predict(X0[ote])
            res["b"]["acc"].append(accuracy_score(y[ote], p)); res["b"]["f1"].append(f1_score(y[ote], p, average="macro"))
            p = clf.predict(Xall[~trm])
            res["c"]["acc"].append(accuracy_score(yall[~trm], p)); res["c"]["f1"].append(f1_score(yall[~trm], p, average="macro"))
        name = f"{feat} + {kind}" + (" (k=1)" if kind == "knn" else "")
        raw[name] = {k: {m: [float(x) for x in v] for m, v in d.items()} for k, d in res.items()}
        raw[name]["test_images_with_copy_in_train_frac"] = float(np.mean(leak_frac))
        A, B, C = (np.array(res[k]["acc"]) for k in "abc")
        rows.append({"Mô hình": name,
                     "(a) tăng cường rồi chia": pm(A.mean(), A.std()),
                     "(b) chia rồi tăng cường train": pm(B.mean(), B.std()),
                     "(c) chia theo ảnh gốc, tăng cường 2 phía": pm(C.mean(), C.std()),
                     "Thổi phồng (a)−(b)": f"{(A - B).mean():+.3f} ± {(A - B).std():.3f}",
                     "Thổi phồng (a)−(c)": f"{(A - C).mean():+.3f} ± {(A - C).std():.3f}",
                     "Macro-F1 (a)": pm(np.mean(res['a']['f1']), np.std(res['a']['f1'])),
                     "Macro-F1 (b)": pm(np.mean(res['b']['f1']), np.std(res['b']['f1']))})
        print(rows[-1], flush=True)

    # D1b: ảnh gần trùng tự nhiên
    grp = pd.read_csv(RESULTS_DIR / "data" / "near_dup_groups.csv").set_index("idx")
    rows_b = []
    for feat, fp, kind, cp in MODELS:
        X = extract(feat, dev_imgs, "dev", **fp)
        r0 = cv_eval(X, y, lambda: make_clf(kind, **cp))
        row = {"Mô hình": f"{feat} + {kind}" + (" (k=1)" if kind == "knn" else ""),
               "StratifiedKFold": pm(r0["macro_f1"], r0["macro_f1_std"])}
        for th in (2, 4):
            g = grp.loc[ids, f"group_th{th}"].values
            rg = cv_eval(X, y, lambda: make_clf(kind, **cp), groups=g)
            row[f"GroupKFold (Hamming ≤ {th})"] = pm(rg["macro_f1"], rg["macro_f1_std"])
            row[f"Chênh (≤ {th})"] = f"{r0['macro_f1'] - rg['macro_f1']:+.3f}"
        rows_b.append(row); print(row, flush=True)

    save_json(exp_dir("D1") / "metrics.json", {"repeats": args.repeats, "n_dev_images": int(n),
                                                "augmentations": ["hflip", "rotate ±[10,30] độ"],
                                                "accuracy_and_f1_per_repeat": raw,
                                                "table": rows, "near_duplicate_cv_macro_f1": rows_b})
    save_table("D1", rows, "table", title="D1 — Accuracy (trung bình ± độ lệch chuẩn qua %d lần chia 80/20)" % args.repeats)
    save_table("D1", rows_b, "table_near_duplicates", title="D1b — Macro-F1 5-fold: chia thường so với chia giữ nhóm ảnh gần trùng")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(rows)); w = 0.27
    for k, (key, lab) in enumerate([("a", "(a) tăng cường rồi chia"), ("b", "(b) chia rồi tăng cường train"),
                                    ("c", "(c) chia theo ảnh gốc, tăng cường 2 phía")]):
        m = [np.mean(raw[r["Mô hình"]][key]["acc"]) for r in rows]
        s = [np.std(raw[r["Mô hình"]][key]["acc"]) for r in rows]
        ax.bar(x + (k - 1) * w, m, w, yerr=s, capsize=3, label=lab)
        for xi, v in zip(x + (k - 1) * w, m):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([r["Mô hình"] for r in rows], fontsize=8)
    ax.set_ylim(0.8, 1.04); ax.set_ylabel("Accuracy"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("D1 — Rò rỉ dữ liệu do tăng cường ảnh trước khi chia tập")
    fig.tight_layout(); fig.savefig(FIG_DIR / "D1_leakage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
