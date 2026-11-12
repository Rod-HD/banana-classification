"""Bước 1: chia dữ liệu stratified 70/15/15 (seed 42), kiểm tra ảnh gần trùng bằng
perceptual hash và thống kê dữ liệu.

    python -m src.split_data
"""
from __future__ import annotations

import argparse
import json
import time

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import CLASS_NAMES, CLASSES, FIG_DIR, RESULTS_DIR, ROOT, SEED, SPLIT_DIR
from .data import list_images, load_all_images
from .metrics import write_json


def make_splits(df: pd.DataFrame, seed: int = SEED):
    tr, rest = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["label_id"], random_state=seed)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part = part.sort_values("idx").reset_index(drop=True)
        part.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        out[name] = part
    return out


def phash_all(df):
    hashes = []
    for p in df["path"]:
        with Image.open(ROOT / p) as im:
            hashes.append(imagehash.phash(im.convert("RGB"), hash_size=8).hash.flatten())
    return np.array(hashes, dtype=np.uint8)  # (N, 64) bit


def pairwise_hamming(bits):
    b = bits.astype(np.int32)
    # khoảng cách Hamming = số bit khác nhau = a·(1-b) + (1-a)·b
    return b @ (1 - b).T + (1 - b) @ b.T


def near_duplicates(df, split_of):
    t0 = time.time()
    bits = phash_all(df)
    D = pairwise_hamming(bits)
    iu = np.triu_indices(len(df), k=1)
    d = D[iu]
    sp = split_of[iu[0]], split_of[iu[1]]
    same_cls = df["label_id"].values[iu[0]] == df["label_id"].values[iu[1]]
    rep = {"n_images": int(len(df)), "hash": "phash 64 bit", "time_s": round(time.time() - t0, 1),
           "thresholds": {}}
    for th in (0, 2, 4, 6, 8):
        m = d <= th
        cross = m & (sp[0] != sp[1])
        rep["thresholds"][f"<= {th}"] = {
            "pairs_total": int(m.sum()),
            "pairs_same_class": int((m & same_cls).sum()),
            "pairs_diff_class": int((m & ~same_cls).sum()),
            "pairs_cross_split": int(cross.sum()),
            "pairs_train_test": int((m & (((sp[0] == "train") & (sp[1] == "test")) | ((sp[0] == "test") & (sp[1] == "train")))).sum()),
            "pairs_train_val": int((m & (((sp[0] == "train") & (sp[1] == "val")) | ((sp[0] == "val") & (sp[1] == "train")))).sum()),
            "pairs_val_test": int((m & (((sp[0] == "val") & (sp[1] == "test")) | ((sp[0] == "test") & (sp[1] == "val")))).sum()),
            "test_images_with_neighbor_in_train": int(len(set(
                np.r_[iu[0][m & (sp[0] == "test") & (sp[1] == "train")], iu[1][m & (sp[1] == "test") & (sp[0] == "train")]].tolist()))),
        }
    hist = np.bincount(d, minlength=65)
    rep["distance_histogram_first_16"] = hist[:16].tolist()
    order = np.argsort(d, kind="stable")[:12]
    rep["closest_pairs"] = [{"a": df["path"].iloc[iu[0][k]], "b": df["path"].iloc[iu[1][k]],
                             "dist": int(d[k]), "split_a": sp[0][k], "split_b": sp[1][k]} for k in order]
    return rep, D


def near_dup_groups(D, th):
    """Nhóm ảnh gần trùng = thành phần liên thông của đồ thị nối các cặp có Hamming <= th."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    A = csr_matrix(D <= th)
    _, lab = connected_components(A, directed=False)
    return lab


def fig_pairs_by_distance(df, imgs, D, path, dists=(2, 4, 6, 8), n=8):
    iu = np.triu_indices(len(df), k=1)
    d = D[iu]
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2 * len(dists), n, figsize=(2 * n, 4.3 * len(dists)))
    for r, th in enumerate(dists):
        ks = rng.choice(np.where(d == th)[0], n, replace=False)
        for j, k in enumerate(ks):
            a, b = iu[0][k], iu[1][k]
            axes[2 * r, j].imshow(imgs[a]); axes[2 * r + 1, j].imshow(imgs[b])
            axes[2 * r, j].axis("off"); axes[2 * r + 1, j].axis("off")
        axes[2 * r, 0].set_title(f"Hamming = {th} (cặp: hàng trên ↔ hàng dưới)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def fig_closest_pairs(rep, path):
    pairs = rep["closest_pairs"][:8]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for j, pr in enumerate(pairs):
        for i, key in enumerate(("a", "b")):
            ax = axes[i, j]
            ax.imshow(Image.open(ROOT / pr[key]).convert("RGB"))
            ax.set_title(f"{pr['split_' + key]}: {pr[key].split('/')[-1]}", fontsize=7)
            ax.axis("off")
        axes[1, j].text(0.5, -0.12, f"Hamming = {pr['dist']}", transform=axes[1, j].transAxes,
                        ha="center", fontsize=8)
    fig.suptitle("8 cặp ảnh có perceptual hash gần nhau nhất (hàng trên ↔ hàng dưới)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def data_stats(df, splits, imgs):
    counts = {s: splits[s]["label"].value_counts().reindex(CLASSES).astype(int).tolist() for s in splits}
    stats = {"class_names": CLASS_NAMES, "counts_per_split": counts,
             "total_per_class": df["label"].value_counts().reindex(CLASSES).astype(int).tolist()}
    # màu trung bình (RGB) mỗi lớp trên ảnh 256×256
    stats["mean_rgb_per_class"] = {CLASS_NAMES[c]: imgs[df["label_id"].values == c].reshape(-1, 3).mean(0).round(1).tolist()
                                   for c in range(len(CLASSES))}
    # biểu đồ số ảnh theo lớp và tập
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(CLASSES)); w = 0.27
    for k, s in enumerate(["train", "val", "test"]):
        ax.bar(x + (k - 1) * w, counts[s], w, label=f"{s} ({sum(counts[s])})")
        for xi, v in zip(x + (k - 1) * w, counts[s]):
            ax.text(xi, v + 5, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố số ảnh theo lớp sau khi chia 70/15/15"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "data_split_counts.png", dpi=120); plt.close(fig)
    # lưới ảnh mẫu
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(4, 6, figsize=(12, 8.4))
    for c in range(4):
        ids = rng.choice(np.where(df["label_id"].values == c)[0], 6, replace=False)
        for j, i in enumerate(ids):
            axes[c, j].imshow(imgs[i]); axes[c, j].axis("off")
        axes[c, 0].set_title(CLASS_NAMES[c], loc="left", fontsize=11)
    fig.suptitle("Ảnh mẫu của 4 giống chuối")
    fig.tight_layout(); fig.savefig(FIG_DIR / "data_samples.png", dpi=110); plt.close(fig)
    return stats


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = list_images()
    splits = make_splits(df)
    split_of = np.empty(len(df), dtype=object)
    for s, part in splits.items():
        split_of[part["idx"].values] = s
    imgs = load_all_images(256)
    out = RESULTS_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    stats = data_stats(df, splits, imgs)
    write_json(out / "data_stats.json", stats)
    rep, D = near_duplicates(df, split_of)
    for th in (2, 4):
        g = near_dup_groups(D, th)
        sizes = np.bincount(g)
        rep[f"groups_th{th}"] = {"n_groups": int(len(sizes)), "n_groups_size_gt1": int((sizes > 1).sum()),
                                 "images_in_groups_size_gt1": int(sizes[sizes > 1].sum()),
                                 "largest_group": int(sizes.max())}
        df[f"group_th{th}"] = g
    df[["idx", "path", "label", "group_th2", "group_th4"]].to_csv(out / "near_dup_groups.csv", index=False)
    write_json(out / "near_duplicates.json", rep)
    fig_pairs_by_distance(df, imgs, D, FIG_DIR / "near_duplicates_by_distance.png")
    fig_closest_pairs(rep, FIG_DIR / "near_duplicates_closest_pairs.png")
    print(json.dumps(stats["counts_per_split"]), json.dumps(rep["thresholds"], indent=1))


if __name__ == "__main__":
    main()
