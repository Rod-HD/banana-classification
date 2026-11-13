"""Phân tích lỗi trên tập test từ dự đoán đã lưu ở results/final/test_predictions.npz (không chạy lại mô hình).

    python -m src.error_analysis
Kết quả: results/final/table_error_pairs.md, results/final/table_error_context.md, hình results/figures/final_errors_all.png.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_NAMES, FIG_DIR, RESULTS_DIR
from .data import list_images, load_all_images
from .exp_segment import all_masks
from .report_utils import save_table

SHORT = [c.replace(" Kola", "") for c in CLASS_NAMES]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    z = np.load(RESULTS_DIR / "final" / "test_predictions.npz")
    idx, y = z["idx"], z["y"]; models = [k for k in z.files if k not in ("idx", "y")]
    P = {k: z[k] for k in models}
    wrong = np.array([P[k] != y for k in models])            # (số mô hình, số ảnh)
    n_wrong = wrong.sum(0)
    # 1) cặp lớp bị nhầm, cộng dồn trên mọi mô hình
    pairs = {}
    for k in models:
        for t, p in zip(y[P[k] != y], P[k][P[k] != y]):
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    rows = [{"Thực tế": CLASS_NAMES[t], "Đoán thành": CLASS_NAMES[p], "Số lượt sai (cộng trên 9 mô hình)": n}
            for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])]
    print(save_table("final", rows, "table_error_pairs", title="Các cặp lớp bị nhầm trên test, cộng dồn qua 9 mô hình đã chốt"))
    # 2) bối cảnh của ảnh bị sai: diện tích quả chuối trong ảnh so với trung bình của lớp đúng
    masks = all_masks(); imgs = load_all_images(256)
    area = np.array([(masks[i] > 0).mean() for i in idx])
    df = list_images().set_index("idx")
    ctx = []
    for c in range(4):
        sel = y == c
        ctx.append({"Lớp": CLASS_NAMES[c], "Số ảnh test": int(sel.sum()), "Ảnh bị ≥ 1 mô hình đoán sai": int((sel & (n_wrong > 0)).sum()),
                    "Diện tích quả TB (ảnh đúng hết)": f"{area[sel & (n_wrong == 0)].mean():.3f}",
                    "Diện tích quả TB (ảnh bị sai)": f"{area[sel & (n_wrong > 0)].mean():.3f}" if (sel & (n_wrong > 0)).any() else "-"})
    print(save_table("final", ctx, "table_error_context", title="Diện tích quả chuối (tỉ lệ trong ảnh, theo mask S1) của ảnh test bị đoán sai so với ảnh đoán đúng"))
    # 3) hình: mọi ảnh test bị ít nhất một mô hình đoán sai, sắp theo số mô hình sai
    order = [j for j in np.argsort(-n_wrong, kind="stable") if n_wrong[j] > 0][:24]
    cols = 6; rws = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(2.4 * cols, 2.9 * rws), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, j in zip(axes.ravel(), order):
        ax.imshow(imgs[idx[j]])
        guess = {}
        for k in models:
            if P[k][j] != y[j]:
                guess.setdefault(SHORT[P[k][j]], []).append(k)
        g = ", ".join(f"{c}×{len(v)}" for c, v in guess.items())
        ax.set_title(f"đúng: {SHORT[y[j]]} | sai: {n_wrong[j]}/9\nđoán: {g}", fontsize=7.5)
    fig.suptitle(f"Ảnh test bị ít nhất 1 trong 9 mô hình đoán sai ({int((n_wrong > 0).sum())} ảnh; hiện tối đa 24, sắp theo số mô hình sai)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "final_errors_all.png", dpi=110); plt.close(fig)
    print("số ảnh sai ít nhất 1 mô hình:", int((n_wrong > 0).sum()))


if __name__ == "__main__":
    main()
