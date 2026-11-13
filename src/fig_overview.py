"""Hai hình minh họa dùng trong báo cáo và slide:

  results/figures/problem_io.png        phát biểu bài toán: ảnh vào, mô hình f, xác suất 4 lớp và nhãn ra
  results/figures/method_overview.png   sơ đồ hai hướng giải quyết (truyền thống và học sâu)

    python -m src.fig_overview
Ảnh minh họa là 4 ảnh test trong app/examples; xác suất do mô hình models/b_mobilenetv3.keras tính ra.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import CLASS_NAMES, FIG_DIR, MODELS_DIR, ROOT
from .data import read_rgb

BLUE, ORANGE, GRAY = "#1E489F", "#F68921", "#4A4A4A"


def problem_io():
    import keras
    files = sorted((ROOT / "app" / "examples").glob("*.jpg"))
    model = keras.models.load_model(MODELS_DIR / "b_mobilenetv3.keras")
    fig, axes = plt.subplots(len(files), 3, figsize=(9.2, 1.75 * len(files)), gridspec_kw={"width_ratios": [1.0, 0.9, 2.2]})
    for r, f in enumerate(files):
        img = read_rgb(f, 256)
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)[None]
        p = np.asarray(model(x, training=False))[0]
        a = axes[r, 0]; a.imshow(img); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(f"x{r + 1}", rotation=0, labelpad=14, fontsize=11, va="center")
        if r == 0:
            a.set_title("Input: ảnh màu x\n(256 × 256 × 3)", fontsize=10, color=BLUE)
        a = axes[r, 1]; a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.add_patch(FancyBboxPatch((0.22, 0.25), 0.56, 0.5, boxstyle="round,pad=0.03", fc="#EAF0FA", ec=BLUE, lw=1.5))
        a.text(0.5, 0.5, "f", ha="center", va="center", fontsize=20, style="italic", color=BLUE)
        for x0, x1 in [(0.0, 0.17), (0.83, 1.0)]:
            a.add_patch(FancyArrowPatch((x0, 0.5), (x1, 0.5), arrowstyle="-|>", mutation_scale=14, color=GRAY, lw=1.5))
        if r == 0:
            a.set_title("Mô hình phân loại", fontsize=10, color=BLUE)
        a = axes[r, 2]
        k = int(p.argmax())
        bars = a.barh(range(4), p, color=[ORANGE if i == k else "#B8C4D9" for i in range(4)])
        a.set_yticks(range(4), CLASS_NAMES, fontsize=9); a.invert_yaxis(); a.set_xlim(0, 1.8); a.set_xticks([0, 0.5, 1.0])
        a.tick_params(axis="x", labelsize=8)
        for i, b in enumerate(bars):
            a.text(p[i] + 0.02, i, f"{p[i]:.3f}", va="center", fontsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        if r == 0:
            a.set_title("Output: vector xác suất p = f(x) và nhãn ŷ = argmax p", fontsize=10, color=BLUE)
        a.text(1.22, k, f"ŷ = {CLASS_NAMES[k]}", ha="left", va="center", fontsize=10, color=ORANGE, weight="bold")
    fig.tight_layout()
    out = FIG_DIR / "problem_io.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print("đã ghi", out.relative_to(ROOT))


def _box(ax, x, y, w, h, text, fc="#EAF0FA", ec=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.3))


def method_overview():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    img = read_rgb(sorted((ROOT / "app" / "examples").glob("*.jpg"))[3], 256)
    ax.imshow(img, extent=(0.01, 0.13, 0.36, 0.64), aspect="auto", zorder=2)
    ax.text(0.07, 0.69, "Ảnh vào x", ha="center", fontsize=9.5, color=BLUE, weight="bold")
    ax.text(0.07, 0.31, "256 × 256 × 3", ha="center", fontsize=8, color=GRAY)
    # hướng truyền thống
    ax.text(0.47, 0.965, "Hướng 1: truyền thống (đặc trưng thiết kế bằng tay + bộ phân lớp)", ha="center", fontsize=10, color=BLUE, weight="bold")
    _box(ax, 0.20, 0.62, 0.27, 0.27, "Rút feature vector\nMàu: histogram RGB, HSV, H-S,\nmoment màu, màu chủ đạo, CCV\nHình dạng: cạnh (Sobel, Prewitt,\nRoberts, Laplacian), HOG", fs=8.5)
    _box(ax, 0.52, 0.66, 0.11, 0.19, "Chuẩn hóa,\nghép\nđặc trưng", fs=8.5)
    _box(ax, 0.68, 0.62, 0.14, 0.27, "Bộ phân lớp\nKNN\nLogistic Regression\nLinear SVM", fs=8.5)
    # hướng học sâu
    ax.text(0.47, 0.42, "Hướng 2: học sâu (CNN học đặc trưng từ dữ liệu)", ha="center", fontsize=10, color=ORANGE, weight="bold")
    _box(ax, 0.20, 0.08, 0.27, 0.27, "Hướng A (theo bài giảng)\nLeNet và CNN có Dropout: học từ đầu\nVGG16 đóng băng + lớp Dense", fc="#FFF3E6", ec=ORANGE, fs=8.5)
    _box(ax, 0.52, 0.08, 0.30, 0.27, "Hướng B (transfer learning)\nMobileNetV3-Small, ResNet50\ngiai đoạn 1: đóng băng thân mạng\ngiai đoạn 2: fine-tune", fc="#FFF3E6", ec=ORANGE, fs=8.5)
    # đầu ra
    _box(ax, 0.87, 0.36, 0.12, 0.28, "Output\np ∈ [0,1]⁴\nŷ = argmax p\n(1 trong 4 giống)", fc="#F4F7FB", ec=GRAY, fs=8.5, bold=True)
    _arrow(ax, (0.135, 0.56), (0.195, 0.72)); _arrow(ax, (0.135, 0.44), (0.195, 0.24))
    _arrow(ax, (0.475, 0.755), (0.515, 0.755)); _arrow(ax, (0.635, 0.755), (0.675, 0.755))
    _arrow(ax, (0.825, 0.72), (0.865, 0.58)); _arrow(ax, (0.825, 0.24), (0.865, 0.42))
    ax.text(0.5, 0.005, "Hai hướng dùng chung một cách chia dữ liệu (train / val / test) và một bộ độ đo (macro-F1, accuracy, precision, recall, ma trận nhầm lẫn)",
            ha="center", fontsize=8.5, color=GRAY)
    out = FIG_DIR / "method_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("đã ghi", out.relative_to(ROOT))


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    problem_io()
    method_overview()


if __name__ == "__main__":
    main()
