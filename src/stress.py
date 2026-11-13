"""Bộ kiểm tra khó dùng để so sánh các nhánh (mục 12.1 của kế hoạch): 6 phiên bản biến đổi của tập chấm.
Phép biến đổi áp lên ảnh thô 256×256, trước mọi bước tiền xử lý riêng của từng nhánh."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CACHE_DIR, SEED
from .data import load_all_images, load_split
from .transforms import apply_transform

STRESS = ["swapbg", "v0.6", "v1.4", "noise10", "rot30", "occlude25"]
STRESS_LABEL = {"clean": "Ảnh gốc", "swapbg": "Nền hoán đổi", "v0.6": "V×0.6", "v1.4": "V×1.4", "noise10": "Nhiễu σ=10",
                "rot30": "Xoay +30°", "occlude25": "Che 25%"}
_DIL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


def background_of(img, mask):
    """Nền của một ảnh: lấp vùng quả chuối (mask nới rộng 12 px) bằng inpainting."""
    return cv2.inpaint(img, cv2.dilate(mask, _DIL), 7, cv2.INPAINT_TELEA)


def paste(img, mask, bg):
    """Ghép quả chuối của `img` (theo mask, viền làm mềm) lên nền `bg`."""
    m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    a = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32)[..., None] / 255.0
    return np.clip(a * img.astype(np.float32) + (1 - a) * bg.astype(np.float32), 0, 255).astype(np.uint8)


def swap_background(ids, labels, seed: int = SEED):
    """Mỗi ảnh được ghép lên nền của một ảnh khác lớp, chọn ngẫu nhiên trong cùng tập."""
    from .exp_segment import all_masks_v2
    imgs, masks = load_all_images(256), all_masks_v2()
    rng = np.random.default_rng([seed, 99])
    out = np.empty((len(ids), 256, 256, 3), np.uint8)
    for k, (i, y) in enumerate(zip(ids, labels)):
        j = int(rng.choice(ids[labels != y]))
        out[k] = paste(np.asarray(imgs[i]), masks[i], background_of(np.asarray(imgs[j]), masks[j]))
    return out


def stress_sets(split: str = "test") -> dict:
    """{tên phép: mảng ảnh uint8} cho tập val hoặc test; cache vào data/cache/."""
    f = CACHE_DIR / f"stress_{split}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in STRESS}
    df = load_split(split); ids, y = df["idx"].values, df["label_id"].values
    raw = np.stack([load_all_images(256)[i] for i in ids])
    out = {"swapbg": swap_background(ids, y)}
    for name in STRESS[1:]:
        out[name] = apply_transform(name, raw, ids)
    np.savez(f, **out)
    return out


def fig_examples():
    """Hình minh họa bộ kiểm tra khó: mỗi lớp một ảnh test, ảnh gốc và 6 phiên bản biến đổi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .config import CLASS_NAMES, FIG_DIR
    df = load_split("test"); ids, y = df["idx"].values, df["label_id"].values
    st = stress_sets("test"); raw = load_all_images(256)
    pick = [int(np.where(y == c)[0][7]) for c in range(4)]
    names = ["clean"] + STRESS
    fig, axes = plt.subplots(4, len(names), figsize=(1.9 * len(names), 7.8))
    for r, k in enumerate(pick):
        for c, n in enumerate(names):
            axes[r, c].imshow(raw[ids[k]] if n == "clean" else st[n][k]); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(STRESS_LABEL[n], fontsize=9)
        axes[r, 0].set_ylabel(CLASS_NAMES[r], fontsize=9)
    fig.suptitle("Bộ kiểm tra khó: ảnh test gốc và 6 phiên bản biến đổi (nền hoán đổi = ghép quả lên nền của ảnh thuộc lớp khác)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "E_stress_examples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    fig_examples()
    print("đã ghi results/figures/E_stress_examples.png")
