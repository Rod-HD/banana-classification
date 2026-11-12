"""Tăng cường dữ liệu của nhánh X (mục 12.2 của kế hoạch), chỉ áp dụng trên tập train.

  geo      hình học + quang học: lật ngang/dọc, xoay bội số 90°, xoay ±15°, độ sáng ±15%, làm mờ Gauss,
           nhiễu muối tiêu trên tối đa 1,5% điểm ảnh (cùng loại phép với công thức tác giả bộ dữ liệu công bố)
  geo+bg   như trên, thêm thay nền ngẫu nhiên: ghép quả chuối lên nền lấy từ một ảnh train bất kỳ (mọi lớp),
           để màu và kết cấu nền không còn đi kèm với nhãn
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import SEED
from .stress import background_of, paste

MODES = ["none", "geo", "geo+bg"]
MODE_NAME = {"none": "Không tăng cường", "geo": "Hình học + quang học", "geo+bg": "Hình học + quang học + thay nền ngẫu nhiên"}


def photo_geo(img, rng):
    out = img
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1]
    out = np.ascontiguousarray(np.rot90(out, int(rng.integers(0, 4))))
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-15, 15)), 1.0)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.85, 1.15), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.25)))
    frac = rng.uniform(0, 0.015)
    n = int(frac * h * w)
    if n:
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        out = out.copy(); out[ys, xs] = np.where(rng.random((n, 1)) < 0.5, 0, 255).astype(np.uint8)
    return out


def background_pool(X, masks, n: int = 200, seed: int = SEED):
    """Nền (đã lấp vùng chuối) của n ảnh train chọn ngẫu nhiên."""
    rng = np.random.default_rng([seed, 5])
    pick = rng.choice(len(X), min(n, len(X)), replace=False)
    return [background_of(np.asarray(X[i]), masks[i]) for i in pick]


def make_augmented(X, y, mode: str = "geo", k: int = 2, masks=None, seed: int = SEED):
    """Tạo k bản tăng cường cho mỗi ảnh train. Trả về (X_aug, y_aug) với k·N ảnh (không gồm ảnh gốc)."""
    if mode == "none":
        return np.empty((0,) + X.shape[1:], np.uint8), np.empty(0, y.dtype)
    pool = background_pool(X, masks) if mode == "geo+bg" else None
    out = np.empty((k * len(X),) + X.shape[1:], np.uint8)
    for i in range(len(X)):
        for j in range(k):
            rng = np.random.default_rng([seed, 31, i, j])
            im = np.asarray(X[i])
            if pool is not None:
                g = rng.uniform(0.85, 1.1)
                bg = np.clip(pool[int(rng.integers(len(pool)))].astype(np.float32) * g, 0, 255).astype(np.uint8)
                im = paste(im, masks[i], bg)
            out[i * k + j] = photo_geo(im, rng)
    return out, np.tile(y[:, None], (1, k)).ravel()
