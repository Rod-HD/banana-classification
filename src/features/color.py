"""Đặc trưng màu: histogram RGB, histogram HSV, histogram H-S, moment màu,
màu chủ đạo (k-means) và Color Coherence Vector."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label as cc_label


def rgb_hist(img, bins: int = 32, mask=None):
    """Histogram từng kênh R, G, B (mỗi kênh `bins` bin), mỗi kênh chuẩn hóa tổng = 1."""
    out = []
    for c in range(3):
        h = cv2.calcHist([img], [c], mask, [bins], [0, 256]).ravel()
        out.append(h / max(h.sum(), 1e-9))
    return np.concatenate(out)


def hsv_hist(img, bins=(8, 8, 8), mask=None):
    """Histogram 3 chiều trên HSV (OpenCV: H∈[0,180), S,V∈[0,256)), chuẩn hóa tổng = 1."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1e-9)


def hs_hist(img, bins=(8, 8), mask=None):
    """Histogram 2 chiều chỉ dùng H và S (bỏ kênh độ sáng V)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256]).ravel()
    return h / max(h.sum(), 1e-9)


def color_moments(img, mask=None):
    """Moment màu trên HSV: mean, std, skewness của từng kênh (9 chiều)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    if mask is not None:
        hsv = hsv[mask.reshape(-1) > 0]
        if len(hsv) == 0:
            return np.zeros(9)
    x = hsv.astype(np.float32) * np.array([1 / 179.0, 1 / 255.0, 1 / 255.0], np.float32)
    mean = x.mean(0, dtype=np.float64)
    d = x - mean.astype(np.float32)
    std = np.sqrt((d * d).mean(0, dtype=np.float64))
    skew = np.cbrt((d * d * d).mean(0, dtype=np.float64))
    return np.concatenate([mean, std, skew])


def dominant_colors(img, k: int = 3, size: int = 64):
    """Màu chủ đạo: k-means (k=3) trên điểm ảnh RGB của ảnh thu nhỏ; mỗi cụm cho
    (tỉ lệ, R, G, B), sắp theo tỉ lệ giảm dần -> 4k chiều."""
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    Z = small.reshape(-1, 3).astype(np.float32) / 255.0
    cv2.setRNGSeed(42)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
    _, lab, cen = cv2.kmeans(Z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    prop = np.bincount(lab.ravel(), minlength=k) / len(Z)
    order = np.argsort(-prop)
    return np.concatenate([np.r_[prop[i], cen[i]] for i in order])


def _quantize(img, n_levels: int = 4, blur: bool = True):
    if blur:
        img = cv2.blur(img, (3, 3))
    q = (img.astype(np.int32) * n_levels) // 256
    return q[..., 0] * n_levels * n_levels + q[..., 1] * n_levels + q[..., 2]


def quant_hist(img, n_levels: int = 4):
    """Histogram trên cùng mức lượng tử hóa với CCV (n_levels^3 màu) để so sánh công bằng."""
    idx = _quantize(img, n_levels)
    h = np.bincount(idx.ravel(), minlength=n_levels ** 3).astype(np.float64)
    return h / h.sum()


def ccv(img, n_levels: int = 4, tau_frac: float = 0.01):
    """Color Coherence Vector: với mỗi màu lượng tử, đếm số điểm ảnh 'liên kết' (thuộc vùng
    liên thông 8 hướng có diện tích >= tau) và 'không liên kết'. Vector 2·n_levels^3 chiều."""
    idx = _quantize(img, n_levels)
    n_colors = n_levels ** 3
    lab = cc_label(idx, connectivity=2, background=-1)
    sizes = np.bincount(lab.ravel())
    tau = tau_frac * idx.size
    coherent = sizes[lab] >= tau
    alpha = np.bincount(idx[coherent], minlength=n_colors)
    beta = np.bincount(idx[~coherent], minlength=n_colors)
    return np.concatenate([alpha, beta]).astype(np.float64) / idx.size
