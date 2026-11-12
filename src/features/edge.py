"""Đặc trưng cạnh: gradient bằng Roberts, Prewitt, Sobel, Laplacian; vector đặc trưng là
tổng độ lớn gradient theo từng hàng và từng cột của ảnh 256×256 (512 chiều)."""
from __future__ import annotations

import cv2
import numpy as np

KERNELS = {
    "roberts": (np.array([[1, 0], [0, -1]], np.float32), np.array([[0, 1], [-1, 0]], np.float32)),
    "prewitt": (np.array([[-1, 0, 1]] * 3, np.float32), np.array([[-1, 0, 1]] * 3, np.float32).T),
    "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
              np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32).T),
    "laplacian": (np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32), None),
}


def gradient_magnitude(gray: np.ndarray, op: str = "sobel") -> np.ndarray:
    """Độ lớn gradient của ảnh xám (float32). Laplacian: lấy trị tuyệt đối đạo hàm bậc hai."""
    g = gray.astype(np.float32)
    kx, ky = KERNELS[op]
    gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
    if ky is None:
        return np.abs(gx)
    gy = cv2.filter2D(g, cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(gx * gx + gy * gy)


def edge_vector(img, op: str = "sobel"):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    if gray.shape != (256, 256):
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    mag = gradient_magnitude(gray, op)
    return np.concatenate([mag.sum(1), mag.sum(0)])
