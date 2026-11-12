"""HOG tự cài đặt theo các bước trong bài giảng: resize -> gradient -> histogram hướng
theo cell -> chuẩn hóa theo block -> ghép vector. Xử lý theo lô bằng numpy.

Tham số `norm='none'` bỏ bước chuẩn hóa block (dùng cho thí nghiệm C2).
Với soft=False và color=False, kết quả trùng với skimage.feature.hog(block_norm='L2')
(kiểm tra bằng `python -m src.features.hog`)."""
from __future__ import annotations

import cv2
import numpy as np


def _resize_batch(imgs, size, color):
    w, h = size
    out = np.empty((len(imgs), h, w, 3 if color else 1), np.float32)
    for i, im in enumerate(imgs):
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(np.asarray(im), (w, h), interpolation=cv2.INTER_AREA)
        if color:
            out[i] = im
        else:
            out[i, ..., 0] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
    return out


def hog_cells(x, cell=8, bins=9, soft=True):
    """x: (N, H, W, C) float32. Trả về histogram hướng của từng cell: (N, H/cell, W/cell, bins)."""
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:, :, 1:-1] = x[:, :, 2:] - x[:, :, :-2]          # mặt nạ [-1, 0, 1]
    gy[:, 1:-1] = x[:, 2:] - x[:, :-2]
    mag = np.sqrt(gx * gx + gy * gy)
    if x.shape[-1] > 1:                                   # ảnh màu: lấy kênh có gradient lớn nhất
        k = mag.argmax(-1)[..., None]
        mag = np.take_along_axis(mag, k, -1); gx = np.take_along_axis(gx, k, -1); gy = np.take_along_axis(gy, k, -1)
    mag, gx, gy = mag[..., 0], gx[..., 0], gy[..., 0]
    ori = np.rad2deg(np.arctan2(gy, gx)) % 180.0          # hướng không dấu [0, 180)
    n, h, w = mag.shape
    ny, nx = h // cell, w // cell
    width = 180.0 / bins
    hist = np.zeros((n, ny, nx, bins), np.float32)

    def cellsum(a):
        return a[:, :ny * cell, :nx * cell].reshape(n, ny, cell, nx, cell).sum((2, 4))

    if soft:   # chia phiếu cho hai bin gần nhất theo khoảng cách tới tâm bin (như bài giảng)
        pos = ori / width - 0.5
        lo = np.floor(pos).astype(np.int32)
        w_hi = (pos - lo).astype(np.float32)
        lo_b = lo % bins; hi_b = (lo + 1) % bins
        for b in range(bins):
            hist[..., b] = cellsum(mag * (1 - w_hi) * (lo_b == b)) + cellsum(mag * w_hi * (hi_b == b))
    else:
        bi = np.minimum((ori / width).astype(np.int32), bins - 1)
        for b in range(bins):
            hist[..., b] = cellsum(mag * (bi == b))
    return hist / (cell * cell)


def hog_from_cells(hist, block=2, norm="L2", eps=1e-5):
    n, ny, nx, bins = hist.shape
    if norm == "none":
        return hist.reshape(n, -1)
    v = np.lib.stride_tricks.sliding_window_view(hist, (block, block), axis=(1, 2))  # (n, ny-b+1, nx-b+1, bins, b, b)
    v = np.moveaxis(v, 3, -1).reshape(n, ny - block + 1, nx - block + 1, -1)
    v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
    return v.reshape(n, -1)


def hog_batch(imgs, size=(64, 128), cell=8, bins=9, block=2, norm="L2", soft=True, color=False,
              batch=256):
    """size = (rộng, cao). Trả về (N, D) float32."""
    out = []
    for a in range(0, len(imgs), batch):
        x = _resize_batch(imgs[a:a + batch], size, color)
        out.append(hog_from_cells(hog_cells(x, cell, bins, soft), block, norm))
    return np.concatenate(out).astype(np.float32)


def hog_dim(size=(64, 128), cell=8, bins=9, block=2, norm="L2"):
    nx, ny = size[0] // cell, size[1] // cell
    if norm == "none":
        return nx * ny * bins
    return (nx - block + 1) * (ny - block + 1) * block * block * bins


if __name__ == "__main__":
    # Đối chiếu với skimage trên 20 ảnh
    from skimage.feature import hog as sk_hog
    from ..data import load_all_images
    imgs = np.asarray(load_all_images(256)[:20])
    mine = hog_batch(imgs, size=(64, 128), soft=False)
    ref = []
    for im in imgs:
        g = cv2.cvtColor(cv2.resize(im, (64, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
        ref.append(sk_hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2"))
    ref = np.array(ref)
    print("số chiều:", mine.shape[1], "| skimage:", ref.shape[1], "| công thức:", hog_dim())
    print("sai khác tuyệt đối lớn nhất so với skimage:", float(np.abs(mine - ref).max()))
