"""Các phép biến đổi ảnh dùng để kiểm chứng giả thuyết (M1, M2, M6, C1, C4, N8).
Mọi hàm nhận ảnh RGB uint8 và trả về ảnh RGB uint8 cùng kích thước. Phép nào có yếu tố
ngẫu nhiên thì nhận `rng` để kết quả lặp lại được."""
from __future__ import annotations

import cv2
import numpy as np

from .config import SEED


def shuffle_grid(img, rng, n: int = 4):
    """Cắt ảnh thành lưới n×n rồi xáo trộn vị trí các ô."""
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    tiles = [img[i * th:(i + 1) * th, j * tw:(j + 1) * tw] for i in range(n) for j in range(n)]
    perm = rng.permutation(len(tiles))
    out = np.empty_like(img[:th * n, :tw * n])
    for k, p in enumerate(perm):
        i, j = divmod(k, n)
        out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tiles[p]
    return out


def scale_v(img, rng=None, factor: float = 1.0):
    """Nhân kênh V (HSV) với `factor`, cắt ngưỡng ở 255."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def to_gray(img, rng=None):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)


def gauss_noise(img, rng, sigma: float = 10.0):
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def occlude(img, rng, frac: float = 0.25):
    """Che một hình vuông chiếm `frac` diện tích ảnh (tô đen) ở vị trí ngẫu nhiên."""
    h, w = img.shape[:2]
    side = int(round(np.sqrt(frac * h * w)))
    y = rng.integers(0, h - side + 1); x = rng.integers(0, w - side + 1)
    out = img.copy()
    out[y:y + side, x:x + side] = 0
    return out


def rotate(img, rng=None, angle: float = 15.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def hflip(img, rng=None):
    return img[:, ::-1].copy()


TRANSFORMS = {
    "shuffle4": (shuffle_grid, {"n": 4}),
    "v0.6": (scale_v, {"factor": 0.6}),
    "v1.4": (scale_v, {"factor": 1.4}),
    "gray": (to_gray, {}),
    "noise2": (gauss_noise, {"sigma": 2.0}),
    "noise5": (gauss_noise, {"sigma": 5.0}),
    "noise10": (gauss_noise, {"sigma": 10.0}),
    "noise20": (gauss_noise, {"sigma": 20.0}),
    "occlude25": (occlude, {"frac": 0.25}),
    "rot-30": (rotate, {"angle": -30.0}),
    "rot-15": (rotate, {"angle": -15.0}),
    "rot15": (rotate, {"angle": 15.0}),
    "rot30": (rotate, {"angle": 30.0}),
}


def apply_transform(name: str, imgs: np.ndarray, ids=None) -> np.ndarray:
    """Áp dụng phép biến đổi `name` cho cả mảng ảnh. Mỗi ảnh dùng bộ sinh số ngẫu nhiên
    riêng, khởi tạo từ (SEED, tên phép, chỉ số ảnh) nên kết quả không phụ thuộc thứ tự gọi."""
    fn, kw = TRANSFORMS[name]
    ids = np.arange(len(imgs)) if ids is None else ids
    key = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    out = np.empty((len(imgs),) + imgs.shape[1:], np.uint8)
    for k, i in enumerate(ids):
        rng = np.random.default_rng([SEED, key, int(i)])
        out[k] = fn(np.asarray(imgs[k]), rng, **kw)
    return out


def transformed_images(name: str, size: int = 256) -> np.ndarray:
    """Toàn bộ ảnh (theo thứ tự list_images) sau phép biến đổi; cache vào data/cache/."""
    from .config import CACHE_DIR
    from .data import load_all_images
    if name == "clean":
        return load_all_images(size)
    f = CACHE_DIR / f"images_{size}_{name}.npy"
    if not f.exists():
        np.save(f, apply_transform(name, load_all_images(size)))
    return np.load(f, mmap_mode="r")
