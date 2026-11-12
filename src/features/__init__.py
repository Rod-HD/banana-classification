"""Rút đặc trưng thủ công. Hàm extract() chạy song song và cache kết quả vào data/cache/."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from joblib import Parallel, delayed

from ..config import CACHE_DIR
from . import color, edge, hog as hogmod

# tên đặc trưng -> hàm nhận một ảnh RGB uint8 (256×256×3) và trả về vector 1 chiều
PER_IMAGE = {
    "rgb_hist": color.rgb_hist,
    "hsv_hist": color.hsv_hist,
    "hs_hist": color.hs_hist,
    "moments": color.color_moments,
    "dominant": color.dominant_colors,
    "ccv": color.ccv,
    "quant_hist": color.quant_hist,
    "edge": edge.edge_vector,
}


def _chunk(fn, imgs, params):
    return np.stack([fn(im, **params) for im in imgs]).astype(np.float32)


def compute(name: str, imgs: np.ndarray, n_jobs: int = 4, **params) -> np.ndarray:
    """Rút đặc trưng `name` cho mảng ảnh (N, H, W, 3) uint8, không cache."""
    if name == "hog":
        return hogmod.hog_batch(imgs, **params)
    fn = PER_IMAGE[name]
    n = len(imgs)
    if n < 64 or n_jobs == 1:
        return _chunk(fn, imgs, params)
    bounds = np.linspace(0, n, n_jobs * 4 + 1).astype(int)
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(fn, np.asarray(imgs[a:b]), params)
                                    for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
    return np.concatenate(parts)


def extract(name: str, imgs: np.ndarray, tag: str, return_time: bool = False, **params):
    """Như compute() nhưng cache theo (tên đặc trưng, tham số, tag của bộ ảnh).
    `tag` mô tả bộ ảnh đầu vào, ví dụ 'all256' hoặc 'all256_shuffle4'."""
    key = hashlib.md5(json.dumps([name, params, tag, len(imgs)], sort_keys=True, default=str).encode()).hexdigest()[:10]
    f = CACHE_DIR / f"feat_{name}_{tag}_{key}.npy"
    ft = f.with_suffix(".time")
    if f.exists():
        X = np.load(f)
        t = float(ft.read_text()) if ft.exists() else float("nan")
    else:
        t0 = time.perf_counter()
        X = compute(name, imgs, **params)
        t = time.perf_counter() - t0
        np.save(f, X)
        ft.write_text(str(t))
    return (X, t) if return_time else X
