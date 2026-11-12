"""Mô tả một cấu hình truyền thống (đặc trưng + cách ghép + bộ phân lớp) dưới dạng dict,
dùng chung cho bước chọn siêu tham số, đánh giá cuối, thí nghiệm I2/S2 và demo.

    cfg = {"name": ..., "parts": [["hsv_hist", {}], ["hog", {"size": [64, 128]}]],
           "fusion": "balanced" | "plain", "clf": "logreg", "cparams": {"C": 1.0}, "scale": "standard"}
"""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classical import BlockScaler, make_clf
from .features import compute, extract
from .transforms import transformed_images


def _params(fp):
    fp = dict(fp)
    if "size" in fp:
        fp["size"] = tuple(fp["size"])
    if "bins" in fp and isinstance(fp["bins"], list):
        fp["bins"] = tuple(fp["bins"])
    return fp


def features_all(cfg, variant: str = "clean") -> tuple[np.ndarray, list[int]]:
    """Đặc trưng của toàn bộ 2.471 ảnh (theo thứ tự list_images) cho cấu hình cfg."""
    Xs = [extract(f, transformed_images(variant), f"all256_{variant}", **_params(fp)) for f, fp in cfg["parts"]]
    return np.concatenate(Xs, axis=1), [x.shape[1] for x in Xs]


def features_of_images(cfg, imgs: np.ndarray) -> np.ndarray:
    """Đặc trưng của một mảng ảnh bất kỳ (N, 256, 256, 3) uint8, không cache."""
    return np.concatenate([compute(f, imgs, n_jobs=1, **_params(fp)) for f, fp in cfg["parts"]], axis=1)


def build_clf(cfg, dims):
    kw = dict(cfg.get("cparams", {}))
    if len(cfg["parts"]) > 1 and cfg.get("fusion", "balanced") == "balanced":
        base = make_clf(cfg["clf"], scale="none", **kw)
        return make_pipeline(BlockScaler(dims), *[s for _, s in base.steps])
    return make_clf(cfg["clf"], scale=cfg.get("scale", "standard"), **kw)
