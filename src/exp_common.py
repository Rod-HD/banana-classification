"""Hàm dùng chung cho các thí nghiệm của nhóm truyền thống."""
from __future__ import annotations

import time

import numpy as np

from .classical import cv_eval, make_clf, strip
from .data import dev_and_test, load_all_images
from .features import compute, extract
from .report_utils import exp_dir, save_json
from .transforms import transformed_images

_DEV = None


def dev_set():
    """(ids, y) của tập phát triển train+val, theo thứ tự idx tăng dần."""
    global _DEV
    if _DEV is None:
        dev, _ = dev_and_test()
        _DEV = (dev["idx"].values, dev["label_id"].values)
    return _DEV


def dev_features(name: str, variant: str = "clean", **params) -> np.ndarray:
    """Đặc trưng `name` của tập phát triển, tính trên ảnh sạch hoặc ảnh đã qua phép biến đổi `variant`."""
    ids, _ = dev_set()
    X = extract(name, transformed_images(variant), f"all256_{variant}", **params)
    return X[ids]


def all_features(name: str, variant: str = "clean", **params) -> np.ndarray:
    return extract(name, transformed_images(variant), f"all256_{variant}", **params)


def extract_ms(name: str, n: int = 100, **params) -> float:
    """Thời gian rút đặc trưng một ảnh (ms), đo đơn luồng trên n ảnh đầu của tập phát triển."""
    ids, _ = dev_set()
    imgs = np.stack([load_all_images(256)[i] for i in ids[:n]])
    compute(name, imgs[:4], n_jobs=1, **params)        # làm nóng
    t0 = time.perf_counter()
    compute(name, imgs, n_jobs=1, **params)
    return 1000 * (time.perf_counter() - t0) / n


def run(feat, clf="knn", variants=(), fparams=None, cparams=None, scale="standard", **kw):
    """Chạy 5-fold CV cho (đặc trưng, bộ phân lớp); `variants` là tên các phép biến đổi áp lên fold giữ lại."""
    fparams = fparams or {}; cparams = cparams or {}
    _, y = dev_set()
    X = dev_features(feat, **fparams)
    V = {v: dev_features(feat, v, **fparams) for v in variants}
    return cv_eval(X, y, lambda: make_clf(clf, scale=scale, **cparams), variants=V, **kw)


def summary_row(experiment, config, res, condition="clean", group="truyền thống", eval_set="5-fold CV (train+val)", **extra):
    r = res if condition == "clean" else res["variants"][condition]
    row = {"experiment": experiment, "group": group, "config": config, "condition": condition,
           "eval_set": eval_set, "macro_f1": round(r["macro_f1"], 4), "macro_f1_std": round(r.get("macro_f1_std", float("nan")), 4),
           "accuracy": round(r["accuracy"], 4), "n_features": res.get("n_features"),
           "fit_time_s": round(res.get("fit_time_s", float("nan")), 4),
           "predict_ms_per_image": round(res.get("predict_ms_per_image", float("nan")), 4)}
    row.update(extra)
    return row


def register(exp: str, rows: list[dict]):
    """Lưu các dòng tổng hợp của thí nghiệm để src.summarize gom vào results/summary.csv."""
    save_json(exp_dir(exp) / "summary_rows.json", rows)


__all__ = ["dev_set", "dev_features", "all_features", "extract_ms", "run", "summary_row", "register",
           "cv_eval", "make_clf", "strip"]
