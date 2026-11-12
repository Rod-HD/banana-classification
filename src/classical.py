"""Khung đánh giá cho nhóm truyền thống: bộ phân lớp + 5-fold stratified cross-validation
trên train+val, có đo thời gian và hỗ trợ đánh giá trên ảnh đã biến đổi."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from .config import N_FOLDS, SEED

warnings.filterwarnings("ignore", category=FutureWarning)


def _sqrt_l1(X):
    """Căn bậc hai của histogram đã chuẩn hóa L1: Euclidean trên kết quả = Hellinger (sai khác hằng số)."""
    X = np.maximum(X, 0)
    return np.sqrt(X / np.maximum(X.sum(1, keepdims=True), 1e-12))


def make_scaler(scale: str):
    if scale == "standard":
        return [StandardScaler()]
    if scale == "unit":          # chuẩn hóa mỗi vector về vector đơn vị (L2)
        return [Normalizer("l2")]
    if scale == "hellinger":
        return [FunctionTransformer(_sqrt_l1)]
    if scale == "none":
        return []
    raise ValueError(scale)


class BlockScaler(BaseEstimator, TransformerMixin):
    """Chuẩn hóa từng phần trước khi ghép: z-score từng chiều, rồi chia mỗi phần cho căn bậc hai
    số chiều của phần đó, để mỗi phần (màu, HOG) đóng góp ngang nhau vào khoảng cách."""

    def __init__(self, dims):
        self.dims = dims

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        self.w_ = np.concatenate([np.full(d, 1.0 / np.sqrt(d)) for d in self.dims]).astype(np.float32)
        return self

    def transform(self, X):
        return self.scaler_.transform(X) * self.w_


def make_clf(kind: str = "knn", scale: str = "standard", **kw):
    """kind: knn | logreg | svm. scale: standard | unit | hellinger | none."""
    steps = make_scaler(scale)
    if kind == "knn":
        p = dict(n_neighbors=5, weights="uniform", metric="euclidean", algorithm="brute")
        p.update(kw)
        steps.append(KNeighborsClassifier(**p))
    elif kind == "logreg":
        p = dict(C=1.0, max_iter=2000)
        p.update(kw)
        steps.append(LogisticRegression(**p))
    elif kind == "svm":
        p = dict(C=0.01, max_iter=5000, random_state=SEED)
        p.update(kw)
        steps.append(LinearSVC(**p))
    else:
        raise ValueError(kind)
    return make_pipeline(*steps)


def folds(y, groups=None, n_splits: int = N_FOLDS, seed: int = SEED):
    if groups is None:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def cv_eval(X, y, clf_factory, variants: dict | None = None, groups=None, fit_transform=None) -> dict:
    """5-fold CV. `variants`: {tên: X_biến_đổi} cùng thứ tự mẫu với X; mô hình học trên X sạch,
    được chấm trên fold giữ lại của X và của từng biến thể.
    `fit_transform(Xtr, ytr, tr_idx) -> (Xtr2, ytr2)`: lấy mẫu lại chỉ trên phần huấn luyện.
    Trả về macro-F1/accuracy trung bình ± độ lệch chuẩn, thời gian, và dự đoán out-of-fold."""
    y = np.asarray(y)
    variants = variants or {}
    names = ["clean"] + list(variants)
    f1 = {n: [] for n in names}; acc = {n: [] for n in names}
    oof = np.full(len(y), -1)
    t_fit = t_pred = 0.0
    for tr, te in folds(y, groups):
        Xtr, ytr = X[tr], y[tr]
        if fit_transform is not None:
            Xtr, ytr = fit_transform(Xtr, ytr, tr)
        clf = clf_factory()
        t0 = time.perf_counter(); clf.fit(Xtr, ytr); t_fit += time.perf_counter() - t0
        t0 = time.perf_counter(); p = clf.predict(X[te]); t_pred += time.perf_counter() - t0
        oof[te] = p
        f1["clean"].append(f1_score(y[te], p, average="macro")); acc["clean"].append(accuracy_score(y[te], p))
        for n, Xv in variants.items():
            pv = clf.predict(Xv[te])
            f1[n].append(f1_score(y[te], pv, average="macro")); acc[n].append(accuracy_score(y[te], pv))
    res = {"macro_f1": float(np.mean(f1["clean"])), "macro_f1_std": float(np.std(f1["clean"])),
           "accuracy": float(np.mean(acc["clean"])), "accuracy_std": float(np.std(acc["clean"])),
           "fit_time_s": t_fit / N_FOLDS, "predict_time_s": t_pred / N_FOLDS,
           "predict_ms_per_image": 1000 * t_pred / len(y),
           "fold_macro_f1": [float(v) for v in f1["clean"]], "n_features": int(X.shape[1]), "oof_pred": oof}
    if variants:
        res["variants"] = {}
        for n in variants:
            drop = np.array(f1["clean"]) - np.array(f1[n])
            res["variants"][n] = {"macro_f1": float(np.mean(f1[n])), "macro_f1_std": float(np.std(f1[n])),
                                  "accuracy": float(np.mean(acc[n])),
                                  "drop_macro_f1": float(drop.mean()), "drop_macro_f1_std": float(drop.std())}
    return res


def strip(res: dict) -> dict:
    """Bỏ mảng dự đoán trước khi ghi JSON."""
    return {k: v for k, v in res.items() if k != "oof_pred"}


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"
