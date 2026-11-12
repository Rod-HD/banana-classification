"""Tính và lưu độ đo: Accuracy, Precision/Recall/F1 từng lớp, macro-F1, Confusion Matrix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from .config import CLASS_NAMES

K = len(CLASS_NAMES)


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(K), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(K), zero_division=0)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "per_class": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])} for i in range(K)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(K)).tolist(),
    }


def plot_confusion(cm, path, title=""):
    cm = np.asarray(cm)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    short = [c.replace(" Kola", "") for c in CLASS_NAMES]
    for ax, m, fmt, t in [(axes[0], cm, "d", "Số lượng"), (axes[1], cmn, ".2f", "Chuẩn hóa theo hàng")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if fmt == "d" else 1)
        for i in range(K):
            for j in range(K):
                v = m[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > (m.max() if fmt == "d" else 1) * 0.6 else "black", fontsize=10)
        ax.set_xticks(range(K)); ax.set_xticklabels(short)
        ax.set_yticks(range(K)); ax.set_yticklabels(short)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(t)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_result(exp_dir: Path, y_true, y_pred, extra: dict | None = None, title: str = "") -> dict:
    """Lưu metrics.json, classification_report.txt và hình confusion matrix vào exp_dir."""
    exp_dir = Path(exp_dir); exp_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, y_pred)
    if extra:
        m.update(extra)
    (exp_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    rep = classification_report(y_true, y_pred, labels=range(K), target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    (exp_dir / "classification_report.txt").write_text(rep)
    plot_confusion(m["confusion_matrix"], exp_dir / "confusion_matrix.png", title)
    return m


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
