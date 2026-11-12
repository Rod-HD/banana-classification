"""Nạp mô hình CNN đã huấn luyện và dự đoán trên ảnh RGB uint8 bất kỳ (tự resize về kích thước của mô hình)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np

from ..config import CKPT_DIR, RESULTS_DIR
from . import models as M
from .train import resize_batch


def run_meta(run: str) -> dict:
    return json.loads((RESULTS_DIR / "cnn" / run / "metrics.json").read_text())


def full_model(run: str):
    """Mô hình đầy đủ ảnh → xác suất 4 lớp (ghép thân đóng băng với phần đầu nếu cần)."""
    meta = run_meta(run)
    kind = meta["model"]
    if kind == "a_vgg16":
        base = M.vgg16_features(meta["input_size"])
        head = keras.models.load_model(CKPT_DIR / run / "best.keras")
        return keras.Model(base.input, head(base.output), name="a_vgg16")
    if kind.startswith("b_") and meta.get("frozen_backbone"):
        model, base = M.b_model(kind, meta["input_size"], augment=False)
        head = M.b_head(base.output_shape[-1]); head.load_weights(CKPT_DIR / run / "best.weights.h5")
        model.get_layer("softmax").set_weights(head.get_layer("softmax").get_weights())
        return model
    return keras.models.load_model(CKPT_DIR / run / "best.keras")


def embedding_model(run: str):
    """Mô hình ảnh → embedding ở lớp áp chót (a_vgg16: Dense(256); hướng B: vector sau GlobalAveragePooling)."""
    meta = run_meta(run)
    if meta["model"] == "a_vgg16":
        base = M.vgg16_features(meta["input_size"])
        head = keras.models.load_model(CKPT_DIR / run / "best.keras")
        emb = keras.Model(head.input, head.get_layer("fc2").output)
        return keras.Model(base.input, emb(base.output))
    m = full_model(run)
    return keras.Model(m.input, m.get_layer("embedding").output)


class Predictor:
    def __init__(self, run: str, model=None):
        self.run, self.size = run, run_meta(run)["input_size"]
        self.model = model or full_model(run)

    def __call__(self, imgs: np.ndarray, batch: int = 32) -> np.ndarray:
        x = resize_batch(imgs, self.size).astype(np.float32)
        return self.model.predict(x, batch_size=batch, verbose=0)
