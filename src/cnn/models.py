"""Kiến trúc CNN của đồ án.

Hướng A (theo bài giảng): a_lenet, a_dropout, a_vgg16 (transfer learning, đóng băng phần tích chập).
Hướng B (lớn hơn, fine-tune): b_mobilenetv3, b_resnet50.

Mọi mô hình nhận ảnh RGB giá trị 0-255; phần chuẩn hóa nằm trong mô hình.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
from keras import layers

from ..config import SEED

N_CLASSES = 4
INPUT_SIZE = {"a_lenet": 64, "a_dropout": 128, "a_vgg16": 224, "b_mobilenetv3": 224, "b_resnet50": 224}
IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]


def augmentation(name="augment"):
    """Tăng cường ảnh chỉ hoạt động lúc huấn luyện: lật, xoay ±30°, phóng to/thu nhỏ ±10%."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(30 / 360, fill_mode="nearest", seed=SEED + 1),
        layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED + 2),
    ], name=name)


def a_lenet(size: int = 64, augment: bool = False):
    """Kiểu LeNet-5: Conv(6, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(120) → Dense(84) → Softmax(4)."""
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation()(inp) if augment else inp
    x = layers.Rescaling(1 / 255.0, name="rescale")(x)
    x = layers.Conv2D(6, 5, activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Conv2D(16, 5, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(120, activation="relu", name="fc1")(x)
    x = layers.Dense(84, activation="relu", name="fc2")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="a_lenet")


def a_dropout(size: int = 128, dropout: bool = True, augment: bool = False):
    """Conv(32, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(64) → Softmax(4)."""
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation()(inp) if augment else inp
    x = layers.Rescaling(1 / 255.0, name="rescale")(x)
    x = layers.Conv2D(32, 5, activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Conv2D(16, 5, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    if dropout:
        x = layers.Dropout(0.5, seed=SEED, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="a_dropout")


def _rgb_backbone(kind: str, size: int = 224):
    """VGG16 / ResNet50 ImageNet không có phần đầu, sửa để nhận thẳng ảnh RGB 0-255:
    trừ trung bình ImageNet bằng lớp Normalization và đảo trục kênh của kernel tích chập đầu tiên
    (trọng số gốc học trên ảnh BGR). Tương đương preprocess_input kiểu 'caffe'."""
    fn = {"vgg16": keras.applications.VGG16, "resnet50": keras.applications.ResNet50}[kind]
    base = fn(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    first = next(l for l in base.layers if isinstance(l, layers.Conv2D))
    w = first.get_weights()
    w[0] = w[0][:, :, ::-1, :]
    first.set_weights(w)
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Normalization(mean=IMAGENET_MEAN_RGB, variance=[1.0, 1.0, 1.0], name="imagenet_mean")(inp)
    return keras.Model(inp, base(x, training=False), name=f"{kind}_rgb"), base


def vgg16_features(size: int = 224):
    """Phần tích chập VGG16 đóng băng: ảnh RGB 0-255 → bản đồ đặc trưng 7×7×512."""
    m, base = _rgb_backbone("vgg16", size)
    m.trainable = False
    return m


def a_vgg16_head(feat_shape=(7, 7, 512)):
    """Phần đầu của a_vgg16 đúng ví dụ bài giảng: Flatten → Dense(1024) → Dense(256) → Softmax(4)."""
    inp = keras.Input(feat_shape, name="vgg16_features")
    x = layers.Flatten(name="flatten")(inp)
    x = layers.Dense(1024, activation="relu", name="fc1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="a_vgg16_head")


def b_backbone(kind: str, size: int = 224):
    """Thân mạng hướng B: ảnh RGB 0-255 → vector sau GlobalAveragePooling."""
    if kind == "b_mobilenetv3":
        base = keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_shape=(size, size, 3),
                                                   include_preprocessing=True)
        return base, None
    wrapped, base = _rgb_backbone("resnet50", size)
    return base, wrapped


def b_model(kind: str, size: int = 224, augment: bool = True):
    """Mô hình hướng B đầy đủ: (tăng cường) → thân ImageNet (BatchNorm ở chế độ suy luận) → GAP → Dropout(0.2) → Softmax(4).
    Trả về (model, base) để đóng/mở băng base."""
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation()(inp) if augment else inp
    if kind == "b_mobilenetv3":
        base = keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_shape=(size, size, 3),
                                                   include_preprocessing=True)
    else:
        base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        first = next(l for l in base.layers if isinstance(l, layers.Conv2D))
        w = first.get_weights(); w[0] = w[0][:, :, ::-1, :]; first.set_weights(w)
        x = layers.Normalization(mean=IMAGENET_MEAN_RGB, variance=[1.0, 1.0, 1.0], name="imagenet_mean")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="embedding")(x)
    x = layers.Dropout(0.2, seed=SEED, name="dropout")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name=kind), base


def unfreeze(base, kind: str):
    """Mở băng thân mạng cho giai đoạn 2. Các lớp BatchNormalization luôn giữ đóng băng (trainable=False)
    để chúng chạy ở chế độ suy luận, dùng thống kê đã học trên ImageNet; nếu không, thống kê theo lô nhỏ
    sẽ làm lệch đặc trưng mà đầu phân lớp của giai đoạn 1 đã học.
    b_mobilenetv3: mở toàn mạng. b_resnet50: chỉ mở khối tích chập cuối (conv5)."""
    base.trainable = True
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
        elif kind == "b_resnet50":
            l.trainable = l.name.startswith("conv5")


def b_head(dim: int):
    """Đầu phân lớp của hướng B huấn luyện trên đặc trưng đã tính sẵn (giai đoạn 1)."""
    inp = keras.Input((dim,), name="embedding")
    x = layers.Dropout(0.2, seed=SEED, name="dropout")(inp)
    out = layers.Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="b_head")


def count_params(model) -> dict:
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"total": tr + nt, "trainable": tr, "non_trainable": nt}
