"""Phần xử lý của demo (không phụ thuộc giao diện): nạp mô hình, dự đoán, Grad-CAM, biến đổi ảnh,
tính các metric cơ bản (accuracy, precision, recall, F1, ma trận nhầm lẫn) trên tập val/test.

Mọi số liệu hiển thị trong demo hoặc được tính trực tiếp từ ảnh trong data/raw và mô hình trong models/,
hoặc đọc từ file trong results/ (tab "Kết quả thực nghiệm").
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR, SEED
from src.data import load_split, read_rgb
from src.segmentation import banana_mask_v2
from src.stress import STRESS, STRESS_LABEL, background_of, paste
from src.transforms import apply_transform, gauss_noise, hflip, occlude, rotate, scale_v, to_gray

APP_DIR = Path(__file__).resolve().parent
INFO = json.loads((APP_DIR / "variety_info.json").read_text(encoding="utf-8"))
REGISTRY = {e["key"]: e for e in json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))}
LABEL_TO_KEY = {e["label"]: k for k, e in REGISTRY.items()}
CNN_KEYS = [k for k, e in REGISTRY.items() if e["type"] == "keras"]
EXAMPLES = sorted((APP_DIR / "examples").glob("*.jpg"))
LOW_CONF = 0.6      # quy ước của demo để nhắc người dùng khi mô hình không chắc chắn; không rút ra từ thí nghiệm


# ---------------------------------------------------------------- mô hình

class Model:
    """Một mô hình trong models/: nhận lô ảnh RGB uint8 256×256, trả về xác suất 4 lớp."""

    def __init__(self, key: str):
        self.key, self.entry = key, REGISTRY[key]
        self.is_cnn = self.entry["type"] == "keras"
        if self.is_cnn:
            import keras
            import tensorflow as tf
            self.net = keras.models.load_model(MODELS_DIR / self.entry["file"])
            self.size = self.entry["input_size"]
            # biên dịch hàm dự đoán một ảnh thành đồ thị (như cách đo ở N6); gọi trực tiếp kiểu eager chậm hơn khoảng 10 lần
            self._one = tf.function(lambda x: self.net(x, training=False),
                                    input_signature=[tf.TensorSpec([1, self.size, self.size, 3], tf.float32)], reduce_retracing=True)
        else:
            import joblib
            self.pack = joblib.load(MODELS_DIR / self.entry["file"])
            self.size = 256
        self.predict(np.zeros((1, 256, 256, 3), np.uint8))       # chạy nháp một lần để đo thời gian cho đúng

    def prepare(self, imgs: np.ndarray) -> np.ndarray:
        """Ảnh 256×256 -> đầu vào của mạng (resize về kích thước của mô hình, giống lúc huấn luyện)."""
        return np.stack([cv2.resize(np.asarray(im), (self.size, self.size), interpolation=cv2.INTER_AREA) for im in imgs]).astype(np.float32)

    def predict(self, imgs: np.ndarray) -> np.ndarray:
        imgs = np.asarray(imgs)
        if self.is_cnn:
            x = self.prepare(imgs)
            if len(x) == 1:
                return self._one(x).numpy()
            return np.asarray(self.net.predict(x, batch_size=32, verbose=0))
        from src.trad_config import features_of_images
        X = features_of_images(self.pack["cfg"], imgs)
        clf = self.pack["clf"]
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        P = np.zeros((len(X), len(CLASS_NAMES)))
        P[np.arange(len(X)), clf.predict(X).astype(int)] = 1.0
        return P

    def gradcam(self, img: np.ndarray, cls: int | None = None):
        """Grad-CAM trên lớp tích chập cuối: trọng số mỗi kênh = trung bình gradient của điểm lớp (trước softmax)
        theo bản đồ đặc trưng; bản đồ nhiệt = ReLU(tổng có trọng số các kênh). Trả về (bản đồ HxW trong [0,1], xác suất, lớp)."""
        import keras
        import tensorflow as tf
        layers = [l for l in self.net.layers if not isinstance(l, keras.layers.InputLayer)]
        k = max(i for i, l in enumerate(layers)
                if (isinstance(l, keras.layers.Conv2D) or (isinstance(l, keras.Model) and not isinstance(l, keras.Sequential)))
                and len(l.output.shape) == 4)
        x = tf.constant(self.prepare(img[None]))
        with tf.GradientTape() as tape:
            h = x
            for l in layers[:k + 1]:
                h = l(h, training=False)
            A = h
            tape.watch(A)
            for l in layers[k + 1:-1]:
                h = l(h, training=False)
            last = layers[-1]
            logits = tf.matmul(h, last.kernel) + last.bias
            probs = tf.nn.softmax(logits)
            c = int(tf.argmax(probs[0])) if cls is None else int(cls)
            score = logits[:, c]
        g = tape.gradient(score, A)
        w = tf.reduce_mean(g, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(A * w[:, None, None, :], axis=-1))[0].numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, probs[0].numpy(), c, A.shape[1:3]


_LOCK = threading.Lock()
_MODELS: dict[str, Model] = {}


def get_model(key: str) -> Model:
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = Model(key)
        return _MODELS[key]


def to_rgb256(image) -> np.ndarray:
    """Ảnh PIL / numpy / đường dẫn -> RGB uint8 256×256 (cùng kích thước với ảnh của bộ dữ liệu)."""
    if isinstance(image, (str, Path)):
        from PIL import Image
        image = Image.open(image)
    img = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.uint8)
    if img.shape[:2] != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return img


def predict_one(img256: np.ndarray, key: str) -> tuple[np.ndarray, float]:
    """(xác suất 4 lớp, thời gian dự đoán tính bằng ms)."""
    m = get_model(key)
    t0 = time.perf_counter()
    p = m.predict(img256[None])[0]
    return p, 1000 * (time.perf_counter() - t0)


def probs_dict(p) -> dict:
    return {c: float(v) for c, v in zip(CLASS_NAMES, p)}


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, img.shape[1::-1], interpolation=cv2.INTER_CUBIC).clip(0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255).astype(np.uint8)


def info_md(name: str) -> str:
    v = INFO.get(name, {})
    src = INFO["_nguon_du_lieu"]
    lines = [f"#### Thông tin giống {name}", f"- Số ảnh gốc trong bộ dữ liệu: {v.get('so_anh_goc_trong_bo_du_lieu', 'chưa có')}"]
    for k, lab in [("ten_khac", "Tên khác"), ("dac_diem_hinh_thai", "Đặc điểm hình thái"), ("vung_trong", "Vùng trồng"), ("cach_dung", "Cách dùng")]:
        if k in v:
            lines.append(f"- {lab}: {'chưa có thông tin có nguồn dẫn' if v[k] == 'TODO' else v[k]}")
    lines.append(f"- Nguồn: {src['ten']} ({src['phien_ban']}), DOI {src['doi']}, giấy phép {src['giay_phep']}.")
    return "\n".join(lines)


def label_from_name(name: str) -> int | None:
    """Đoán nhãn thật từ tên file (ảnh của bộ dữ liệu có dạng Bangla_Kola_102.jpg); không đoán được thì None."""
    s = name.lower()
    for i, keys in enumerate([("bangla",), ("champa", "chompa"), ("sabri",), ("sagor",)]):
        if any(k in s for k in keys):
            return i
    return None


# ---------------------------------------------------------------- biến đổi ảnh (tab "Thử độ bền")

def test_background(i: int) -> tuple[np.ndarray, str]:
    """Nền của ảnh test thứ i (quả chuối được xóa bằng inpainting, giống phép "nền hoán đổi" của bộ kiểm tra khó)."""
    df, raw = split_data("test")
    i = int(np.clip(i, 0, len(df) - 1))
    return background_of(raw[i], banana_mask_v2(raw[i])), f"{CLASS_NAMES[df['label_id'].values[i]]} ({Path(df['path'].values[i]).name})"


def transform(img: np.ndarray, v: float = 1.0, noise: float = 0.0, angle: float = 0.0, occl: float = 0.0,
              gray: bool = False, flip: bool = False, bg: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Áp lần lượt các phép của bộ kiểm tra khó (cùng hàm trong src/transforms.py, src/stress.py) lên một ảnh 256×256."""
    rng = np.random.default_rng([SEED, int(seed)])
    out = np.asarray(img, np.uint8).copy()
    if bg is not None:
        out = paste(out, banana_mask_v2(out), cv2.resize(np.asarray(bg, np.uint8)[..., :3], (256, 256), interpolation=cv2.INTER_AREA))
    if v != 1.0:
        out = scale_v(out, factor=float(v))
    if angle:
        out = rotate(out, angle=float(angle))
    if noise > 0:
        out = gauss_noise(out, rng, sigma=float(noise))
    if occl > 0:
        out = occlude(out, rng, frac=float(occl))
    if gray:
        out = to_gray(out)
    if flip:
        out = hflip(out)
    return out


# ---------------------------------------------------------------- đánh giá trên val/test (tab "Đánh giá theo metric")

@lru_cache(maxsize=None)
def split_data(split: str):
    """(bảng chia của tập, mảng ảnh 256×256) đọc thẳng từ data/raw theo data/splits/<split>.csv."""
    df = load_split(split)
    return df, np.stack([read_rgb(p, 256) for p in df["path"]])


@lru_cache(maxsize=None)
def split_masks(split: str) -> np.ndarray:
    return np.stack([banana_mask_v2(im) for im in split_data(split)[1]])


@lru_cache(maxsize=4)
def eval_images(split: str, stress: str) -> np.ndarray:
    """Ảnh của tập `split` sau phép thử `stress` ("clean" = ảnh gốc). Cùng cách sinh với src/stress.stress_sets
    (cùng seed), nên kết quả trên test trùng với bộ kiểm tra khó đã dùng trong thí nghiệm."""
    df, raw = split_data(split)
    ids, y = df["idx"].values, df["label_id"].values
    if stress == "clean":
        return raw
    if stress != "swapbg":
        return apply_transform(stress, raw, ids)
    masks = split_masks(split)
    pos = {int(i): k for k, i in enumerate(ids)}
    rng = np.random.default_rng([SEED, 99])
    out = np.empty_like(raw)
    for k, (i, c) in enumerate(zip(ids, y)):
        j = pos[int(rng.choice(ids[y != c]))]
        out[k] = paste(raw[k], masks[k], background_of(raw[j], masks[j]))
    return out


@lru_cache(maxsize=32)
def evaluate(key: str, split: str, stress: str) -> dict:
    df, _ = split_data(split)
    X = eval_images(split, stress)
    m = get_model(key)
    t0 = time.perf_counter()
    P = m.predict(X)
    dt = time.perf_counter() - t0
    return {"y": df["label_id"].values, "pred": P.argmax(1), "probs": P, "paths": df["path"].values, "seconds": dt}


def metric_tables(r: dict):
    """Các metric cơ bản từ nhãn thật và nhãn dự đoán: bảng theo lớp, bảng tổng hợp, ma trận nhầm lẫn."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y, pred = r["y"], r["pred"]
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y, pred, labels=labels)
    p, rc, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per = pd.DataFrame({"Lớp": CLASS_NAMES, "TP": np.diag(cm), "FP": cm.sum(0) - np.diag(cm), "FN": cm.sum(1) - np.diag(cm),
                        "Precision": p.round(4), "Recall": rc.round(4), "F1": f.round(4), "Số ảnh (support)": s})
    rows = []
    for avg, name in [("macro", "Trung bình macro (mỗi lớp như nhau)"), ("weighted", "Trung bình weighted (theo số ảnh)")]:
        a = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        rows.append({"Cách tính": name, "Precision": round(a[0], 4), "Recall": round(a[1], 4), "F1": round(a[2], 4)})
    summary = {"accuracy": float(accuracy_score(y, pred)), "n": int(len(y)), "wrong": int((pred != y).sum()),
               "macro_f1": float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[2])}
    return per, pd.DataFrame(rows), cm, summary


def confusion_figure(cm: np.ndarray, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = [c.replace(" Kola", "") for c in CLASS_NAMES]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, M, t, fmt in [(axes[0], cm, "Số ảnh", "d"), (axes[1], norm, "Tỉ lệ theo hàng (= recall trên đường chéo)", ".2f")]:
        ax.imshow(M, cmap="Blues", vmin=0)
        ax.set_xticks(range(4), short); ax.set_yticks(range(4), short)
        ax.set_xlabel("Lớp dự đoán"); ax.set_ylabel("Lớp thật"); ax.set_title(t, fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def recorded_score(key: str, split: str, stress: str) -> str | None:
    """Kết quả đã ghi trong repo cho đúng mô hình và tập này (để đối chiếu), nếu có."""
    e = REGISTRY[key]
    if stress != "clean":
        return None
    if split == "test" and e.get("test"):
        return f"macro-F1 {e['test']['macro_f1_test']:.4f}, accuracy {e['test']['accuracy_test']:.4f} (models/models.json, lấy từ results/final)"
    f = RESULTS_DIR / "cnn" / e.get("run", "-") / "metrics.json"
    if split == "val" and f.exists():
        m = json.loads(f.read_text())
        return f"macro-F1 {m['macro_f1']:.4f}, accuracy {m['accuracy']:.4f} ({f.relative_to(ROOT)})"
    return None


DATA_VERSIONS = {"Ảnh gốc": "clean", **{f"Bộ kiểm tra khó: {STRESS_LABEL[s]}": s for s in STRESS}}


# ---------------------------------------------------------------- kết quả đã lưu (tab "Kết quả thực nghiệm")

def result_tables() -> dict:
    """{nhãn hiển thị: đường dẫn} cho mọi bảng results/**/table*.md; nhãn = đường dẫn + dòng tiêu đề."""
    out = {}
    for f in sorted(RESULTS_DIR.glob("**/table*.md")):
        first = f.read_text(encoding="utf-8").lstrip().splitlines()[0] if f.stat().st_size else ""
        title = first.lstrip("#").strip()
        rel = str(f.relative_to(ROOT))
        out[f"{rel}: {title[:90]}" if title else rel] = f
    return out


def result_figures() -> dict:
    figs = sorted((RESULTS_DIR / "figures").glob("*.png")) + sorted(RESULTS_DIR.glob("final/*/confusion_matrix.png"))
    return {str(f.relative_to(ROOT)): f for f in figs}
