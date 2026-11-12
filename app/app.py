"""Demo nhận diện giống chuối: tải ảnh lên, nhận giống dự đoán và xác suất 4 lớp.

    python app/app.py                 # mở giao diện web tại http://127.0.0.1:7860
    python app/app.py --image duong_dan_anh.jpg [--model trad_best]   # dự đoán một ảnh trên dòng lệnh

Mô hình nằm trong models/ (tạo bằng `python -m src.export_demo`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from src.config import CLASS_NAMES, MODELS_DIR

INFO = json.loads((Path(__file__).parent / "variety_info.json").read_text(encoding="utf-8"))
REGISTRY = {e["key"]: e for e in json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))}
_LOADED = {}


def load(key: str):
    """Nạp mô hình (một lần) và trả về hàm: ảnh RGB uint8 256×256 -> xác suất 4 lớp."""
    if key in _LOADED:
        return _LOADED[key]
    e = REGISTRY[key]
    if e["type"] == "keras":
        import keras
        model = keras.models.load_model(MODELS_DIR / e["file"])
        size = e["input_size"]

        def fn(img):
            x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)[None]
            return np.asarray(model(x, training=False))[0]
    else:
        import joblib
        from src.trad_config import features_of_images
        pack = joblib.load(MODELS_DIR / e["file"])

        def fn(img):
            X = features_of_images(pack["cfg"], img[None])
            clf = pack["clf"]
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X)[0]
            p = np.zeros(len(CLASS_NAMES)); p[int(clf.predict(X)[0])] = 1.0
            return p
    _LOADED[key] = fn
    return fn


def to_rgb256(image) -> np.ndarray:
    """Ảnh PIL / numpy bất kỳ -> RGB uint8 256×256 (cùng kích thước với ảnh của bộ dữ liệu)."""
    img = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.uint8)
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)


def predict(image, key: str):
    probs = load(key)(to_rgb256(image))
    return {c: float(p) for c, p in zip(CLASS_NAMES, probs)}


def info_md(name: str) -> str:
    v = INFO.get(name, {})
    src = INFO["_nguon_du_lieu"]
    lines = [f"### {name}", f"- Số ảnh gốc trong bộ dữ liệu: {v.get('so_anh_goc_trong_bo_du_lieu', 'TODO')}"]
    for k, lab in [("ten_khac", "Tên khác"), ("dac_diem_hinh_thai", "Đặc điểm hình thái"), ("vung_trong", "Vùng trồng"), ("cach_dung", "Cách dùng")]:
        if k in v:
            lines.append(f"- {lab}: {'chưa có thông tin có nguồn dẫn' if v[k] == 'TODO' else v[k]}")
    lines.append(f"- Nguồn: {src['ten']} ({src['phien_ban']}), DOI {src['doi']}, giấy phép {src['giay_phep']}.")
    return "\n".join(lines)


def ui_predict(image, label):
    if image is None:
        return None, "Hãy tải một ảnh lên."
    key = next(k for k, e in REGISTRY.items() if e["label"] == label)
    probs = predict(image, key)
    top = max(probs, key=probs.get)
    e = REGISTRY[key]
    note = (f"**Dự đoán: {top}** (xác suất {probs[top]:.2f}).\n\n"
            + (f"Mô hình này đạt macro-F1 {e['test']['macro_f1_test']:.3f}, accuracy {e['test']['accuracy_test']:.3f} trên tập test 371 ảnh.\n\n" if e.get("test") else "")
            + "Lưu ý: mô hình chỉ học ảnh một quả chuối chụp trên nền trơn của 4 giống trong bộ dữ liệu. "
              "Với ảnh khác bối cảnh hoặc giống khác, mô hình vẫn trả về một trong 4 lớp và kết quả không đáng tin.\n\n" + info_md(top))
    return probs, note


def build_ui():
    import gradio as gr
    labels = [e["label"] for e in REGISTRY.values()]
    with gr.Blocks(title="Nhận diện giống chuối") as demo:
        gr.Markdown("# Nhận diện giống chuối từ ảnh\nTải lên ảnh một quả chuối (Bangla Kola, Champa Kola, Sabri Kola hoặc Sagor Kola), chọn mô hình rồi bấm **Dự đoán**.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Ảnh chuối")
                model = gr.Dropdown(labels, value=labels[0], label="Mô hình")
                btn = gr.Button("Dự đoán", variant="primary")
                examples = sorted((ROOT / "app" / "examples").glob("*.jpg"))
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=[image], label="Ảnh ví dụ (lấy từ tập test)")
            with gr.Column():
                out = gr.Label(num_top_classes=4, label="Xác suất 4 lớp")
                note = gr.Markdown()
        btn.click(ui_predict, [image, model], [out, note])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="dự đoán một ảnh trên dòng lệnh thay vì mở giao diện")
    ap.add_argument("--model", default=next(iter(REGISTRY)), choices=list(REGISTRY))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    if a.image:
        from PIL import Image
        probs = predict(Image.open(a.image), a.model)
        print(f"Mô hình: {REGISTRY[a.model]['label']}")
        for c, p in sorted(probs.items(), key=lambda t: -t[1]):
            print(f"  {c:12s} {p:.4f}")
        return
    build_ui().launch(server_name="0.0.0.0", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
