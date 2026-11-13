"""Demo nhận diện giống chuối (Gradio, 7 tab).

    python app/app.py                 # mở giao diện web tại http://127.0.0.1:7860
    python app/app.py --share         # thêm link công khai tạm thời do Gradio cấp
    python app/app.py --image duong_dan_anh.jpg [--model trad_best]   # dự đoán một ảnh trên dòng lệnh

Các tab: 1. Dự đoán · 2. So sánh mô hình · 3. Giải thích (Grad-CAM) · 4. Thử độ bền · 5. Nhiều ảnh ·
6. Đánh giá theo metric (accuracy, precision, recall, F1, ma trận nhầm lẫn trên val/test) · 7. Kết quả thực nghiệm.
Mô hình nằm trong models/ (tạo bằng `python -m src.export_demo`); phần xử lý nằm trong app/core.py.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import core as C
from core import CLASS_NAMES, REGISTRY

INTRO = ("# Nhận diện giống chuối từ ảnh\n"
         "Bốn giống: **Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola** (bộ dữ liệu BananaImageBD). "
         "Mô hình chỉ học ảnh **một quả chuối đặt trên nền trơn**; với ảnh khác bối cảnh hoặc giống khác, "
         "mô hình vẫn trả về một trong 4 lớp và kết quả không đáng tin.")

GRADCAM_NOTE = """**Grad-CAM** tô màu vùng ảnh mà CNN dựa vào nhiều nhất để cho điểm lớp được chọn (đỏ = ảnh hưởng mạnh, xanh = yếu).
Cách tính: lấy bản đồ đặc trưng A của lớp tích chập cuối, tính gradient của điểm lớp (trước softmax) theo A,
lấy trung bình gradient trên mỗi kênh làm trọng số, cộng có trọng số các kênh rồi giữ phần dương (ReLU), phóng về kích thước ảnh.
Bản đồ của MobileNetV3 chỉ có 7×7 ô nên thô; của LeNet là 26×26 ô. Mô hình truyền thống (HOG + histogram màu) không có Grad-CAM."""

METRIC_NOTE = """**Các metric cơ bản** (tính trên từng lớp c, coi c là lớp dương, 3 lớp còn lại là âm):
- TP = ảnh lớp c đoán đúng là c; FP = ảnh lớp khác bị đoán là c; FN = ảnh lớp c bị đoán thành lớp khác.
- **Precision** = TP / (TP + FP): trong các ảnh mô hình gọi là c, bao nhiêu phần trăm đúng.
- **Recall** = TP / (TP + FN): trong các ảnh thật sự là c, mô hình tìm ra bao nhiêu phần trăm.
- **F1** = 2·P·R / (P + R): trung bình điều hòa của precision và recall.
- **Accuracy** = số ảnh đoán đúng / tổng số ảnh.
- **Macro** = trung bình cộng 4 lớp (lớp ít ảnh như Bangla Kola có trọng số bằng lớp nhiều ảnh như Champa Kola); **weighted** = trung bình theo số ảnh mỗi lớp.
  Đồ án chọn **macro-F1** làm metric chính vì dữ liệu lệch lớp (Champa Kola gấp hơn 2 lần Bangla Kola).
- **Ma trận nhầm lẫn**: hàng = lớp thật, cột = lớp dự đoán; đường chéo là số ảnh đúng."""


def key_of(label: str) -> str:
    return C.LABEL_TO_KEY[label]


def fmt(df: pd.DataFrame, cols, nd: int = 4) -> pd.DataFrame:
    """Hiện các cột số với cùng số chữ số thập phân (để 1.0000 không thành 1)."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: f"{float(v):.{nd}f}")
    return df


# ---------------------------------------------------------------- tab 1: dự đoán

def ui_predict(image, label):
    if image is None:
        return None, "Hãy tải, chụp hoặc dán một ảnh rồi bấm **Dự đoán**."
    key = key_of(label)
    p, ms = C.predict_one(C.to_rgb256(image), key)
    top = CLASS_NAMES[int(p.argmax())]
    e = REGISTRY[key]
    lines = [f"### Dự đoán: {top} (xác suất {p.max():.3f})",
             f"Thời gian dự đoán: {ms:.1f} ms trên CPU (không tính thời gian nạp mô hình)."]
    if p.max() < C.LOW_CONF:
        lines.append(f"⚠️ Xác suất cao nhất dưới {C.LOW_CONF:.1f}: mô hình không chắc chắn; ảnh có thể khác bối cảnh huấn luyện "
                     f"(ngưỡng {C.LOW_CONF:.1f} là quy ước của demo, không rút ra từ thí nghiệm).")
    if e.get("test"):
        lines.append(f"Mô hình này đạt macro-F1 {e['test']['macro_f1_test']:.3f}, accuracy {e['test']['accuracy_test']:.3f} trên tập test 371 ảnh.")
    lines.append(C.info_md(top))
    return C.probs_dict(p), "\n\n".join(lines)


# ---------------------------------------------------------------- tab 2: so sánh mô hình

def ui_compare(image):
    if image is None:
        return None, None, "Hãy tải một ảnh lên."
    img = C.to_rgb256(image)
    rows, probs = [], {}
    for key, e in REGISTRY.items():
        p, ms = C.predict_one(img, key)
        probs[key] = p
        rows.append({"Mô hình": e["label"], "Giống dự đoán": CLASS_NAMES[int(p.argmax())], "Xác suất cao nhất": round(float(p.max()), 4),
                     **{c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)}, "Thời gian (ms)": round(ms, 1),
                     "Dung lượng file (MB)": e.get("size_mb")})
    df = pd.DataFrame(rows)
    preds = set(df["Giống dự đoán"])
    note = (f"Cả {len(df)} mô hình cùng dự đoán **{preds.pop()}**." if len(preds) == 1
            else f"Các mô hình **không thống nhất**: {', '.join(sorted(preds))}. Ảnh có thể khó hoặc khác bối cảnh huấn luyện.")
    return fmt(df, CLASS_NAMES + ["Xác suất cao nhất"]), compare_figure(probs), note


def compare_figure(probs: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6))
    n, w = len(probs), 0.8 / len(probs)
    for i, (key, p) in enumerate(probs.items()):
        ax.bar(np.arange(4) + (i - (n - 1) / 2) * w, p, w, label=key)
    ax.set_xticks(range(4), CLASS_NAMES); ax.set_ylim(0, 1.05); ax.set_ylabel("Xác suất")
    ax.legend(fontsize=8, ncol=2); ax.set_title("Xác suất 4 lớp theo từng mô hình", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- tab 3: Grad-CAM

def ui_gradcam(image, label, target, alpha):
    if image is None:
        return None, None, None, "Hãy tải một ảnh lên."
    key = key_of(label)
    m = C.get_model(key)
    img = C.to_rgb256(image)
    cls = None if target == "Lớp được dự đoán" else CLASS_NAMES.index(target)
    cam, p, c, grid = m.gradcam(img, cls)
    seen = C.cv2.resize(m.prepare(img[None])[0].astype(np.uint8), (256, 256), interpolation=C.cv2.INTER_NEAREST)
    note = (f"Bản đồ nhiệt cho lớp **{CLASS_NAMES[c]}** (lưới {grid[0]}×{grid[1]} ô của lớp tích chập cuối). "
            f"Mô hình dự đoán **{CLASS_NAMES[int(p.argmax())]}** với xác suất {p.max():.3f}.")
    return seen, C.overlay(img, cam, float(alpha)), C.probs_dict(p), note


# ---------------------------------------------------------------- tab 4: thử độ bền

PRESETS = {  # tên: (V, nhiễu, góc, che, xám, lật, nền)
    "Tự chỉnh": None,
    "Ảnh gốc": (1.0, 0, 0, 0.0, False, False, "Giữ nền gốc"),
    "Nền hoán đổi": (1.0, 0, 0, 0.0, False, False, "Nền của một ảnh test"),
    "V×0.6 (tối hơn)": (0.6, 0, 0, 0.0, False, False, "Giữ nền gốc"),
    "V×1.4 (sáng hơn)": (1.4, 0, 0, 0.0, False, False, "Giữ nền gốc"),
    "Nhiễu σ=10": (1.0, 10, 0, 0.0, False, False, "Giữ nền gốc"),
    "Xoay +30°": (1.0, 0, 30, 0.0, False, False, "Giữ nền gốc"),
    "Che 25%": (1.0, 0, 0, 0.25, False, False, "Giữ nền gốc"),
    "Ảnh xám": (1.0, 0, 0, 0.0, True, False, "Giữ nền gốc"),
}
BG_MODES = ["Giữ nền gốc", "Nền của một ảnh test", "Nền tự tải lên"]
BG_DEFAULT = 278   # data/raw/sabri_kola/Sabri_Kola_480.jpg: nền xanh dương, khác hẳn nền trắng/xám của đa số ảnh


def ui_preset(name):
    import gradio as gr
    v = PRESETS.get(name)
    if v is None:
        return [gr.update()] * 7
    return [gr.update(value=x) for x in v]


def ui_stress(image, label, v, noise, angle, occl, gray, flip, bg_mode, bg_idx, bg_img, seed):
    if image is None:
        return None, None, None, "Hãy tải một ảnh lên."
    img = C.to_rgb256(image)
    bg, bg_note = None, ""
    if bg_mode == "Nền của một ảnh test":
        bg, src = C.test_background(int(bg_idx))
        bg_note = f" Nền lấy từ ảnh test số {int(bg_idx)}: {src}."
    elif bg_mode == "Nền tự tải lên":
        if bg_img is None:
            return None, None, None, "Chọn \"Nền tự tải lên\" thì cần tải ảnh nền."
        bg = C.to_rgb256(bg_img)
    out = C.transform(img, v, noise, angle, occl, gray, flip, bg, seed)
    key = key_of(label)
    p0, _ = C.predict_one(img, key)
    p1, _ = C.predict_one(out, key)
    a, b = CLASS_NAMES[int(p0.argmax())], CLASS_NAMES[int(p1.argmax())]
    note = (f"Trước biến đổi: **{a}** ({p0.max():.3f}). Sau biến đổi: **{b}** ({p1.max():.3f}). "
            + ("Dự đoán **không đổi**." if a == b else "Dự đoán **đã thay đổi**.") + bg_note
            + "\n\nBảng `results/so_sanh_nhanh/table_stress.md` (tab 7) ghi macro-F1 của từng nhánh trên cả 371 ảnh test với 6 phép thử này; "
              "tab 6 tính lại được trên toàn tập.")
    return out, C.probs_dict(p0), C.probs_dict(p1), note


# ---------------------------------------------------------------- tab 5: nhiều ảnh

def ui_batch(files, label):
    import gradio as gr
    if not files:
        return None, "Hãy chọn một hoặc nhiều ảnh.", gr.update(visible=False)
    key = key_of(label)
    paths = [Path(f if isinstance(f, str) else f.name) for f in files]
    imgs, ok_paths, bad = [], [], []
    for pth in paths:
        try:
            imgs.append(C.to_rgb256(str(pth))); ok_paths.append(pth)
        except Exception:
            bad.append(pth.name)
    if not imgs:
        return None, "Không đọc được ảnh nào.", gr.update(visible=False)
    t0 = time.perf_counter()
    P = C.get_model(key).predict(np.stack(imgs))
    dt = time.perf_counter() - t0
    rows = []
    for pth, p in zip(ok_paths, P):
        t = C.label_from_name(pth.name)
        rows.append({"File": pth.name, "Giống dự đoán": CLASS_NAMES[int(p.argmax())], "Xác suất": round(float(p.max()), 4),
                     "Nhãn thật (đoán từ tên file)": CLASS_NAMES[t] if t is not None else "",
                     "Đúng?": ("đúng" if t == int(p.argmax()) else "sai") if t is not None else "",
                     **{c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)}})
    df = pd.DataFrame(rows)
    lab = df[df["Đúng?"] != ""]
    lines = [f"Đã dự đoán {len(df)} ảnh bằng **{REGISTRY[key]['label']}** trong {dt:.2f} s ({1000 * dt / len(df):.1f} ms/ảnh)."]
    if len(lab):
        lines.append(f"{len(lab)} ảnh có tên file chứa tên giống: đúng {int((lab['Đúng?'] == 'đúng').sum())}/{len(lab)} ảnh.")
    if bad:
        lines.append(f"Không đọc được: {', '.join(bad)}.")
    lines.append("Số ảnh mỗi giống dự đoán: " + ", ".join(f"{c}: {n}" for c, n in df["Giống dự đoán"].value_counts().items()) + ".")
    out = Path(tempfile.mkdtemp()) / "du_doan.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return fmt(df, CLASS_NAMES + ["Xác suất"]), "\n\n".join(lines), gr.update(value=str(out), visible=True)


# ---------------------------------------------------------------- tab 6: đánh giá theo metric

def ui_metrics(label, split, version):
    key, stress = key_of(label), C.DATA_VERSIONS[version]
    r = C.evaluate(key, split, stress)
    per, avg, cm, s = C.metric_tables(r)
    name = f"{REGISTRY[key]['label']}, tập {split} ({s['n']} ảnh), {version.lower()}"
    lines = [f"### {name}",
             f"**Accuracy** {s['accuracy']:.4f} · **Macro-F1** {s['macro_f1']:.4f} · số ảnh sai {s['wrong']}/{s['n']} · "
             f"thời gian dự đoán {r['seconds']:.1f} s ({1000 * r['seconds'] / s['n']:.1f} ms/ảnh)"]
    rec = C.recorded_score(key, split, stress)
    if rec:
        lines.append(f"Kết quả đã ghi trong repo để đối chiếu: {rec}.")
    if split == "test":
        lines.append("Tập test chỉ dùng để báo cáo; các lựa chọn mô hình trong đồ án đều dựa trên val.")
    wrong = np.where(r["pred"] != r["y"])[0]
    X = C.eval_images(split, stress)
    gallery = [(X[i], f"Thật: {CLASS_NAMES[r['y'][i]]} → Đoán: {CLASS_NAMES[r['pred'][i]]} ({r['probs'][i].max():.2f}) · {Path(r['paths'][i]).name}")
               for i in wrong[:60]]
    if len(wrong) > 60:
        lines.append(f"Thư viện ảnh bên dưới chỉ hiện 60/{len(wrong)} ảnh sai đầu tiên.")
    if not len(wrong):
        lines.append("Không có ảnh nào bị đoán sai.")
    cols = ["Precision", "Recall", "F1"]
    return "\n\n".join(lines), fmt(per, cols), fmt(avg, cols), C.confusion_figure(cm, name), gallery


def ui_metrics_all(split, version):
    stress = C.DATA_VERSIONS[version]
    rows = []
    for key, e in REGISTRY.items():
        r = C.evaluate(key, split, stress)
        per, avg, _, s = C.metric_tables(r)
        rows.append({"Mô hình": e["label"], "Accuracy": round(s["accuracy"], 4),
                     "Precision macro": avg.loc[0, "Precision"], "Recall macro": avg.loc[0, "Recall"], "F1 macro": avg.loc[0, "F1"],
                     "F1 weighted": avg.loc[1, "F1"], "Số ảnh sai": f"{s['wrong']}/{s['n']}",
                     "ms/ảnh": round(1000 * r["seconds"] / s["n"], 2)})
    return fmt(pd.DataFrame(rows), ["Accuracy", "Precision macro", "Recall macro", "F1 macro", "F1 weighted"])


# ---------------------------------------------------------------- tab 7: kết quả thực nghiệm

def ui_table(name):
    f = C.result_tables().get(name)
    return f.read_text(encoding="utf-8") if f else ""


def ui_figure(name):
    from PIL import Image
    f = C.result_figures().get(name)
    return Image.open(f) if f else None


# ---------------------------------------------------------------- giao diện

def build_ui():
    import gradio as gr
    labels = [e["label"] for e in REGISTRY.values()]
    cnn_labels = [REGISTRY[k]["label"] for k in C.CNN_KEYS]
    ex = [[str(p)] for p in C.EXAMPLES]
    tables, figures = C.result_tables(), C.result_figures()
    t0 = next((k for k in tables if k.startswith("results/final/table.md")), next(iter(tables), None))
    f0 = next((k for k in figures if k.endswith("final_test_macro_f1.png")), next(iter(figures), None))
    n_test = len(C.load_split("test"))

    with gr.Blocks(title="Nhận diện giống chuối") as demo:
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("1. Dự đoán"):
                with gr.Row():
                    with gr.Column():
                        p_img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Ảnh chuối (tải lên, chụp webcam hoặc dán)")
                        p_model = gr.Dropdown(labels, value=labels[0], label="Mô hình")
                        with gr.Row():
                            p_btn = gr.Button("Dự đoán", variant="primary")
                            p_clear = gr.ClearButton(value="Xóa")
                        if ex:
                            gr.Examples(ex, inputs=[p_img], label="Ảnh ví dụ (lấy từ tập test)")
                    with gr.Column():
                        p_out = gr.Label(num_top_classes=4, label="Xác suất 4 lớp")
                        p_note = gr.Markdown()
                p_clear.add([p_img, p_out, p_note])
                p_btn.click(ui_predict, [p_img, p_model], [p_out, p_note])

            with gr.Tab("2. So sánh mô hình"):
                gr.Markdown("Chạy cùng một ảnh qua cả 4 mô hình trong `models/` để xem các mô hình có thống nhất không.")
                with gr.Row():
                    with gr.Column(scale=1):
                        c_img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Ảnh chuối")
                        c_btn = gr.Button("Chạy tất cả mô hình", variant="primary")
                        if ex:
                            gr.Examples(ex, inputs=[c_img], label="Ảnh ví dụ")
                    with gr.Column(scale=2):
                        c_note = gr.Markdown()
                        c_plot = gr.Plot(label="Xác suất theo mô hình")
                c_df = gr.Dataframe(label="Kết quả từng mô hình", wrap=True)
                c_btn.click(ui_compare, [c_img], [c_df, c_plot, c_note])

            with gr.Tab("3. Giải thích (Grad-CAM)"):
                gr.Markdown(GRADCAM_NOTE)
                with gr.Row():
                    with gr.Column():
                        g_img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Ảnh chuối")
                        g_model = gr.Dropdown(cnn_labels, value=cnn_labels[0], label="Mô hình CNN")
                        g_target = gr.Radio(["Lớp được dự đoán"] + CLASS_NAMES, value="Lớp được dự đoán", label="Giải thích cho lớp")
                        g_alpha = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Độ đậm lớp phủ")
                        g_btn = gr.Button("Tạo bản đồ nhiệt", variant="primary")
                        if ex:
                            gr.Examples(ex, inputs=[g_img], label="Ảnh ví dụ")
                    with gr.Column():
                        with gr.Row():
                            g_seen = gr.Image(label="Ảnh mô hình nhìn thấy (sau khi resize)", interactive=False)
                            g_cam = gr.Image(label="Grad-CAM", interactive=False)
                        g_out = gr.Label(num_top_classes=4, label="Xác suất 4 lớp")
                        g_note = gr.Markdown()
                g_btn.click(ui_gradcam, [g_img, g_model, g_target, g_alpha], [g_seen, g_cam, g_out, g_note])

            with gr.Tab("4. Thử độ bền"):
                gr.Markdown("Biến đổi ảnh bằng đúng các phép của **bộ kiểm tra khó** dùng trong thí nghiệm (độ sáng, nhiễu, xoay, che, thay nền) "
                            "rồi xem dự đoán thay đổi thế nào. Chọn một mẫu có sẵn hoặc tự chỉnh từng thanh trượt.")
                with gr.Row():
                    with gr.Column():
                        s_img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Ảnh chuối")
                        s_model = gr.Dropdown(labels, value=labels[0], label="Mô hình")
                        s_preset = gr.Dropdown(list(PRESETS), value="Tự chỉnh", label="Mẫu phép biến đổi")
                        s_v = gr.Slider(0.3, 1.7, value=1.0, step=0.05, label="Nhân độ sáng V (kênh V của HSV)")
                        s_noise = gr.Slider(0, 40, value=0, step=1, label="Nhiễu Gauss σ")
                        s_angle = gr.Slider(-90, 90, value=0, step=5, label="Góc xoay (độ)")
                        s_occl = gr.Slider(0.0, 0.5, value=0.0, step=0.05, label="Tỉ lệ diện tích bị che (ô đen)")
                        with gr.Row():
                            s_gray = gr.Checkbox(label="Chuyển ảnh xám")
                            s_flip = gr.Checkbox(label="Lật ngang")
                        s_bg = gr.Radio(BG_MODES, value=BG_MODES[0], label="Nền")
                        s_bg_idx = gr.Slider(0, n_test - 1, value=BG_DEFAULT, step=1, label="Lấy nền của ảnh test số (mặc định: ảnh Sabri Kola nền xanh)")
                        s_bg_img = gr.Image(type="pil", label="Ảnh nền tự tải lên", height=160)
                        s_seed = gr.Slider(0, 99, value=0, step=1, label="Seed (vị trí ô che, mẫu nhiễu)")
                        with gr.Row():
                            s_btn = gr.Button("Áp dụng và dự đoán", variant="primary")
                            s_reset = gr.Button("Đặt lại")
                        if ex:
                            gr.Examples(ex, inputs=[s_img], label="Ảnh ví dụ")
                    with gr.Column():
                        s_out_img = gr.Image(label="Ảnh sau biến đổi", interactive=False)
                        with gr.Row():
                            s_before = gr.Label(num_top_classes=4, label="Trước biến đổi")
                            s_after = gr.Label(num_top_classes=4, label="Sau biến đổi")
                        s_note = gr.Markdown()
                ctrls = [s_v, s_noise, s_angle, s_occl, s_gray, s_flip, s_bg]
                s_preset.change(ui_preset, [s_preset], ctrls)
                s_reset.click(lambda: ["Tự chỉnh", 1.0, 0, 0, 0.0, False, False, BG_MODES[0], BG_DEFAULT, None, 0, None, None, None, ""], None,
                              [s_preset] + ctrls + [s_bg_idx, s_bg_img, s_seed, s_out_img, s_before, s_after, s_note])
                s_btn.click(ui_stress, [s_img, s_model] + ctrls + [s_bg_idx, s_bg_img, s_seed], [s_out_img, s_before, s_after, s_note])

            with gr.Tab("5. Nhiều ảnh"):
                gr.Markdown("Tải nhiều ảnh cùng lúc. Nếu tên file chứa tên giống (ví dụ `Sabri_Kola_104.jpg` như trong bộ dữ liệu), "
                            "demo dùng tên đó làm nhãn thật để đếm số ảnh đúng.")
                with gr.Row():
                    b_files = gr.File(file_count="multiple", file_types=["image"], label="Các ảnh chuối")
                    with gr.Column():
                        b_model = gr.Dropdown(labels, value=labels[0], label="Mô hình")
                        b_btn = gr.Button("Dự đoán tất cả", variant="primary")
                        b_dl = gr.DownloadButton("Tải CSV", visible=False)
                b_note = gr.Markdown()
                b_df = gr.Dataframe(label="Kết quả", wrap=True)
                b_btn.click(ui_batch, [b_files, b_model], [b_df, b_note, b_dl])

            with gr.Tab("6. Đánh giá theo metric"):
                gr.Markdown("Tính lại các metric cơ bản của môn học trên tập val hoặc test của đồ án (đọc ảnh từ `data/raw` theo `data/splits/`), "
                            "với ảnh gốc hoặc từng phép của bộ kiểm tra khó. Lần đầu mất vài giây đến vài chục giây để đọc ảnh.")
                with gr.Row():
                    m_model = gr.Dropdown(labels, value=labels[0], label="Mô hình")
                    m_split = gr.Radio(["test", "val"], value="test", label="Tập đánh giá")
                    m_ver = gr.Dropdown(list(C.DATA_VERSIONS), value="Ảnh gốc", label="Phiên bản ảnh")
                with gr.Row():
                    m_btn = gr.Button("Tính metric", variant="primary")
                    m_all = gr.Button("So sánh mọi mô hình")
                with gr.Accordion("Công thức các metric", open=False):
                    gr.Markdown(METRIC_NOTE)
                m_note = gr.Markdown()
                with gr.Row():
                    m_per = gr.Dataframe(label="Theo từng lớp")
                    m_avg = gr.Dataframe(label="Trung bình")
                m_plot = gr.Plot(label="Ma trận nhầm lẫn")
                m_gal = gr.Gallery(label="Ảnh bị đoán sai", columns=6, height="auto")
                m_cmp = gr.Dataframe(label="So sánh mọi mô hình trên cùng tập", wrap=True)
                m_btn.click(ui_metrics, [m_model, m_split, m_ver], [m_note, m_per, m_avg, m_plot, m_gal])
                m_all.click(ui_metrics_all, [m_split, m_ver], [m_cmp])

            with gr.Tab("7. Kết quả thực nghiệm"):
                gr.Markdown("Xem các bảng và hình do code sinh ra trong `results/` (đọc thẳng từ file, không nhập tay).")
                r_tab = gr.Dropdown(list(tables), value=t0, label=f"Bảng ({len(tables)} bảng)", filterable=True)
                r_md = gr.Markdown(ui_table(t0) if t0 else "")
                r_fig = gr.Dropdown(list(figures), value=f0, label=f"Hình ({len(figures)} hình)", filterable=True)
                r_img = gr.Image(value=ui_figure(f0) if f0 else None, label="Hình", interactive=False)
                r_tab.change(ui_table, [r_tab], [r_md])
                r_fig.change(ui_figure, [r_fig], [r_img])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="dự đoán một ảnh trên dòng lệnh thay vì mở giao diện")
    ap.add_argument("--model", default=next(iter(REGISTRY)), choices=list(REGISTRY))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai tạm thời do Gradio cấp")
    a = ap.parse_args()
    if a.image:
        p, ms = C.predict_one(C.to_rgb256(a.image), a.model)
        print(f"Mô hình: {REGISTRY[a.model]['label']}")
        for c, v in sorted(zip(CLASS_NAMES, p), key=lambda t: -t[1]):
            print(f"  {c:12s} {v:.4f}")
        print(f"Thời gian dự đoán: {ms:.1f} ms")
        return
    build_ui().launch(server_name="0.0.0.0", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
