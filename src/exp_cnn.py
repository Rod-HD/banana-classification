"""Thí nghiệm CNN N1–N8 (mục 6.5 của kế hoạch). Đọc kết quả các lần huấn luyện trong results/cnn/
(tạo bởi src.cnn.train) và chạy các phép đo bổ sung (đếm tham số, thời gian dự đoán, embedding, độ bền).

    python -m src.exp_cnn --only N1 N2 N3     # sau khi huấn luyện hướng A
    python -m src.exp_cnn                     # sau khi huấn luyện cả hai hướng
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .classical import make_clf
from .config import CACHE_DIR, CKPT_DIR, FIG_DIR, RESULTS_DIR, SEED
from .data import load_all_images, load_split
from .exp_common import register
from .report_utils import exp_dir, save_json, save_table
from .trad_config import build_clf, features_all, features_of_images
from .transforms import transformed_images

A_RUNS = ["a_lenet", "a_dropout", "a_vgg16"]
B_RUNS = ["b_mobilenetv3", "b_resnet50"]
LABEL = {"a_lenet": "a_lenet (LeNet, từ đầu)", "a_dropout": "a_dropout (từ đầu)", "a_vgg16": "a_vgg16 (đóng băng)",
         "b_mobilenetv3": "b_mobilenetv3 (fine-tune)", "b_resnet50": "b_resnet50"}


def meta(run):
    f = RESULTS_DIR / "cnn" / run / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def hist(run):
    return json.loads((RESULTS_DIR / "cnn" / run / "history.json").read_text())


def srow(exp, run, m, condition="clean", group=None, **extra):
    return dict({"experiment": exp, "group": group or ("CNN hướng A" if run.startswith("a_") else "CNN hướng B"), "config": run,
                 "condition": condition, "eval_set": "val", "macro_f1": round(m["macro_f1"], 4), "macro_f1_std": None,
                 "accuracy": round(m["accuracy"], 4), "n_features": None,
                 "fit_time_s": round(m.get("train_time_s", float("nan")), 1) if "train_time_s" in m else None,
                 "predict_ms_per_image": None}, **extra)


def best_run(prefix_runs):
    """Run có macro-F1 trên val cao nhất trong danh sách (hòa thì lấy run đứng trước)."""
    ms = [(r, meta(r)) for r in prefix_runs if meta(r)]
    return max(ms, key=lambda t: t[1]["macro_f1"])[0]


def final_run(model: str) -> str:
    """Biến thể được chốt của mỗi mô hình: chọn theo macro-F1 trên val trong các biến thể đã huấn luyện."""
    cands = {"a_lenet": ["a_lenet"], "a_dropout": ["a_dropout", "a_dropout_aug"], "a_vgg16": ["a_vgg16", "a_vgg16_lr1e-3"],
             "b_mobilenetv3": ["b_mobilenetv3", "b_mobilenetv3_lr1e-4"], "b_resnet50": ["b_resnet50", "b_resnet50_s1"]}[model]
    return best_run(cands)


# ---------------------------------------------------------------- N1
def n1():
    from .cnn import models as M

    def conv(name, F, C, K): return (name, f"({F}×{F}×{C}+1)×{K}", (F * F * C + 1) * K)
    def fc(name, n_in, n_out): return (name, f"({n_in}+1)×{n_out}", (n_in + 1) * n_out)
    def flat(size, k2): s = ((size - 4) // 2 - 4) // 2; return s, s * s * k2

    out, raw = [], {}
    for mname, build, size, spec in [
        ("a_lenet", M.a_lenet, 64, lambda n: [conv("conv1", 5, 3, 6), conv("conv2", 5, 6, 16), fc("fc1", n, 120), fc("fc2", 120, 84), fc("softmax", 84, 4)]),
        ("a_dropout", M.a_dropout, 128, lambda n: [conv("conv1", 5, 3, 32), conv("conv2", 5, 32, 16), fc("fc1", n, 128), fc("fc2", 128, 64), fc("softmax", 64, 4)]),
    ]:
        model = build(size)
        k2 = 16
        s, n_flat = flat(size, k2)
        rows, total = [], 0
        for lname, formula, val in spec(n_flat):
            keras_val = model.get_layer(lname).count_params(); total += val
            note = f"Flatten: {s}×{s}×{k2} = {n_flat}" if lname == "fc1" else ""
            rows.append({"Mô hình": mname, "Lớp": lname, "Công thức": formula, "Tính tay": val, "model.summary()": keras_val,
                         "Khớp": "có" if val == keras_val else "KHÔNG", "Ghi chú": note})
        rows.append({"Mô hình": mname, "Lớp": "Tổng", "Công thức": "", "Tính tay": total, "model.summary()": model.count_params(),
                     "Khớp": "có" if total == model.count_params() else "KHÔNG", "Ghi chú": f"đầu vào {size}×{size}×3"})
        out += rows; raw[mname] = rows
    totals = []
    full = {"a_lenet": M.a_lenet(), "a_dropout": M.a_dropout()}
    base = M.vgg16_features(); head = M.a_vgg16_head()
    pc = {k: M.count_params(v) for k, v in full.items()}
    pc["a_vgg16"] = {"total": base.count_params() + head.count_params(), "trainable": head.count_params(), "non_trainable": base.count_params()}
    for k in B_RUNS:
        m, b = M.b_model(k, augment=False)
        tot = m.count_params()
        M.unfreeze(b, k)
        s2 = M.count_params(m)["trainable"]
        pc[k] = {"total": tot, "trainable_stage1": m.get_layer("softmax").count_params(), "trainable_stage2": s2}
    for k, v in pc.items():
        totals.append({"Mô hình": k, "Tổng số tham số": v["total"],
                       "Tham số được huấn luyện": v.get("trainable", f"giai đoạn 1: {v.get('trainable_stage1')}; giai đoạn 2: {v.get('trainable_stage2')}")})
    raw["totals"] = pc
    save_json(exp_dir("N1") / "metrics.json", raw); register("N1", [])
    print(save_table("N1", out, title="N1 — Số tham số từng lớp: tính tay theo công thức so với Keras"))
    print(save_table("N1", totals, "table_totals", title="N1 — Tổng số tham số của năm mô hình"))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(pc); vals = [pc[k]["total"] for k in names]
    ax.bar(names, vals, color=["tab:blue"] * 3 + ["tab:orange"] * 2)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v / 1e6:.2f} triệu", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(top=max(vals) * 4); ax.set_ylabel("Số tham số (thang log)"); ax.set_title("N1 — Tổng số tham số (xanh: hướng A, cam: hướng B)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "N1_params.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- N2
def n2():
    rows, srows, raw = [], [], {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in ["a_lenet", "a_vgg16", "b_mobilenetv3"]:
        pts = []
        for frac, suffix in [(0.25, "_frac25"), (0.5, "_frac50"), (1.0, "")]:
            m = meta(model + suffix)
            if m is None:
                continue
            pts.append((frac, m)); srows.append(srow("N2", model + suffix, m))
        if not pts:
            continue
        raw[model] = {str(f): {"n_train": m["n_train"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]} for f, m in pts}
        row = {"Mô hình": LABEL[model]}
        for f, m in pts:
            row[f"{int(f * 100)}% train ({m['n_train']} ảnh)"] = f"{m['macro_f1']:.3f}"
        if len(pts) == 3:
            row["Tăng từ 25% lên 100%"] = f"{pts[2][1]['macro_f1'] - pts[0][1]['macro_f1']:+.3f}"
        rows.append(row)
        ax.plot([m["n_train"] for _, m in pts], [m["macro_f1"] for _, m in pts], marker="o", label=LABEL[model])
        for _, m in pts:
            ax.annotate(f"{m['macro_f1']:.3f}", (m["n_train"], m["macro_f1"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Số ảnh huấn luyện"); ax.set_ylabel("Macro-F1 trên val (371 ảnh)"); ax.set_title("N2 — Đường cong học theo lượng dữ liệu"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "N2_learning_curve.png", dpi=130); plt.close(fig)
    save_json(exp_dir("N2") / "metrics.json", raw); register("N2", srows)
    print(save_table("N2", rows, title="N2 — Macro-F1 trên val theo tỉ lệ tập train được dùng"))


# ---------------------------------------------------------------- N3, N5 (dùng chung cách đo khoảng cách train–val)
def gap_row(run, name):
    m, h = meta(run), hist(run)
    b = h[m["best_epoch"] - 1]; last = h[-1]
    return {"Cấu hình": name, "Epoch chọn / tổng": f"{m['best_epoch']} / {m['epochs_run']}", "Macro-F1 val": f"{m['macro_f1']:.3f}",
            "Accuracy train (epoch chọn)": f"{b.get('train_eval_accuracy', m['train_accuracy_inference_mode']):.3f}", "Accuracy val (epoch chọn)": f"{b['val_accuracy']:.3f}",
            "Loss train (epoch cuối)": f"{last.get('train_eval_loss', last['loss']):.4f}", "Loss val (epoch cuối)": f"{last['val_loss']:.4f}",
            "Khoảng cách loss val − train (epoch cuối)": f"{last['val_loss'] - last.get('train_eval_loss', last['loss']):.4f}",
            "Loss val thấp nhất": f"{min(x['val_loss'] for x in h):.4f}"}


def curves_compare(runs, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for (run, name), color in zip(runs, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        h = hist(run); ep = [x["epoch"] for x in h]
        tl = "train_eval_loss" if "train_eval_loss" in h[0] else "loss"
        ta = "train_eval_accuracy" if "train_eval_accuracy" in h[0] else "accuracy"
        axes[0].plot(ep, [x[tl] for x in h], color=color, ls="--"); axes[0].plot(ep, [x["val_loss"] for x in h], color=color, label=name)
        axes[1].plot(ep, [x[ta] for x in h], color=color, ls="--"); axes[1].plot(ep, [x["val_accuracy"] for x in h], color=color, label=name)
    axes[0].set_yscale("log"); axes[0].set_title("Loss (đứt: train, liền: val)"); axes[1].set_title("Accuracy (đứt: train, liền: val)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def n3():
    runs = [("a_dropout", "a_dropout có Dropout(0,5)"), ("a_dropout_nodrop", "a_dropout không Dropout")]
    rows = [gap_row(r, n) for r, n in runs]
    save_json(exp_dir("N3") / "metrics.json", {r: {"metrics": meta(r), "history": hist(r)} for r, _ in runs})
    register("N3", [srow("N3", r, meta(r)) for r, _ in runs])
    print(save_table("N3", rows, title="N3 — Dropout và khoảng cách train–val (seed 42, 40 epoch, train chấm ở chế độ suy luận)"))
    curves_compare(runs, FIG_DIR / "N3_dropout.png", "N3 — a_dropout có và không có Dropout")
    # Lặp lại với 3 seed khởi tạo (42, 43, 44) để biết chênh lệch có vượt nhiễu giữa các lần huấn luyện không
    seeds = []
    for name, base in [("Có Dropout(0,5)", "a_dropout"), ("Không Dropout", "a_dropout_nodrop")]:
        ms = [(r, meta(r), hist(r)) for r in [base, base + "_seed43", base + "_seed44"] if meta(r)]
        f1 = [m["macro_f1"] for _, m, _ in ms]; gap = [h[-1]["val_loss"] - h[-1]["train_eval_loss"] for _, _, h in ms]
        vmin = [min(x["val_loss"] for x in h) for _, _, h in ms]
        seeds.append({"Cấu hình": name, "Số lần chạy": len(ms), "Macro-F1 val từng seed": ", ".join(f"{v:.3f}" for v in f1),
                      "Macro-F1 val (TB ± ĐLC)": f"{np.mean(f1):.3f} ± {np.std(f1):.3f}",
                      "Khoảng cách loss val − train ở epoch 40 (TB ± ĐLC)": f"{np.mean(gap):.3f} ± {np.std(gap):.3f}",
                      "Loss val thấp nhất (TB ± ĐLC)": f"{np.mean(vmin):.3f} ± {np.std(vmin):.3f}"})
    print(save_table("N3", seeds, "table_seeds", title="N3 — Lặp lại với 3 seed khởi tạo (cùng cách chia dữ liệu)"))


def n5():
    runs = [("a_dropout", "a_dropout, không tăng cường"), ("a_dropout_aug", "a_dropout, có tăng cường")]
    runs_b = [("b_mobilenetv3_noaug", "b_mobilenetv3 giai đoạn 2, không tăng cường"), ("b_mobilenetv3", "b_mobilenetv3 giai đoạn 2, có tăng cường")]
    runs_b = [(r, n) for r, n in runs_b if meta(r)]
    rows = [gap_row(r, n) for r, n in runs + runs_b]
    save_json(exp_dir("N5") / "metrics.json", {r: {"metrics": meta(r), "history": hist(r)} for r, _ in runs + runs_b})
    register("N5", [srow("N5", r, meta(r)) for r, _ in runs + runs_b])
    print(save_table("N5", rows, title="N5 — Tăng cường ảnh (lật, xoay ±30°, zoom ±10%) lúc huấn luyện"))
    curves_compare(runs, FIG_DIR / "N5_augment_a_dropout.png", "N5 — a_dropout có và không có tăng cường ảnh")
    if runs_b:
        curves_compare(runs_b, FIG_DIR / "N5_augment_b_mobilenetv3.png", "N5 — b_mobilenetv3 (giai đoạn 2) có và không có tăng cường ảnh")


# ---------------------------------------------------------------- N4
def n4():
    spec = [("b_mobilenetv3_s1", "MobileNetV3-Small, giai đoạn 1 (đóng băng thân)"),
            ("b_mobilenetv3", "MobileNetV3-Small, giai đoạn 2, lr = 1e-5"),
            ("b_mobilenetv3_lr1e-4", "MobileNetV3-Small, giai đoạn 2, lr = 1e-4"),
            ("b_mobilenetv3_lr1e-3", "MobileNetV3-Small, giai đoạn 2, lr = 1e-3"),
            ("b_resnet50_s1", "ResNet50, giai đoạn 1 (đóng băng thân)"),
            ("b_resnet50", "ResNet50, giai đoạn 2 (mở băng conv5), lr = 1e-5")]
    rows, srows, raw = [], [], {}
    for run, name in spec:
        m = meta(run)
        if m is None:
            continue
        h = hist(run); raw[run] = {"metrics": m, "history": h}
        rows.append({"Cấu hình": name, "Tham số được huấn luyện": m["params"]["trainable"], "Macro-F1 val": f"{m['macro_f1']:.3f}",
                     "Accuracy val": f"{m['accuracy']:.3f}", "Loss val (epoch chọn)": f"{m['val_loss']:.4f}",
                     "Macro-F1 val sau epoch 1": f"{h[0]['val_macro_f1']:.3f}", "Macro-F1 val thấp nhất": f"{min(x['val_macro_f1'] for x in h):.3f}",
                     "Epoch chọn / tổng": f"{m['best_epoch']} / {m['epochs_run']}", "Thời gian mỗi epoch (s)": f"{m['time_per_epoch_s']:.1f}"})
        srows.append(srow("N4", run, m))
    save_json(exp_dir("N4") / "metrics.json", raw); register("N4", srows)
    print(save_table("N4", rows, title="N4 — Đóng băng (giai đoạn 1) so với fine-tune (giai đoạn 2) và ảnh hưởng của learning rate"))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for run, name in spec:
        if run in raw and "mobilenet" in run:
            h = raw[run]["history"]
            ax.plot([x["epoch"] for x in h], [x["val_macro_f1"] for x in h], marker="o", ms=3, label=name.replace("MobileNetV3-Small, ", ""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 trên val"); ax.set_title("N4 — b_mobilenetv3: giai đoạn 1 và giai đoạn 2 theo learning rate"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "N4_finetune_lr.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- N6
def time_single(fn, imgs, warm=3):
    for i in range(warm):
        fn(imgs[i:i + 1])
    ts = []
    for i in range(len(imgs)):
        t0 = time.perf_counter(); fn(imgs[i:i + 1]); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def bench_step_s(model, X, y, steps=18, batch=32):
    """Thời gian trung bình một bước huấn luyện (lô 32 ảnh): chạy 1 epoch để biên dịch và làm nóng,
    rồi đo epoch thứ hai trên cùng tập dữ liệu (không biên dịch lại)."""
    import keras
    import tensorflow as tf
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy")
    ds = tf.data.Dataset.from_tensor_slices((X[:batch * steps], y[:batch * steps])).batch(batch).map(lambda a, b: (tf.cast(a, tf.float32), b)).cache()
    model.fit(ds, epochs=1, verbose=0)
    t0 = time.perf_counter(); model.fit(ds, epochs=1, verbose=0)
    return (time.perf_counter() - t0) / steps


def n6():
    """So sánh hướng A và B. Thời gian được đo lại riêng trong hàm này (không có tác vụ khác chạy cùng):
    thời gian một bước huấn luyện, thời gian thân đóng băng xử lý một ảnh, thời gian dự đoán một ảnh.
    Thời gian huấn luyện = (số bước mỗi epoch × số epoch đã chạy × giây/bước) + thời gian tính đặc trưng thân đóng băng."""
    from .cnn import models as M
    from .cnn.predict import full_model
    from .cnn.train import backbone_net, images_at, resize_batch
    tr, va = load_split("train"), load_split("val")
    imgs = np.stack([load_all_images(256)[i] for i in va["idx"].values[:60]])
    ytr = tr["label_id"].values; n_tr = len(ytr); steps_ep = int(np.ceil(n_tr / 32))
    rows, raw, srows = [], {}, []
    for model in A_RUNS + B_RUNS:
        run = final_run(model); m = meta(run); size = m["input_size"]
        fm = full_model(run)
        f = CKPT_DIR / run / "full.keras"; fm.save(f)
        import tensorflow as tf
        pred_fn = tf.function(lambda x: fm(x, training=False), reduce_retracing=True)   # đồ thị đã biên dịch, như khi triển khai
        ms = time_single(lambda x: pred_fn(tf.constant(resize_batch(x, size).astype(np.float32))).numpy(), imgs)
        xb = tf.constant(resize_batch(imgs[:32], size).astype(np.float32)); pred_fn(xb)
        t0 = time.perf_counter(); [pred_fn(xb).numpy() for _ in range(3)]; ms_batch = 1000 * (time.perf_counter() - t0) / 96
        Xb = np.asarray(images_at(size)[tr["idx"].values[:32 * 18]])
        t_feat = 0.0; parts = []
        if model in ("a_lenet", "a_dropout"):
            fresh = M.a_lenet(size) if model == "a_lenet" else M.a_dropout(size, augment=m["augment"])
            step = bench_step_s(fresh, Xb, ytr); t_train = step * steps_ep * m["epochs_run"]
            parts.append(f"{m['epochs_run']} epoch × {step * steps_ep:.1f} s")
        else:
            net = backbone_net(model); x64 = Xb[:64].astype(np.float32)
            net.predict(x64[:8], verbose=0); t0 = time.perf_counter(); F = net.predict(x64, batch_size=32, verbose=0)
            t_feat = (time.perf_counter() - t0) / 64 * n_tr
            parts.append(f"tính đặc trưng thân đóng băng cho {n_tr} ảnh train: {t_feat:.0f} s")
            if model == "a_vgg16":
                Fb = np.repeat(F, 9, axis=0)[:32 * 18]
                step = bench_step_s(M.a_vgg16_head(F.shape[1:]), Fb, ytr); t_train = t_feat + step * steps_ep * m["epochs_run"]
                parts.append(f"đầu phân lớp {m['epochs_run']} epoch × {step * steps_ep:.1f} s")
            else:
                s1 = meta(model + "_s1"); t_train = t_feat + s1["train_time_s"]
                parts.append(f"giai đoạn 1: {s1['epochs_run']} epoch, {s1['train_time_s']:.0f} s")
                if not m.get("frozen_backbone"):
                    fresh, base = M.b_model(model, size, augment=m["augment"]); M.unfreeze(base, model)
                    step = bench_step_s(fresh, Xb, ytr); t_train += step * steps_ep * m["epochs_run"]
                    parts.append(f"giai đoạn 2: {m['epochs_run']} epoch × {step * steps_ep:.1f} s")
        raw[model] = {"run": run, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "params_total": fm.count_params(),
                      "file_mb": os.path.getsize(f) / 1e6, "train_time_s": t_train, "predict_ms_single_image": ms, "predict_ms_per_image_batch32": ms_batch, "epochs": m["epochs_run"],
                      "train_time_breakdown": "; ".join(parts), "recorded_train_time_s_during_run": m["train_time_s"]}
        rows.append({"Hướng": "A" if model.startswith("a_") else "B", "Mô hình": model, "Biến thể được chốt": run, "Macro-F1 val": f"{m['macro_f1']:.3f}",
                     "Accuracy val": f"{m['accuracy']:.3f}", "Số tham số": fm.count_params(), "File mô hình (MB)": f"{os.path.getsize(f) / 1e6:.1f}",
                     "Thời gian huấn luyện (s)": f"{t_train:.0f}", "Dự đoán 1 ảnh (ms, CPU)": f"{ms:.1f}", "Dự đoán theo lô 32 (ms/ảnh)": f"{ms_batch:.1f}", "Cách tính thời gian huấn luyện": "; ".join(parts)})
        srows.append(srow("N6", run, m, predict_ms_per_image=round(ms, 2)))
        print(rows[-1], flush=True)
    save_json(exp_dir("N6") / "metrics.json", raw); register("N6", srows)
    print(save_table("N6", rows, title="N6 — So sánh hướng A với hướng B (chấm trên val; thời gian đo riêng trên CPU 4 nhân)"))
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    names = list(raw); col = ["tab:blue" if n.startswith("a_") else "tab:orange" for n in names]
    for ax, key, title, log in [(axes[0], "macro_f1", "Macro-F1 trên val", False), (axes[1], "file_mb", "Dung lượng file mô hình (MB, thang log)", True),
                                (axes[2], "train_time_s", "Thời gian huấn luyện (s, thang log)", True),
                                (axes[3], "predict_ms_single_image", "Dự đoán 1 ảnh (ms, thang log)", True)]:
        v = [raw[n][key] for n in names]
        ax.bar(names, v, color=col)
        for i, x in enumerate(v):
            ax.text(i, x * (1.08 if log else 1.0) + (0 if log else 0.002), f"{x:.3f}" if key == "macro_f1" else f"{x:.1f}", ha="center", fontsize=8)
        if log:
            ax.set_yscale("log"); ax.set_ylim(top=max(v) * 3)
        else:
            ax.set_ylim(min(v) - 0.03, 1.01)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelsize=7, rotation=15)
    fig.suptitle("N6 — Hướng A (xanh) và hướng B (cam)"); fig.tight_layout(); fig.savefig(FIG_DIR / "N6_a_vs_b.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- N7
def n7():
    from .cnn.predict import embedding_model
    from .cnn.train import backbone_features, images_at
    tr, va = load_split("train"), load_split("val")
    itr, ytr, iva, yva = tr["idx"].values, tr["label_id"].values, va["idx"].values, va["label_id"].values
    X224 = images_at(224)

    def embed(run):
        f = CACHE_DIR / f"embedding_{run}.npy"
        if not f.exists():
            em = embedding_model(run)
            np.save(f, np.concatenate([em.predict(np.asarray(X224[a:a + 256]).astype(np.float32), batch_size=32, verbose=0) for a in range(0, len(X224), 256)]))
        return np.load(f)

    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    feats = []
    for name, parts in [("Hist HSV (8,8,8)", [["hsv_hist", {}]]), ("HOG 64×128", [["hog", {"size": [64, 128]}]])]:
        X, dims = features_all({"parts": parts}); feats.append((name, "thủ công", X, None))
    Xf, dimsf = features_all(sel["trad_fusion"]); feats.append(("Hist H-S + HOG (ghép, chuẩn hóa từng phần)", "thủ công", Xf, dimsf))
    vg = np.asarray(backbone_features("a_vgg16")).mean((1, 2)); feats.append(("VGG16 ImageNet, trung bình 7×7 (chưa học trên chuối)", "embedding", vg, None))
    feats.append(("a_vgg16, lớp Dense(256) áp chót", "embedding", embed(final_run("a_vgg16")), None))
    feats.append(("MobileNetV3 ImageNet, sau GAP (chưa fine-tune)", "embedding", np.asarray(backbone_features("b_mobilenetv3")), None))
    rb = final_run("b_mobilenetv3")
    feats.append((f"{rb}, sau GAP (đã fine-tune)", "embedding", embed(rb), None))
    rows, raw, srows = [], {}, []
    for name, kind, X, dims in feats:
        row = {"Đặc trưng": name, "Loại": kind, "Số chiều": X.shape[1]}
        for c, cname in [("knn", "KNN (k=5)"), ("logreg", "LogReg")]:
            if dims:
                from .exp_shape import fusion_clf
                clf = fusion_clf(c, dims, True, **({"C": 100.0} if c == "logreg" else {}))
            else:
                clf = make_clf(c)
            p = clf.fit(X[itr], ytr).predict(X[iva])
            f1, acc = f1_score(yva, p, average="macro"), accuracy_score(yva, p)
            row[cname] = f"{f1:.3f}"; raw[f"{name}|{c}"] = {"macro_f1": float(f1), "accuracy": float(acc), "n_features": int(X.shape[1])}
            srows.append({"experiment": "N7", "group": kind, "config": f"{name} + {c}", "condition": "clean", "eval_set": "val (học trên train)",
                          "macro_f1": round(float(f1), 4), "macro_f1_std": None, "accuracy": round(float(acc), 4), "n_features": int(X.shape[1])})
        rows.append(row)
    save_json(exp_dir("N7") / "metrics.json", raw); register("N7", srows)
    print(save_table("N7", rows, title="N7 — Feature vector thủ công so với embedding của CNN: học trên train (1729 ảnh), macro-F1 trên val (371 ảnh)"))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(rows)); h = 0.38
    for j, cname in enumerate(["KNN (k=5)", "LogReg"]):
        v = [float(r[cname]) for r in rows]
        ax.barh(y + (j - 0.5) * h, v, h, label=cname)
        for yi, x in zip(y + (j - 0.5) * h, v):
            ax.text(x + 0.002, yi, f"{x:.3f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['Đặc trưng']}\n({r['Loại']}, {r['Số chiều']} chiều)" for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlim(0.9, 1.012); ax.set_xlabel("Macro-F1 trên val"); ax.set_title("N7 — Đặc trưng thủ công và embedding với cùng bộ phân lớp"); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "N7_embedding.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- N8
N8_T = [("shuffle4", "Xáo trộn 4×4"), ("v0.6", "V×0,6"), ("v1.4", "V×1,4"), ("gray", "Ảnh xám"), ("noise5", "Nhiễu σ=5"), ("noise10", "Nhiễu σ=10"),
        ("noise20", "Nhiễu σ=20"), ("occlude25", "Che 25%"), ("rot-30", "Xoay −30°"), ("rot-15", "Xoay −15°"), ("rot15", "Xoay +15°"), ("rot30", "Xoay +30°")]


def n8():
    from .cnn.predict import Predictor
    tr, va = load_split("train"), load_split("val")
    itr, ytr, iva, yva = tr["idx"].values, tr["label_id"].values, va["idx"].values, va["label_id"].values
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    best_a = best_run([final_run(m) for m in A_RUNS]); best_b = best_run([final_run(m) for m in B_RUNS if meta(m)])
    models = {}
    for key, lab in [("trad_color", "Truyền thống: "), ("trad_hog", "Truyền thống: "), ("trad_fusion", "Truyền thống: ")]:
        cfg = sel[key]; X, dims = features_all(cfg)
        clf = build_clf(cfg, dims).fit(X[itr], ytr)
        models[lab + cfg["name"].split(" + LogReg")[0].split(" + KNN")[0]] = (lambda imgs, cfg=cfg, clf=clf: clf.predict(features_of_images(cfg, imgs)))
    extra = [("b_mobilenetv3", "Hướng B, lr 1e-5, có tăng cường (N5): "), ("b_mobilenetv3_noaug", "Hướng B, lr 1e-5, không tăng cường (N5): ")]
    for run, lab in [(best_a, "Hướng A tốt nhất: "), (best_b, "Hướng B tốt nhất: ")] + [e for e in extra if meta(e[0])]:
        P = Predictor(run); models[lab + run] = (lambda imgs, P=P: P(imgs).argmax(1))
    raw, rows, srows = {}, [], []
    for name, fn in models.items():
        res = {}
        for t, _ in [("clean", "Ảnh gốc")] + N8_T:
            imgs = np.stack([transformed_images(t)[i] for i in iva])
            p = fn(imgs)
            res[t] = {"macro_f1": float(f1_score(yva, p, average="macro")), "accuracy": float(accuracy_score(yva, p))}
            srows.append({"experiment": "N8", "group": name.split(":")[0], "config": name, "condition": t, "eval_set": "val (học trên train)",
                          "macro_f1": round(res[t]["macro_f1"], 4), "macro_f1_std": None, "accuracy": round(res[t]["accuracy"], 4)})
        raw[name] = res
        rows.append(dict({"Mô hình": name, "Ảnh gốc": f"{res['clean']['macro_f1']:.3f}"}, **{lab: f"{res[t]['macro_f1']:.3f}" for t, lab in N8_T}))
        print(rows[-1], flush=True)
    save_json(exp_dir("N8") / "metrics.json", dict(raw, _best_a=best_a, _best_b=best_b)); register("N8", srows)
    print(save_table("N8", rows, title="N8 — Macro-F1 trên val khi ảnh chấm bị biến đổi (mô hình học trên ảnh train sạch)"))
    names = list(raw); M = np.array([[raw[n][t]["macro_f1"] for t, _ in [("clean", "")] + N8_T] for n in names])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(["Ảnh gốc"] + [l for _, l in N8_T], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.grid(False)
    ax.set_title("N8 — Macro-F1 trên val theo phép biến đổi áp lên ảnh chấm"); fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "N8_robustness.png", dpi=130); plt.close(fig)


def runs():
    """Bảng tổng hợp mọi lần huấn luyện CNN trong results/cnn/."""
    rows, srows = [], []
    for d in sorted((RESULTS_DIR / "cnn").iterdir()):
        m = meta(d.name)
        if m is None:
            continue
        rows.append({"Run": m["run"], "Mô hình": m["model"], "Ảnh vào": f"{m['input_size']}×{m['input_size']}", "Số ảnh train": m["n_train"],
                     "lr": f"{m['lr']:g}", "Tăng cường": "có" if m["augment"] else "không", "Tham số huấn luyện": m["params"]["trainable"],
                     "Epoch chọn / tổng": f"{m['best_epoch']} / {m['epochs_run']}", "Macro-F1 val": f"{m['macro_f1']:.3f}", "Accuracy val": f"{m['accuracy']:.3f}",
                     "Accuracy train": f"{m['train_accuracy_inference_mode']:.3f}", "Giây / epoch": f"{m['time_per_epoch_s']:.1f}",
                     "Thời gian huấn luyện (s)": f"{m['train_time_s']:.0f}"})
        srows.append(srow("CNN", m["run"], m))
    register("cnn", srows)
    print(save_table("cnn", rows, "table_runs", title="Tất cả các lần huấn luyện CNN (chấm trên val, 371 ảnh)"))


def n4_bn():
    """N4 (phụ): vì sao phải giữ BatchNormalization đóng băng khi fine-tune. So đầu ra GAP của thân mạng
    ở chế độ huấn luyện (training=True) với chế độ suy luận, và accuracy của đầu phân lớp giai đoạn 1
    trên cùng 64 ảnh train khi BN để mở so với khi BN đóng băng."""
    from .cnn import models as M
    from .cnn.train import images_at
    import keras
    tr = load_split("train"); idx, y = tr["idx"].values[:64], tr["label_id"].values[:64]
    x = np.asarray(images_at(224)[idx]).astype(np.float32)
    rows, raw = [], {}
    for kind in B_RUNS:
        head = M.b_head(576 if kind == "b_mobilenetv3" else 2048); head.load_weights(CKPT_DIR / f"{kind}_s1" / "best.weights.h5")
        for mode in ("BN mở (trainable=True)", "BN đóng băng (cách đồ án dùng)"):
            m, base = M.b_model(kind, augment=False)
            if mode.startswith("BN mở"):
                base.trainable = True
                if kind == "b_resnet50":
                    for l in base.layers:
                        l.trainable = l.name.startswith("conv5") or isinstance(l, keras.layers.BatchNormalization)
            else:
                M.unfreeze(base, kind)
            m.get_layer("softmax").set_weights(head.get_layer("softmax").get_weights())
            emb = keras.Model(m.input, m.get_layer("embedding").output)
            e_inf = emb(x, training=False).numpy(); e_tr = emb(x, training=True).numpy()
            acc_inf = float((m(x, training=False).numpy().argmax(1) == y).mean())
            acc_tr = float((head(e_tr, training=False).numpy().argmax(1) == y).mean())
            rel = float(np.linalg.norm(e_tr - e_inf) / np.linalg.norm(e_inf))
            raw[f"{kind}|{mode}"] = {"relative_embedding_change": rel, "head_acc_inference_mode": acc_inf, "head_acc_on_training_mode_embedding": acc_tr}
            rows.append({"Thân mạng": kind, "Cách mở băng": mode, "Thay đổi tương đối của embedding (training=True so với suy luận)": f"{rel:.3f}",
                         "Accuracy đầu phân lớp GĐ1, embedding chế độ suy luận": f"{acc_inf:.3f}",
                         "Accuracy đầu phân lớp GĐ1, embedding chế độ huấn luyện": f"{acc_tr:.3f}"})
    save_json(exp_dir("N4") / "bn_check.json", raw)
    print(save_table("N4", rows, "table_batchnorm", title="N4 (phụ) — BatchNorm khi fine-tune: 64 ảnh train, lô 64"))


EXPS = {"RUNS": runs, "N1": n1, "N4BN": n4_bn, "N2": n2, "N3": n3, "N4": n4, "N5": n5, "N6": n6, "N7": n7, "N8": n8}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(EXPS), nargs="*")
    args = ap.parse_args()
    for k in (args.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
