"""Thí nghiệm đặc trưng màu M1-M6 (mục 6.2 của kế hoạch). 5-fold CV trên train+val;
phép biến đổi chỉ áp lên fold giữ lại.

    python -m src.exp_color            # chạy tất cả
    python -m src.exp_color --only M3
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR
from .exp_common import dev_features, dev_set, extract_ms, register, run, strip, summary_row
from .classical import cv_eval, make_clf
from .report_utils import exp_dir, pm, save_json, save_table

CLFS = ["knn", "logreg"]
CLF_NAME = {"knn": "KNN (k=5)", "logreg": "LogReg", "svm": "Linear SVM"}
FEAT_NAME = {"rgb_hist": "Hist RGB (3×32)", "hsv_hist": "Hist HSV (8,8,8)", "hs_hist": "Hist H-S (8,8)",
             "moments": "Moment màu", "dominant": "Màu chủ đạo (k=3)", "ccv": "CCV (64 màu)",
             "quant_hist": "Hist 64 màu", "edge": "Cạnh Sobel", "hog": "HOG 64×128"}


def bar_groups(ax, labels, series, ylabel, title, ylim=None, fmt="{:.3f}"):
    """Biểu đồ cột theo nhóm; nhãn giá trị đặt phía trên thanh sai số, chú giải nằm ngoài vùng vẽ."""
    x = np.arange(len(labels)); w = 0.8 / len(series)
    top = max(max(np.add(m, s)) for m, s in series.values())
    span = (ylim[1] - ylim[0]) if ylim else max(top, 1e-6)
    for k, (name, (m, s)) in enumerate(series.items()):
        xs = x + (k - (len(series) - 1) / 2) * w
        ax.bar(xs, m, w, yerr=s, capsize=2, label=name)
        for xi, v, e in zip(xs, m, s):
            ax.text(xi, max(v, 0) + e + 0.015 * span, fmt.format(v), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(top=top + 0.16 * span)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))


def m1():
    feats = ["rgb_hist", "hsv_hist", "moments", "ccv", "edge", "hog"]
    rows, srows, raw = [], [], {}
    for f in feats:
        for c in CLFS:
            r = run(f, c, variants=["shuffle4"])
            v = r["variants"]["shuffle4"]
            raw[f"{f}+{c}"] = strip(r)
            rows.append({"Đặc trưng": FEAT_NAME[f], "Bộ phân lớp": CLF_NAME[c], "Macro-F1 ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]),
                         "Macro-F1 ảnh xáo trộn 4×4": pm(v["macro_f1"], v["macro_f1_std"]),
                         "Độ sụt": f"{v['drop_macro_f1']:+.3f} ± {v['drop_macro_f1_std']:.3f}"})
            srows += [summary_row("M1", f"{f} + {c}", r), summary_row("M1", f"{f} + {c}", r, "shuffle4")]
    save_json(exp_dir("M1") / "metrics.json", raw); register("M1", srows)
    print(save_table("M1", rows, title="M1: Xáo trộn lưới 4×4 trên ảnh chấm"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, c in zip(axes, CLFS):
        ser = {"Ảnh gốc": ([raw[f"{f}+{c}"]["macro_f1"] for f in feats], [raw[f"{f}+{c}"]["macro_f1_std"] for f in feats]),
               "Xáo trộn 4×4": ([raw[f"{f}+{c}"]["variants"]["shuffle4"]["macro_f1"] for f in feats],
                                [raw[f"{f}+{c}"]["variants"]["shuffle4"]["macro_f1_std"] for f in feats])}
        bar_groups(ax, [FEAT_NAME[f] for f in feats], ser, "Macro-F1 (5-fold)", f"M1: {CLF_NAME[c]}", (0, 1.2))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / "M1_shuffle.png", dpi=130); plt.close(fig)


def m2():
    feats = ["rgb_hist", "hsv_hist", "hs_hist"]
    rows, srows, raw = [], [], {}
    for f in feats:
        for c in CLFS:
            r = run(f, c, variants=["v0.6", "v1.4"])
            raw[f"{f}+{c}"] = strip(r)
            a, b = r["variants"]["v0.6"], r["variants"]["v1.4"]
            rows.append({"Đặc trưng": FEAT_NAME[f], "Bộ phân lớp": CLF_NAME[c], "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]),
                         "V×0.6": pm(a["macro_f1"], a["macro_f1_std"]), "Sụt (V×0.6)": f"{a['drop_macro_f1']:+.3f} ± {a['drop_macro_f1_std']:.3f}",
                         "V×1.4": pm(b["macro_f1"], b["macro_f1_std"]), "Sụt (V×1.4)": f"{b['drop_macro_f1']:+.3f} ± {b['drop_macro_f1_std']:.3f}"})
            srows += [summary_row("M2", f"{f} + {c}", r)] + [summary_row("M2", f"{f} + {c}", r, v) for v in ("v0.6", "v1.4")]
    save_json(exp_dir("M2") / "metrics.json", raw); register("M2", srows)
    print(save_table("M2", rows, title="M2: Đổi độ sáng (nhân kênh V) trên ảnh chấm; macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, c in zip(axes, CLFS):
        ser = {lab: ([raw[f"{f}+{c}"]["variants"][v]["drop_macro_f1"] for f in feats],
                     [raw[f"{f}+{c}"]["variants"][v]["drop_macro_f1_std"] for f in feats])
               for lab, v in [("V×0.6", "v0.6"), ("V×1.4", "v1.4")]}
        bar_groups(ax, [FEAT_NAME[f] for f in feats], ser, "Độ sụt macro-F1", f"M2: {CLF_NAME[c]}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "M2_brightness.png", dpi=130); plt.close(fig)


def m3():
    _, y = dev_set()
    rows, srows, raw = [], [], {"rgb_hist": {}, "hsv_hist": {}}
    for b in [4, 8, 16, 32, 64, 256]:
        row = {"Đặc trưng": "Hist RGB từng kênh", "Số bin mỗi kênh": b, "Số chiều": 3 * b}
        for c in CLFS:
            r = run("rgb_hist", c, fparams={"bins": b})
            raw["rgb_hist"].setdefault(c, {})[b] = strip(r)
            row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"])
            srows.append(summary_row("M3", f"rgb_hist bins={b} + {c}", r))
        X = dev_features("rgb_hist", bins=b)
        row["Tỉ lệ phần tử bằng 0"] = f"{(X == 0).mean():.2f}"
        rows.append(row)
    for b in [2, 4, 8, 16]:
        row = {"Đặc trưng": "Hist HSV 3 chiều", "Số bin mỗi kênh": b, "Số chiều": b ** 3}
        for c in CLFS:
            r = run("hsv_hist", c, fparams={"bins": (b, b, b)})
            raw["hsv_hist"].setdefault(c, {})[b] = strip(r)
            row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"])
            srows.append(summary_row("M3", f"hsv_hist bins=({b},{b},{b}) + {c}", r))
        X = dev_features("hsv_hist", bins=(b, b, b))
        row["Tỉ lệ phần tử bằng 0"] = f"{(X == 0).mean():.2f}"
        rows.append(row)
    save_json(exp_dir("M3") / "metrics.json", raw); register("M3", srows)
    print(save_table("M3", rows, title="M3: Quét số bin; macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, f, t in [(axes[0], "rgb_hist", "Hist RGB từng kênh"), (axes[1], "hsv_hist", "Hist HSV 3 chiều")]:
        for c in CLFS:
            bs = sorted(raw[f][c])
            ax.errorbar(bs, [raw[f][c][b]["macro_f1"] for b in bs], yerr=[raw[f][c][b]["macro_f1_std"] for b in bs],
                        marker="o", capsize=3, label=CLF_NAME[c])
        ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
        ax.set_xlabel("Số bin mỗi kênh"); ax.set_ylabel("Macro-F1 (5-fold)"); ax.set_title(f"M3: {t}"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "M3_bins.png", dpi=130); plt.close(fig)


def m4():
    feats = [("rgb_hist", {}), ("hsv_hist", {}), ("hs_hist", {}), ("moments", {}), ("dominant", {}), ("ccv", {}), ("quant_hist", {})]
    rows, srows, raw = [], [], {}
    for f, fp in feats:
        ms = extract_ms(f, **fp)
        row = {"Đặc trưng": FEAT_NAME[f], "Số chiều": None, "Rút đặc trưng (ms/ảnh)": f"{ms:.2f}"}
        for c in ["knn", "logreg", "svm"]:
            r = run(f, c, fparams=fp)
            raw[f"{f}+{c}"] = dict(strip(r), extract_ms_per_image=ms)
            row["Số chiều"] = r["n_features"]
            row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"])
            if c != "svm":
                row[f"Huấn luyện {CLF_NAME[c]} (s)"] = f"{r['fit_time_s']:.3f}"
                row[f"Dự đoán {CLF_NAME[c]} (ms/ảnh)"] = f"{r['predict_ms_per_image']:.3f}"
            srows.append(summary_row("M4", f"{f} + {c}", r, extract_ms_per_image=round(ms, 3)))
        rows.append(row)
    save_json(exp_dir("M4") / "metrics.json", raw); register("M4", srows)
    print(save_table("M4", rows, title="M4: So sánh các đặc trưng màu: macro-F1 5-fold, số chiều, thời gian"))
    names = [FEAT_NAME[f] for f, _ in feats]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ser = {CLF_NAME[c]: ([raw[f"{f}+{c}"]["macro_f1"] for f, _ in feats], [raw[f"{f}+{c}"]["macro_f1_std"] for f, _ in feats])
           for c in ["knn", "logreg", "svm"]}
    bar_groups(ax, [f"{n}\n{raw[f + '+knn']['n_features']} chiều, {raw[f + '+knn']['extract_ms_per_image']:.1f} ms"
                    for n, (f, _) in zip(names, feats)], ser, "Macro-F1 (5-fold)", "M4: Các đặc trưng màu", (0.75, 1.08))
    fig.tight_layout(); fig.savefig(FIG_DIR / "M4_color_features.png", dpi=130); plt.close(fig)


def m5():
    rows, srows, raw = [], [], {}
    for f in ["quant_hist", "ccv"]:
        for c in CLFS:
            r = run(f, c, variants=["shuffle4"])
            raw[f"{f}+{c}"] = strip(r); v = r["variants"]["shuffle4"]
            rows.append({"Đặc trưng": FEAT_NAME[f], "Số chiều": r["n_features"], "Bộ phân lớp": CLF_NAME[c],
                         "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]), "Xáo trộn 4×4": pm(v["macro_f1"], v["macro_f1_std"]),
                         "Độ sụt": f"{v['drop_macro_f1']:+.3f} ± {v['drop_macro_f1_std']:.3f}"})
            srows += [summary_row("M5", f"{f} + {c}", r), summary_row("M5", f"{f} + {c}", r, "shuffle4")]
    # mức thay đổi của chính vector đặc trưng khi xáo trộn (khoảng cách L1 trung bình giữa ảnh gốc và ảnh xáo trộn)
    chg = {f: float(np.abs(dev_features(f) - dev_features(f, "shuffle4")).sum(1).mean()) for f in ["quant_hist", "ccv"]}
    raw["mean_l1_change_after_shuffle"] = chg
    save_json(exp_dir("M5") / "metrics.json", raw); register("M5", srows)
    print(save_table("M5", rows, title="M5: CCV so với histogram cùng mức lượng tử hóa (64 màu); macro-F1 5-fold"), chg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, c in zip(axes, CLFS):
        fs = ["quant_hist", "ccv"]
        ser = {"Ảnh gốc": ([raw[f"{f}+{c}"]["macro_f1"] for f in fs], [raw[f"{f}+{c}"]["macro_f1_std"] for f in fs]),
               "Xáo trộn 4×4": ([raw[f"{f}+{c}"]["variants"]["shuffle4"]["macro_f1"] for f in fs],
                                [raw[f"{f}+{c}"]["variants"]["shuffle4"]["macro_f1_std"] for f in fs])}
        bar_groups(ax, [FEAT_NAME[f] for f in fs], ser, "Macro-F1 (5-fold)", f"M5: {CLF_NAME[c]}", (0, 1.2))
    fig.tight_layout(); fig.savefig(FIG_DIR / "M5_ccv.png", dpi=130); plt.close(fig)


def m6():
    _, y = dev_set()
    feats = [("rgb_hist", {}), ("hsv_hist", {}), ("moments", {}), ("ccv", {}), ("edge", {}), ("hog", {}), ("hog", {"color": True})]
    rows, srows, raw = [], [], {}
    for f, fp in feats:
        name = FEAT_NAME[f] + (" (màu, kênh gradient lớn nhất)" if fp.get("color") else "")
        for c in CLFS:
            r = run(f, c, variants=["gray"], fparams=fp)
            v = r["variants"]["gray"]
            Xg = dev_features(f, "gray", **fp)
            rg = cv_eval(Xg, y, lambda: make_clf(c))        # huấn luyện lại trên ảnh xám
            key = f"{f}{'_color' if fp.get('color') else ''}+{c}"
            raw[key] = {"color_train_color_test": strip(r), "gray_train_gray_test": strip(rg)}
            rows.append({"Đặc trưng": name, "Bộ phân lớp": CLF_NAME[c],
                         "Học màu, chấm màu": pm(r["macro_f1"], r["macro_f1_std"]),
                         "Học màu, chấm xám": pm(v["macro_f1"], v["macro_f1_std"]),
                         "Học xám, chấm xám": pm(rg["macro_f1"], rg["macro_f1_std"]),
                         "Sụt khi bỏ màu (học lại)": f"{r['macro_f1'] - rg['macro_f1']:+.3f}"})
            srows += [summary_row("M6", f"{key}", r), summary_row("M6", f"{key}", r, "gray"),
                      summary_row("M6", f"{key} (học lại trên ảnh xám)", rg)]
    save_json(exp_dir("M6") / "metrics.json", raw); register("M6", srows)
    print(save_table("M6", rows, title="M6: Bỏ thông tin màu (ảnh xám); macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
    keys = [f"{f}{'_color' if fp.get('color') else ''}" for f, fp in feats]
    labels = [FEAT_NAME[f].replace(" 64×128", "") + ("\n(màu)" if fp.get("color") else "") for f, fp in feats]
    for ax, c in zip(axes, CLFS):
        ser = {"Học màu, chấm màu": ([raw[f"{k}+{c}"]["color_train_color_test"]["macro_f1"] for k in keys],
                                     [raw[f"{k}+{c}"]["color_train_color_test"]["macro_f1_std"] for k in keys]),
               "Học màu, chấm xám": ([raw[f"{k}+{c}"]["color_train_color_test"]["variants"]["gray"]["macro_f1"] for k in keys],
                                     [raw[f"{k}+{c}"]["color_train_color_test"]["variants"]["gray"]["macro_f1_std"] for k in keys]),
               "Học xám, chấm xám": ([raw[f"{k}+{c}"]["gray_train_gray_test"]["macro_f1"] for k in keys],
                                     [raw[f"{k}+{c}"]["gray_train_gray_test"]["macro_f1_std"] for k in keys])}
        bar_groups(ax, labels, ser, "Macro-F1 (5-fold)", f"M6: {CLF_NAME[c]}", (0, 1.2))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / "M6_grayscale.png", dpi=130); plt.close(fig)


EXPS = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(EXPS), nargs="*")
    args = ap.parse_args()
    for k in (args.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
