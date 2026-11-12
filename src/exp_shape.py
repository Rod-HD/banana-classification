"""Thí nghiệm đặc trưng cạnh và HOG C1–C5 (mục 6.3 của kế hoạch). 5-fold CV trên train+val;
phép biến đổi chỉ áp lên fold giữ lại.

    python -m src.exp_shape            # chạy tất cả
    python -m src.exp_shape --only C3
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classical import BlockScaler, cv_eval, make_clf
from .config import FIG_DIR
from .exp_color import CLF_NAME, CLFS, bar_groups
from .exp_common import dev_features, dev_set, extract_ms, register, run, strip, summary_row
from .features.hog import hog_dim
from .report_utils import exp_dir, pm, save_json, save_table

OPS = [("roberts", "Roberts (2×2)"), ("prewitt", "Prewitt (3×3)"), ("sobel", "Sobel (3×3)"), ("laplacian", "Laplacian (bậc hai)")]
NOISES = ["noise2", "noise5", "noise10", "noise20"]


def c1():
    rows, srows, raw = [], [], {}
    for op, oname in OPS:
        for c in CLFS:
            r = run("edge", c, variants=NOISES, fparams={"op": op})
            raw[f"{op}+{c}"] = strip(r); v = r["variants"]
            rows.append(dict({"Toán tử": oname, "Bộ phân lớp": CLF_NAME[c], "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"])},
                             **{f"Sụt σ={n[5:]}": f"{v[n]['drop_macro_f1']:+.3f} ± {v[n]['drop_macro_f1_std']:.3f}" for n in NOISES}))
            srows += [summary_row("C1", f"edge[{op}] + {c}", r)] + [summary_row("C1", f"edge[{op}] + {c}", r, n) for n in NOISES]
    # Mức thay đổi của chính vector đặc trưng: ||f(ảnh nhiễu) − f(ảnh gốc)|| / ||f(ảnh gốc)||, trung bình trên train+val
    rel = []
    for op, oname in OPS:
        X = dev_features("edge", op=op); row = {"Toán tử": oname}
        for n in NOISES:
            Xn = dev_features("edge", n, op=op)
            row[f"σ={n[5:]}"] = f"{float((np.linalg.norm(Xn - X, axis=1) / np.linalg.norm(X, axis=1)).mean()):.2f}"
        raw[f"{op}|relative_change"] = row; rel.append(row)
    save_json(exp_dir("C1") / "metrics.json", raw); register("C1", srows)
    print(save_table("C1", rows, title="C1 — Vector cạnh (512 chiều) theo toán tử; nhiễu Gauss trên ảnh chấm; macro-F1 5-fold"))
    print(save_table("C1", rel, "table_feature_change", title="C1 — Thay đổi tương đối của vector đặc trưng khi thêm nhiễu (‖f(nhiễu) − f(gốc)‖ / ‖f(gốc)‖)"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    sig = [0, 2, 5, 10, 20]
    for ax, c in zip(axes, CLFS):
        for op, oname in OPS:
            r = raw[f"{op}+{c}"]
            m = [r["macro_f1"]] + [r["variants"][n]["macro_f1"] for n in NOISES]
            s = [r["macro_f1_std"]] + [r["variants"][n]["macro_f1_std"] for n in NOISES]
            ax.errorbar(sig, m, yerr=s, marker="o", capsize=3, label=oname)
        ax.set_xticks(sig); ax.set_xlabel("Độ lệch chuẩn nhiễu Gauss σ"); ax.set_title(f"C1 — {CLF_NAME[c]}")
    axes[0].set_ylabel("Macro-F1 (5-fold)"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "C1_edge_noise.png", dpi=130); plt.close(fig)


def c2():
    confs = [("hog", {"norm": "L2"}, "HOG, chuẩn hóa block L2"), ("hog", {"norm": "none"}, "HOG, không chuẩn hóa block"),
             ("rgb_hist", {}, "Hist RGB (3×32)"), ("hsv_hist", {}, "Hist HSV (8,8,8)"), ("hs_hist", {}, "Hist H-S (8,8)")]
    rows, srows, raw = [], [], {}
    for f, fp, name in confs:
        for c in CLFS:
            r = run(f, c, variants=["v0.6", "v1.4"], fparams=fp)
            raw[f"{name}+{c}"] = strip(r); a, b = r["variants"]["v0.6"], r["variants"]["v1.4"]
            rows.append({"Đặc trưng": name, "Số chiều": r["n_features"], "Bộ phân lớp": CLF_NAME[c], "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]),
                         "Sụt (V×0,6)": f"{a['drop_macro_f1']:+.3f} ± {a['drop_macro_f1_std']:.3f}",
                         "Sụt (V×1,4)": f"{b['drop_macro_f1']:+.3f} ± {b['drop_macro_f1_std']:.3f}"})
            if f == "hog":
                srows += [summary_row("C2", f"{name} + {c}", r)] + [summary_row("C2", f"{name} + {c}", r, v) for v in ("v0.6", "v1.4")]
    # Phần phụ: tỉ lệ cell 8×8 gần như không có gradient (năng lượng < 1% trung vị của ảnh gốc) và
    # mức thay đổi tương đối của vector HOG sau khi đổi độ sáng.
    extra = []
    for norm, name in [("L2", "HOG, chuẩn hóa block L2"), ("none", "HOG, không chuẩn hóa block")]:
        X = dev_features("hog", norm=norm); row = {"Đặc trưng": name}
        for v, lab in [("v0.6", "V×0,6"), ("v1.4", "V×1,4")]:
            Xv = dev_features("hog", v, norm=norm)
            row[f"Thay đổi tương đối của vector ({lab})"] = f"{float((np.linalg.norm(Xv - X, axis=1) / np.linalg.norm(X, axis=1)).mean()):.2f}"
        extra.append(row)
    cells = {v: dev_features("hog", v, norm="none") for v in ("clean", "v0.6", "v1.4")}
    energy = {v: c.reshape(len(c), -1, 9).sum(-1) for v, c in cells.items()}
    thr = 0.01 * np.median(energy["clean"])
    flat = {"Ảnh gốc": float((energy["clean"] < thr).mean()), "V×0,6": float((energy["v0.6"] < thr).mean()),
            "V×1,4": float((energy["v1.4"] < thr).mean())}
    from .data import load_all_images
    from .transforms import transformed_images
    ids, _ = dev_set()
    sat = {lab: float(np.mean([(np.asarray(transformed_images(v)[i]).max(-1) == 255).mean() for i in ids[::10]]))
           for lab, v in [("Ảnh gốc", "clean"), ("V×0,6", "v0.6"), ("V×1,4", "v1.4")]}
    raw["_extra"] = {"relative_change": extra, "flat_cell_fraction": flat, "saturated_pixel_fraction": sat}
    extra2 = [{"Ảnh": k, "Tỉ lệ cell gần như không có gradient": f"{flat[k]:.3f}", "Tỉ lệ điểm ảnh bão hòa (kênh lớn nhất = 255)": f"{sat[k]:.3f}"} for k in flat]
    print(save_table("C2", extra, "table_feature_change", title="C2 — Thay đổi tương đối của vector HOG khi đổi độ sáng"))
    print(save_table("C2", extra2, "table_saturation", title="C2 — Bão hòa và vùng phẳng khi đổi độ sáng (tập train+val)"))
    save_json(exp_dir("C2") / "metrics.json", raw); register("C2", srows)
    print(save_table("C2", rows, title="C2 — Đổi độ sáng (nhân kênh V) với HOG có và không chuẩn hóa block, so với histogram màu; macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9))
    for ax, c in zip(axes, CLFS):
        ser = {lab: ([raw[f"{n}+{c}"]["variants"][v]["drop_macro_f1"] for _, _, n in confs],
                     [raw[f"{n}+{c}"]["variants"][v]["drop_macro_f1_std"] for _, _, n in confs]) for lab, v in [("V×0,6", "v0.6"), ("V×1,4", "v1.4")]}
        bar_groups(ax, [n.replace(", ", ",\n").replace("không chuẩn hóa block", "không chuẩn hóa\nblock").replace("chuẩn hóa block L2", "chuẩn hóa\nblock L2")
                        for _, _, n in confs], ser, "Độ sụt macro-F1", f"C2 — {CLF_NAME[c]}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "C2_hog_brightness.png", dpi=130); plt.close(fig)


def c3():
    rows, srows, raw = [], [], {}
    for size in [(64, 128), (128, 128)]:
        for cell in (8, 16):
            for bins in (6, 9, 12):
                fp = {"size": size, "cell": cell, "bins": bins}
                key = f"{size[0]}x{size[1]}_c{cell}_b{bins}"
                row = {"Kích thước ảnh (rộng×cao)": f"{size[0]}×{size[1]}", "Cell": f"{cell}×{cell}", "Số bin": bins,
                       "Số chiều": hog_dim(size, cell, bins), "Rút đặc trưng (ms/ảnh)": f"{extract_ms('hog', **fp):.2f}"}
                for c in ["knn", "logreg", "svm"]:
                    r = run("hog", c, fparams=fp)
                    assert r["n_features"] == row["Số chiều"]
                    raw[f"{key}+{c}"] = strip(r); row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"])
                    srows.append(summary_row("C3", f"hog {key} + {c}", r))
                rows.append(row)
    # biến thể: phiếu cứng (không chia cho hai bin lân cận), ảnh màu
    for label, fp in [("64×128, cell 8, 9 bin, phiếu cứng (như skimage)", {"soft": False}),
                      ("64×128, cell 8, 9 bin, ảnh màu (kênh gradient lớn nhất)", {"color": True})]:
        row = {"Biến thể": label}
        for c in ["knn", "logreg", "svm"]:
            r = run("hog", c, fparams=fp); raw[f"{label}+{c}"] = strip(r); row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"])
            srows.append(summary_row("C3", f"hog {label} + {c}", r))
        raw.setdefault("_variants", []).append(row)
    save_json(exp_dir("C3") / "metrics.json", raw); register("C3", srows)
    print(save_table("C3", rows, title="C3 — Quét tham số HOG (ảnh xám, block 2×2 cell, chuẩn hóa L2, chia phiếu cho hai bin gần nhất); macro-F1 5-fold"))
    print(save_table("C3", raw["_variants"], "table_variants", title="C3 — Biến thể cài đặt của HOG 64×128"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, mk in [("knn", "o"), ("logreg", "s"), ("svm", "^")]:
        d = [r["Số chiều"] for r in rows]; m = [raw[k]["macro_f1"] for k in raw if k.endswith("+" + c) and "x" in k.split("_")[0] and "bin," not in k]
        s = [raw[k]["macro_f1_std"] for k in raw if k.endswith("+" + c) and "x" in k.split("_")[0] and "bin," not in k]
        ax.errorbar(d, m, yerr=s, fmt=mk, capsize=2, label=CLF_NAME[c], alpha=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Số chiều của vector HOG (thang log)"); ax.set_ylabel("Macro-F1 (5-fold)")
    ax.set_title("C3 — 12 cấu hình HOG: độ chính xác theo số chiều"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "C3_hog_params.png", dpi=130); plt.close(fig)


def c4():
    confs = [("rgb_hist", {}, "Hist RGB (3×32)"), ("hsv_hist", {}, "Hist HSV (8,8,8)"), ("hs_hist", {}, "Hist H-S (8,8)"),
             ("moments", {}, "Moment màu"), ("hog", {}, "HOG 64×128"), ("edge", {}, "Cạnh Sobel")]
    rows, srows, raw = [], [], {}
    for f, fp, name in confs:
        for c in CLFS:
            r = run(f, c, variants=["occlude25"], fparams=fp); v = r["variants"]["occlude25"]
            raw[f"{name}+{c}"] = strip(r)
            rows.append({"Đặc trưng": name, "Loại": "cục bộ ghép lại" if f in ("hog", "edge") else "toàn cục", "Bộ phân lớp": CLF_NAME[c],
                         "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]), "Che 25%": pm(v["macro_f1"], v["macro_f1_std"]),
                         "Độ sụt": f"{v['drop_macro_f1']:+.3f} ± {v['drop_macro_f1_std']:.3f}"})
            srows += [summary_row("C4", f"{f} + {c}", r), summary_row("C4", f"{f} + {c}", r, "occlude25")]
    # Phần phụ: histogram sập là do đặc trưng hay do bước z-score? Chấm lại KNN không z-score và Hellinger.
    rows2 = []
    for f, name in [("rgb_hist", "Hist RGB (3×32)"), ("hsv_hist", "Hist HSV (8,8,8)"), ("hs_hist", "Hist H-S (8,8)")]:
        for scale, sname in [("standard", "z-score"), ("none", "không chuẩn hóa"), ("hellinger", "Hellinger")]:
            r = run(f, "knn", variants=["occlude25"], scale=scale); v = r["variants"]["occlude25"]
            raw[f"{name}+knn|{scale}"] = strip(r)
            rows2.append({"Đặc trưng": name, "KNN (k=5), chuẩn hóa": sname, "Ảnh gốc": pm(r["macro_f1"], r["macro_f1_std"]),
                          "Che 25%": pm(v["macro_f1"], v["macro_f1_std"]), "Độ sụt": f"{v['drop_macro_f1']:+.3f} ± {v['drop_macro_f1_std']:.3f}"})
    print(save_table("C4", rows2, "table_hist_scaling", title="C4 (phụ) — Histogram dưới phép che 25% theo cách chuẩn hóa của KNN"))
    save_json(exp_dir("C4") / "metrics.json", raw); register("C4", srows)
    print(save_table("C4", rows, title="C4 — Che một hình vuông 25% diện tích (tô đen, vị trí ngẫu nhiên) trên ảnh chấm; macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, c in zip(axes, CLFS):
        ser = {"Ảnh gốc": ([raw[f"{n}+{c}"]["macro_f1"] for _, _, n in confs], [raw[f"{n}+{c}"]["macro_f1_std"] for _, _, n in confs]),
               "Che 25%": ([raw[f"{n}+{c}"]["variants"]["occlude25"]["macro_f1"] for _, _, n in confs],
                           [raw[f"{n}+{c}"]["variants"]["occlude25"]["macro_f1_std"] for _, _, n in confs])}
        bar_groups(ax, [n for _, _, n in confs], ser, "Macro-F1 (5-fold)", f"C4 — {CLF_NAME[c]}", (0, 1.2))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / "C4_occlusion.png", dpi=130); plt.close(fig)


def fused(parts):
    """parts: danh sách (tên đặc trưng, tham số). Trả về (X ghép, danh sách số chiều)."""
    Xs = [dev_features(f, **fp) for f, fp in parts]
    return np.concatenate(Xs, axis=1), [x.shape[1] for x in Xs]


def fusion_clf(kind, dims, balanced=True, **kw):
    base = make_clf(kind, scale="none", **kw)
    return make_pipeline(BlockScaler(dims) if balanced else StandardScaler(), *[s for _, s in base.steps])


def c5():
    _, y = dev_set()
    hogp = {"size": (64, 128)}
    singles = [("Hist HSV (8,8,8)", [("hsv_hist", {})]), ("Hist H-S (8,8)", [("hs_hist", {})]), ("HOG 64×128", [("hog", hogp)])]
    combos = [("Hist HSV + HOG", [("hsv_hist", {}), ("hog", hogp)]), ("Hist H-S + HOG", [("hs_hist", {}), ("hog", hogp)]),
              ("Hist HSV + moment + HOG", [("hsv_hist", {}), ("moments", {}), ("hog", hogp)])]
    rows, srows, raw = [], [], {}
    for name, parts in singles:
        X, dims = fused(parts); row = {"Đặc trưng": name, "Cách ghép": "—", "Số chiều": X.shape[1]}
        for c in ["knn", "logreg", "svm"]:
            r = cv_eval(X, y, lambda: make_clf(c)); raw[f"{name}|single|{c}"] = strip(r)
            row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"]); srows.append(summary_row("C5", f"{name} + {c}", r))
        rows.append(row)
    for name, parts in combos:
        X, dims = fused(parts)
        for mode, bal in [("ghép thẳng, z-score chung", False), ("chuẩn hóa từng phần rồi ghép", True)]:
            row = {"Đặc trưng": name, "Cách ghép": mode, "Số chiều": X.shape[1]}
            for c in ["knn", "logreg", "svm"]:
                r = cv_eval(X, y, lambda: fusion_clf(c, dims, bal)); raw[f"{name}|{mode}|{c}"] = strip(r)
                row[CLF_NAME[c]] = pm(r["macro_f1"], r["macro_f1_std"]); srows.append(summary_row("C5", f"{name} ({mode}) + {c}", r))
            rows.append(row)
    save_json(exp_dir("C5") / "metrics.json", raw); register("C5", srows)
    print(save_table("C5", rows, title="C5 — Ghép đặc trưng màu và HOG; macro-F1 5-fold"))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    short = {"ghép thẳng, z-score chung": "(z-score chung)", "chuẩn hóa từng phần rồi ghép": "(chuẩn hóa từng phần)"}
    labels = [f"{r['Đặc trưng'].replace(' + ', ' +\n')}\n{short[r['Cách ghép']]}" if r["Cách ghép"] != "—" else r["Đặc trưng"] for r in rows]
    keys = [f"{r['Đặc trưng']}|{'single' if r['Cách ghép'] == '—' else r['Cách ghép']}" for r in rows]
    ser = {CLF_NAME[c]: ([raw[f"{k}|{c}"]["macro_f1"] for k in keys], [raw[f"{k}|{c}"]["macro_f1_std"] for k in keys]) for c in ["knn", "logreg", "svm"]}
    bar_groups(ax, labels, ser, "Macro-F1 (5-fold)", "C5 — Đặc trưng riêng lẻ và ghép", (0.9, 1.03))
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "C5_fusion.png", dpi=130); plt.close(fig)


EXPS = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(EXPS), nargs="*")
    args = ap.parse_args()
    for k in (args.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
