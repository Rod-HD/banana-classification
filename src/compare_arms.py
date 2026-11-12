"""So sánh bốn nhánh của mục 12 trên cùng tập test (371 ảnh gốc) và bộ kiểm tra khó:
G (pipeline gốc), X (xử lý dữ liệu, hai biến thể X-a và X-b), T (bản Augmented của tác giả, không rò rỉ), B (phương pháp của bài báo).

    python -m src.compare_arms
Kết quả: results/so_sanh_nhanh/table_models.md, table_arms.md, table_stress.md; hình results/figures/E_compare_arms.png.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, RESULTS_DIR
from .pipeline import CNN, TRAD
from .report_utils import exp_dir, save_json, save_table
from .stress import STRESS, STRESS_LABEL

ARMS = {"G": ("G — pipeline gốc", RESULTS_DIR / "so_sanh_nhanh" / "G"),
        "X-a": ("X-a — xử lý dữ liệu: tiền xử lý lúc suy luận", RESULTS_DIR / "E1_xu_ly" / "pipeline"),
        "X-b": ("X-b — xử lý dữ liệu: tăng cường có thay nền", RESULTS_DIR / "E1_xu_ly" / "pipeline_b"),
        "T": ("T — Augmented của tác giả (không rò rỉ)", RESULTS_DIR / "E2_aug_tac_gia" / "khong_ro_ri")}
SHORT = {"trad_color": "Hist HSV + LogReg", "trad_edge": "Cạnh Roberts + KNN", "trad_hog": "HOG + LogReg", "trad_fusion": "Hist H-S + HOG + LogReg",
         "a_lenet": "a_lenet", "a_dropout": "a_dropout", "a_vgg16": "a_vgg16", "b_mobilenetv3": "b_mobilenetv3", "b_resnet50": "b_resnet50"}


def load(arm_dir):
    out = {}
    for k in TRAD + CNN:
        f = arm_dir / k / "metrics.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    exp_dir("so_sanh_nhanh")
    data = {a: load(d) for a, (_, d) in ARMS.items()}
    data = {a: m for a, m in data.items() if m}
    # 1) từng cấu hình qua các nhánh
    rows = []
    for k in TRAD + CNN:
        row = {"Cấu hình": SHORT[k]}
        for a in data:
            m = data[a].get(k)
            row[f"{a}: macro-F1 test"] = f"{m['macro_f1']:.3f}" if m else "—"
            row[f"{a}: số ảnh sai"] = m["n_errors"] if m else "—"
            row[f"{a}: điểm bền"] = f"{m['robust_score']:.3f}" if m else "—"
        rows.append(row)
    common = [k for k in TRAD + CNN if all(k in data[a] for a in data)]
    for label, keys, need_all in [("**Trung bình 9 cấu hình**", TRAD + CNN, True), (f"**Trung bình {len(common)} cấu hình có ở mọi nhánh**", common, False)]:
        if not need_all and len(common) == len(TRAD + CNN):
            continue
        mean_row = {"Cấu hình": label}
        for a in data:
            ms = [data[a][k] for k in keys if k in data[a]]
            ok = len(ms) == len(keys)
            mean_row[f"{a}: macro-F1 test"] = f"**{np.mean([m['macro_f1'] for m in ms]):.3f}**" if ok else "—"
            mean_row[f"{a}: số ảnh sai"] = f"**{np.mean([m['n_errors'] for m in ms]):.1f}**" if ok else "—"
            mean_row[f"{a}: điểm bền"] = f"**{np.mean([m['robust_score'] for m in ms]):.3f}**" if ok else "—"
        rows.append(mean_row)
    print(save_table("so_sanh_nhanh", rows, "table_models", title="So sánh từng cấu hình qua các nhánh: macro-F1 trên test gốc (371 ảnh), số ảnh sai, điểm bền (trung bình macro-F1 trên 6 phép thử khó)"))

    # 2) đại diện mỗi nhánh: cấu hình có macro-F1 trên val cao nhất (hòa thì lấy cấu hình đứng trước trong danh sách)
    reps = []
    for a in data:
        k = max(data[a], key=lambda kk: (round(data[a][kk]["val"]["macro_f1"], 4), -(TRAD + CNN).index(kk)))
        reps.append((ARMS[a][0], SHORT[k], data[a][k]["val"]["macro_f1"], data[a][k]["macro_f1"], data[a][k]["macro_f1_ci95"], data[a][k]["n_errors"],
                     {s: data[a][k]["stress"][s]["macro_f1"] for s in STRESS}, data[a][k]["robust_score"], "macro-F1 trên val"))
    fb = RESULTS_DIR / "E3_bai_bao" / "metrics.json"
    if fb.exists():
        raw = json.loads(fb.read_text()); summ = json.loads((RESULTS_DIR / "E3_bai_bao" / "summary.json").read_text())
        from .exp_papers import CLF_NAME, EXTRACTORS
        picks = [("B1 — bài báo, trong phạm vi môn học (CNN + bộ phân lớp)", summ["best_by_dev_cv"].get("B1")),
                 ("B2 — bài báo, đúng phương pháp (ViT + SVM)", summ.get("paper_main"))]
        for name, key in picks:
            if key:
                r = raw[key]; ex, c = key.split("|")
                reps.append((name, f"{EXTRACTORS[ex][1].split(' (')[0]} + {CLF_NAME[c].split(' (')[0]}", r["dev_cv"]["macro_f1"], r["test"]["macro_f1"], r["test"]["macro_f1_ci95"],
                             r["test"]["n_errors"], r["stress"], r["robust_score"], "5-fold trên train+val"))
    arm_rows, stress_rows = [], []
    for name, cfg, dev, f1, ci, ne, st, rob, how in reps:
        arm_rows.append({"Nhánh": name, "Cấu hình đại diện": cfg, "Điểm lúc chọn": f"{dev:.3f} ({how})", "Macro-F1 test": f"{f1:.3f}",
                         "KTC 95%": f"[{ci[0]:.3f}; {ci[1]:.3f}]", "Số ảnh sai / 371": ne, "Điểm bền": f"{rob:.3f}"})
        stress_rows.append(dict({"Nhánh": name, "Cấu hình đại diện": cfg, "Ảnh gốc": f"{f1:.3f}"}, **{STRESS_LABEL[s]: f"{st[s]:.3f}" for s in STRESS}, **{"Điểm bền": f"{rob:.3f}"}))
    print(save_table("so_sanh_nhanh", arm_rows, "table_arms", title="So sánh các nhánh qua cấu hình đại diện (chọn theo điểm lúc phát triển, không theo test)"))
    print(save_table("so_sanh_nhanh", stress_rows, "table_stress", title="Cấu hình đại diện của mỗi nhánh trên từng phép thử của bộ kiểm tra khó (macro-F1, 371 ảnh test)"))
    save_json(RESULTS_DIR / "so_sanh_nhanh" / "summary.json", {"representatives": [{"arm": r[0], "config": r[1], "test_macro_f1": r[3], "n_errors": r[5], "robust": r[7]} for r in reps]})
    from .exp_common import register
    register("so_sanh_nhanh", [{"experiment": "so_sanh_nhanh", "group": f"Nhánh {a}", "config": SHORT[k], "condition": "clean", "eval_set": "test",
                                 "macro_f1": round(m["macro_f1"], 4), "macro_f1_std": None, "accuracy": round(m["accuracy"], 4)} for a in data for k, m in data[a].items()]
             + [{"experiment": "so_sanh_nhanh", "group": f"Nhánh {a}", "config": SHORT[k], "condition": s, "eval_set": "test",
                 "macro_f1": round(m["stress"][s]["macro_f1"], 4), "macro_f1_std": None, "accuracy": round(m["stress"][s]["accuracy"], 4)}
                for a in data for k, m in data[a].items() for s in STRESS])

    # 2b) phép che 25%: accuracy theo mức quả chuối bị che (mô hình còn đoán đúng khi không thấy quả thì đang dựa vào nền)
    from .data import load_split
    from .exp_segment import all_masks_v2
    from .stress import stress_sets
    te = load_split("test"); masks = all_masks_v2()[te["idx"].values]
    occ = stress_sets("test")["occlude25"].sum(-1) == 0
    cover = np.array([(o & (m > 0)).sum() / max((m > 0).sum(), 1) for o, m in zip(occ, masks)])
    bins = [("Quả bị che ≥ 90%", cover >= 0.9), ("Quả bị che 30–90%", (cover >= 0.3) & (cover < 0.9)), ("Quả bị che < 30%", cover < 0.3)]
    occ_rows = []
    for a in data:
        for k in ("b_mobilenetv3", "trad_fusion"):
            f = ARMS[a][1] / k / "predictions.npz"
            if f.exists():
                z = np.load(f); ok = z["occlude25"] == z["y"]
                occ_rows.append(dict({"Nhánh": a, "Cấu hình": SHORT[k]}, **{f"{name} ({int(sel.sum())} ảnh)": f"{ok[sel].mean():.3f}" for name, sel in bins}))
    from .config import CLASS_NAMES
    yte = te["label_id"].values
    comp = ", ".join(f"{int(n)} ảnh {CLASS_NAMES[c]}" for c, n in enumerate(np.bincount(yte[bins[0][1]], minlength=4)) if n)
    print(save_table("so_sanh_nhanh", occ_rows, "table_occlusion_split",
                     title=f"Phép che 25%: accuracy trên test theo tỉ lệ diện tích quả chuối bị ô đen che (nhóm bị che ≥ 90% gồm {comp})"))

    # 3) hình
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    keys = TRAD + CNN; x = np.arange(len(keys)); w = 0.8 / len(data); col = {"G": "tab:gray", "X-a": "tab:green", "X-b": "tab:olive", "T": "tab:purple"}
    for ax, field, title in [(axes[0], "macro_f1", "Macro-F1 trên test gốc (371 ảnh)"), (axes[1], "robust_score", "Điểm bền (trung bình 6 phép thử khó)")]:
        for j, a in enumerate(data):
            v = [data[a][k][field] if k in data[a] else np.nan for k in keys]
            ax.bar(x + (j - (len(data) - 1) / 2) * w, v, w, label=ARMS[a][0], color=col[a])
            for xi, val in zip(x + (j - (len(data) - 1) / 2) * w, v):
                if not np.isnan(val):
                    ax.text(xi, val + 0.004, f"{val:.3f}", ha="center", fontsize=6.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[k].replace(" + ", "\n+ ") for k in keys], fontsize=7)
        ax.set_ylim((0.8, 1.04) if field == "macro_f1" else (0.3, 1.1)); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, ncol=len(data), loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle("So sánh các nhánh dữ liệu (G, X-a, X-b, T) trên 9 cấu hình của pipeline gốc")
    fig.tight_layout(); fig.savefig(FIG_DIR / "E_compare_arms.png", dpi=130); plt.close(fig)
    if len(reps) > 1:
        fig, ax = plt.subplots(figsize=(12, 4.6)); tests = ["clean"] + STRESS; x = np.arange(len(tests)); w = 0.8 / len(reps)
        for j, (name, cfg, dev, f1, ci, ne, st, rob, how) in enumerate(reps):
            v = [f1] + [st[s] for s in STRESS]
            ax.bar(x + (j - (len(reps) - 1) / 2) * w, v, w, label=f"{name.split(' — ')[0]}: {cfg}")
        ax.set_xticks(x); ax.set_xticklabels([STRESS_LABEL[t] for t in tests]); ax.set_ylabel("Macro-F1 trên test"); ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False); ax.set_title("Cấu hình đại diện của mỗi nhánh trên ảnh gốc và 6 phép thử khó")
        fig.tight_layout(); fig.savefig(FIG_DIR / "E_compare_stress.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
