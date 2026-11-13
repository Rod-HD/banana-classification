"""Thí nghiệm mất cân bằng dữ liệu I1-I2 (mục 6.6 của kế hoạch).

I1: Accuracy so với macro-F1 và recall từng lớp ở mô hình chưa xử lý mất cân bằng.
I2: không xử lý / class weight / oversampling / tăng cường ảnh cho lớp ít, trên mô hình tốt nhất
    của nhóm truyền thống (5-fold CV), của hướng A và của hướng B (val). Mọi phép lấy mẫu lại
    chỉ áp dụng trên phần huấn luyện.

    python -m src.exp_imbalance
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_fscore_support

from .classical import cv_eval, folds, make_clf
from .config import CLASS_NAMES, FIG_DIR, RESULTS_DIR, SEED
from .data import load_split
from .exp_common import dev_features, dev_set, register, strip, summary_row
from .metrics import compute_metrics
from .report_utils import exp_dir, pm, save_json, save_table
from .trad_config import build_clf, features_all

SHORT = [c.replace(" Kola", "") for c in CLASS_NAMES]
def cnn_i2_runs():
    """Run của hướng A và hướng B cho I2: biến thể đã chốt (không xử lý) và ba run huấn luyện lại
    với class weight (_cw), oversampling (_os), tăng cường lớp ít (_minaug)."""
    from .exp_cnn import final_run
    out = {}
    for d, model in [("A", "a_vgg16"), ("B", "b_mobilenetv3")]:
        base = final_run(model)
        out[d] = (base, {"Không xử lý": base, "Class weight": base + "_cw", "Oversampling": base + "_os", "Tăng cường lớp ít": base + "_minaug"})
    return out


def cnn_meta(run):
    f = RESULTS_DIR / "cnn" / run / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def per_class(y, p):
    pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    return pr, rc


def i1():
    ids, y = dev_set()
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    entries = []
    maj = np.full(len(y), np.bincount(y).argmax())
    entries.append(("Luôn đoán lớp đông nhất (Champa Kola)", "train+val", compute_metrics(y, maj)))
    weak = [("HOG 64×128 + KNN k=31", "hog", {}, "knn", {"n_neighbors": 31}), ("Hist HSV + KNN k=31", "hsv_hist", {}, "knn", {"n_neighbors": 31}),
            ("Moment màu + LogReg", "moments", {}, "logreg", {}), ("Cạnh Sobel + LogReg", "edge", {}, "logreg", {})]
    for name, f, fp, c, cp in weak:
        r = cv_eval(dev_features(f, **fp), y, lambda: make_clf(c, **cp))
        entries.append((name, "out-of-fold 5-fold", compute_metrics(y, r["oof_pred"])))
    for key in ("trad_edge", "trad_hog", "trad_color", "trad_fusion"):
        cfg = sel[key]; X, dims = features_all(cfg)
        r = cv_eval(X[ids], y, lambda: build_clf(cfg, dims))
        entries.append((cfg["name"], "out-of-fold 5-fold", compute_metrics(y, r["oof_pred"])))
    from .exp_cnn import final_run
    for run in ["a_lenet_frac25", "a_lenet", "a_dropout", final_run("a_vgg16"), "b_mobilenetv3_s1", final_run("b_mobilenetv3"), "b_resnet50_s1", "b_resnet50"]:
        m = cnn_meta(run)
        if m:
            entries.append((f"CNN {run}", "val", m))
    rows, raw = [], {}
    for name, ev, m in entries:
        rc = [m["per_class"][c]["recall"] for c in CLASS_NAMES]
        raw[name] = {"eval": ev, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "recall": rc}
        rows.append(dict({"Mô hình (chưa xử lý mất cân bằng)": name, "Chấm trên": ev, "Accuracy": f"{m['accuracy']:.3f}", "Macro-F1": f"{m['macro_f1']:.3f}",
                          "Accuracy − Macro-F1": f"{m['accuracy'] - m['macro_f1']:+.3f}"},
                         **{f"Recall {s}": f"{r:.3f}" for s, r in zip(SHORT, rc)}, **{"Recall thấp nhất": f"{min(rc):.3f} ({SHORT[int(np.argmin(rc))]})"}))
    save_json(exp_dir("I1") / "metrics.json", raw); register("I1", [])
    print(save_table("I1", rows, title="I1: Accuracy so với macro-F1 và recall từng lớp (mô hình chưa xử lý mất cân bằng)"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    a = [raw[n]["accuracy"] for n in raw]; f = [raw[n]["macro_f1"] for n in raw]; mn = [min(raw[n]["recall"]) for n in raw]
    ax.scatter(a, f, label="Macro-F1", marker="o"); ax.scatter(a, mn, label="Recall của lớp kém nhất", marker="x", color="tab:red")
    lo = min(min(a), min(mn)) - 0.02
    ax.plot([lo, 1], [lo, 1], color="gray", lw=1, ls="--", label="bằng Accuracy")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("Chỉ số"); ax.set_title("I1: Accuracy che khuất lớp bị nhận kém"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "I1_accuracy_vs_macro_f1.png", dpi=130); plt.close(fig)


AUG_VARIANTS = ["hflip", "vflip", "rot15", "rot-15"]


def trad_i2(name, cfg, key):
    """Bốn cách xử lý trên một cấu hình truyền thống, 5-fold CV; trả về danh sách dòng bảng."""
    ids, y = dev_set()
    X, dims = features_all(cfg); X = X[ids]
    Xaug = [features_all(cfg, v)[0][ids] for v in AUG_VARIANTS]

    def oversample(Xtr, ytr, tr):
        return RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)

    def minority_aug(Xtr, ytr, tr):
        rng = np.random.default_rng(SEED); n = np.bincount(ytr).max(); xs, ys = [Xtr], [ytr]
        for c in range(4):
            pool = tr[ytr == c]; need = n - len(pool)
            pairs = [(i, v) for v in range(len(AUG_VARIANTS)) for i in pool]
            pick = rng.choice(len(pairs), need, replace=need > len(pairs))
            xs.append(np.stack([Xaug[pairs[k][1]][pairs[k][0]] for k in pick]) if need else np.empty((0, X.shape[1]), X.dtype)); ys.append(np.full(need, c))
        return np.concatenate(xs), np.concatenate(ys)

    cw_cfg = dict(cfg, cparams=dict(cfg["cparams"], class_weight="balanced"))
    methods = [("Không xử lý", cfg, None), ("Class weight", cw_cfg, None), ("Oversampling", cfg, oversample), ("Tăng cường lớp ít", cfg, minority_aug)]
    if cfg["clf"] == "knn":
        methods.pop(1)
    out = []
    for mname, c, ft in methods:
        r = cv_eval(X, y, lambda: build_clf(c, dims), fit_transform=ft)
        pr, rc = per_class(y, r["oof_pred"])
        fold_pr, fold_rc = [], []
        for _, te in folds(y):
            a, b = per_class(y[te], r["oof_pred"][te]); fold_pr.append(a); fold_rc.append(b)
        out.append({"model": name, "group": key, "method": mname, "eval": "5-fold CV", "macro_f1": r["macro_f1"], "macro_f1_std": r["macro_f1_std"],
                    "accuracy": r["accuracy"], "precision": pr.tolist(), "recall": rc.tolist(),
                    "precision_std": np.std(fold_pr, 0).tolist(), "recall_std": np.std(fold_rc, 0).tolist(), "res": strip(r)})
    return out


def i2():
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    recs = trad_i2("Truyền thống tốt nhất: " + sel["trad_fusion"]["name"], sel["trad_fusion"], "truyền thống")
    weak = {"parts": [["edge", {"op": "sobel"}]], "clf": "logreg", "cparams": {"C": 1.0}, "scale": "standard"}
    recs += trad_i2("Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg", weak, "truyền thống (minh họa)")
    for direction, (base, runs) in cnn_i2_runs().items():
        for mname, run in runs.items():
            m = cnn_meta(run)
            if m is None:
                continue
            recs.append({"model": f"Hướng {direction} tốt nhất: {base}", "group": f"CNN hướng {direction}", "method": mname, "eval": "val",
                         "macro_f1": m["macro_f1"], "macro_f1_std": None, "accuracy": m["accuracy"],
                         "precision": [m["per_class"][c]["precision"] for c in CLASS_NAMES], "recall": [m["per_class"][c]["recall"] for c in CLASS_NAMES],
                         "run": run})
    rows, srows = [], []
    for r in recs:
        f1 = pm(r["macro_f1"], r["macro_f1_std"]) if r["macro_f1_std"] is not None else f"{r['macro_f1']:.3f}"
        rows.append({"Mô hình": r["model"], "Cách xử lý": r["method"], "Chấm": r["eval"], "Macro-F1": f1, "Accuracy": f"{r['accuracy']:.3f}",
                     "Precision Bangla (ít nhất)": f"{r['precision'][0]:.3f}", "Recall Bangla (ít nhất)": f"{r['recall'][0]:.3f}",
                     "Precision Champa (đông nhất)": f"{r['precision'][1]:.3f}", "Recall Champa (đông nhất)": f"{r['recall'][1]:.3f}",
                     "Recall thấp nhất": f"{min(r['recall']):.3f} ({SHORT[int(np.argmin(r['recall']))]})"})
        srows.append({"experiment": "I2", "group": r["group"], "config": f"{r['model']} | {r['method']}", "condition": "clean", "eval_set": r["eval"],
                      "macro_f1": round(r["macro_f1"], 4), "macro_f1_std": None if r["macro_f1_std"] is None else round(r["macro_f1_std"], 4),
                      "accuracy": round(r["accuracy"], 4)})
    save_json(exp_dir("I2") / "metrics.json", [{k: v for k, v in r.items() if k != "res"} for r in recs]); register("I2", srows)
    print(save_table("I2", rows, title="I2: Các cách xử lý mất cân bằng (lấy mẫu lại chỉ trên phần huấn luyện)"))
    full = [dict({"Mô hình": r["model"], "Cách xử lý": r["method"]}, **{f"P {s}": f"{v:.3f}" for s, v in zip(SHORT, r["precision"])},
                 **{f"R {s}": f"{v:.3f}" for s, v in zip(SHORT, r["recall"])}) for r in recs]
    save_table("I2", full, "table_per_class", title="I2: Precision (P) và Recall (R) từng lớp")
    models = list(dict.fromkeys(r["model"] for r in recs))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.9), sharey=False)
    for ax, mdl in zip(np.atleast_1d(axes), models):
        rs = [r for r in recs if r["model"] == mdl]; x = np.arange(len(rs)); w = 0.2
        for j, (key, cls, lab) in enumerate([("precision", 0, "Precision Bangla"), ("recall", 0, "Recall Bangla"), ("precision", 1, "Precision Champa"), ("recall", 1, "Recall Champa")]):
            ax.bar(x + (j - 1.5) * w, [r[key][cls] for r in rs], w, label=lab)
        lo = min(min(r["precision"][0], r["recall"][0], r["precision"][1], r["recall"][1]) for r in rs)
        ax.set_ylim(max(0, lo - 0.05), 1.005); ax.set_xticks(x); ax.set_xticklabels([r["method"].replace(" ", "\n", 1) for r in rs], fontsize=7)
        ax.set_title(mdl.split(":")[0] + ":\n" + mdl.split(": ")[1][:38], fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("I2: Precision/Recall của lớp ít nhất (Bangla Kola) và lớp đông nhất (Champa Kola)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "I2_imbalance.png", dpi=130); plt.close(fig)


EXPS = {"I1": i1, "I2": i2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(EXPS), nargs="*")
    args = ap.parse_args()
    for k in (args.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
