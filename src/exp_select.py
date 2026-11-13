"""Chốt cấu hình cho nhóm truyền thống: quét siêu tham số của KNN, Logistic Regression và
Linear SVM trên các đặc trưng ứng viên bằng 5-fold CV (train+val), rồi chọn cấu hình tốt nhất
cho từng họ đặc trưng (màu, cạnh, HOG, ghép). Kết quả: results/select_traditional/.

    python -m src.exp_select
"""
from __future__ import annotations

import argparse

from .classical import cv_eval
from .exp_common import dev_set, register, strip, summary_row
from .metrics import save_result
from .report_utils import exp_dir, pm, save_json, save_table
from .trad_config import build_clf, features_all

HOG = ["hog", {"size": [64, 128], "cell": 8, "bins": 9}]
FAMILIES = {
    "màu": [("Hist HSV (8,8,8)", [["hsv_hist", {}]]), ("Hist H-S (8,8)", [["hs_hist", {}]]), ("Moment màu", [["moments", {}]])],
    "cạnh": [("Cạnh Roberts", [["edge", {"op": "roberts"}]]), ("Cạnh Sobel", [["edge", {"op": "sobel"}]])],
    "HOG": [("HOG 64×128", [HOG])],
    "ghép": [("Hist HSV + HOG", [["hsv_hist", {}], HOG]), ("Hist H-S + HOG", [["hs_hist", {}], HOG])],
}


def grid(family, parts):
    nonneg = all(p[0] in ("hsv_hist", "hs_hist", "rgb_hist", "hog") for p in parts)
    out = []
    multi = len(parts) > 1
    fusions = ["plain", "balanced"] if multi else [None]
    for fu in fusions:
        base = {"parts": parts, "fusion": fu} if multi else {"parts": parts}
        for k in (1, 3, 5):
            for metric, scale in [("euclidean", "standard"), ("manhattan", "standard")] + ([("euclidean", "hellinger")] if nonneg and not multi else []):
                out.append(dict(base, clf="knn", scale=scale, cparams={"n_neighbors": k, "metric": metric, "weights": "distance"}))
        if family == "cạnh":
            cs_lr, cs_svm = (0.01, 0.1, 1.0), (0.001, 0.01, 0.1)
        elif fu == "balanced":
            cs_lr, cs_svm = (1.0, 10.0, 100.0), (0.1, 1.0, 10.0)
        else:
            cs_lr, cs_svm = (0.01, 0.1, 1.0, 10.0), (0.001, 0.01, 0.1, 1.0)
        out += [dict(base, clf="logreg", scale="standard", cparams={"C": c}) for c in cs_lr]
        out += [dict(base, clf="svm", scale="standard", cparams={"C": c}) for c in cs_svm]
    return out


def describe(cfg):
    cp = cfg["cparams"]
    if cfg["clf"] == "knn":
        m = "Hellinger" if cfg["scale"] == "hellinger" else cp["metric"].capitalize()
        s = f"KNN k={cp['n_neighbors']}, {m}, ưu tiên điểm gần"
    else:
        s = f"{'LogReg' if cfg['clf'] == 'logreg' else 'Linear SVM'} C={cp['C']:g}"
    if cfg.get("fusion"):
        s += ", " + ("chuẩn hóa từng phần" if cfg["fusion"] == "balanced" else "z-score chung")
    return s


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ids, y = dev_set()
    rows, srows, selected, allres = [], [], {}, []
    for family, feats in FAMILIES.items():
        best = None
        for fname, parts in feats:
            Xall, dims = features_all({"parts": parts}); X = Xall[ids]
            per_clf = {}
            for cfg in grid(family, parts):
                r = cv_eval(X, y, lambda: build_clf(cfg, dims))
                rec = {"family": family, "feature": fname, "config": describe(cfg), "cfg": cfg, **strip(r)}
                allres.append(rec)
                srows.append(summary_row("select_traditional", f"{fname} + {describe(cfg)}", r))
                if cfg["clf"] not in per_clf or r["macro_f1"] > per_clf[cfg["clf"]][0]["macro_f1"]:
                    per_clf[cfg["clf"]] = (r, cfg)
                if best is None or r["macro_f1"] > best[0]["macro_f1"]:
                    best = (r, cfg, fname)
                print(f"{family:5s} {fname:18s} {describe(cfg):55s} {pm(r['macro_f1'], r['macro_f1_std'])}", flush=True)
            for c, (r, cfg) in per_clf.items():
                rows.append({"Họ": family, "Đặc trưng": fname, "Cấu hình tốt nhất của bộ phân lớp": describe(cfg), "Số chiều": r["n_features"],
                             "Macro-F1": pm(r["macro_f1"], r["macro_f1_std"]), "Accuracy": pm(r["accuracy"], r["accuracy_std"]),
                             "Huấn luyện (s)": f"{r['fit_time_s']:.2f}", "Dự đoán (ms/ảnh)": f"{r['predict_ms_per_image']:.3f}"})
        r, cfg, fname = best
        key = {"màu": "trad_color", "cạnh": "trad_edge", "HOG": "trad_hog", "ghép": "trad_fusion"}[family]
        selected[key] = dict(cfg, name=f"{fname} + {describe(cfg)}", family=family, cv_macro_f1=r["macro_f1"], cv_macro_f1_std=r["macro_f1_std"],
                             cv_accuracy=r["accuracy"])
        save_result(exp_dir("select_traditional") / key, y, r["oof_pred"], extra={"config": selected[key]["name"], "eval": "out-of-fold, 5-fold CV trên train+val",
                    "cv_macro_f1_mean": r["macro_f1"], "cv_macro_f1_std": r["macro_f1_std"]}, title=f"{selected[key]['name']} (out-of-fold)")
    best_key = max(selected, key=lambda k: selected[k]["cv_macro_f1"])
    selected["_best_overall"] = best_key
    save_json(exp_dir("select_traditional") / "selected.json", selected)
    save_json(exp_dir("select_traditional") / "metrics.json", [{k: v for k, v in r.items() if k != "cfg"} for r in allres])
    register("select_traditional", srows)
    print(save_table("select_traditional", rows, title="Chốt cấu hình truyền thống: cấu hình tốt nhất của mỗi bộ phân lớp trên từng đặc trưng (5-fold CV)"))
    sel_rows = [{"Vai trò": k, "Cấu hình": v["name"], "Macro-F1 (5-fold)": pm(v["cv_macro_f1"], v["cv_macro_f1_std"]),
                 "Accuracy": f"{v['cv_accuracy']:.3f}"} for k, v in selected.items() if not k.startswith("_")]
    print(save_table("select_traditional", sel_rows, "table_selected", title="Cấu hình truyền thống đã chốt"), "tốt nhất:", best_key)


if __name__ == "__main__":
    main()
