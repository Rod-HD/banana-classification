"""Thí nghiệm bộ phân lớp P1-P6 (mục 6.4 của kế hoạch). 5-fold CV trên train+val.

    python -m src.exp_classifier            # chạy tất cả
    python -m src.exp_classifier --only P4
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from .classical import cv_eval, folds, make_clf
from .config import CLASS_NAMES, FIG_DIR, SEED
from .exp_common import dev_features, dev_set, register, run, strip, summary_row
from .report_utils import exp_dir, pm, save_json, save_table

FEATS = [("moments", "Moment màu (9 chiều)"), ("hsv_hist", "Hist HSV (512 chiều)"), ("hog", "HOG 64×128 (3780 chiều)")]
KS = list(range(1, 32, 2))


def p1():
    rows, srows, raw = [], [], {}
    fig, ax = plt.subplots(figsize=(8, 4))
    for f, fname in FEATS:
        res = {k: run(f, "knn", cparams={"n_neighbors": k}) for k in KS}
        raw[f] = {k: strip(r) for k, r in res.items()}
        m = np.array([res[k]["macro_f1"] for k in KS]); s = np.array([res[k]["macro_f1_std"] for k in KS])
        kb = KS[int(m.argmax())]
        rows.append({"Đặc trưng": fname, "k = số lớp + 1 = 5": pm(res[5]["macro_f1"], res[5]["macro_f1_std"]),
                     "k tốt nhất theo CV": kb, "Macro-F1 tại k tốt nhất": pm(res[kb]["macro_f1"], res[kb]["macro_f1_std"]),
                     "Chênh so với k=5": f"{res[kb]['macro_f1'] - res[5]['macro_f1']:+.3f}",
                     "k=1": pm(res[1]["macro_f1"], res[1]["macro_f1_std"]), "k=31": pm(res[31]["macro_f1"], res[31]["macro_f1_std"])})
        srows += [summary_row("P1", f"{f} + knn k={k}", res[k]) for k in KS]
        ax.errorbar(KS, m, yerr=s, marker="o", ms=4, capsize=2, label=fname)
        ax.scatter([kb], [m.max()], s=120, facecolors="none", edgecolors="red", zorder=5)
    ax.axvline(5, color="gray", ls="--", lw=1); ax.text(5.2, ax.get_ylim()[0] + 0.002, "k = số lớp + 1", fontsize=8, color="gray")
    ax.set_xticks(KS); ax.set_xlabel("k"); ax.set_ylabel("Macro-F1 (5-fold)")
    ax.set_title("P1: KNN theo k (vòng đỏ: k tốt nhất theo cross-validation)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "P1_knn_k.png", dpi=130); plt.close(fig)
    save_json(exp_dir("P1") / "metrics.json", raw); register("P1", srows)
    full = [dict({"k": k}, **{fname: pm(raw[f][k]["macro_f1"], raw[f][k]["macro_f1_std"]) for f, fname in FEATS}) for k in KS]
    save_table("P1", full, "table_all_k", title="P1: Macro-F1 5-fold theo k (Euclidean, chuẩn hóa z-score, bình chọn đa số)")
    print(save_table("P1", rows, title="P1: Quy tắc k = số lớp + 1 so với k chọn bằng cross-validation"))


def neighbor_rank_agreement(X, k=10):
    """Sau khi chuẩn hóa về vector đơn vị: tỉ lệ điểm truy vấn mà danh sách k láng giềng theo
    Euclidean trùng (đúng thứ tự) với danh sách theo cosine."""
    Xu = normalize(X)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)
    e = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(Xu[tr]).kneighbors(Xu[te], return_distance=False)
    c = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(Xu[tr]).kneighbors(Xu[te], return_distance=False)
    c_raw = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X[tr]).kneighbors(X[te], return_distance=False)
    e_raw = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(X[tr]).kneighbors(X[te], return_distance=False)
    return {"unit_euclid_vs_cosine_same_order": float((e == c).all(1).mean()),
            "unit_euclid_vs_cosine_same_set": float(np.mean([set(a) == set(b) for a, b in zip(e, c)])),
            "raw_euclid_vs_cosine_same_order": float((e_raw == c_raw).all(1).mean()),
            "raw_euclid_vs_cosine_same_set": float(np.mean([set(a) == set(b) for a, b in zip(e_raw, c_raw)]))}


def p2():
    combos = [("Euclidean", "none", "euclidean"), ("Euclidean", "unit", "euclidean"), ("Euclidean", "standard", "euclidean"),
              ("Manhattan", "none", "manhattan"), ("Manhattan", "unit", "manhattan"), ("Manhattan", "standard", "manhattan"),
              ("Cosine", "none", "cosine"), ("Cosine", "unit", "cosine"), ("Hellinger", "hellinger", "euclidean")]
    scale_name = {"none": "không", "unit": "vector đơn vị (L2)", "standard": "z-score", "hellinger": "L1 rồi lấy căn"}
    rows, srows, raw = [], [], {"rank_agreement": {}}
    for mname, scale, metric in combos:
        row = {"Độ đo": mname, "Chuẩn hóa": scale_name[scale]}
        for f, fname in FEATS:
            if mname == "Hellinger" and f == "moments":
                row[fname] = "không áp dụng (có giá trị âm)"; continue
            r = run(f, "knn", scale=scale, cparams={"metric": metric})
            raw[f"{f}|{mname}|{scale}"] = strip(r)
            row[fname] = pm(r["macro_f1"], r["macro_f1_std"])
            srows.append(summary_row("P2", f"{f} + knn k=5 {mname} / chuẩn hóa {scale}", r))
        rows.append(row)
    agree = []
    for f, fname in FEATS:
        a = neighbor_rank_agreement(dev_features(f)); raw["rank_agreement"][f] = a
        agree.append({"Đặc trưng": fname,
                      "Đã chuẩn hóa đơn vị: cùng 10 láng giềng, cùng thứ tự": f"{100 * a['unit_euclid_vs_cosine_same_order']:.1f}%",
                      "Đã chuẩn hóa đơn vị: cùng tập 10 láng giềng": f"{100 * a['unit_euclid_vs_cosine_same_set']:.1f}%",
                      "Chưa chuẩn hóa: cùng tập 10 láng giềng": f"{100 * a['raw_euclid_vs_cosine_same_set']:.1f}%"})
    save_json(exp_dir("P2") / "metrics.json", raw); register("P2", srows)
    print(save_table("P2", rows, title="P2: KNN (k=5) theo độ đo và cách chuẩn hóa; macro-F1 5-fold"))
    print(save_table("P2", agree, "table_rank_agreement",
                     title="P2: Euclidean so với cosine: tỉ lệ ảnh truy vấn có cùng 10 láng giềng gần nhất (420 ảnh truy vấn, 1680 ảnh tham chiếu)"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (f, fname) in zip(axes, FEATS):
        labs, m, s = [], [], []
        for mname, scale, _ in combos:
            key = f"{f}|{mname}|{scale}"
            if key in raw:
                labs.append(f"{mname}\n{scale_name[scale]}"); m.append(raw[key]["macro_f1"]); s.append(raw[key]["macro_f1_std"])
        ax.barh(range(len(m)), m, xerr=s, capsize=2, color="tab:blue")
        for i, v in enumerate(m):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_yticks(range(len(m))); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
        ax.set_xlim(0.6, 1.06); ax.set_title(fname, fontsize=9); ax.set_xlabel("Macro-F1 (5-fold)")
    fig.suptitle("P2: KNN (k=5): độ đo khoảng cách và chuẩn hóa")
    fig.tight_layout(); fig.savefig(FIG_DIR / "P2_metrics.png", dpi=130); plt.close(fig)


def p3():
    _, y = dev_set()
    rows, srows, raw = [], [], {}
    for f, fname in FEATS:
        for k in (5, 15):
            rec = {}
            for w in ("uniform", "distance"):
                r = run(f, "knn", cparams={"n_neighbors": k, "weights": w})
                pc = recall_score(y, r["oof_pred"], average=None)
                fold_rec = [recall_score(y[te], r["oof_pred"][te], labels=[0], average=None)[0] for _, te in folds(y)]
                raw[f"{f}|k={k}|{w}"] = dict(strip(r), recall_per_class=pc.tolist(), recall_bangla_fold_std=float(np.std(fold_rec)))
                rec[w] = (r, pc, np.std(fold_rec))
                srows.append(summary_row("P3", f"{f} + knn k={k} weights={w}", r))
            (ru, pu, su), (rd, pd_, sd) = rec["uniform"], rec["distance"]
            rows.append({"Đặc trưng": fname, "k": k, "Macro-F1 uniform": pm(ru["macro_f1"], ru["macro_f1_std"]),
                         "Macro-F1 distance": pm(rd["macro_f1"], rd["macro_f1_std"]),
                         "Recall Bangla Kola (lớp ít ảnh nhất) uniform": pm(pu[0], su),
                         "Recall Bangla Kola distance": pm(pd_[0], sd),
                         "Recall thấp nhất trong 4 lớp uniform → distance": f"{pu.min():.3f} → {pd_.min():.3f}"})
    save_json(exp_dir("P3") / "metrics.json", raw); register("P3", srows)
    print(save_table("P3", rows, title="P3: Bình chọn đa số (uniform) so với ưu tiên điểm gần (distance); 5-fold"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (f, fname) in zip(axes, FEATS):
        x = np.arange(4); w = 0.2
        for j, (k, wt) in enumerate([(5, "uniform"), (5, "distance"), (15, "uniform"), (15, "distance")]):
            ax.bar(x + (j - 1.5) * w, raw[f"{f}|k={k}|{wt}"]["recall_per_class"], w, label=f"k={k}, {wt}")
        ax.set_xticks(x); ax.set_xticklabels([c.replace(" Kola", "") for c in CLASS_NAMES]); ax.set_ylim(0.7, 1.02)
        ax.set_title(fname, fontsize=9)
    axes[0].set_ylabel("Recall từng lớp (out-of-fold)"); axes[0].legend(fontsize=7)
    fig.suptitle("P3: Recall từng lớp của KNN: uniform so với distance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "P3_weights.png", dpi=130); plt.close(fig)


def p4(repeats: int = 3):
    _, y = dev_set()
    rows, raw = [], {}
    for f, fname in [("moments", "Moment màu (9 chiều)"), ("hsv_hist", "Hist HSV (512 chiều)"), ("hog", "HOG 64×128 (3780 chiều)")]:
        X = StandardScaler().fit_transform(dev_features(f))
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
        ref = None; row = {"Đặc trưng": fname}
        for alg in ("brute", "kd_tree", "ball_tree"):
            tb, tp = [], []
            for _ in range(repeats):
                t0 = time.perf_counter(); knn = KNeighborsClassifier(5, algorithm=alg, n_jobs=1).fit(X[tr], y[tr]); tb.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); p = knn.predict(X[te]); tp.append(time.perf_counter() - t0)
            dist, idx = knn.kneighbors(X[te])
            if ref is None:
                ref = (p, dist)
            same_pred = bool((p == ref[0]).all()); same_dist = bool(np.allclose(dist, ref[1], rtol=1e-4, atol=1e-4))
            raw[f"{f}|{alg}"] = {"build_time_s": float(np.median(tb)), "predict_time_s": float(np.median(tp)),
                                 "predict_ms_per_image": float(1000 * np.median(tp) / len(te)),
                                 "same_predictions_as_brute": same_pred, "same_neighbor_distances_as_brute": same_dist}
            row[f"{alg}: dựng (ms)"] = f"{1000 * np.median(tb):.1f}"
            row[f"{alg}: dự đoán 420 ảnh (ms)"] = f"{1000 * np.median(tp):.1f}"
            if alg != "brute":
                row[f"{alg}: kết quả trùng brute"] = "có" if (same_pred and same_dist) else "KHÔNG"
        rows.append(row)
    # Phần phụ: cùng đặc trưng 9 chiều nhưng tập tham chiếu lớn dần (nhân bản kèm nhiễu nhỏ)
    # để xem K-D Tree / Ball Tree bắt đầu có lợi từ cỡ dữ liệu nào.
    X = StandardScaler().fit_transform(dev_features("moments"))
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    rng = np.random.default_rng(SEED)
    rows_n = []
    for mult in (1, 5, 20, 60):
        Xr = np.concatenate([X[tr] + (rng.normal(0, 0.05, X[tr].shape) if m else 0) for m in range(mult)])
        yr = np.tile(y[tr], mult)
        row = {"Số điểm tham chiếu": len(Xr)}
        for alg in ("brute", "kd_tree", "ball_tree"):
            tp = []
            for _ in range(repeats):
                knn = KNeighborsClassifier(5, algorithm=alg, n_jobs=1).fit(Xr, yr)
                t0 = time.perf_counter(); knn.predict(X[te]); tp.append(time.perf_counter() - t0)
            raw[f"moments_x{mult}|{alg}"] = {"n_reference": int(len(Xr)), "predict_time_s": float(np.median(tp))}
            row[f"{alg}: dự đoán 420 ảnh (ms)"] = f"{1000 * np.median(tp):.1f}"
        rows_n.append(row)
    print(save_table("P4", rows_n, "table_scaling", title="P4 (phụ): Đặc trưng 9 chiều, tập tham chiếu nhân bản kèm nhiễu Gauss σ=0.05 (dữ liệu tổng hợp, chỉ để đo thời gian)"))
    save_json(exp_dir("P4") / "metrics.json", raw); register("P4", [])
    print(save_table("P4", rows, title="P4: Brute Force, K-D Tree, Ball Tree (k=5, 1680 ảnh tham chiếu, 420 ảnh truy vấn, 1 luồng, trung vị 3 lần)"))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fs = ["moments", "hsv_hist", "hog"]; x = np.arange(3); w = 0.26
    for j, alg in enumerate(("brute", "kd_tree", "ball_tree")):
        v = [1000 * raw[f"{f}|{alg}"]["predict_time_s"] for f in fs]
        ax.bar(x + (j - 1) * w, v, w, label=alg)
        for xi, t in zip(x + (j - 1) * w, v):
            ax.text(xi, t * 1.1, f"{t:.0f}" if t >= 10 else f"{t:.1f}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["9 chiều (moment)", "512 chiều (hist HSV)", "3780 chiều (HOG)"])
    ax.set_ylabel("Thời gian dự đoán 420 ảnh (ms, thang log)"); ax.set_title("P4: Thuật toán tìm láng giềng"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "P4_knn_algorithms.png", dpi=130); plt.close(fig)


def p5_p6():
    rows5, rows6, srows, raw = [], [], [], {}
    for f, fname in FEATS:
        row6 = {"Đặc trưng": fname}
        for c, cname in [("knn", "KNN (k=5)"), ("logreg", "Logistic Regression"), ("svm", "Linear SVM")]:
            r = run(f, c)
            raw[f"{f}|{c}"] = strip(r)
            rows5.append({"Đặc trưng": fname, "Bộ phân lớp": cname, "Huấn luyện 1680 ảnh (ms)": f"{1000 * r['fit_time_s']:.1f}",
                          "Dự đoán 420 ảnh (ms)": f"{1000 * r['predict_time_s']:.1f}",
                          "Dự đoán (ms/ảnh)": f"{r['predict_ms_per_image']:.4f}", "Macro-F1": pm(r["macro_f1"], r["macro_f1_std"])})
            row6[cname] = pm(r["macro_f1"], r["macro_f1_std"])
            srows.append(summary_row("P5", f"{f} + {c}", r))
        k, l = raw[f"{f}|knn"], raw[f"{f}|logreg"]
        row6["LogReg − KNN"] = f"{l['macro_f1'] - k['macro_f1']:+.3f}"
        rows6.append(row6)
    save_json(exp_dir("P5") / "metrics.json", raw); register("P5", srows)
    save_json(exp_dir("P6") / "metrics.json", {k: v for k, v in raw.items() if not k.endswith("svm")}); register("P6", [])
    print(save_table("P5", rows5, title="P5: Thời gian huấn luyện và dự đoán (trung bình mỗi fold, đặc trưng đã rút sẵn)"))
    print(save_table("P6", rows6, title="P6: Bộ phân lớp tuyến tính so với KNN theo số chiều của đặc trưng; macro-F1 5-fold"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(3); w = 0.26
    for j, (c, cname) in enumerate([("knn", "KNN (k=5)"), ("logreg", "LogReg"), ("svm", "Linear SVM")]):
        axes[0].bar(x + (j - 1) * w, [1000 * raw[f"{f}|{c}"]["fit_time_s"] for f, _ in FEATS], w, label=cname)
        axes[1].bar(x + (j - 1) * w, [1000 * raw[f"{f}|{c}"]["predict_time_s"] for f, _ in FEATS], w, label=cname)
    for ax, t in zip(axes, ["Huấn luyện 1680 ảnh (ms)", "Dự đoán 420 ảnh (ms)"]):
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["Moment (9)", "Hist HSV (512)", "HOG (3780)"]); ax.set_title("P5: " + t); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "P5_time.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for j, (c, cname) in enumerate([("knn", "KNN (k=5)"), ("logreg", "Logistic Regression")]):
        m = [raw[f"{f}|{c}"]["macro_f1"] for f, _ in FEATS]; s = [raw[f"{f}|{c}"]["macro_f1_std"] for f, _ in FEATS]
        ax.bar(x + (j - 0.5) * 0.38, m, 0.38, yerr=s, capsize=3, label=cname)
        for xi, v in zip(x + (j - 0.5) * 0.38, m):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Moment (9 chiều)", "Hist HSV (512)", "HOG (3780)"]); ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("Macro-F1 (5-fold)"); ax.set_title("P6: KNN so với bộ phân lớp tuyến tính"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "P6_linear_vs_knn.png", dpi=130); plt.close(fig)


EXPS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5_p6}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(EXPS) + ["P6"])
    args = ap.parse_args()
    todo = [("P5" if k == "P6" else k) for k in (args.only or EXPS)]
    for k in dict.fromkeys(todo):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
