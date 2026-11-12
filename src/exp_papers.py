"""Nhánh B (mục 12.4): phương pháp của bài báo trên bộ BananaImageBD.

Ergün (2025, Scientific Reports 15, 10366): rút đặc trưng bằng mạng pretrained rồi phân lớp bằng SVM/GBM/MLP;
tốt nhất là ViT + SVM, chấm bằng 5-fold cross-validation trên bản Augmented.
  B1  trong phạm vi môn học: bộ rút đặc trưng là CNN (ResNet50, MobileNetV3-Small, SqueezeNet).
  B2  đúng như bài báo: ViT-B/16.
Ba giao thức chấm: (i) như bài báo; (ii) 5-fold giữ các file cùng ảnh gốc trong một fold; (iii) giao thức của đồ án.

    python -m src.exp_papers
Cần PyTorch + torchvision cho ViT và SqueezeNet (pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu).
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classical import make_clf
from .config import CACHE_DIR, FIG_DIR, RESULTS_DIR, SEED
from .pipeline import base_split
from .report_utils import exp_dir, save_json, save_table
from .stress import STRESS, STRESS_LABEL, stress_sets

EXP = "E3_bai_bao"
OUT = RESULTS_DIR / EXP
EXTRACTORS = {  # tên: (hướng, mô tả, thư viện)
    "resnet50": ("B1", "ResNet50 (ImageNet), sau GAP, 2048 chiều", "keras"),
    "mobilenetv3": ("B1", "MobileNetV3-Small (ImageNet), sau GAP, 576 chiều", "keras"),
    "squeezenet": ("B1", "SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều", "torch"),
    "vit_b16": ("B2", "ViT-B/16 (ImageNet), token phân lớp, 768 chiều", "torch"),
}
PAPER = {  # accuracy (%) bài báo công bố trên BananaImageBD, Bảng 3 của Ergün (2025)
    ("vit_b16", "svm_rbf"): 99.70, ("vit_b16", "gbm"): 97.39, ("vit_b16", "mlp"): 99.29,
    ("resnet50", "svm_rbf"): 96.69, ("resnet50", "gbm"): 96.29, ("resnet50", "mlp"): 97.76,
    ("squeezenet", "svm_rbf"): 97.63, ("squeezenet", "gbm"): 93.08, ("squeezenet", "mlp"): 95.88,
}
CLF_NAME = {"svm_rbf": "SVM (RBF, C=1)", "svm_linear": "SVM tuyến tính (C=0,01)", "logreg": "Logistic Regression", "knn": "KNN (k=5)",
            "mlp": "MLP (1 lớp ẩn 256)", "gbm": "GBM (HistGradientBoosting)"}


def classifiers(extractor: str) -> dict:
    c = {"svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
         "svm_linear": lambda: make_clf("svm"),
         "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=SEED))}
    if EXTRACTORS[extractor][0] == "B1":            # bộ phân lớp đã học trong môn
        c["logreg"] = lambda: make_clf("logreg"); c["knn"] = lambda: make_clf("knn")
    else:                                           # như bài báo
        c["gbm"] = lambda: HistGradientBoostingClassifier(max_iter=100, random_state=SEED)
    return c


def _torch_extract(name: str, X: np.ndarray, batch: int = 32) -> np.ndarray:
    import torch
    import torchvision
    torch.set_num_threads(4); torch.manual_seed(SEED)
    if name == "vit_b16":
        m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1); m.heads = torch.nn.Identity()
        fwd = lambda t: m(t)
    else:
        m = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        fwd = lambda t: m.features(t).mean((2, 3))
    m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            t = torch.from_numpy(np.asarray(X[a:a + batch])).permute(0, 3, 1, 2).float() / 255.0
            t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            out.append(fwd((t - mean) / std).numpy())
    return np.concatenate(out)


def _keras_extract(name: str, X: np.ndarray) -> np.ndarray:
    from .cnn.train import backbone_net, resize_batch
    net = backbone_net({"resnet50": "b_resnet50", "mobilenetv3": "b_mobilenetv3"}[name])
    return np.concatenate([net.predict(resize_batch(X[a:a + 512], 224).astype(np.float32), batch_size=32, verbose=0) for a in range(0, len(X), 512)])


def features(name: str, tag: str, X: np.ndarray) -> np.ndarray:
    f = CACHE_DIR / f"paperfeat_{name}_{tag}.npy"
    if not f.exists():
        t0 = time.perf_counter()
        F = _torch_extract(name, X) if EXTRACTORS[name][2] == "torch" else _keras_extract(name, X)
        np.save(f, F); f.with_suffix(".time").write_text(f"{(time.perf_counter() - t0) / len(X) * 1000:.2f}")
    return np.load(f)


def cv_scores(X, y, fac, splits):
    acc, f1, rec = [], [], []
    for tr, te in splits:
        p = fac().fit(X[tr], y[tr]).predict(X[te])
        acc.append(accuracy_score(y[te], p)); f1.append(f1_score(y[te], p, average="macro")); rec.append(recall_score(y[te], p, average="macro"))
    return {"accuracy": float(np.mean(acc)), "accuracy_std": float(np.std(acc)), "macro_f1": float(np.mean(f1)), "macro_f1_std": float(np.std(f1)),
            "macro_recall": float(np.mean(rec)), "fold_accuracy": [float(a) for a in acc]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractors", nargs="*", choices=list(EXTRACTORS))
    a = ap.parse_args()
    from .exp_arm_t import load_aug_author
    from .final_eval import bootstrap_ci
    exp_dir(EXP)
    Xaug, idx = load_aug_author(); yaug, groups = idx["label_id"].values, idx["orig_idx"].values
    b = base_split()
    Xdev, ydev = np.concatenate([b["train"][0], b["val"][0]]), np.concatenate([b["train"][1], b["val"][1]])
    Xte, yte = b["test"][0], b["test"][1]
    st = stress_sets("test")
    n = np.arange(len(yaug))
    sp_paper = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(n, yaug))
    sp_group = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(n, yaug, groups))
    sp_dev = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.arange(len(ydev)), ydev))
    raw_f = OUT / "metrics.json"
    raw = json.loads(raw_f.read_text()) if raw_f.exists() else {}
    for ex in (a.extractors or EXTRACTORS):
        Fa = features(ex, "aug", Xaug); Fd = features(ex, "dev", Xdev); Ft = features(ex, "test", Xte)
        Fs = {s: features(ex, f"test_{s}", st[s]) for s in STRESS}
        for cname, fac in classifiers(ex).items():
            key = f"{ex}|{cname}"
            if key in raw:
                continue
            r = {"extractor": ex, "direction": EXTRACTORS[ex][0], "classifier": cname, "n_features": int(Fa.shape[1]),
                 "paper_protocol": cv_scores(Fa, yaug, fac, sp_paper), "group_protocol": cv_scores(Fa, yaug, fac, sp_group),
                 "dev_cv": cv_scores(Fd, ydev, fac, sp_dev), "paper_reported_accuracy_pct": PAPER.get((ex, cname))}
            clf = fac().fit(Fd, ydev); p = clf.predict(Ft)
            ci, _ = bootstrap_ci(yte, p)
            r["test"] = {"macro_f1": float(f1_score(yte, p, average="macro")), "accuracy": float(accuracy_score(yte, p)), "n_errors": int((p != yte).sum()), "macro_f1_ci95": ci}
            r["stress"] = {s: float(f1_score(yte, clf.predict(Fs[s]), average="macro")) for s in STRESS}
            r["robust_score"] = float(np.mean(list(r["stress"].values())))
            raw[key] = r; save_json(raw_f, raw)
            print(key, f"(i) {100 * r['paper_protocol']['accuracy']:.2f}% (ii) {100 * r['group_protocol']['accuracy']:.2f}% test F1 {r['test']['macro_f1']:.3f} bền {r['robust_score']:.3f}", flush=True)

    rows1, rows2, srows = [], [], []
    for key, r in raw.items():
        ex, cname = key.split("|"); pp, gp = r["paper_protocol"], r["group_protocol"]
        rows1.append({"Hướng": r["direction"], "Bộ rút đặc trưng": EXTRACTORS[ex][1], "Bộ phân lớp": CLF_NAME[cname],
                      "Bài báo công bố: accuracy (%)": f"{r['paper_reported_accuracy_pct']:.2f}" if r["paper_reported_accuracy_pct"] else "—",
                      "(i) Như bài báo: accuracy (%)": f"{100 * pp['accuracy']:.2f} ± {100 * pp['accuracy_std']:.2f}", "(i) Macro-F1": f"{pp['macro_f1']:.4f}",
                      "(ii) Giữ nhóm theo ảnh gốc: accuracy (%)": f"{100 * gp['accuracy']:.2f} ± {100 * gp['accuracy_std']:.2f}", "(ii) Macro-F1": f"{gp['macro_f1']:.4f}",
                      "Thổi phồng (i) − (ii), điểm %": f"{100 * (pp['accuracy'] - gp['accuracy']):+.2f}"})
        row = {"Hướng": r["direction"], "Bộ rút đặc trưng": EXTRACTORS[ex][1].split(",")[0], "Bộ phân lớp": CLF_NAME[cname],
               "Macro-F1 5-fold (train+val)": f"{r['dev_cv']['macro_f1']:.3f} ± {r['dev_cv']['macro_f1_std']:.3f}", "Macro-F1 test": f"{r['test']['macro_f1']:.3f}",
               "KTC 95%": f"[{r['test']['macro_f1_ci95'][0]:.3f}; {r['test']['macro_f1_ci95'][1]:.3f}]", "Số ảnh sai / 371": r["test"]["n_errors"]}
        row.update({STRESS_LABEL[s]: f"{r['stress'][s]:.3f}" for s in STRESS}); row["Điểm bền"] = f"{r['robust_score']:.3f}"
        rows2.append(row)
        srows.append({"experiment": "E3", "group": f"Bài báo — hướng {r['direction']}", "config": f"{ex} + {cname}", "condition": "clean", "eval_set": "test",
                      "macro_f1": round(r["test"]["macro_f1"], 4), "macro_f1_std": None, "accuracy": round(r["test"]["accuracy"], 4), "n_features": r["n_features"]})
    from .exp_common import register
    register(EXP, srows)
    print(save_table(EXP, rows1, "table_protocols", title="E3 — Phương pháp của bài báo trên bản Augmented (7.413 ảnh), 5-fold: như bài báo (i) so với giữ nhóm theo ảnh gốc (ii)"))
    print(save_table(EXP, rows2, "table_test", title="E3 — Phương pháp của bài báo theo giao thức của đồ án: học trên train+val gốc (2.100 ảnh), chấm trên test gốc (371 ảnh) và bộ kiểm tra khó"))
    # đại diện mỗi hướng: chọn theo 5-fold trên train+val
    best = {}
    for d in ("B1", "B2"):
        ks = [k for k, r in raw.items() if r["direction"] == d]
        if ks:
            best[d] = max(ks, key=lambda k: raw[k]["dev_cv"]["macro_f1"])
    paper_main = "vit_b16|svm_rbf" if "vit_b16|svm_rbf" in raw else None
    save_json(OUT / "summary.json", {"best_by_dev_cv": best, "paper_main": paper_main,
                                     "extract_ms_per_image": {ex: float((CACHE_DIR / f"paperfeat_{ex}_aug.time").read_text()) for ex in EXTRACTORS if (CACHE_DIR / f"paperfeat_{ex}_aug.time").exists()}})
    # hình: (i) so với (ii)
    keys = [k for k in raw if raw[k]["classifier"] in ("svm_rbf", "mlp", "gbm", "logreg")]
    fig, ax = plt.subplots(figsize=(12, 4.2)); x = np.arange(len(keys)); w = 0.38
    v1 = [100 * raw[k]["paper_protocol"]["accuracy"] for k in keys]; v2 = [100 * raw[k]["group_protocol"]["accuracy"] for k in keys]
    ax.bar(x - w / 2, v1, w, yerr=[100 * raw[k]["paper_protocol"]["accuracy_std"] for k in keys], capsize=2, label="(i) 5-fold như bài báo (không giữ nhóm)")
    ax.bar(x + w / 2, v2, w, yerr=[100 * raw[k]["group_protocol"]["accuracy_std"] for k in keys], capsize=2, label="(ii) 5-fold giữ nhóm theo ảnh gốc")
    for xi, k in zip(x, keys):
        pr = raw[k]["paper_reported_accuracy_pct"]
        if pr:
            ax.plot([xi - w, xi + w], [pr, pr], color="red", lw=1.5)
    ax.plot([], [], color="red", lw=1.5, label="Số bài báo công bố")
    ax.set_xticks(x); ax.set_xticklabels([f"{k.split('|')[0]}\n{CLF_NAME[k.split('|')[1]].split(' (')[0]}" for k in keys], fontsize=7)
    lo = min(min(v2), min([raw[k]["paper_reported_accuracy_pct"] or 100 for k in keys])) - 1.5
    ax.set_ylim(max(lo, 80), 100.6); ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("E3 — Đặc trưng pretrained + bộ phân lớp trên bản Augmented: hai cách chia fold")
    fig.tight_layout(); fig.savefig(FIG_DIR / "E3_papers_protocols.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
