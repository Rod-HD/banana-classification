"""Sinh số liệu cho các ví dụ tính tay trong báo cáo (mục 9 của kế hoạch):
(1) khoảng cách giữa hai vector moment màu, (2) Precision/Recall một lớp từ Confusion Matrix trên test,
(3) số tham số một lớp CNN (lấy từ results/N1). Kết quả: results/hand_calc/.

    python -m src.hand_calc
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .config import CLASS_NAMES, RESULTS_DIR
from .data import dev_and_test, list_images, load_all_images
from .features.color import color_moments
from .report_utils import exp_dir, save_json, save_table


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = exp_dir("hand_calc"); df = list_images(); imgs = load_all_images(256)
    dev, _ = dev_and_test()
    a = int(dev[dev.label_id == 0].idx.iloc[0]); b = int(dev[dev.label_id == 1].idx.iloc[0])
    va, vb = color_moments(np.asarray(imgs[a])), color_moments(np.asarray(imgs[b]))
    names = ["mean H", "mean S", "mean V", "std H", "std S", "std V", "skew H", "skew S", "skew V"]
    rows = [{"Thành phần": n, f"x ({df.path[a].split('/')[-1]})": f"{x:.4f}", f"y ({df.path[b].split('/')[-1]})": f"{z:.4f}",
             "x − y": f"{x - z:+.4f}", "(x − y)²": f"{(x - z) ** 2:.6f}", "Trị tuyệt đối của x − y": f"{abs(x - z):.4f}"} for n, x, z in zip(names, va, vb)]
    d = va - vb
    eu, man = float(np.sqrt((d ** 2).sum())), float(np.abs(d).sum())
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    ua, ub = va / np.linalg.norm(va), vb / np.linalg.norm(vb)
    eu_unit = float(np.linalg.norm(ua - ub))
    res = {"image_x": df.path[a], "image_y": df.path[b], "class_x": CLASS_NAMES[0], "class_y": CLASS_NAMES[1],
           "x": va.tolist(), "y": vb.tolist(), "sum_sq": float((d ** 2).sum()), "euclidean": eu, "manhattan": man,
           "dot": float(va @ vb), "norm_x": float(np.linalg.norm(va)), "norm_y": float(np.linalg.norm(vb)), "cosine_similarity": cos,
           "euclidean_after_unit_norm": eu_unit, "check_sqrt_2_minus_2cos": float(np.sqrt(2 - 2 * cos))}
    save_table("hand_calc", rows, "table_moments", title="Ví dụ 1 — Vector moment màu (HSV, đã chia về [0, 1]) của hai ảnh")
    # Precision / Recall từ confusion matrix của a_vgg16 trên test
    m = json.loads((RESULTS_DIR / "final" / "a_vgg16" / "metrics.json").read_text())
    cm = np.array(m["confusion_matrix"])
    k = 0
    tp = int(cm[k, k]); fp = int(cm[:, k].sum() - tp); fn = int(cm[k, :].sum() - tp)
    res["confusion_matrix_a_vgg16_test"] = cm.tolist()
    res["bangla_tp_fp_fn"] = [tp, fp, fn]
    res["bangla_precision"] = tp / (tp + fp); res["bangla_recall"] = tp / (tp + fn)
    res["bangla_f1"] = 2 * res["bangla_precision"] * res["bangla_recall"] / (res["bangla_precision"] + res["bangla_recall"])
    res["reported_per_class"] = m["per_class"]["Bangla Kola"]
    cm_rows = [dict({"Thực tế \\ Dự đoán": CLASS_NAMES[i]}, **{CLASS_NAMES[j]: int(cm[i, j]) for j in range(4)}) for i in range(4)]
    save_table("hand_calc", cm_rows, "table_cm_a_vgg16", title="Ví dụ 2 — Confusion Matrix của a_vgg16 trên test")
    n1 = json.loads((RESULTS_DIR / "N1" / "metrics.json").read_text())
    res["param_example"] = [r for r in n1["a_lenet"] if r["Lớp"] == "conv2"][0]
    save_json(out / "metrics.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("x", "y", "confusion_matrix_a_vgg16_test", "reported_per_class")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
