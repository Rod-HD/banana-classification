"""Xuất mô hình dùng cho demo vào models/ (mỗi file dưới 50 MB) và ghi models/models.json.

    python -m src.export_demo
- CNN: b_mobilenetv3 (biến thể được chốt) và a_lenet, lưu dạng .keras (học trên train).
- Truyền thống: cấu hình tốt nhất (results/select_traditional/selected.json), học trên train+val, lưu dạng .joblib.
a_vgg16 không xuất được vì file mô hình vượt 50 MB.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib

from .config import MODELS_DIR, RESULTS_DIR
from .data import dev_and_test
from .trad_config import build_clf, features_all


def test_metrics(key):
    f = RESULTS_DIR / "final" / key / "metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    return {"macro_f1_test": round(m["macro_f1"], 4), "accuracy_test": round(m["accuracy"], 4)}


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from .cnn.predict import full_model, run_meta
    from .exp_cnn import final_run
    MODELS_DIR.mkdir(exist_ok=True)
    entries = []
    for model, label in [("b_mobilenetv3", "CNN hướng B — MobileNetV3-Small fine-tune"), ("a_lenet", "CNN hướng A — LeNet huấn luyện từ đầu")]:
        run = final_run(model)
        f = MODELS_DIR / f"{model}.keras"
        full_model(run).save(f)
        entries.append({"key": model, "label": label, "type": "keras", "file": f.name, "input_size": run_meta(run)["input_size"],
                        "run": run, "size_mb": round(os.path.getsize(f) / 1e6, 2), "test": test_metrics(model)})
    sel = json.loads((RESULTS_DIR / "select_traditional" / "selected.json").read_text())
    cfg = sel[sel["_best_overall"]]
    dev, _ = dev_and_test()
    X, dims = features_all(cfg)
    clf = build_clf(cfg, dims).fit(X[dev["idx"].values], dev["label_id"].values)
    f = MODELS_DIR / "trad_best.joblib"
    joblib.dump({"cfg": {k: cfg[k] for k in ("parts", "fusion", "clf", "cparams", "scale", "name") if k in cfg}, "clf": clf}, f, compress=3)
    entries.append({"key": "trad_best", "label": "Truyền thống — " + cfg["name"], "type": "joblib", "file": f.name, "input_size": 256,
                    "size_mb": round(os.path.getsize(f) / 1e6, 2), "test": test_metrics(sel["_best_overall"])})
    for e in entries:
        assert e["size_mb"] < 50, e
        print(e)
    (MODELS_DIR / "models.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
