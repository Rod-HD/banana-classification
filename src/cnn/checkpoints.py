"""Kiểm tra các bộ trọng số CNN đã huấn luyện trong checkpoints/ (có sẵn trong repo, không cần huấn luyện lại).

    python -m src.cnn.checkpoints            # dự đoán lại trên val bằng trọng số đã lưu, so với kết quả đã ghi
    python -m src.cnn.checkpoints --slim     # chuyển checkpoint kiểu cũ (kèm trạng thái Adam) sang file chỉ để suy luận

Mỗi run có checkpoints/<run>/best.keras (kiến trúc + trọng số tốt nhất theo macro-F1 trên val) và state.json
(lịch sử huấn luyện). File không kèm trạng thái của bộ tối ưu Adam: hai mảng moment cho mỗi trọng số huấn luyện
được chỉ cần khi huấn luyện tiếp, và làm file lớn gấp ba.
Kết quả: results/cnn/table_checkpoints.md (kiểm tra), results/cnn/table_slim.md (chuyển đổi).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..config import CKPT_DIR, RESULTS_DIR
from ..report_utils import md_table

OUT = RESULTS_DIR / "cnn"


def weight_arrays(path) -> tuple[dict, int]:
    """Đọc mọi mảng trong file trọng số (.weights.h5 hoặc .keras). Trả về (mảng của các lớp, số phần tử của bộ tối ưu)."""
    src = path
    if str(path).endswith(".keras"):
        with zipfile.ZipFile(path) as z:
            src = io.BytesIO(z.read("model.weights.h5"))
    layers, n_opt = {}, 0
    with h5py.File(src, "r") as f:
        def visit(name, obj):
            nonlocal n_opt
            if isinstance(obj, h5py.Dataset):
                if name.startswith("optimizer"):
                    n_opt += obj.size
                else:
                    layers[name] = obj[()]
        f.visititems(visit)
    return layers, n_opt


def n_diff(a: dict, b: dict) -> int:
    """Số mảng khác nhau giữa hai file (so từng bit; thiếu mảng cũng tính là khác)."""
    return sum(1 for k in set(a) | set(b) if k not in a or k not in b or not np.array_equal(a[k], b[k]))


def slim():
    """Lưu lại best.keras không kèm trạng thái Adam, kiểm tra trọng số không đổi bit nào, rồi mới xóa file cũ."""
    import keras
    rows, tot = [], {"before": 0.0, "after": 0.0, "arrays": 0, "diff": 0}
    for d in sorted(p for p in CKPT_DIR.iterdir() if (p / "best.keras").exists()):
        bk, bw = d / "best.keras", d / "best.weights.h5"
        old, n_opt = weight_arrays(bk)
        if n_opt == 0 and not bw.exists():
            continue                                    # đã ở dạng suy luận
        before = os.path.getsize(bk) / 1e6
        tmp = d / "slim.tmp.keras"
        keras.models.load_model(bk, compile=False).save(tmp)
        new, n_opt_new = weight_arrays(tmp)
        diff = n_diff(old, new) + (n_diff(old, weight_arrays(bw)[0]) if bw.exists() else 0)
        assert diff == 0 and n_opt_new == 0, (d.name, diff, n_opt_new)
        os.replace(tmp, bk)
        for f in (bw, d / "full.keras", d / "last.keras"):   # bản trùng và file dẫn xuất (N6 tự tạo lại full.keras)
            if f.exists():
                f.unlink()
        after = os.path.getsize(bk) / 1e6
        rows.append({"Run": d.name, "Số tham số": sum(v.size for v in new.values()), "Trước (MB)": f"{before:.2f}", "Sau (MB)": f"{after:.2f}",
                     "Số mảng trọng số": len(new), "Số mảng khác (so từng bit)": diff})
        tot["before"] += before; tot["after"] += after; tot["arrays"] += len(new); tot["diff"] += diff
        print(rows[-1], flush=True)
    if rows:
        rows.append({"Run": f"**Tổng {len(rows)} run**", "Số tham số": "", "Trước (MB)": f"{tot['before']:.0f}", "Sau (MB)": f"{tot['after']:.0f}",
                     "Số mảng trọng số": tot["arrays"], "Số mảng khác (so từng bit)": tot["diff"]})
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "table_slim.md").write_text("### Chuyển best.keras sang file chỉ để suy luận (bỏ trạng thái Adam): dung lượng và kiểm tra trọng số\n\n"
                                           + md_table(rows) + "\n")
    print(f"đã chuyển {max(len(rows) - 1, 0)} run")


# Nhánh mở rộng chấm trên đúng ảnh test gốc (không xử lý ảnh lúc suy luận): so được trực tiếp với kết quả đã ghi
ARM_DIRS = {"T-kr": "E2_aug_tac_gia/khong_ro_ri", "Xb": "E1_xu_ly/pipeline_b"}


def verify():
    """Nạp best.keras của từng run, dự đoán lại và so với kết quả đã ghi: run ở bước 4-6 chấm trên val
    (results/cnn/<run>/), run của nhánh T-kr và X-b chấm trên test (results/<nhánh>/<mô hình>/)."""
    import keras
    from ..data import load_all_images, load_split
    from ..pipeline import CNN_CFG
    from . import models as M
    from .predict import Predictor, run_meta
    from .train import resize_batch
    data = {}
    for sp in ("val", "test"):
        d = load_split(sp)
        data[sp] = (np.stack([load_all_images(256)[i] for i in d["idx"].values]), d["label_id"].values)
    rows, bad, n = [], 0, 0
    for run in sorted(p.name for p in CKPT_DIR.iterdir() if (p / "best.keras").exists()):
        size_mb = os.path.getsize(CKPT_DIR / run / "best.keras") / 1e6
        arm, _, model = run.partition("__")
        row = {"Run": run, "File (MB)": f"{size_mb:.2f}", "Tập chấm": "-", "Macro-F1 đã ghi": "-", "Macro-F1 tính lại": "-",
               "Số ảnh đoán sai (đã ghi / tính lại)": "-", "Khớp": ""}
        if (OUT / run / "metrics.json").exists():
            sp, m = "val", run_meta(run)
            predict = Predictor(run)
        elif arm in ARM_DIRS and (RESULTS_DIR / ARM_DIRS[arm] / model / "metrics.json").exists():
            sp, m = "test", json.loads((RESULTS_DIR / ARM_DIRS[arm] / model / "metrics.json").read_text())
            net = keras.models.load_model(CKPT_DIR / run / "best.keras")
            if model == "a_vgg16":
                base = M.vgg16_features(CNN_CFG[model][0]); net = keras.Model(base.input, net(base.output))
            predict = lambda X, net=net, size=CNN_CFG[model][0]: net.predict(resize_batch(X, size).astype(np.float32), batch_size=32, verbose=0)
        else:       # đầu phân lớp trung gian của giai đoạn 1, hoặc nhánh có tập chấm riêng (X-a: ảnh đã tách nền; T-tl: tập test riêng)
            row["Khớp"] = "không chấm lại ở đây; trọng số đã so từng bit (table_slim.md)"
            rows.append(row); continue
        X, y = data[sp]
        pred = predict(X).argmax(1)
        f1, acc = float(f1_score(y, pred, average="macro")), float(accuracy_score(y, pred))
        ok = abs(f1 - m["macro_f1"]) < 1e-9 and abs(acc - m["accuracy"]) < 1e-9
        n += 1; bad += not ok
        row.update({"Tập chấm": f"{sp} ({len(y)} ảnh)", "Macro-F1 đã ghi": f"{m['macro_f1']:.6f}", "Macro-F1 tính lại": f"{f1:.6f}",
                    "Số ảnh đoán sai (đã ghi / tính lại)": f"{round((1 - m['accuracy']) * len(y))} / {int((pred != y).sum())}", "Khớp": "có" if ok else "KHÔNG"})
        rows.append(row); print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_checkpoints.md").write_text(
        f"### Kiểm tra trọng số trong checkpoints/: dự đoán lại bằng best.keras, so với kết quả đã ghi, khớp {n - bad}/{n} run được chấm lại\n\n"
        + md_table(rows) + "\n")
    print(f"khớp {n - bad}/{n} run được chấm lại; {len(rows) - n} run còn lại chỉ liệt kê dung lượng")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slim", action="store_true", help="chuyển checkpoint kiểu cũ sang file chỉ để suy luận")
    a = ap.parse_args()
    if a.slim:
        slim()
    else:
        raise SystemExit(1 if verify() else 0)


if __name__ == "__main__":
    main()
