"""Nhánh T (mục 12.3): chạy 9 cấu hình của pipeline gốc với bản "Augmented Banana Classification Dataset"
của tác giả bộ dữ liệu (data/aug_author/, tải bằng `bash scripts/download_data.sh augmented`).

  T-kr  không rò rỉ: train = 3 file của mỗi ảnh gốc thuộc tập train gốc; val, test = ảnh gốc.
  T-tl  theo thông lệ: gộp 7.413 file, chia ngẫu nhiên phân tầng 70/15/15, không quan tâm ảnh gốc.

    python -m src.exp_arm_t              # cả hai
    python -m src.exp_arm_t --only kr
"""
from __future__ import annotations

import argparse
import json
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CACHE_DIR, CLASSES, RESULTS_DIR, ROOT, SEED
from .data import list_images, load_split
from .pipeline import CNN, TRAD, ArmData, base_split, run_arm
from .report_utils import save_table
from .stress import stress_sets

AUG_DIR = ROOT / "data" / "aug_author"
OUT = RESULTS_DIR / "E2_aug_tac_gia"


def load_aug_author():
    """Ảnh của bản Augmented (N, 256, 256, 3) cùng nhãn và chỉ số ảnh gốc tương ứng (suy ra từ tên file Roboflow
    `<tên ảnh gốc>_jpg.rf.<mã>.jpg`)."""
    fx, fi = CACHE_DIR / "aug_author_256.npy", CACHE_DIR / "aug_author_index.csv"
    if not (fx.exists() and fi.exists()):
        if not AUG_DIR.exists():
            raise SystemExit("Chưa có data/aug_author/. Chạy: bash scripts/download_data.sh augmented")
        df = list_images()
        stem = {p.split("/")[-1].rsplit(".", 1)[0]: i for i, p in zip(df["idx"], df["path"])}
        rows, imgs = [], []
        for label_id, cls in enumerate(CLASSES):
            for p in sorted((AUG_DIR / cls).iterdir()):
                m = re.match(r"(.+)_(?:jpg|JPG|jpeg|png)\.rf\.[0-9a-f]+\.jpg$", p.name)
                rows.append({"file": f"{cls}/{p.name}", "label_id": label_id, "orig_idx": stem[m.group(1)]})
                im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                imgs.append(im if im.shape[:2] == (256, 256) else cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
        np.save(fx, np.stack(imgs)); pd.DataFrame(rows).to_csv(fi, index=False)
    return np.load(fx, mmap_mode="r"), pd.read_csv(fi)


def arm_kr() -> ArmData:
    X, idx = load_aug_author(); b = base_split()
    tr = np.where(idx["orig_idx"].isin(set(b["train"][2])).values)[0]
    return ArmData("T-kr", OUT / "khong_ro_ri", np.asarray(X[tr]), idx["label_id"].values[tr], b["val"][0], b["val"][1], b["test"][0], b["test"][1],
                   stress=stress_sets("test"), note="bản Augmented của tác giả, chia theo ảnh gốc (không rò rỉ)")


def arm_tl() -> ArmData:
    X, idx = load_aug_author(); y = idx["label_id"].values; n = np.arange(len(y))
    tr, rest = train_test_split(n, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    pd.DataFrame({"i": np.r_[tr, va, te], "split": ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te)}).to_csv(OUT / "thong_le_split.csv", index=False)
    return ArmData("T-tl", OUT / "thong_le", np.asarray(X[tr]), y[tr], np.asarray(X[va]), y[va], np.asarray(X[te]), y[te],
                   note="bản Augmented của tác giả, chia ngẫu nhiên 70/15/15 không quan tâm ảnh gốc")


def leak_report():
    """T-tl: tỉ lệ ảnh test có file cùng ảnh gốc nằm trong train, và accuracy tách theo hai nhóm."""
    _, idx = load_aug_author()
    sp = pd.read_csv(OUT / "thong_le_split.csv")
    tr, te = sp[sp.split == "train"].i.values, sp[sp.split == "test"].i.values
    orig = idx["orig_idx"].values
    sib = np.isin(orig[te], orig[tr])
    n_sib = np.array([(orig[tr] == o).sum() for o in orig[te]])
    rows = [{"Chỉ số": "Số ảnh test", "Giá trị": len(te)}, {"Chỉ số": "Ảnh test có ít nhất 1 file cùng ảnh gốc trong train", "Giá trị": f"{int(sib.sum())} ({100 * sib.mean():.1f}%)"},
            {"Chỉ số": "… có đúng 2 file cùng ảnh gốc trong train", "Giá trị": f"{int((n_sib == 2).sum())} ({100 * (n_sib == 2).mean():.1f}%)"},
            {"Chỉ số": "Ảnh test không có file cùng ảnh gốc trong train", "Giá trị": int((~sib).sum())}]
    save_table("E2_aug_tac_gia", rows, "table_leak_stats", title="T-tl — Mức trùng nguồn giữa train và test khi chia ngẫu nhiên bản Augmented")
    acc = []
    for key in TRAD + CNN:
        f = OUT / "thong_le" / key / "predictions.npz"
        if f.exists():
            z = np.load(f); ok = z["test"] == z["y"]
            acc.append({"Mô hình": json.loads((OUT / "thong_le" / key / "metrics.json").read_text())["config"],
                        "Accuracy (ảnh test có anh em trong train)": f"{ok[sib].mean():.3f}", "Số ảnh": int(sib.sum()),
                        "Accuracy (ảnh test không có)": f"{ok[~sib].mean():.3f}" if (~sib).any() else "—", "Số ảnh ": int((~sib).sum())})
    if acc:
        save_table("E2_aug_tac_gia", acc, "table_leak_accuracy", title="T-tl — Accuracy trên test tách theo việc ảnh có file cùng ảnh gốc trong train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["kr", "tl"])
    ap.add_argument("--models", nargs="*", choices=TRAD + CNN)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # T-tl chỉ để đo mức thổi phồng do rò rỉ nên mặc định bỏ hai mô hình huấn luyện lâu nhất (a_dropout, b_resnet50);
    # muốn chạy đủ: python -m src.exp_arm_t --only tl --models <danh sách>.
    tl_default = TRAD + ["a_lenet", "a_vgg16", "b_mobilenetv3"]
    for k in (a.only or ["kr", "tl"]):
        if k == "kr":
            run_arm(arm_kr(), a.models)
        else:
            run_arm(arm_tl(), a.models or tl_default)
            leak_report()


if __name__ == "__main__":
    main()
