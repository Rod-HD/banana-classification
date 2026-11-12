"""Đọc danh sách ảnh, đọc ảnh và cache ảnh đã resize vào data/cache/."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import CACHE_DIR, CLASSES, RAW_DIR, ROOT, SPLIT_DIR

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images() -> pd.DataFrame:
    """Danh sách toàn bộ ảnh gốc, sắp theo lớp rồi theo tên file (thứ tự cố định).
    Nhận cả đuôi .jpg và .JPG."""
    rows = []
    for label_id, cls in enumerate(CLASSES):
        for p in sorted((RAW_DIR / cls).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": str(p.relative_to(ROOT)), "label": cls, "label_id": label_id})
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    return df


def read_rgb(path, size: int | None = None) -> np.ndarray:
    """Đọc ảnh thành mảng RGB uint8; resize về size×size nếu cần (2 ảnh không phải 256×256)."""
    p = ROOT / path if not str(path).startswith("/") else path
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"không đọc được ảnh {p}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None and img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_all_images(size: int = 256) -> np.ndarray:
    """Mảng (N, size, size, 3) uint8 theo thứ tự của list_images(); cache ra .npy."""
    f = CACHE_DIR / f"images_{size}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    df = list_images()
    arr = np.stack([read_rgb(p, size) for p in df["path"]])
    np.save(f, arr)
    return np.load(f, mmap_mode="r")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"{name}.csv")


def dev_and_test():
    """dev = train+val (dùng cho cross-validation của nhóm truyền thống), test riêng."""
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    dev = pd.concat([tr, va], ignore_index=True).sort_values("idx").reset_index(drop=True)
    return dev, te
