"""Tiền xử lý ảnh của nhánh X (mục 12.2 của kế hoạch). Mọi hàm nhận và trả về ảnh RGB uint8 256×256.

Các mức:
  none          chỉ resize (như pipeline gốc)
  wb            cân bằng trắng và hiệu chỉnh sáng theo nền giấy trắng (chia ảnh cho mô hình nền bậc hai)
  bg            nền phẳng: giữ nguyên màu quả chuối, thay toàn bộ nền bằng một màu trắng phẳng (không cân bằng trắng)
  wb+bg         cân bằng trắng rồi thay nền bằng màu trắng phẳng
  wb+bg+crop    thêm: cắt theo khung bao của quả chuối (lề 12%), đệm thành hình vuông, resize về 256×256
"""
from __future__ import annotations

import cv2
import numpy as np
from joblib import Parallel, delayed

from . import segmentation as S

LEVELS = ["none", "wb", "bg", "wb+bg", "wb+bg+crop"]
LEVEL_NAME = {"none": "Không xử lý", "wb": "Cân bằng trắng theo nền", "bg": "Nền phẳng (không cân bằng trắng)",
              "wb+bg": "Cân bằng trắng + nền phẳng", "wb+bg+crop": "Cân bằng trắng + nền phẳng + cắt theo quả"}
WHITE = 235          # mức xám mà nền giấy được đưa về


def white_balance(img):
    """Chia ảnh cho mô hình nền bậc hai (ước lượng màu giấy trắng tại từng vị trí) rồi nhân với WHITE.
    Khử cả sắc màu của nguồn sáng lẫn độ sáng không đều (vệt sáng giữa ảnh, tối ở mép)."""
    bg_lab, _ = S.background_surface(img)
    bg = cv2.cvtColor(np.clip(bg_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    gain = WHITE / np.maximum(bg, 40.0)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def flatten_background(img, mask):
    """Giữ quả chuối (mask, viền làm mềm), phần còn lại tô màu trắng phẳng."""
    a = cv2.GaussianBlur(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), (5, 5), 0).astype(np.float32)[..., None] / 255.0
    return np.clip(a * img.astype(np.float32) + (1 - a) * WHITE, 0, 255).astype(np.uint8)


def crop_to_object(img, mask, margin: float = 0.12):
    """Cắt theo khung bao của mask, thêm lề, đệm nền trắng thành hình vuông, resize về kích thước ban đầu.
    Nếu mask rỗng hoặc quá nhỏ (phân đoạn hỏng) thì giữ nguyên ảnh."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) < 0.002 * h * w:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(round(max(y1 - y0, x1 - x0) * (1 + 2 * margin)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = np.full((side, side, 3), WHITE, np.uint8)
    sy0, sx0 = cy - side // 2, cx - side // 2
    ya, yb, xa, xb = max(sy0, 0), min(sy0 + side, h), max(sx0, 0), min(sx0 + side, w)
    canvas[ya - sy0:yb - sy0, xa - sx0:xb - sx0] = img[ya:yb, xa:xb]
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA if side > w else cv2.INTER_LINEAR)


def process(img, level: str = "none", return_mask: bool = False):
    if level == "none":
        return (img, None) if return_mask else img
    out = white_balance(img) if level.startswith("wb") else img
    mask = None
    if level != "wb":
        mask = S.banana_mask_v2(out)
        out = flatten_background(out, mask)
        if level.endswith("crop"):
            out = crop_to_object(out, mask)
    return (out, mask) if return_mask else out


def _chunk(imgs, level):
    return np.stack([process(np.asarray(im), level) for im in imgs])


def process_batch(imgs: np.ndarray, level: str, n_jobs: int = 4) -> np.ndarray:
    if level == "none":
        return np.asarray(imgs)
    n = len(imgs); b = np.linspace(0, n, n_jobs * 4 + 1).astype(int)
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(np.asarray(imgs[a:c]), level) for a, c in zip(b[:-1], b[1:]) if c > a)
    return np.concatenate(parts)
