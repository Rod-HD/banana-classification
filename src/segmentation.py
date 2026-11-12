"""Các phương pháp phân đoạn tách quả chuối khỏi nền (mục 6.7): ngưỡng toàn cục, Otsu,
ngưỡng thích nghi, k-means. Mọi hàm nhận ảnh RGB uint8 và trả về mask uint8 (255 = chuối)."""
from __future__ import annotations

import cv2
import numpy as np


def _fg_is_minority_on_border(mask: np.ndarray) -> np.ndarray:
    """Quy ước chọn phía 'vật': nền là phía chiếm đa số trên viền ảnh (quả chuối nằm giữa ảnh)."""
    b = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return mask if (b > 0).mean() < 0.5 else cv2.bitwise_not(mask)


def global_threshold(img, t: int = 127):
    """Ngưỡng toàn cục cố định trên ảnh xám."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(g, t, 255, cv2.THRESH_BINARY_INV)
    return _fg_is_minority_on_border(m)


def otsu_gray(img):
    """Otsu trên ảnh xám (tự chọn ngưỡng tối đa hóa phương sai giữa hai lớp)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _fg_is_minority_on_border(m)


def adaptive(img, block: int = 51, c: int = 5):
    """Ngưỡng thích nghi: ngưỡng của mỗi điểm = trung bình Gauss vùng lân cận block×block trừ c."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)


def kmeans_color(img, k: int = 2):
    """Gom cụm k-means (k=2) trên màu Lab của từng điểm ảnh; cụm chiếm đa số trên viền là nền."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lab_id, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    L = lab_id.reshape(img.shape[:2])
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    bg = np.bincount(border, minlength=k).argmax()
    return ((L != bg) * 255).astype(np.uint8)


def otsu_color_distance(img, ring: int = 8):
    """Otsu trên khoảng cách màu tới nền: lấy màu Lab trung vị của viền ảnh làm màu nền,
    tính khoảng cách Euclidean của từng điểm ảnh tới màu đó rồi tách ngưỡng bằng Otsu."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3), lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(b, axis=0), axis=2)
    d8 = np.clip(d * 255.0 / max(d.max(), 1e-6), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def background_surface(img, order: int = 2, iters: int = 3, small: int = 128):
    """Mô hình nền bậc hai: khớp một mặt bậc hai theo (x, y) cho từng kênh Lab của nền, bắt đầu từ
    vành ngoài của ảnh rồi lặp lại trên các điểm ảnh gần mô hình. Mặt cong mô tả được nền sáng không đều
    (vệt sáng của đèn ở giữa, tối dần ra mép), thứ mà "một màu nền duy nhất" không mô tả được.
    Trả về (ảnh nền Lab float32 cùng kích thước ảnh, bản đồ khoảng cách màu tới nền)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    ls = cv2.resize(lab, (small, small), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:small, 0:small].astype(np.float32) / (small - 1) * 2 - 1
    terms = [np.ones_like(xx), xx, yy] + ([xx * xx, xx * yy, yy * yy] if order >= 2 else [])
    A = np.stack([t.ravel() for t in terms], 1)
    Y = ls.reshape(-1, 3)
    r = int(0.12 * small)
    inl = np.ones((small, small), bool); inl[r:-r, r:-r] = False
    inl = inl.ravel()
    wgt = np.array([0.6, 1.0, 1.0], np.float32)          # giảm trọng số kênh sáng L để bóng đổ ít bị coi là vật
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[inl], Y[inl], rcond=None)
        d = np.linalg.norm((Y - A @ coef) * wgt, axis=1)
        d8 = np.clip(d * 255.0 / max(d.max(), 1e-6), 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new = d8 < 0.5 * t
        if new.sum() < 0.2 * new.size:                   # quá ít điểm nền: giữ tập cũ
            break
        inl = new
    bg = cv2.resize((A @ coef).reshape(small, small, 3), (w, h), interpolation=cv2.INTER_LINEAR)
    dist = np.linalg.norm((lab - bg) * wgt, axis=2)
    return bg, dist


def otsu_background_surface(img):
    """Otsu trên khoảng cách màu tới mô hình nền bậc hai (dùng cho phần mở rộng, mục 12)."""
    _, d = background_surface(img)
    d8 = np.clip(d * 255.0 / max(d.max(), 1e-6), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def banana_mask_v2(img):
    """Mask dùng cho phần mở rộng: mô hình nền bậc hai + Otsu + hậu xử lý."""
    return clean(otsu_background_surface(img))


def clean(mask, keep_largest: bool = True):
    """Hậu xử lý: đóng/mở hình thái, giữ thành phần liên thông lớn nhất, lấp lỗ."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    if keep_largest:
        m = ((lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


METHODS = {
    "Ngưỡng toàn cục (T=127)": global_threshold,
    "Otsu (ảnh xám)": otsu_gray,
    "Ngưỡng thích nghi": adaptive,
    "K-means màu (k=2)": kmeans_color,
    "Otsu trên khoảng cách màu tới nền": otsu_color_distance,
}


def banana_mask(img):
    """Mask dùng cho S2: Otsu trên khoảng cách màu tới nền + hậu xử lý (phương pháp được chọn ở S1)."""
    return clean(otsu_color_distance(img))
