"""Thí nghiệm phân đoạn tách nền S1-S2 (mục 6.7 của kế hoạch).

S1: so các phương pháp phân đoạn trên ảnh mẫu (hình định tính) và hai chỉ số không cần nhãn mask.
S2: rút đặc trưng trên toàn ảnh / chỉ vùng chuối / chỉ vùng nền để xem mô hình có dựa vào nền không.

    python -m src.exp_segment
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from . import segmentation as S
from .classical import cv_eval, make_clf
from .config import CACHE_DIR, CLASS_NAMES, FIG_DIR, SEED
from .data import list_images, load_all_images
from .exp_common import dev_set, register, strip, summary_row
from .exp_shape import fusion_clf
from .features import color as C
from .features.hog import hog_batch
from .report_utils import exp_dir, pm, save_json, save_table


def s1(n_per_class: int = 100):
    df = list_images(); imgs = load_all_images(256)
    rng = np.random.default_rng(SEED)
    # Hình định tính: 2 ảnh mỗi lớp
    ids = np.concatenate([rng.choice(np.where(df["label_id"].values == c)[0], 2, replace=False) for c in range(4)])
    methods = dict(S.METHODS); methods["Otsu khoảng cách màu + hậu xử lý (dùng cho S2)"] = S.banana_mask
    fig, axes = plt.subplots(len(ids), len(methods) + 1, figsize=(2.1 * (len(methods) + 1), 2.1 * len(ids)))
    for r, i in enumerate(ids):
        im = np.asarray(imgs[i]); axes[r, 0].imshow(im); axes[r, 0].set_ylabel(CLASS_NAMES[df["label_id"].values[i]], fontsize=8)
        for j, (name, fn) in enumerate(methods.items()):
            axes[r, j + 1].imshow(fn(im), cmap="gray", vmin=0, vmax=255)
            if r == 0:
                axes[r, j + 1].set_title(name.replace(" (", "\n(").replace(" + ", "\n+ "), fontsize=7)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_title("Ảnh gốc", fontsize=8)
    fig.suptitle("S1: Các phương pháp phân đoạn trên ảnh mẫu (trắng = vùng được coi là chuối)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "S1_segmentation.png", dpi=110); plt.close(fig)

    # Chỉ số không cần nhãn: số mảnh rời và tỉ lệ lỗ thủng bên trong vật, trên n_per_class ảnh mỗi lớp
    sample = np.concatenate([rng.choice(np.where(df["label_id"].values == c)[0], n_per_class, replace=False) for c in range(4)])
    ref = {i: S.banana_mask(np.asarray(imgs[i])) for i in sample}
    rows, raw = [], {}
    for name, fn in methods.items():
        ncomp, holes, area, iou = [], [], [], []
        for i in sample:
            m = fn(np.asarray(imgs[i]))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            ncomp.append(n - 1)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m); cv2.drawContours(filled, cnts, -1, 255, -1)
            holes.append(1 - (m > 0).sum() / max((filled > 0).sum(), 1)); area.append((m > 0).mean())
            inter = ((m > 0) & (ref[i] > 0)).sum(); union = ((m > 0) | (ref[i] > 0)).sum(); iou.append(inter / max(union, 1))
        raw[name] = {"mean_components": float(np.mean(ncomp)), "median_components": float(np.median(ncomp)),
                     "mean_hole_fraction": float(np.mean(holes)), "mean_foreground_fraction": float(np.mean(area)),
                     "mean_iou_with_selected_method": float(np.mean(iou))}
        rows.append({"Phương pháp": name, "Số mảnh rời (trung bình)": f"{np.mean(ncomp):.1f}", "Số mảnh rời (trung vị)": f"{np.median(ncomp):.0f}",
                     "Tỉ lệ lỗ thủng trong vật": f"{np.mean(holes):.3f}", "Tỉ lệ diện tích vật": f"{np.mean(area):.3f}",
                     "IoU với phương pháp được chọn": f"{np.mean(iou):.3f}"})
    save_json(exp_dir("S1") / "metrics.json", dict(raw, _note="Không có nhãn mask: các chỉ số này không phải IoU/Dice so với nhãn thật.",
                                                    _n_images=int(len(sample)))); register("S1", [])
    print(save_table("S1", rows, title=f"S1: Chỉ số không cần nhãn mask trên {len(sample)} ảnh ({n_per_class} ảnh mỗi lớp)"))


def all_masks():
    f = CACHE_DIR / "masks_256.npy"
    if not f.exists():
        imgs = load_all_images(256)
        ms = Parallel(n_jobs=4)(delayed(lambda a, b: np.stack([S.banana_mask(np.asarray(imgs[i])) for i in range(a, b)]))(a, min(a + 100, len(imgs)))
                                for a in range(0, len(imgs), 100))
        np.save(f, np.concatenate(ms))
    return np.load(f)


def all_masks_v2():
    """Mask theo mô hình nền bậc hai (segmentation.banana_mask_v2) cho toàn bộ ảnh; dùng ở phần mở rộng."""
    f = CACHE_DIR / "masks_v2_256.npy"
    if not f.exists():
        imgs = load_all_images(256)
        ms = Parallel(n_jobs=4)(delayed(lambda a, b: np.stack([S.banana_mask_v2(np.asarray(imgs[i])) for i in range(a, b)]))(a, min(a + 100, len(imgs)))
                                for a in range(0, len(imgs), 100))
        np.save(f, np.concatenate(ms))
    return np.load(f)


def s2(mask_fn=None, sfx: str = ""):
    """sfx rỗng: mask của bước S1 (kết quả gốc của S2). sfx khác rỗng: chạy lại với mask khác (ví dụ mask mô hình nền bậc hai)
    và ghi vào các file có hậu tố sfx, không ghi đè kết quả gốc."""
    ids, y = dev_set()
    imgs = load_all_images(256); masks = (mask_fn or all_masks)()
    I = [np.asarray(imgs[i]) for i in ids]; Mk = [masks[i] for i in ids]
    # Vùng nền lấy từ mask đã nới rộng 12 điểm ảnh, để phần mép quả và bóng đổ sát quả
    # (chỗ phân đoạn hay sai) không bị tính là nền.
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    Mk_dil = [cv2.dilate(m, kd) for m in Mk]
    Bg = [cv2.bitwise_not(m) for m in Mk_dil]

    def feat(fn, ms):
        return np.stack([fn(im, mask=m) for im, m in zip(I, ms)]).astype(np.float32)

    def bg_color(im, m):       # màu Lab trung bình của nền: 3 con số
        return cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[m == 0].mean(0)

    def filled(im, m, keep_fg):  # giữ một vùng, vùng còn lại tô bằng màu nền trung vị (chỉ chuối: nền phẳng; chỉ nền: lấp chỗ quả chuối)
        out = im.copy(); med = np.median(im[m == 0], axis=0).astype(np.uint8) if (m == 0).any() else np.array([128, 128, 128], np.uint8)
        out[(m == 0) if keep_fg else (m > 0)] = med
        return out

    def inpainted(im, m):        # chỉ nền: lấp vùng chuối (mask đã nới rộng) bằng inpainting Telea, giảm vết viền hình bóng
        return cv2.inpaint(im, m, 7, cv2.INPAINT_TELEA)

    none = [None] * len(I)
    versions = {"Toàn ảnh": (none, I), "Chỉ vùng chuối": (Mk, [filled(im, m, True) for im, m in zip(I, Mk)]),
                "Chỉ vùng nền": (Bg, [inpainted(im, m) for im, m in zip(I, Mk_dil)])}
    rows, srows, raw = [], [], {}
    hogp = dict(size=(64, 128))
    for vname, (ms, ims) in versions.items():
        Xhsv = feat(C.hsv_hist, ms); Xhs = feat(C.hs_hist, ms); Xhog = hog_batch(np.stack(ims), **hogp)
        confs = [("Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất]", Xhsv, lambda: make_clf("logreg", C=0.1)),
                 ("HOG 64×128 + LogReg C=0.1", Xhog, lambda: make_clf("logreg", C=0.1)),
                 ("Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất]", np.concatenate([Xhs, Xhog], 1),
                  lambda: fusion_clf("logreg", [Xhs.shape[1], Xhog.shape[1]], True, C=100.0))]
        for cname, X, fac in confs:
            r = cv_eval(X, y, fac); raw[f"{cname}|{vname}"] = strip(r)
            rows.append({"Đặc trưng + bộ phân lớp": cname, "Phiên bản ảnh": vname, "Macro-F1": pm(r["macro_f1"], r["macro_f1_std"]),
                         "Accuracy": pm(r["accuracy"], r["accuracy_std"])})
            srows.append(summary_row("S2", f"{cname} | {vname}", r))
    # Kiểm tra rò rỉ: trên ảnh "chỉ vùng nền" (đã lấp vùng chuối), độ lớn gradient trung bình dọc viền mask
    # so với vùng nền xa quả. Tỉ số > 1 nghĩa là còn vết viền hình bóng mà HOG có thể dùng.
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ratios = []
    for im, m in zip(versions["Chỉ vùng nền"][1], Mk_dil):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
        ring = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, ring_k) > 0
        far = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) == 0
        if ring.any() and far.any():
            ratios.append(float(mag[ring].mean() / max(mag[far].mean(), 1e-6)))
    raw["_silhouette_leak_gradient_ratio"] = {"median": float(np.median(ratios)), "p25": float(np.percentile(ratios, 25)),
                                              "p75": float(np.percentile(ratios, 75)), "n": len(ratios)}
    # Phép thử không phụ thuộc mask: ghép 4 góc ảnh (mỗi góc 64×64) thành ảnh 128×128.
    c = 64
    corners = np.stack([np.vstack([np.hstack([im[:c, :c], im[:c, -c:]]), np.hstack([im[-c:, :c], im[-c:, -c:]])]) for im in I])
    corner_fg = float(np.mean([np.concatenate([m[:c, :c].ravel(), m[:c, -c:].ravel(), m[-c:, :c].ravel(), m[-c:, -c:].ravel()]).mean() / 255 for m in Mk_dil]))
    Xc_hog = hog_batch(corners, size=(128, 128))
    Xc_hsv = np.stack([C.hsv_hist(im) for im in corners]).astype(np.float32)
    raw["_corner_foreground_fraction"] = corner_fg
    corner_rows = []
    for cname, X, fac in [("HOG của 4 góc ảnh (128×128) + LogReg C=0.1", Xc_hog, lambda: make_clf("logreg", C=0.1)),
                          ("Hist HSV (8,8,8) của 4 góc ảnh + LogReg C=0.1", Xc_hsv, lambda: make_clf("logreg", C=0.1))]:
        r = cv_eval(X, y, fac); raw[cname] = strip(r)
        corner_rows.append({"Đặc trưng + bộ phân lớp": cname, "Số chiều": X.shape[1], "Macro-F1": pm(r["macro_f1"], r["macro_f1_std"]),
                            "Accuracy": pm(r["accuracy"], r["accuracy_std"])})
        srows.append(summary_row("S2", cname, r))
    print(save_table("S2", corner_rows, "table_corners" + sfx,
                     title=f"S2: Chỉ dùng 4 góc ảnh 64×64 (không cần mask; trung bình {100 * corner_fg:.1f}% điểm ảnh ở góc thuộc vùng chuối đã nới rộng)"))
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6))
    for cl in range(4):
        k = int(np.where(y == cl)[0][5]); axes[cl].imshow(corners[k]); axes[cl].set_title(CLASS_NAMES[cl], fontsize=9); axes[cl].axis("off")
    fig.suptitle("S2: Ảnh ghép từ 4 góc 64×64 (đầu vào của phép thử 'chỉ góc ảnh')", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"S2_corners{sfx}.png", dpi=110); plt.close(fig)
    # Hai phép thử cực đoan: chỉ 3 con số màu nền; chỉ hình bóng (mask nhị phân)
    Xbg = np.stack([bg_color(im, cv2.bitwise_not(b)) for im, b in zip(I, Bg)]).astype(np.float32)
    Xsil = hog_batch(np.stack([np.repeat(m[..., None], 3, 2) for m in Mk]), **hogp)
    Xarea = np.array([[(m > 0).mean()] for m in Mk], np.float32)
    extra = []
    for cname, X, fac in [("Màu Lab trung bình của nền (3 số) + KNN k=5", Xbg, lambda: make_clf("knn")),
                          ("Màu Lab trung bình của nền (3 số) + LogReg", Xbg, lambda: make_clf("logreg")),
                          ("Tỉ lệ diện tích quả chuối trong ảnh (1 số) + KNN k=5", Xarea, lambda: make_clf("knn")),
                          ("HOG của mask nhị phân (chỉ hình bóng) + LogReg C=0.1", Xsil, lambda: make_clf("logreg", C=0.1))]:
        r = cv_eval(X, y, fac); raw[cname] = strip(r)
        extra.append({"Đặc trưng + bộ phân lớp": cname, "Số chiều": X.shape[1], "Macro-F1": pm(r["macro_f1"], r["macro_f1_std"]),
                      "Accuracy": pm(r["accuracy"], r["accuracy_std"])})
        srows.append(summary_row("S2", cname, r))
    # Thống kê nền theo lớp
    Xbg64 = Xbg.astype(np.float64)
    lab_mean = {CLASS_NAMES[c]: np.round(Xbg64[y == c].mean(0), 1).tolist() for c in range(4)}
    lab_std = {CLASS_NAMES[c]: np.round(Xbg64[y == c].std(0), 1).tolist() for c in range(4)}
    area = {CLASS_NAMES[c]: float(Xarea[y == c].mean()) for c in range(4)}
    raw["_background_lab_mean"] = lab_mean; raw["_background_lab_std"] = lab_std; raw["_banana_area_fraction"] = area
    stat_rows = [{"Lớp": k, "Nền: L trung bình ± std": f"{lab_mean[k][0]} ± {lab_std[k][0]}", "Nền: a": f"{lab_mean[k][1]} ± {lab_std[k][1]}",
                  "Nền: b": f"{lab_mean[k][2]} ± {lab_std[k][2]}", "Tỉ lệ diện tích quả chuối": f"{area[k]:.3f}"} for k in CLASS_NAMES]
    save_json(exp_dir("S2") / f"metrics{sfx}.json", raw)
    if not sfx:
        register("S2", srows)
    lk = raw["_silhouette_leak_gradient_ratio"]
    save_table("S2", [{"Chỉ số": "Gradient trung bình dọc viền vùng đã lấp / gradient trung bình vùng nền xa quả",
                       "Trung vị": f"{lk['median']:.2f}", "Phân vị 25%": f"{lk['p25']:.2f}", "Phân vị 75%": f"{lk['p75']:.2f}", "Số ảnh": lk["n"]}],
               "table_leak" + sfx, title="S2: Vết viền hình bóng còn lại trên ảnh 'chỉ vùng nền' (1 = không còn vết)")
    print(save_table("S2", rows, "table" + sfx, title="S2: Đặc trưng rút trên toàn ảnh, chỉ vùng chuối, chỉ vùng nền; macro-F1 5-fold (train+val)" + (", mask mô hình nền bậc hai" if sfx else "")))
    print(save_table("S2", extra, "table_extreme" + sfx, title="S2: Hai phép thử cực đoan: chỉ màu nền, chỉ kích thước/hình bóng"))
    print("rò rỉ hình bóng (tỉ số gradient viền / nền xa):", raw["_silhouette_leak_gradient_ratio"])
    print(save_table("S2", stat_rows, "table_background_stats" + sfx, title="S2: Màu nền (Lab của OpenCV, thang 0-255) và kích thước quả theo lớp (train+val)"))

    fig, ax = plt.subplots(figsize=(11, 4))
    cn = [c for c, _, _ in confs]; x = np.arange(len(cn)); w = 0.26
    for j, v in enumerate(versions):
        m = [raw[f"{c}|{v}"]["macro_f1"] for c in cn]; s = [raw[f"{c}|{v}"]["macro_f1_std"] for c in cn]
        ax.bar(x + (j - 1) * w, m, w, yerr=s, capsize=3, label=v)
        for xi, val, e in zip(x + (j - 1) * w, m, s):
            ax.text(xi, val + e + 0.004, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" [", "\n[") for c in cn], fontsize=8); ax.set_ylim(0.6, 1.04)
    ax.set_ylabel("Macro-F1 (5-fold)"); ax.set_title("S2: Mô hình dựa vào quả chuối hay vào nền?"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"S2_background_bias{sfx}.png", dpi=130); plt.close(fig)
    # Hình: ba phiên bản ảnh của một mẫu mỗi lớp
    fig, axes = plt.subplots(3, 4, figsize=(9, 6.8))
    for c in range(4):
        k = int(np.where(y == c)[0][5])
        for r, (v, (_, ims)) in enumerate(versions.items()):
            axes[r, c].imshow(ims[k]); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0: axes[r, c].set_ylabel(v)
        axes[0, c].set_title(CLASS_NAMES[c], fontsize=9)
    fig.suptitle("S2: Ba phiên bản ảnh (chỉ chuối: nền tô màu trung vị; chỉ nền: vùng chuối nới rộng 12 px rồi inpainting)")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"S2_versions{sfx}.png", dpi=110); plt.close(fig)


def s2_cnn(mask_fn=None, sfx: str = ""):
    """S2 với CNN: mô hình đã huấn luyện trên ảnh đầy đủ (train), dự đoán trên ảnh val chỉ còn quả chuối
    (nền tô màu trung vị) hoặc chỉ còn nền (vùng chuối nới rộng 12 px rồi inpainting)."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.metrics import accuracy_score, f1_score
    from .cnn.predict import Predictor
    from .data import load_split
    from .exp_cnn import final_run
    va = load_split("val"); ids, y = va["idx"].values, va["label_id"].values
    imgs = load_all_images(256); masks = (mask_fn or all_masks)()
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    full, fg, bg = [], [], []
    for i in ids:
        im = np.asarray(imgs[i]); m = masks[i]; md = cv2.dilate(m, kd)
        med = np.median(im[m == 0], axis=0).astype(np.uint8)
        a = im.copy(); a[m == 0] = med
        full.append(im); fg.append(a); bg.append(cv2.inpaint(im, md, 7, cv2.INPAINT_TELEA))
    rows, raw = [], {}
    for model in ("a_vgg16", "b_mobilenetv3", "b_resnet50"):
        run = final_run(model); P = Predictor(run); row = {"Mô hình (học trên ảnh đầy đủ)": run}
        for name, X in [("Toàn ảnh", full), ("Chỉ vùng chuối", fg), ("Chỉ vùng nền", bg)]:
            p = P(np.stack(X)).argmax(1)
            f1 = float(f1_score(y, p, average="macro")); raw[f"{run}|{name}"] = {"macro_f1": f1, "accuracy": float(accuracy_score(y, p))}
            row[name] = f"{f1:.3f}"
        rows.append(row); print(row, flush=True)
    save_json(exp_dir("S2") / f"cnn_metrics{sfx}.json", raw)
    print(save_table("S2", rows, "table_cnn" + sfx, title="S2: CNN học trên ảnh đầy đủ, dự đoán trên ảnh val đã bỏ nền hoặc bỏ quả chuối (macro-F1, 371 ảnh)" + (", mask mô hình nền bậc hai" if sfx else "")))


def s2_recheck():
    """Chạy lại S2 (đặc trưng thủ công và CNN) với mask mô hình nền bậc hai của phần mở rộng."""
    s2(all_masks_v2, "_mask_v2"); s2_cnn(all_masks_v2, "_mask_v2")


EXPS = {"S1": s1, "S2": s2, "S2CNN": s2_cnn, "S2V2": s2_recheck}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(EXPS), nargs="*")
    args = ap.parse_args()
    for k in (args.only or EXPS):
        print(f"===== {k} =====", flush=True)
        EXPS[k]()


if __name__ == "__main__":
    main()
