"""Gom kết quả của mọi thí nghiệm thành results/summary.csv và results/summary.md.

    python -m src.summarize
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import RESULTS_DIR

ORDER = ["D1", "M1", "M2", "M3", "M4", "M5", "M6", "C1", "C2", "C3", "C4", "C5", "P1", "P2", "P3", "P4", "P5", "P6", "select_traditional",
         "cnn", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "I1", "I2", "S1", "S2", "final", "so_sanh_nhanh", "E3_bai_bao"]
TITLE = {"D1": "Rò rỉ do tăng cường trước khi chia", "M1": "Xáo trộn lưới 4×4", "M2": "Đổi độ sáng (kênh V)", "M3": "Số bin histogram",
         "M4": "So sánh đặc trưng màu", "M5": "CCV", "M6": "Ảnh xám", "C1": "Toán tử cạnh và nhiễu", "C2": "HOG và độ sáng",
         "C3": "Tham số HOG", "C4": "Che 25%", "C5": "Ghép màu + HOG", "P1": "KNN theo k", "P2": "Độ đo và chuẩn hóa", "P3": "Bình chọn uniform/distance",
         "P4": "Brute, K-D Tree, Ball Tree", "P5": "Thời gian huấn luyện/dự đoán", "P6": "Tuyến tính so với KNN", "N1": "Số tham số", "N2": "Lượng dữ liệu",
         "N3": "Dropout", "N4": "Đóng băng và fine-tune", "N5": "Tăng cường ảnh", "N6": "Hướng A so với hướng B", "N7": "Embedding", "N8": "Độ bền của CNN",
         "I1": "Accuracy và macro-F1", "I2": "Xử lý mất cân bằng", "S1": "Phương pháp phân đoạn", "S2": "Thiên lệch nền"}


def tables_of(exp):
    d = RESULTS_DIR / exp
    return sorted(p.name for p in d.glob("table*.md")) if d.exists() else []


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = []
    for exp in ORDER:
        f = RESULTS_DIR / exp / "summary_rows.json"
        if f.exists():
            rows += json.loads(f.read_text())
    df = pd.DataFrame(rows)
    cols = ["experiment", "group", "config", "condition", "eval_set", "macro_f1", "macro_f1_std", "accuracy", "n_features",
            "extract_ms_per_image", "fit_time_s", "predict_ms_per_image", "macro_f1_ci95_low", "macro_f1_ci95_high"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    md = ["# Tổng hợp kết quả", "",
          "File này do `python -m src.summarize` sinh ra từ các file trong `results/`. Bảng đầy đủ mọi cấu hình: [`summary.csv`](summary.csv) "
          f"({len(df)} dòng). Số trong bảng dùng dấu chấm thập phân.", ""]
    fin = RESULTS_DIR / "final" / "table.md"
    if fin.exists():
        md += ["## 1. Kết quả chính: đánh giá cuối trên tập test", "", "\n".join(l for l in fin.read_text().splitlines() if l.startswith("|")), "",
               "Chi tiết từng mô hình (classification report, confusion matrix): `results/final/<tên>/`. "
               "So sánh từng cặp: [`final/table_pairs.md`](final/table_pairs.md). Ảnh sai: [`final/errors.csv`](final/errors.csv).", ""]
    sel = RESULTS_DIR / "select_traditional" / "table_selected.md"
    if sel.exists():
        md += ["## 2. Cấu hình truyền thống đã chốt (5-fold CV trên train+val)", "", "\n".join(l for l in sel.read_text().splitlines() if l.startswith("|")), ""]
    runs = RESULTS_DIR / "cnn" / "table_runs.md"
    if runs.exists():
        md += ["## 3. Các lần huấn luyện CNN (chấm trên val)", "", "\n".join(l for l in runs.read_text().splitlines() if l.startswith("|")), ""]
    md += ["## 4. Danh mục 30 thí nghiệm kiểm chứng", "", "| Mã | Nội dung | Bảng số liệu | Hình |", "|---|---|---|---|"]
    figs = sorted(p.name for p in (RESULTS_DIR / "figures").glob("*.png"))
    for exp in ORDER:
        if exp in TITLE:
            t = ", ".join(f"[`{exp}/{n}`]({exp}/{n})" for n in tables_of(exp)) or "chưa có"
            g = ", ".join(f"[`{n}`](figures/{n})" for n in figs if n.startswith(exp + "_")) or "chưa có"
            md.append(f"| {exp} | {TITLE[exp]} | {t} | {g} |")
    ext = [("so_sanh_nhanh/table_arms.md", "Cấu hình đại diện của mỗi nhánh (cùng 371 ảnh test gốc; điểm bền = trung bình macro-F1 trên 6 phép thử khó)"),
           ("so_sanh_nhanh/table_models.md", "Từng cấu hình qua các nhánh G (gốc), X (xử lý dữ liệu), T (bản Augmented của tác giả, không rò rỉ)"),
           ("E3_bai_bao/table_protocols.md", "Nhánh B: phương pháp của bài báo trên bản Augmented, 5-fold như bài báo so với 5-fold giữ nhóm theo ảnh gốc")]
    if any((RESULTS_DIR / f).exists() for f, _ in ext):
        md += ["", "## 5. Phần mở rộng: ba nhánh so sánh độc lập với pipeline gốc (mục 12 của kế hoạch)", "",
               "Chi tiết: `results/so_sanh_nhanh/`, `results/E1_xu_ly/`, `results/E2_aug_tac_gia/`, `results/E3_bai_bao/`.", ""]
        for f, cap in ext:
            if (RESULTS_DIR / f).exists():
                md += [f"**{cap}** ([`{f}`]({f})):", "", "\n".join(l for l in (RESULTS_DIR / f).read_text().splitlines() if l.startswith("|")), ""]
    md += ["", "## 6. Số dòng trong summary.csv theo thí nghiệm", "", "| Thí nghiệm | Số cấu hình | Macro-F1 cao nhất (ảnh gốc) | Cấu hình đạt cao nhất |", "|---|---|---|---|"]
    for exp, g in df.groupby("experiment", sort=False):
        c = g[g["condition"] == "clean"]
        if len(c):
            b = c.loc[c["macro_f1"].idxmax()]
            md.append(f"| {exp} | {len(g)} | {b['macro_f1']:.3f} | {b['config']} ({b['eval_set']}) |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print(f"summary.csv: {len(df)} dòng; summary.md đã ghi")


if __name__ == "__main__":
    main()
