### S2 — Hai phép thử cực đoan: chỉ màu nền, chỉ kích thước/hình bóng

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| Màu Lab trung bình của nền (3 số) + KNN k=5 | 3 | 0.934 ± 0.012 | 0.945 ± 0.011 |
| Màu Lab trung bình của nền (3 số) + LogReg | 3 | 0.637 ± 0.024 | 0.706 ± 0.020 |
| Tỉ lệ diện tích quả chuối trong ảnh (1 số) + KNN k=5 | 1 | 0.619 ± 0.022 | 0.642 ± 0.016 |
| HOG của mask nhị phân (chỉ hình bóng) + LogReg C=0.1 | 3780 | 0.857 ± 0.017 | 0.861 ± 0.014 |
