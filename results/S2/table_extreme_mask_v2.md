### S2 — Hai phép thử cực đoan: chỉ màu nền, chỉ kích thước/hình bóng

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| Màu Lab trung bình của nền (3 số) + KNN k=5 | 3 | 0.944 ± 0.012 | 0.953 ± 0.011 |
| Màu Lab trung bình của nền (3 số) + LogReg | 3 | 0.606 ± 0.028 | 0.684 ± 0.022 |
| Tỉ lệ diện tích quả chuối trong ảnh (1 số) + KNN k=5 | 1 | 0.609 ± 0.014 | 0.629 ± 0.009 |
| HOG của mask nhị phân (chỉ hình bóng) + LogReg C=0.1 | 3780 | 0.871 ± 0.004 | 0.874 ± 0.003 |
