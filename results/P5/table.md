### P5 — Thời gian huấn luyện và dự đoán (trung bình mỗi fold, đặc trưng đã rút sẵn)

| Đặc trưng | Bộ phân lớp | Huấn luyện 1680 ảnh (ms) | Dự đoán 420 ảnh (ms) | Dự đoán (ms/ảnh) | Macro-F1 |
|---|---|---|---|---|---|
| Moment màu (9 chiều) | KNN (k=5) | 1.6 | 3.0 | 0.0072 | 0.972 ± 0.008 |
| Moment màu (9 chiều) | Logistic Regression | 15.2 | 0.5 | 0.0011 | 0.908 ± 0.009 |
| Moment màu (9 chiều) | Linear SVM | 5.3 | 0.5 | 0.0012 | 0.838 ± 0.011 |
| Hist HSV (512 chiều) | KNN (k=5) | 8.8 | 19.4 | 0.0461 | 0.965 ± 0.014 |
| Hist HSV (512 chiều) | Logistic Regression | 528.9 | 1.5 | 0.0035 | 0.991 ± 0.005 |
| Hist HSV (512 chiều) | Linear SVM | 491.3 | 2.1 | 0.0049 | 0.989 ± 0.008 |
| HOG 64×128 (3780 chiều) | KNN (k=5) | 65.0 | 92.5 | 0.2203 | 0.946 ± 0.007 |
| HOG 64×128 (3780 chiều) | Logistic Regression | 1225.1 | 10.3 | 0.0245 | 0.985 ± 0.004 |
| HOG 64×128 (3780 chiều) | Linear SVM | 2568.0 | 9.5 | 0.0225 | 0.982 ± 0.004 |
