### P5: Thời gian huấn luyện và dự đoán (trung bình mỗi fold, đặc trưng đã rút sẵn)

| Đặc trưng | Bộ phân lớp | Huấn luyện 1680 ảnh (ms) | Dự đoán 420 ảnh (ms) | Dự đoán (ms/ảnh) | Macro-F1 |
|---|---|---|---|---|---|
| Moment màu (9 chiều) | KNN (k=5) | 2.2 | 3.9 | 0.0094 | 0.972 ± 0.008 |
| Moment màu (9 chiều) | Logistic Regression | 14.7 | 0.4 | 0.0010 | 0.908 ± 0.009 |
| Moment màu (9 chiều) | Linear SVM | 5.0 | 0.4 | 0.0011 | 0.838 ± 0.011 |
| Hist HSV (512 chiều) | KNN (k=5) | 8.0 | 21.7 | 0.0518 | 0.965 ± 0.014 |
| Hist HSV (512 chiều) | Logistic Regression | 479.1 | 4.0 | 0.0095 | 0.991 ± 0.005 |
| Hist HSV (512 chiều) | Linear SVM | 464.2 | 2.7 | 0.0065 | 0.989 ± 0.008 |
| HOG 64×128 (3780 chiều) | KNN (k=5) | 70.7 | 84.6 | 0.2014 | 0.946 ± 0.007 |
| HOG 64×128 (3780 chiều) | Logistic Regression | 1019.2 | 13.8 | 0.0328 | 0.985 ± 0.004 |
| HOG 64×128 (3780 chiều) | Linear SVM | 2211.7 | 8.3 | 0.0197 | 0.982 ± 0.004 |
