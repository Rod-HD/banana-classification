### P6 — Bộ phân lớp tuyến tính so với KNN theo số chiều của đặc trưng; macro-F1 5-fold

| Đặc trưng | KNN (k=5) | Logistic Regression | Linear SVM | LogReg − KNN |
|---|---|---|---|---|
| Moment màu (9 chiều) | 0.972 ± 0.008 | 0.908 ± 0.009 | 0.838 ± 0.011 | -0.064 |
| Hist HSV (512 chiều) | 0.965 ± 0.014 | 0.991 ± 0.005 | 0.989 ± 0.008 | +0.027 |
| HOG 64×128 (3780 chiều) | 0.946 ± 0.007 | 0.985 ± 0.004 | 0.982 ± 0.004 | +0.039 |
