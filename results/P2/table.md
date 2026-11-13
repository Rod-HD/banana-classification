### P2: KNN (k=5) theo độ đo và cách chuẩn hóa; macro-F1 5-fold

| Độ đo | Chuẩn hóa | Moment màu (9 chiều) | Hist HSV (512 chiều) | HOG 64×128 (3780 chiều) |
|---|---|---|---|---|
| Euclidean | không | 0.977 ± 0.005 | 0.953 ± 0.006 | 0.954 ± 0.008 |
| Euclidean | vector đơn vị (L2) | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Euclidean | z-score | 0.972 ± 0.008 | 0.965 ± 0.014 | 0.946 ± 0.007 |
| Manhattan | không | 0.977 ± 0.004 | 0.964 ± 0.009 | 0.978 ± 0.007 |
| Manhattan | vector đơn vị (L2) | 0.967 ± 0.006 | 0.956 ± 0.013 | 0.978 ± 0.007 |
| Manhattan | z-score | 0.974 ± 0.008 | 0.961 ± 0.009 | 0.975 ± 0.010 |
| Cosine | không | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Cosine | vector đơn vị (L2) | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Hellinger | L1 rồi lấy căn | không áp dụng (có giá trị âm) | 0.976 ± 0.015 | 0.957 ± 0.011 |
