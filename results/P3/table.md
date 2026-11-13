### P3: Bình chọn đa số (uniform) so với ưu tiên điểm gần (distance); 5-fold

| Đặc trưng | k | Macro-F1 uniform | Macro-F1 distance | Recall Bangla Kola (lớp ít ảnh nhất) uniform | Recall Bangla Kola distance | Recall thấp nhất trong 4 lớp uniform → distance |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 5 | 0.972 ± 0.008 | 0.972 ± 0.008 | 0.984 ± 0.013 | 0.984 ± 0.013 | 0.922 → 0.924 |
| Moment màu (9 chiều) | 15 | 0.955 ± 0.008 | 0.961 ± 0.006 | 0.942 ± 0.026 | 0.968 ± 0.018 | 0.888 → 0.888 |
| Hist HSV (512 chiều) | 5 | 0.965 ± 0.014 | 0.966 ± 0.013 | 0.981 ± 0.007 | 0.981 ± 0.007 | 0.879 → 0.888 |
| Hist HSV (512 chiều) | 15 | 0.931 ± 0.013 | 0.933 ± 0.014 | 0.952 ± 0.018 | 0.955 ± 0.022 | 0.756 → 0.765 |
| HOG 64×128 (3780 chiều) | 5 | 0.946 ± 0.007 | 0.949 ± 0.008 | 0.955 ± 0.021 | 0.952 ± 0.026 | 0.816 → 0.830 |
| HOG 64×128 (3780 chiều) | 15 | 0.912 ± 0.011 | 0.914 ± 0.010 | 0.920 ± 0.033 | 0.923 ± 0.036 | 0.697 → 0.706 |
