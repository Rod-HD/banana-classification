### C5: Ghép đặc trưng màu và HOG; macro-F1 5-fold

| Đặc trưng | Cách ghép | Số chiều | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|---|---|
| Hist HSV (8,8,8) | - | 512 | 0.965 ± 0.014 | 0.991 ± 0.005 | 0.989 ± 0.008 |
| Hist H-S (8,8) | - | 64 | 0.976 ± 0.006 | 0.985 ± 0.008 | 0.958 ± 0.007 |
| HOG 64×128 | - | 3780 | 0.946 ± 0.007 | 0.985 ± 0.004 | 0.982 ± 0.004 |
| Hist HSV + HOG | ghép thẳng, z-score chung | 4292 | 0.943 ± 0.008 | 0.993 ± 0.003 | 0.991 ± 0.003 |
| Hist HSV + HOG | chuẩn hóa từng phần rồi ghép | 4292 | 0.956 ± 0.012 | 0.993 ± 0.004 | 0.961 ± 0.007 |
| Hist H-S + HOG | ghép thẳng, z-score chung | 3844 | 0.948 ± 0.009 | 0.989 ± 0.003 | 0.989 ± 0.003 |
| Hist H-S + HOG | chuẩn hóa từng phần rồi ghép | 3844 | 0.977 ± 0.005 | 0.992 ± 0.003 | 0.966 ± 0.014 |
| Hist HSV + moment + HOG | ghép thẳng, z-score chung | 4301 | 0.945 ± 0.008 | 0.993 ± 0.003 | 0.992 ± 0.003 |
| Hist HSV + moment + HOG | chuẩn hóa từng phần rồi ghép | 4301 | 0.985 ± 0.009 | 0.991 ± 0.004 | 0.959 ± 0.007 |
