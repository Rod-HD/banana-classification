### Cấu hình truyền thống đã chốt

| Vai trò | Cấu hình | Macro-F1 (5-fold) | Accuracy |
|---|---|---|---|
| trad_color | Hist HSV (8,8,8) + LogReg C=0.1 | 0.991 ± 0.005 | 0.992 |
| trad_edge | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.955 ± 0.015 | 0.959 |
| trad_hog | HOG 64×128 + LogReg C=0.1 | 0.986 ± 0.007 | 0.989 |
| trad_fusion | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.996 ± 0.003 | 0.997 |
