### S2: Đặc trưng rút trên toàn ảnh, chỉ vùng chuối, chỉ vùng nền; macro-F1 5-fold (train+val)

| Đặc trưng + bộ phân lớp | Phiên bản ảnh | Macro-F1 | Accuracy |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Toàn ảnh | 0.991 ± 0.005 | 0.992 ± 0.005 |
| HOG 64×128 + LogReg C=0.1 | Toàn ảnh | 0.986 ± 0.007 | 0.989 ± 0.006 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Toàn ảnh | 0.996 ± 0.003 | 0.997 ± 0.003 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng chuối | 0.964 ± 0.010 | 0.967 ± 0.008 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng chuối | 0.884 ± 0.010 | 0.888 ± 0.009 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng chuối | 0.958 ± 0.005 | 0.961 ± 0.005 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng nền | 0.933 ± 0.008 | 0.943 ± 0.008 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng nền | 0.980 ± 0.009 | 0.983 ± 0.008 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng nền | 0.986 ± 0.005 | 0.988 ± 0.005 |
