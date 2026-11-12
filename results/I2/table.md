### I2 — Các cách xử lý mất cân bằng (lấy mẫu lại chỉ trên phần huấn luyện)

| Mô hình | Cách xử lý | Chấm | Macro-F1 | Accuracy | Precision Bangla (ít nhất) | Recall Bangla (ít nhất) | Precision Champa (đông nhất) | Recall Champa (đông nhất) | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Không xử lý | 5-fold CV | 0.996 ± 0.003 | 0.997 | 0.995 | 0.995 | 0.999 | 0.996 | 0.995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Class weight | 5-fold CV | 0.996 ± 0.003 | 0.997 | 0.995 | 0.995 | 0.999 | 0.996 | 0.995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Oversampling | 5-fold CV | 0.995 ± 0.005 | 0.996 | 0.992 | 0.992 | 0.999 | 0.996 | 0.992 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Tăng cường lớp ít | 5-fold CV | 0.996 ± 0.002 | 0.996 | 0.992 | 0.997 | 0.999 | 0.995 | 0.993 (Sagor) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Không xử lý | 5-fold CV | 0.894 ± 0.008 | 0.900 | 0.849 | 0.822 | 0.921 | 0.920 | 0.822 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Class weight | 5-fold CV | 0.888 ± 0.007 | 0.895 | 0.823 | 0.828 | 0.925 | 0.902 | 0.828 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Oversampling | 5-fold CV | 0.888 ± 0.005 | 0.895 | 0.821 | 0.825 | 0.926 | 0.905 | 0.825 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Tăng cường lớp ít | 5-fold CV | 0.884 ± 0.011 | 0.892 | 0.776 | 0.857 | 0.957 | 0.892 | 0.852 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Không xử lý | val | 0.990 | 0.992 | 0.970 | 0.985 | 1.000 | 1.000 | 0.985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Class weight | val | 0.987 | 0.989 | 0.970 | 0.985 | 0.993 | 1.000 | 0.975 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Oversampling | val | 0.987 | 0.989 | 0.956 | 0.985 | 1.000 | 0.993 | 0.985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Tăng cường lớp ít | val | 0.994 | 0.995 | 1.000 | 1.000 | 0.993 | 1.000 | 0.987 (Sabri) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Không xử lý | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Class weight | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Oversampling | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Tăng cường lớp ít | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
