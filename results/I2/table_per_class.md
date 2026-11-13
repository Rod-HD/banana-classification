### I2: Precision (P) và Recall (R) từng lớp

| Mô hình | Cách xử lý | P Bangla | P Champa | P Sabri | P Sagor | R Bangla | R Champa | R Sabri | R Sagor |
|---|---|---|---|---|---|---|---|---|---|
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Không xử lý | 0.995 | 0.999 | 1.000 | 0.991 | 0.995 | 0.996 | 1.000 | 0.996 |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Class weight | 0.995 | 0.999 | 1.000 | 0.991 | 0.995 | 0.996 | 1.000 | 0.996 |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Oversampling | 0.992 | 0.999 | 1.000 | 0.989 | 0.992 | 0.996 | 0.998 | 0.996 |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Tăng cường lớp ít | 0.992 | 0.999 | 1.000 | 0.991 | 0.997 | 0.995 | 1.000 | 0.993 |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Không xử lý | 0.849 | 0.921 | 0.959 | 0.847 | 0.822 | 0.920 | 0.977 | 0.857 |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Class weight | 0.823 | 0.925 | 0.953 | 0.844 | 0.828 | 0.902 | 0.977 | 0.859 |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Oversampling | 0.821 | 0.926 | 0.952 | 0.843 | 0.825 | 0.905 | 0.975 | 0.857 |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Tăng cường lớp ít | 0.776 | 0.957 | 0.935 | 0.844 | 0.857 | 0.892 | 0.965 | 0.852 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Không xử lý | 0.970 | 1.000 | 1.000 | 0.987 | 0.985 | 1.000 | 0.987 | 0.987 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Class weight | 0.970 | 0.993 | 1.000 | 0.987 | 0.985 | 1.000 | 0.987 | 0.975 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Oversampling | 0.956 | 1.000 | 1.000 | 0.987 | 0.985 | 0.993 | 0.987 | 0.987 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Tăng cường lớp ít | 1.000 | 0.993 | 1.000 | 0.987 | 1.000 | 1.000 | 0.987 | 0.987 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Không xử lý | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Class weight | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Oversampling | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Tăng cường lớp ít | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
