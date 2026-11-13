### T-tl: Accuracy trên test tách theo việc ảnh có file cùng ảnh gốc trong train

| Mô hình | Accuracy (ảnh test có anh em trong train) | Số ảnh | Accuracy (ảnh test không có) | Số ảnh  |
|---|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 | 0.993 | 1008 | 0.990 | 104 |
| Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.640 | 1008 | 0.644 | 104 |
| HOG 64×128 + LogReg C=0.1 | 0.859 | 1008 | 0.808 | 104 |
| Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.976 | 1008 | 1.000 | 104 |
| a_lenet | 0.959 | 1008 | 0.971 | 104 |
| a_vgg16 | 0.970 | 1008 | 0.971 | 104 |
| b_mobilenetv3 | 0.997 | 1008 | 0.981 | 104 |
