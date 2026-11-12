### N7 — Feature vector thủ công so với embedding của CNN: học trên train (1729 ảnh), macro-F1 trên val (371 ảnh)

| Đặc trưng | Loại | Số chiều | KNN (k=5) | LogReg |
|---|---|---|---|---|
| Hist HSV (8,8,8) | thủ công | 512 | 0.982 | 0.997 |
| HOG 64×128 | thủ công | 3780 | 0.945 | 0.987 |
| Hist H-S + HOG (ghép, chuẩn hóa từng phần) | thủ công | 3844 | 0.990 | 0.998 |
| VGG16 ImageNet, trung bình 7×7 (chưa học trên chuối) | embedding | 512 | 0.932 | 0.986 |
| a_vgg16, lớp Dense(256) áp chót | embedding | 256 | 0.986 | 0.986 |
| MobileNetV3 ImageNet, sau GAP (chưa fine-tune) | embedding | 576 | 0.981 | 0.993 |
| b_mobilenetv3_lr1e-4, sau GAP (đã fine-tune) | embedding | 576 | 1.000 | 0.996 |
