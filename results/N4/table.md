### N4: Đóng băng (giai đoạn 1) so với fine-tune (giai đoạn 2) và ảnh hưởng của learning rate

| Cấu hình | Tham số được huấn luyện | Macro-F1 val | Accuracy val | Loss val (epoch chọn) | Macro-F1 val sau epoch 1 | Macro-F1 val thấp nhất | Epoch chọn / tổng | Thời gian mỗi epoch (s) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small, giai đoạn 1 (đóng băng thân) | 2308 | 0.987 | 0.989 | 0.0505 | 0.872 | 0.872 | 21 / 31 | 0.1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-5 | 917204 | 0.993 | 0.995 | 0.0203 | 0.981 | 0.978 | 12 / 17 | 20.1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-4 | 917204 | 1.000 | 1.000 | 0.0045 | 0.974 | 0.974 | 7 / 12 | 26.9 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-3 | 917204 | 0.993 | 0.995 | 0.0184 | 0.964 | 0.917 | 4 / 9 | 26.3 |
| ResNet50, giai đoạn 1 (đóng băng thân) | 8196 | 0.996 | 0.997 | 0.0269 | 0.942 | 0.942 | 10 / 20 | 0.2 |
| ResNet50, giai đoạn 2 (mở băng conv5), lr = 1e-5 | 14961668 | 0.998 | 0.997 | 0.0140 | 0.984 | 0.984 | 3 / 8 | 91.2 |
