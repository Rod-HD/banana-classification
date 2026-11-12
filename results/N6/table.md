### N6 — So sánh hướng A với hướng B (chấm trên val; thời gian đo riêng trên CPU 4 nhân)

| Hướng | Mô hình | Biến thể được chốt | Macro-F1 val | Accuracy val | Số tham số | File mô hình (MB) | Thời gian huấn luyện (s) | Dự đoán 1 ảnh (ms, CPU) | Dự đoán theo lô 32 (ms/ảnh) | Cách tính thời gian huấn luyện |
|---|---|---|---|---|---|---|---|---|---|---|
| A | a_lenet | a_lenet | 0.973 | 0.978 | 337976 | 4.1 | 30 | 1.4 | 0.2 | 50 epoch × 0.6 s |
| A | a_dropout | a_dropout_aug | 0.948 | 0.954 | 1746260 | 21.0 | 192 | 2.5 | 0.7 | 40 epoch × 4.8 s |
| A | a_vgg16 | a_vgg16_lr1e-3 | 0.990 | 0.992 | 40669252 | 370.4 | 226 | 130.4 | 67.5 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 129 s; đầu phân lớp 24 epoch × 4.0 s |
| B | b_mobilenetv3 | b_mobilenetv3_lr1e-4 | 1.000 | 1.000 | 941428 | 11.7 | 216 | 12.5 | 2.6 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 25 s; giai đoạn 1: 31 epoch, 4 s; giai đoạn 2: 12 epoch × 15.6 s |
| B | b_resnet50 | b_resnet50 | 0.998 | 0.997 | 23595908 | 214.8 | 591 | 61.1 | 28.0 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 98 s; giai đoạn 1: 20 epoch, 4 s; giai đoạn 2: 8 epoch × 61.2 s |
