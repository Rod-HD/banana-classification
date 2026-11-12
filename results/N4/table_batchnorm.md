### N4 (phụ) — BatchNorm khi fine-tune: 64 ảnh train, lô 64

| Thân mạng | Cách mở băng | Thay đổi tương đối của embedding (training=True so với suy luận) | Accuracy đầu phân lớp GĐ1, embedding chế độ suy luận | Accuracy đầu phân lớp GĐ1, embedding chế độ huấn luyện |
|---|---|---|---|---|
| b_mobilenetv3 | BN mở (trainable=True) | 0.820 | 1.000 | 0.047 |
| b_mobilenetv3 | BN đóng băng (cách đồ án dùng) | 0.000 | 1.000 | 1.000 |
| b_resnet50 | BN mở (trainable=True) | 0.838 | 1.000 | 0.125 |
| b_resnet50 | BN đóng băng (cách đồ án dùng) | 0.000 | 1.000 | 1.000 |
