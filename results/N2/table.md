### N2 — Macro-F1 trên val theo tỉ lệ tập train được dùng

| Mô hình | 25% train (432 ảnh) | 50% train (864 ảnh) | 100% train (1729 ảnh) | Tăng từ 25% lên 100% |
|---|---|---|---|---|
| a_lenet (LeNet, từ đầu) | 0.924 | 0.950 | 0.973 | +0.050 |
| a_vgg16 (đóng băng) | 0.967 | 0.963 | 0.984 | +0.016 |
| b_mobilenetv3 (fine-tune) | 0.975 | 0.974 | 0.993 | +0.018 |
