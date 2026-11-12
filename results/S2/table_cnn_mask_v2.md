### S2 — CNN học trên ảnh đầy đủ, dự đoán trên ảnh val đã bỏ nền hoặc bỏ quả chuối (macro-F1, 371 ảnh) — mask mô hình nền bậc hai

| Mô hình (học trên ảnh đầy đủ) | Toàn ảnh | Chỉ vùng chuối | Chỉ vùng nền |
|---|---|---|---|
| a_vgg16_lr1e-3 | 0.990 | 0.949 | 0.799 |
| b_mobilenetv3_lr1e-4 | 1.000 | 0.967 | 0.676 |
| b_resnet50 | 0.998 | 0.924 | 0.721 |
