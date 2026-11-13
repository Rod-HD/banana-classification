### S2: CNN học trên ảnh đầy đủ, dự đoán trên ảnh val đã bỏ nền hoặc bỏ quả chuối (macro-F1, 371 ảnh)

| Mô hình (học trên ảnh đầy đủ) | Toàn ảnh | Chỉ vùng chuối | Chỉ vùng nền |
|---|---|---|---|
| a_vgg16_lr1e-3 | 0.990 | 0.952 | 0.792 |
| b_mobilenetv3_lr1e-4 | 1.000 | 0.927 | 0.683 |
| b_resnet50 | 0.998 | 0.926 | 0.725 |
