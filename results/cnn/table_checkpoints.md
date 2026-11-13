### Kiểm tra trọng số trong checkpoints/: dự đoán lại bằng best.keras, so với kết quả đã ghi, khớp 40/40 run được chấm lại

| Run | File (MB) | Tập chấm | Macro-F1 đã ghi | Macro-F1 tính lại | Số ảnh đoán sai (đã ghi / tính lại) | Khớp |
|---|---|---|---|---|---|---|
| T-kr__a_dropout | 7.04 | test (371 ảnh) | 0.962761 | 0.962761 | 13 / 13 | có |
| T-kr__a_lenet | 1.39 | test (371 ảnh) | 0.966263 | 0.966263 | 10 / 10 | có |
| T-kr__a_vgg16 | 103.84 | test (371 ảnh) | 0.984792 | 0.984792 | 5 / 5 | có |
| T-kr__b_mobilenetv3 | 4.34 | test (371 ảnh) | 0.997566 | 0.997566 | 1 / 1 | có |
| T-kr__b_mobilenetv3_s1 | 0.03 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| T-kr__b_resnet50 | 95.05 | test (371 ảnh) | 0.996774 | 0.996774 | 1 / 1 | có |
| T-kr__b_resnet50_s1 | 0.05 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| T-tl__a_lenet | 1.39 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| T-tl__a_vgg16 | 103.84 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| T-tl__b_mobilenetv3 | 4.34 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| T-tl__b_mobilenetv3_s1 | 0.03 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__a_dropout | 7.04 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__a_lenet | 1.39 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__a_vgg16 | 103.84 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__b_mobilenetv3 | 4.34 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__b_mobilenetv3_s1 | 0.03 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__b_resnet50 | 95.05 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| X__b_resnet50_s1 | 0.05 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| Xb__a_vgg16 | 103.84 | test (371 ảnh) | 0.976633 | 0.976633 | 8 / 8 | có |
| Xb__b_mobilenetv3 | 4.34 | test (371 ảnh) | 1.000000 | 1.000000 | 0 / 0 | có |
| Xb__b_mobilenetv3_s1 | 0.03 | - | - | - | - | không chấm lại ở đây; trọng số đã so từng bit (table_slim.md) |
| a_dropout | 7.03 | val (371 ảnh) | 0.939909 | 0.939909 | 20 / 20 | có |
| a_dropout_aug | 7.04 | val (371 ảnh) | 0.947960 | 0.947960 | 17 / 17 | có |
| a_dropout_nodrop | 7.02 | val (371 ảnh) | 0.951153 | 0.951153 | 16 / 16 | có |
| a_dropout_nodrop_seed43 | 7.02 | val (371 ảnh) | 0.960085 | 0.960085 | 12 / 12 | có |
| a_dropout_nodrop_seed44 | 7.02 | val (371 ảnh) | 0.946512 | 0.946512 | 18 / 18 | có |
| a_dropout_seed43 | 7.03 | val (371 ảnh) | 0.946800 | 0.946800 | 17 / 17 | có |
| a_dropout_seed44 | 7.03 | val (371 ảnh) | 0.935290 | 0.935290 | 21 / 21 | có |
| a_lenet | 1.39 | val (371 ảnh) | 0.973480 | 0.973480 | 8 / 8 | có |
| a_lenet_frac25 | 1.39 | val (371 ảnh) | 0.923538 | 0.923538 | 26 / 26 | có |
| a_lenet_frac50 | 1.39 | val (371 ảnh) | 0.949996 | 0.949996 | 16 / 16 | có |
| a_lenet_s128 | 6.55 | val (371 ảnh) | 0.956219 | 0.956219 | 14 / 14 | có |
| a_lenet_s32 | 0.28 | val (371 ảnh) | 0.954323 | 0.954323 | 15 / 15 | có |
| a_vgg16 | 103.84 | val (371 ảnh) | 0.983524 | 0.983524 | 5 / 5 | có |
| a_vgg16_frac25 | 103.84 | val (371 ảnh) | 0.967336 | 0.967336 | 10 / 10 | có |
| a_vgg16_frac50 | 103.84 | val (371 ảnh) | 0.962823 | 0.962823 | 12 / 12 | có |
| a_vgg16_lr1e-3 | 103.84 | val (371 ảnh) | 0.989541 | 0.989541 | 3 / 3 | có |
| a_vgg16_lr1e-3_cw | 103.84 | val (371 ảnh) | 0.987098 | 0.987098 | 4 / 4 | có |
| a_vgg16_lr1e-3_minaug | 103.84 | val (371 ảnh) | 0.994349 | 0.994349 | 2 / 2 | có |
| a_vgg16_lr1e-3_os | 103.84 | val (371 ảnh) | 0.986881 | 0.986881 | 4 / 4 | có |
| b_mobilenetv3 | 4.34 | val (371 ảnh) | 0.993048 | 0.993048 | 2 / 2 | có |
| b_mobilenetv3_frac25 | 4.34 | val (371 ảnh) | 0.974549 | 0.974549 | 8 / 8 | có |
| b_mobilenetv3_frac50 | 4.34 | val (371 ảnh) | 0.974340 | 0.974340 | 8 / 8 | có |
| b_mobilenetv3_lr1e-3 | 4.34 | val (371 ảnh) | 0.993281 | 0.993281 | 2 / 2 | có |
| b_mobilenetv3_lr1e-4 | 4.34 | val (371 ảnh) | 1.000000 | 1.000000 | 0 / 0 | có |
| b_mobilenetv3_lr1e-4_cw | 4.34 | val (371 ảnh) | 1.000000 | 1.000000 | 0 / 0 | có |
| b_mobilenetv3_lr1e-4_minaug | 4.34 | val (371 ảnh) | 1.000000 | 1.000000 | 0 / 0 | có |
| b_mobilenetv3_lr1e-4_os | 4.34 | val (371 ảnh) | 1.000000 | 1.000000 | 0 / 0 | có |
| b_mobilenetv3_noaug | 4.33 | val (371 ảnh) | 0.996519 | 0.996519 | 1 / 1 | có |
| b_mobilenetv3_s1 | 0.03 | val (371 ảnh) | 0.987033 | 0.987033 | 4 / 4 | có |
| b_mobilenetv3_s1_frac25 | 0.03 | val (371 ảnh) | 0.974014 | 0.974014 | 8 / 8 | có |
| b_mobilenetv3_s1_frac50 | 0.03 | val (371 ảnh) | 0.980009 | 0.980009 | 6 / 6 | có |
| b_resnet50 | 95.05 | val (371 ảnh) | 0.997577 | 0.997577 | 1 / 1 | có |
| b_resnet50_s1 | 0.05 | val (371 ảnh) | 0.996458 | 0.996458 | 1 / 1 | có |

