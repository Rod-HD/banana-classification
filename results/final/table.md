### Đánh giá cuối trên tập test (371 ảnh; mỗi ảnh sai tương ứng 0,27% accuracy)

| Nhóm | Mô hình | Macro-F1 test | KTC 95% (bootstrap) | Accuracy test | Số ảnh sai / 371 | Recall thấp nhất | Macro-F1 lúc chọn (5-fold CV hoặc val) |
|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.994 | [0.984; 1.000] | 0.995 | 2 | 0.985 (Bangla) | 0.991 ± 0.005 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.964 | [0.944; 0.982] | 0.968 | 12 | 0.885 (Sagor) | 0.955 ± 0.015 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.990 | [0.977; 1.000] | 0.992 | 3 | 0.985 (Bangla) | 0.986 ± 0.007 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 1.000 | [1.000; 1.000] | 1.000 | 0 | 1.000 (Bangla) | 0.996 ± 0.003 |
| CNN hướng A | a_lenet (biến thể: a_lenet) | 0.978 | [0.960; 0.993] | 0.981 | 7 | 0.955 (Bangla) | 0.973 |
| CNN hướng A | a_dropout (biến thể: a_dropout_aug) | 0.942 | [0.917; 0.965] | 0.943 | 21 | 0.872 (Sagor) | 0.948 |
| CNN hướng A | a_vgg16 (biến thể: a_vgg16_lr1e-3) | 0.986 | [0.971; 0.997] | 0.989 | 4 | 0.970 (Bangla) | 0.990 |
| CNN hướng B | b_mobilenetv3 (biến thể: b_mobilenetv3_lr1e-4) | 1.000 | [1.000; 1.000] | 1.000 | 0 | 1.000 (Bangla) | 1.000 |
| CNN hướng B | b_resnet50 (biến thể: b_resnet50) | 0.997 | [0.988; 1.000] | 0.997 | 1 | 0.987 (Sabri) | 0.998 |
