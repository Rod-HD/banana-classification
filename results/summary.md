# Tổng hợp kết quả

File này do `python -m src.summarize` sinh ra từ các file trong `results/`. Bảng đầy đủ mọi cấu hình: [`summary.csv`](summary.csv) (718 dòng). Số trong bảng dùng dấu chấm thập phân.

## 1. Kết quả chính: đánh giá cuối trên tập test

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

Chi tiết từng mô hình (classification report, confusion matrix): `results/final/<tên>/`. So sánh từng cặp: [`final/table_pairs.md`](final/table_pairs.md). Ảnh sai: [`final/errors.csv`](final/errors.csv).

## 2. Cấu hình truyền thống đã chốt (5-fold CV trên train+val)

| Vai trò | Cấu hình | Macro-F1 (5-fold) | Accuracy |
|---|---|---|---|
| trad_color | Hist HSV (8,8,8) + LogReg C=0.1 | 0.991 ± 0.005 | 0.992 |
| trad_edge | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.955 ± 0.015 | 0.959 |
| trad_hog | HOG 64×128 + LogReg C=0.1 | 0.986 ± 0.007 | 0.989 |
| trad_fusion | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.996 ± 0.003 | 0.997 |

## 3. Các lần huấn luyện CNN (chấm trên val)

| Run | Mô hình | Ảnh vào | Số ảnh train | lr | Tăng cường | Tham số huấn luyện | Epoch chọn / tổng | Macro-F1 val | Accuracy val | Accuracy train | Giây / epoch | Thời gian huấn luyện (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_dropout | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 25 / 40 | 0.940 | 0.946 | 0.999 | 4.2 | 168 |
| a_dropout_aug | a_dropout | 128×128 | 1729 | 0.001 | có | 1746260 | 25 / 40 | 0.948 | 0.954 | 0.967 | 4.8 | 194 |
| a_dropout_nodrop | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 33 / 40 | 0.951 | 0.957 | 1.000 | 3.8 | 150 |
| a_dropout_nodrop_seed43 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 18 / 40 | 0.960 | 0.968 | 1.000 | 5.8 | 234 |
| a_dropout_nodrop_seed44 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 37 / 40 | 0.947 | 0.951 | 1.000 | 6.0 | 242 |
| a_dropout_seed43 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 22 / 40 | 0.947 | 0.954 | 0.997 | 6.3 | 253 |
| a_dropout_seed44 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 39 / 40 | 0.935 | 0.943 | 0.999 | 5.8 | 233 |
| a_lenet | a_lenet | 64×64 | 1729 | 0.001 | không | 337976 | 40 / 50 | 0.973 | 0.978 | 1.000 | 0.6 | 31 |
| a_lenet_frac25 | a_lenet | 64×64 | 432 | 0.001 | không | 337976 | 22 / 32 | 0.924 | 0.930 | 1.000 | 0.2 | 7 |
| a_lenet_frac50 | a_lenet | 64×64 | 864 | 0.001 | không | 337976 | 13 / 23 | 0.950 | 0.957 | 0.995 | 0.4 | 8 |
| a_lenet_s128 | a_lenet | 128×128 | 1729 | 0.001 | không | 1628216 | 13 / 23 | 0.956 | 0.962 | 1.000 | 1.7 | 40 |
| a_lenet_s32 | a_lenet | 32×32 | 1729 | 0.001 | không | 61496 | 20 / 30 | 0.954 | 0.960 | 0.984 | 0.3 | 9 |
| a_vgg16 | a_vgg16 | 224×224 | 1729 | 0.0001 | không | 25954564 | 7 / 15 | 0.984 | 0.987 | 1.000 | 3.9 | 59 |
| a_vgg16_frac25 | a_vgg16 | 224×224 | 432 | 0.0001 | không | 25954564 | 3 / 11 | 0.967 | 0.973 | 0.995 | 1.2 | 13 |
| a_vgg16_frac50 | a_vgg16 | 224×224 | 864 | 0.0001 | không | 25954564 | 3 / 11 | 0.963 | 0.968 | 1.000 | 2.1 | 23 |
| a_vgg16_lr1e-3 | a_vgg16 | 224×224 | 1729 | 0.001 | không | 25954564 | 16 / 24 | 0.990 | 0.992 | 0.999 | 3.9 | 94 |
| a_vgg16_lr1e-3_cw | a_vgg16 | 224×224 | 1729 | 0.001 | không | 25954564 | 8 / 16 | 0.987 | 0.989 | 0.999 | 4.1 | 66 |
| a_vgg16_lr1e-3_minaug | a_vgg16 | 224×224 | 2780 | 0.001 | không | 25954564 | 9 / 17 | 0.994 | 0.995 | 0.999 | 6.1 | 104 |
| a_vgg16_lr1e-3_os | a_vgg16 | 224×224 | 2780 | 0.001 | không | 25954564 | 4 / 12 | 0.987 | 0.989 | 0.997 | 6.1 | 74 |
| b_mobilenetv3 | b_mobilenetv3 | 224×224 | 1729 | 1e-05 | có | 917204 | 12 / 17 | 0.993 | 0.995 | 0.997 | 20.1 | 341 |
| b_mobilenetv3_frac25 | b_mobilenetv3 | 224×224 | 432 | 1e-05 | có | 917204 | 12 / 17 | 0.975 | 0.978 | 0.998 | 7.3 | 124 |
| b_mobilenetv3_frac50 | b_mobilenetv3 | 224×224 | 864 | 1e-05 | có | 917204 | 2 / 7 | 0.974 | 0.978 | 0.994 | 16.2 | 114 |
| b_mobilenetv3_lr1e-3 | b_mobilenetv3 | 224×224 | 1729 | 0.001 | có | 917204 | 4 / 9 | 0.993 | 0.995 | 0.998 | 26.3 | 236 |
| b_mobilenetv3_lr1e-4 | b_mobilenetv3 | 224×224 | 1729 | 0.0001 | có | 917204 | 7 / 12 | 1.000 | 1.000 | 1.000 | 26.9 | 323 |
| b_mobilenetv3_lr1e-4_cw | b_mobilenetv3 | 224×224 | 1729 | 0.0001 | có | 917204 | 8 / 13 | 1.000 | 1.000 | 1.000 | 16.3 | 212 |
| b_mobilenetv3_lr1e-4_minaug | b_mobilenetv3 | 224×224 | 2780 | 0.0001 | có | 917204 | 5 / 10 | 1.000 | 1.000 | 0.999 | 25.0 | 250 |
| b_mobilenetv3_lr1e-4_os | b_mobilenetv3 | 224×224 | 2780 | 0.0001 | có | 917204 | 3 / 8 | 1.000 | 1.000 | 0.998 | 28.6 | 229 |
| b_mobilenetv3_noaug | b_mobilenetv3 | 224×224 | 1729 | 1e-05 | không | 917204 | 4 / 9 | 0.997 | 0.997 | 0.999 | 23.1 | 208 |
| b_mobilenetv3_s1 | b_mobilenetv3 | 224×224 | 1729 | 0.001 | không | 2308 | 21 / 31 | 0.987 | 0.989 | 0.997 | 0.1 | 4 |
| b_mobilenetv3_s1_frac25 | b_mobilenetv3 | 224×224 | 432 | 0.001 | không | 2308 | 41 / 51 | 0.974 | 0.978 | 1.000 | 0.1 | 3 |
| b_mobilenetv3_s1_frac50 | b_mobilenetv3 | 224×224 | 864 | 0.001 | không | 2308 | 17 / 27 | 0.980 | 0.984 | 0.993 | 0.1 | 2 |
| b_resnet50 | b_resnet50 | 224×224 | 1729 | 1e-05 | có | 14961668 | 3 / 8 | 0.998 | 0.997 | 1.000 | 91.2 | 730 |
| b_resnet50_s1 | b_resnet50 | 224×224 | 1729 | 0.001 | không | 8196 | 10 / 20 | 0.996 | 0.997 | 1.000 | 0.2 | 4 |

## 4. Danh mục 30 thí nghiệm kiểm chứng

| Mã | Nội dung | Bảng số liệu | Hình |
|---|---|---|---|
| D1 | Rò rỉ do tăng cường trước khi chia | [`D1/table.md`](D1/table.md), [`D1/table_near_duplicates.md`](D1/table_near_duplicates.md) | [`D1_leakage.png`](figures/D1_leakage.png) |
| M1 | Xáo trộn lưới 4×4 | [`M1/table.md`](M1/table.md) | [`M1_shuffle.png`](figures/M1_shuffle.png) |
| M2 | Đổi độ sáng (kênh V) | [`M2/table.md`](M2/table.md) | [`M2_brightness.png`](figures/M2_brightness.png) |
| M3 | Số bin histogram | [`M3/table.md`](M3/table.md) | [`M3_bins.png`](figures/M3_bins.png) |
| M4 | So sánh đặc trưng màu | [`M4/table.md`](M4/table.md) | [`M4_color_features.png`](figures/M4_color_features.png) |
| M5 | CCV | [`M5/table.md`](M5/table.md) | [`M5_ccv.png`](figures/M5_ccv.png) |
| M6 | Ảnh xám | [`M6/table.md`](M6/table.md) | [`M6_grayscale.png`](figures/M6_grayscale.png) |
| C1 | Toán tử cạnh và nhiễu | [`C1/table.md`](C1/table.md), [`C1/table_feature_change.md`](C1/table_feature_change.md) | [`C1_edge_noise.png`](figures/C1_edge_noise.png) |
| C2 | HOG và độ sáng | [`C2/table.md`](C2/table.md), [`C2/table_feature_change.md`](C2/table_feature_change.md), [`C2/table_saturation.md`](C2/table_saturation.md) | [`C2_hog_brightness.png`](figures/C2_hog_brightness.png) |
| C3 | Tham số HOG | [`C3/table.md`](C3/table.md), [`C3/table_variants.md`](C3/table_variants.md) | [`C3_hog_params.png`](figures/C3_hog_params.png) |
| C4 | Che 25% | [`C4/table.md`](C4/table.md), [`C4/table_hist_scaling.md`](C4/table_hist_scaling.md) | [`C4_occlusion.png`](figures/C4_occlusion.png) |
| C5 | Ghép màu + HOG | [`C5/table.md`](C5/table.md) | [`C5_fusion.png`](figures/C5_fusion.png) |
| P1 | KNN theo k | [`P1/table.md`](P1/table.md), [`P1/table_all_k.md`](P1/table_all_k.md) | [`P1_knn_k.png`](figures/P1_knn_k.png) |
| P2 | Độ đo và chuẩn hóa | [`P2/table.md`](P2/table.md), [`P2/table_rank_agreement.md`](P2/table_rank_agreement.md) | [`P2_metrics.png`](figures/P2_metrics.png) |
| P3 | Bình chọn uniform/distance | [`P3/table.md`](P3/table.md) | [`P3_weights.png`](figures/P3_weights.png) |
| P4 | Brute, K-D Tree, Ball Tree | [`P4/table.md`](P4/table.md), [`P4/table_scaling.md`](P4/table_scaling.md) | [`P4_knn_algorithms.png`](figures/P4_knn_algorithms.png) |
| P5 | Thời gian huấn luyện/dự đoán | [`P5/table.md`](P5/table.md) | [`P5_time.png`](figures/P5_time.png) |
| P6 | Tuyến tính so với KNN | [`P6/table.md`](P6/table.md) | [`P6_linear_vs_knn.png`](figures/P6_linear_vs_knn.png) |
| N1 | Số tham số | [`N1/table.md`](N1/table.md), [`N1/table_totals.md`](N1/table_totals.md) | [`N1_params.png`](figures/N1_params.png) |
| N2 | Lượng dữ liệu | [`N2/table.md`](N2/table.md) | [`N2_learning_curve.png`](figures/N2_learning_curve.png) |
| N3 | Dropout | [`N3/table.md`](N3/table.md), [`N3/table_seeds.md`](N3/table_seeds.md) | [`N3_dropout.png`](figures/N3_dropout.png) |
| N4 | Đóng băng và fine-tune | [`N4/table.md`](N4/table.md), [`N4/table_batchnorm.md`](N4/table_batchnorm.md) | [`N4_finetune_lr.png`](figures/N4_finetune_lr.png) |
| N5 | Tăng cường ảnh | [`N5/table.md`](N5/table.md) | [`N5_augment_a_dropout.png`](figures/N5_augment_a_dropout.png), [`N5_augment_b_mobilenetv3.png`](figures/N5_augment_b_mobilenetv3.png) |
| N6 | Hướng A so với hướng B | [`N6/table.md`](N6/table.md) | [`N6_a_vs_b.png`](figures/N6_a_vs_b.png) |
| N7 | Embedding | [`N7/table.md`](N7/table.md) | [`N7_embedding.png`](figures/N7_embedding.png) |
| N8 | Độ bền của CNN | [`N8/table.md`](N8/table.md) | [`N8_robustness.png`](figures/N8_robustness.png) |
| I1 | Accuracy và macro-F1 | [`I1/table.md`](I1/table.md) | [`I1_accuracy_vs_macro_f1.png`](figures/I1_accuracy_vs_macro_f1.png) |
| I2 | Xử lý mất cân bằng | [`I2/table.md`](I2/table.md), [`I2/table_per_class.md`](I2/table_per_class.md) | [`I2_imbalance.png`](figures/I2_imbalance.png) |
| S1 | Phương pháp phân đoạn | [`S1/table.md`](S1/table.md) | [`S1_segmentation.png`](figures/S1_segmentation.png) |
| S2 | Thiên lệch nền | [`S2/table.md`](S2/table.md), [`S2/table_background_stats.md`](S2/table_background_stats.md), [`S2/table_cnn.md`](S2/table_cnn.md), [`S2/table_corners.md`](S2/table_corners.md), [`S2/table_extreme.md`](S2/table_extreme.md), [`S2/table_leak.md`](S2/table_leak.md) | [`S2_background_bias.png`](figures/S2_background_bias.png), [`S2_corners.png`](figures/S2_corners.png), [`S2_versions.png`](figures/S2_versions.png) |

## 5. Số dòng trong summary.csv theo thí nghiệm

| Thí nghiệm | Số cấu hình | Macro-F1 cao nhất (ảnh gốc) | Cấu hình đạt cao nhất |
|---|---|---|---|
| M1 | 24 | 0.991 | hsv_hist + logreg (5-fold CV (train+val)) |
| M2 | 18 | 0.991 | hsv_hist + logreg (5-fold CV (train+val)) |
| M3 | 20 | 0.992 | hsv_hist bins=(16,16,16) + logreg (5-fold CV (train+val)) |
| M4 | 21 | 0.991 | hsv_hist + logreg (5-fold CV (train+val)) |
| M5 | 8 | 0.975 | ccv + logreg (5-fold CV (train+val)) |
| M6 | 42 | 0.991 | hsv_hist+logreg (5-fold CV (train+val)) |
| C1 | 40 | 0.932 | edge[roberts] + logreg (5-fold CV (train+val)) |
| C2 | 12 | 0.985 | HOG, chuẩn hóa block L2 + logreg (5-fold CV (train+val)) |
| C3 | 42 | 0.986 | hog 128x128_c8_b9 + logreg (5-fold CV (train+val)) |
| C4 | 24 | 0.991 | hsv_hist + logreg (5-fold CV (train+val)) |
| C5 | 27 | 0.993 | Hist HSV + HOG (chuẩn hóa từng phần rồi ghép) + logreg (5-fold CV (train+val)) |
| P1 | 48 | 0.977 | moments + knn k=1 (5-fold CV (train+val)) |
| P2 | 26 | 0.978 | hog + knn k=5 Manhattan / chuẩn hóa none (5-fold CV (train+val)) |
| P3 | 12 | 0.972 | moments + knn k=5 weights=distance (5-fold CV (train+val)) |
| P5 | 9 | 0.991 | hsv_hist + logreg (5-fold CV (train+val)) |
| select_traditional | 141 | 0.996 | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần (5-fold CV (train+val)) |
| CNN | 33 | 1.000 | b_mobilenetv3_lr1e-4 (val) |
| N2 | 9 | 0.993 | b_mobilenetv3 (val) |
| N3 | 2 | 0.951 | a_dropout_nodrop (val) |
| N4 | 6 | 1.000 | b_mobilenetv3_lr1e-4 (val) |
| N5 | 4 | 0.997 | b_mobilenetv3_noaug (val) |
| N6 | 5 | 1.000 | b_mobilenetv3_lr1e-4 (val) |
| N7 | 14 | 1.000 | b_mobilenetv3_lr1e-4, sau GAP (đã fine-tune) + knn (val (học trên train)) |
| N8 | 91 | 1.000 | Hướng B tốt nhất: b_mobilenetv3_lr1e-4 (val (học trên train)) |
| I2 | 16 | 1.000 | Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Không xử lý (val) |
| S2 | 15 | 0.996 | Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Toàn ảnh (5-fold CV (train+val)) |
| final | 9 | 1.000 | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần (test) |
