### I1: Accuracy so với macro-F1 và recall từng lớp (mô hình chưa xử lý mất cân bằng)

| Mô hình (chưa xử lý mất cân bằng) | Chấm trên | Accuracy | Macro-F1 | Accuracy − Macro-F1 | Recall Bangla | Recall Champa | Recall Sabri | Recall Sagor | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Luôn đoán lớp đông nhất (Champa Kola) | train+val | 0.402 | 0.143 | +0.259 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 (Bangla) |
| HOG 64×128 + KNN k=31 | out-of-fold 5-fold | 0.884 | 0.871 | +0.014 | 0.899 | 1.000 | 0.988 | 0.552 | 0.552 (Sagor) |
| Hist HSV + KNN k=31 | out-of-fold 5-fold | 0.883 | 0.868 | +0.015 | 0.870 | 0.998 | 0.986 | 0.576 | 0.576 (Sagor) |
| Moment màu + LogReg | out-of-fold 5-fold | 0.925 | 0.908 | +0.017 | 0.870 | 0.992 | 0.986 | 0.785 | 0.785 (Sagor) |
| Cạnh Sobel + LogReg | out-of-fold 5-fold | 0.900 | 0.894 | +0.007 | 0.822 | 0.920 | 0.977 | 0.857 | 0.822 (Bangla) |
| Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | out-of-fold 5-fold | 0.959 | 0.955 | +0.004 | 0.950 | 0.995 | 0.988 | 0.870 | 0.870 (Sagor) |
| HOG 64×128 + LogReg C=0.1 | out-of-fold 5-fold | 0.989 | 0.986 | +0.002 | 0.968 | 0.998 | 0.993 | 0.984 | 0.968 (Bangla) |
| Hist HSV (8,8,8) + LogReg C=0.1 | out-of-fold 5-fold | 0.992 | 0.991 | +0.001 | 0.992 | 0.998 | 0.998 | 0.978 | 0.978 (Sagor) |
| Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | out-of-fold 5-fold | 0.997 | 0.996 | +0.000 | 0.995 | 0.996 | 1.000 | 0.996 | 0.995 (Bangla) |
| CNN a_lenet_frac25 | val | 0.930 | 0.924 | +0.006 | 0.879 | 0.953 | 0.961 | 0.899 | 0.879 (Bangla) |
| CNN a_lenet | val | 0.978 | 0.973 | +0.005 | 1.000 | 0.993 | 0.987 | 0.924 | 0.924 (Sagor) |
| CNN a_dropout | val | 0.946 | 0.940 | +0.006 | 0.924 | 0.987 | 0.987 | 0.848 | 0.848 (Sagor) |
| CNN a_vgg16_lr1e-3 | val | 0.992 | 0.990 | +0.002 | 0.985 | 1.000 | 0.987 | 0.987 | 0.985 (Bangla) |
| CNN b_mobilenetv3_s1 | val | 0.989 | 0.987 | +0.002 | 0.970 | 1.000 | 1.000 | 0.975 | 0.970 (Bangla) |
| CNN b_mobilenetv3_lr1e-4 | val | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| CNN b_resnet50_s1 | val | 0.997 | 0.996 | +0.001 | 0.985 | 1.000 | 1.000 | 1.000 | 0.985 (Bangla) |
| CNN b_resnet50 | val | 0.997 | 0.998 | -0.000 | 1.000 | 1.000 | 1.000 | 0.987 | 0.987 (Sagor) |
