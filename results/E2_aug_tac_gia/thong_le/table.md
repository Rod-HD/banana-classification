### Nhánh T-tl: 9 cấu hình của pipeline gốc (bản Augmented của tác giả, chia ngẫu nhiên 70/15/15 không quan tâm ảnh gốc; 5189 ảnh train, chấm trên 1112 ảnh test)

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 1112 |
|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.980 | 0.991 | [0.984; 0.997] | 0.993 | 8 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.590 | 0.620 | [0.589; 0.650] | 0.640 | 400 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.817 | 0.847 | [0.825; 0.868] | 0.854 | 162 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.978 | 0.976 | [0.966; 0.985] | 0.978 | 24 |
| CNN hướng A | a_lenet | 0.964 | 0.952 | [0.937; 0.965] | 0.960 | 44 |
| CNN hướng A | a_vgg16 | 0.971 | 0.967 | [0.956; 0.978] | 0.970 | 33 |
| CNN hướng B | b_mobilenetv3 | 0.999 | 0.995 | [0.990; 0.999] | 0.996 | 5 |
