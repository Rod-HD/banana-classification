### Nhánh Xb: 9 cấu hình của pipeline gốc (làm sạch; tiền xử lý: không xử lý; tăng cường train: hình học + quang học + thay nền ngẫu nhiên; 5082 ảnh train, chấm trên 371 ảnh test)

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 371 | Nền hoán đổi | V×0,6 | V×1,4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.980 | 0.988 | [0.974; 0.998] | 0.989 | 4 | 0.963 | 0.459 | 0.714 | 0.184 | 0.982 | 0.143 | 0.574 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.934 | 0.960 | [0.938; 0.979] | 0.962 | 14 | 0.799 | 0.691 | 0.892 | 0.179 | 0.708 | 0.617 | 0.648 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.946 | 0.953 | [0.931; 0.974] | 0.957 | 16 | 0.495 | 0.922 | 0.720 | 0.662 | 0.615 | 0.571 | 0.664 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.998 | 0.984 | [0.969; 0.995] | 0.984 | 6 | 0.853 | 0.977 | 0.930 | 0.128 | 0.908 | 0.842 | 0.773 |
| CNN hướng A | a_vgg16 | 0.987 | 0.977 | [0.960; 0.991] | 0.978 | 8 | 0.907 | 0.960 | 0.953 | 0.965 | 0.943 | 0.516 | 0.874 |
| CNN hướng B | b_mobilenetv3 | 1.000 | 1.000 | [1.000; 1.000] | 1.000 | 0 | 0.975 | 0.954 | 0.935 | 0.995 | 1.000 | 0.441 | 0.883 |
