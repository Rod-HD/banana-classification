### S2: Chỉ dùng 4 góc ảnh 64×64 (không cần mask; trung bình 1.9% điểm ảnh ở góc thuộc vùng chuối đã nới rộng)

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| HOG của 4 góc ảnh (128×128) + LogReg C=0.1 | 8100 | 0.968 ± 0.008 | 0.973 ± 0.007 |
| Hist HSV (8,8,8) của 4 góc ảnh + LogReg C=0.1 | 512 | 0.939 ± 0.009 | 0.948 ± 0.008 |
