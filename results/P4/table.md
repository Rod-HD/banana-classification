### P4: Brute Force, K-D Tree, Ball Tree (k=5, 1680 ảnh tham chiếu, 420 ảnh truy vấn, 1 luồng, trung vị 3 lần)

| Đặc trưng | brute: dựng (ms) | brute: dự đoán 420 ảnh (ms) | kd_tree: dựng (ms) | kd_tree: dự đoán 420 ảnh (ms) | kd_tree: kết quả trùng brute | ball_tree: dựng (ms) | ball_tree: dự đoán 420 ảnh (ms) | ball_tree: kết quả trùng brute |
|---|---|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0.7 | 1.8 | 1.5 | 4.0 | có | 1.6 | 4.9 | có |
| Hist HSV (512 chiều) | 1.5 | 8.1 | 38.1 | 363.7 | có | 32.7 | 271.6 | có |
| HOG 64×128 (3780 chiều) | 7.3 | 63.3 | 276.2 | 3795.9 | có | 303.9 | 3196.5 | có |
