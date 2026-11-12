### P4 — Brute Force, K-D Tree, Ball Tree (k=5, 1680 ảnh tham chiếu, 420 ảnh truy vấn, 1 luồng, trung vị 3 lần)

| Đặc trưng | brute: dựng (ms) | brute: dự đoán 420 ảnh (ms) | kd_tree: dựng (ms) | kd_tree: dự đoán 420 ảnh (ms) | kd_tree: kết quả trùng brute | ball_tree: dựng (ms) | ball_tree: dự đoán 420 ảnh (ms) | ball_tree: kết quả trùng brute |
|---|---|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0.6 | 1.6 | 1.4 | 3.2 | có | 1.1 | 4.5 | có |
| Hist HSV (512 chiều) | 1.3 | 9.1 | 38.7 | 361.1 | có | 29.3 | 244.3 | có |
| HOG 64×128 (3780 chiều) | 8.1 | 73.9 | 367.7 | 4533.7 | có | 223.5 | 2231.8 | có |
