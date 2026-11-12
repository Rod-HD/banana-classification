### N3 — Lặp lại với 3 seed khởi tạo (cùng cách chia dữ liệu)

| Cấu hình | Số lần chạy | Macro-F1 val từng seed | Macro-F1 val (TB ± ĐLC) | Khoảng cách loss val − train ở epoch 40 (TB ± ĐLC) | Loss val thấp nhất (TB ± ĐLC) |
|---|---|---|---|---|---|
| Có Dropout(0,5) | 3 | 0.940, 0.947, 0.935 | 0.941 ± 0.005 | 0.456 ± 0.061 | 0.248 ± 0.021 |
| Không Dropout | 3 | 0.951, 0.960, 0.947 | 0.953 ± 0.006 | 0.376 ± 0.081 | 0.202 ± 0.055 |
