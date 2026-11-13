### N5: Tăng cường ảnh (lật, xoay ±30°, zoom ±10%) lúc huấn luyện

| Cấu hình | Epoch chọn / tổng | Macro-F1 val | Accuracy train (epoch chọn) | Accuracy val (epoch chọn) | Loss train (epoch cuối) | Loss val (epoch cuối) | Khoảng cách loss val − train (epoch cuối) | Loss val thấp nhất |
|---|---|---|---|---|---|---|---|---|
| a_dropout, không tăng cường | 25 / 40 | 0.940 | 0.999 | 0.946 | 0.0022 | 0.5444 | 0.5422 | 0.2713 |
| a_dropout, có tăng cường | 25 / 40 | 0.948 | 0.967 | 0.954 | 0.2613 | 0.3150 | 0.0536 | 0.1286 |
| b_mobilenetv3 giai đoạn 2, không tăng cường | 4 / 9 | 0.997 | 0.999 | 0.997 | 0.0045 | 0.0107 | 0.0062 | 0.0107 |
| b_mobilenetv3 giai đoạn 2, có tăng cường | 12 / 17 | 0.993 | 0.997 | 0.995 | 0.0125 | 0.0183 | 0.0058 | 0.0183 |
