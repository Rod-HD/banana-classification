### N3 — Dropout và khoảng cách train–val (seed 42, 40 epoch, train chấm ở chế độ suy luận)

| Cấu hình | Epoch chọn / tổng | Macro-F1 val | Accuracy train (epoch chọn) | Accuracy val (epoch chọn) | Loss train (epoch cuối) | Loss val (epoch cuối) | Khoảng cách loss val − train (epoch cuối) | Loss val thấp nhất |
|---|---|---|---|---|---|---|---|---|
| a_dropout có Dropout(0,5) | 25 / 40 | 0.940 | 0.999 | 0.946 | 0.0022 | 0.5444 | 0.5422 | 0.2713 |
| a_dropout không Dropout | 33 / 40 | 0.951 | 1.000 | 0.957 | 0.0000 | 0.3486 | 0.3486 | 0.1584 |
