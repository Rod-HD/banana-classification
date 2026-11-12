### So sánh các nhánh qua cấu hình đại diện (chọn theo điểm lúc phát triển, không theo test)

| Nhánh | Cấu hình đại diện | Điểm lúc chọn | Macro-F1 test | KTC 95% | Số ảnh sai / 371 | Điểm bền |
|---|---|---|---|---|---|---|
| G — pipeline gốc | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 1.000 | [1.000; 1.000] | 0 | 0.875 |
| T — Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 0.998 | [0.991; 1.000] | 1 | 0.909 |
