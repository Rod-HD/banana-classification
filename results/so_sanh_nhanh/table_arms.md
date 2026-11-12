### So sánh các nhánh qua cấu hình đại diện (chọn theo điểm lúc phát triển, không theo test)

| Nhánh | Cấu hình đại diện | Điểm lúc chọn | Macro-F1 test | KTC 95% | Số ảnh sai / 371 | Điểm bền |
|---|---|---|---|---|---|---|
| G — pipeline gốc | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 1.000 | [1.000; 1.000] | 0 | 0.875 |
| X-a — xử lý dữ liệu: tiền xử lý lúc suy luận | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 0.995 | [0.987; 1.000] | 2 | 0.881 |
| X-b — xử lý dữ liệu: tăng cường có thay nền | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 1.000 | [1.000; 1.000] | 0 | 0.883 |
| T — Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 0.998 | [0.991; 1.000] | 1 | 0.909 |
| B1 — bài báo, trong phạm vi môn học (CNN + bộ phân lớp) | MobileNetV3-Small + SVM tuyến tính | 0.996 (5-fold trên train+val) | 0.993 | [0.982; 1.000] | 2 | 0.836 |
| B2 — bài báo, đúng phương pháp (ViT + SVM) | ViT-B/16 + SVM | 0.994 (5-fold trên train+val) | 0.997 | [0.989; 1.000] | 1 | 0.919 |
