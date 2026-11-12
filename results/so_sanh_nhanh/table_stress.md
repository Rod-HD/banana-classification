### Cấu hình đại diện của mỗi nhánh trên từng phép thử của bộ kiểm tra khó (macro-F1, 371 ảnh test)

| Nhánh | Cấu hình đại diện | Ảnh gốc | Nền hoán đổi | V×0,6 | V×1,4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|
| G — pipeline gốc | b_mobilenetv3 | 1.000 | 0.794 | 0.919 | 0.890 | 0.989 | 1.000 | 0.659 | 0.875 |
| X-a — xử lý dữ liệu: tiền xử lý lúc suy luận | b_mobilenetv3 | 0.995 | 0.995 | 0.943 | 0.915 | 0.988 | 0.998 | 0.450 | 0.881 |
| X-b — xử lý dữ liệu: tăng cường có thay nền | b_mobilenetv3 | 1.000 | 0.975 | 0.954 | 0.935 | 0.995 | 1.000 | 0.441 | 0.883 |
| T — Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 0.998 | 0.867 | 0.954 | 0.904 | 0.993 | 1.000 | 0.736 | 0.909 |
| B1 — bài báo, trong phạm vi môn học (CNN + bộ phân lớp) | MobileNetV3-Small + SVM tuyến tính | 0.993 | 0.742 | 0.965 | 0.856 | 0.897 | 0.981 | 0.575 | 0.836 |
| B2 — bài báo, đúng phương pháp (ViT + SVM) | ViT-B/16 + SVM | 0.997 | 0.857 | 0.990 | 0.978 | 0.986 | 0.978 | 0.728 | 0.919 |
