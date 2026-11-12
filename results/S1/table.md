### S1 — Chỉ số không cần nhãn mask trên 400 ảnh (100 ảnh mỗi lớp)

| Phương pháp | Số mảnh rời (trung bình) | Số mảnh rời (trung vị) | Tỉ lệ lỗ thủng trong vật | Tỉ lệ diện tích vật | IoU với phương pháp được chọn |
|---|---|---|---|---|---|
| Ngưỡng toàn cục (T=127) | 11.6 | 9 | 0.085 | 0.105 | 0.689 |
| Otsu (ảnh xám) | 2.9 | 2 | 0.052 | 0.149 | 0.799 |
| Ngưỡng thích nghi | 16.9 | 14 | 0.282 | 0.081 | 0.561 |
| K-means màu (k=2) | 2.5 | 1 | 0.027 | 0.140 | 0.859 |
| Otsu trên khoảng cách màu tới nền | 2.5 | 1 | 0.010 | 0.145 | 0.971 |
| Otsu khoảng cách màu + hậu xử lý (dùng cho S2) | 1.0 | 1 | 0.000 | 0.144 | 1.000 |
