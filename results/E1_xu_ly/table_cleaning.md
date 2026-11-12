### X1 — Kiểm tra và làm sạch dữ liệu

| Kiểm tra | Kết quả |
|---|---|
| File không đọc được | 0 |
| Ảnh không phải 256×256 (được resize khi đọc) | 2 |
| File trùng tuyệt đối (md5) | 0 |
| Ảnh train gần trùng (pHash, Hamming ≤ 2) với ảnh val hoặc test → loại khỏi train | 15 |
| Bản lặp gần trùng trong nội bộ train → giữ 1 ảnh mỗi nhóm, loại | 20 |
| Số ảnh train sau khi làm sạch | 1694 / 1729 |
| Độ nét (phương sai Laplacian): phân vị 5% / trung vị / phân vị 95% | 56 / 153 / 280 |
| Ảnh có độ nét dưới 1/4 trung vị (nghi mờ; giữ lại, chỉ liệt kê) | 18 |
| Mask của bước S1 lớn hơn mask mới từ 1,5 lần (mask cũ tràn ra nền) | 129 / 2471 (5.2%) |
