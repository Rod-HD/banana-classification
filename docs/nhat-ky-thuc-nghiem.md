# Nhật ký thực nghiệm

Ghi lại từng bước của đồ án: đã làm gì, vì sao, đã cân nhắc phương án nào và bằng chứng nào dẫn tới lựa chọn. Quy tắc ghi nằm ở mục 7 của [kế hoạch thực nghiệm](ke-hoach-thuc-nghiem.md).

Nhãn dùng trong nhật ký:

- **[đo]**: có số liệu so sánh, kèm đường dẫn tới file kết quả.
- **[lập luận]**: dựa trên tính chất phương pháp hoặc giới hạn tài nguyên, chưa đo.

## Mẫu cho mỗi bước

```
## Bước <số> — <tên bước> (<ngày>)

**Mục tiêu.**
**Đã làm.** (file code, lệnh chạy)
**Lý do chọn cách này.**
**Phương án khác và vì sao không chọn.** (mỗi phương án gắn nhãn [đo] hoặc [lập luận])
**Bằng chứng.** (bảng số liệu, đường dẫn results/, hình)
**Kết quả và kết luận.**
**Ảnh hưởng tới bước sau.**
```

---

## Bước 0.1 — Chọn và chuẩn bị dữ liệu (12/11/2026)

**Mục tiêu.** Có bộ ảnh gốc của 4 giống chuối, kiểm tra được tính toàn vẹn, sắp xếp sẵn theo lớp.

**Đã làm.**
- Tải file `Banana Classification Dataset.zip` (12.162.003 byte) của bộ BananaImageBD, Version 2, từ Mendeley Data.
- Kiểm tra SHA-256 của file tải về, khớp với giá trị Mendeley công bố (`df5c7cea…abbf1f`).
- Giải nén, đổi tên thư mục lớp sang dạng `snake_case`, đặt vào `data/raw/`.
- Đếm ảnh từng lớp, đếm đuôi file, kiểm tra kích thước ảnh và tìm file trùng nhau tuyệt đối bằng md5.
- Viết `scripts/download_data.sh` để tải lại và kiểm tra đúng như trên.

**Lý do chọn cách này.** Dùng ảnh gốc từ nguồn phát hành chính thức để số lượng ảnh đối chiếu được với mô tả của tác giả, và giữ bằng chứng toàn vẹn (mã băm) để ai cũng tải lại được đúng bộ dữ liệu.

**Phương án khác và vì sao không chọn.**
- Bản "Augmented" có sẵn (7.413 ảnh): không dùng, vì ảnh tăng cường sinh từ cùng ảnh gốc sẽ nằm ở cả train lẫn test khi chia ngẫu nhiên. **[lập luận]**, sẽ kiểm chứng bằng thí nghiệm D1.
- Version 1 của bộ dữ liệu (2.512 ảnh gốc, Champa Kola 1.035): không dùng, vì Version 2 là bản tác giả cập nhật sau và bỏ bớt 41 ảnh Champa Kola. **[lập luận]**

**Bằng chứng.**

| Kiểm tra | Kết quả |
|---|---|
| Tổng số ảnh | 2.471 (khớp mô tả Version 2) |
| Bangla Kola / Champa Kola / Sabri Kola / Sagor Kola | 444 / 994 / 509 / 524 |
| Đuôi file | 2.378 `.jpg`, 93 `.JPG` |
| File trùng nhau tuyệt đối (md5) | 0 |
| File không đọc được | 0 |
| Chế độ màu | cả 2.471 ảnh là RGB |
| Kích thước ảnh | 2.469 ảnh 256×256; 2 ảnh khác: `bangla_kola/Bangla_Kola_55.jpg` (237×244), `champa_kola/Champa_Kola_324.jpg` (218×256) |

**Kết quả và kết luận.** Dữ liệu đầy đủ và khớp nguồn về số lượng. Mô tả của tác giả ghi ảnh 256×256 nhưng có 2 ảnh lệch kích thước. Lớp lớn nhất (Champa Kola) gấp khoảng 2,2 lần lớp nhỏ nhất (Bangla Kola), nên cần xử lý mất cân bằng và dùng macro-F1 làm chỉ số chính.

**Ảnh hưởng tới bước sau.** Code đọc ảnh phải nhận cả `.jpg` và `.JPG`, và luôn resize về kích thước chuẩn trước khi rút đặc trưng (không giả định ảnh nào cũng 256×256). Bước 1 cần tìm ảnh gần trùng, vì md5 chỉ bắt được file giống hệt nhau.

## Bước 0.2 — Chọn thư viện (12/11/2026)

**Mục tiêu.** Chốt công cụ lập trình cho cả phần truyền thống và phần CNN.

**Đã làm.** Lập `requirements.txt`: OpenCV, scikit-image, scikit-learn, imbalanced-learn cho phần truyền thống; TensorFlow/Keras (bản CPU) cho CNN. Đã kiểm tra các thư viện trong danh sách cài được cùng nhau trên Python 3.13.

**Lý do chọn cách này.** Các ví dụ trong bài giảng dùng OpenCV, scikit-learn và Keras, nên code của đồ án đối chiếu trực tiếp được với nội dung đã học.

**Phương án khác và vì sao không chọn.**
- PyTorch cho phần CNN: không chọn, vì ví dụ CNN và transfer learning trong bài giảng viết bằng Keras; dùng cùng thư viện thì cách khai báo lớp và cách đếm tham số khớp với bài học. **[lập luận]**

**Bằng chứng.** Chưa có số liệu; đây là lựa chọn công cụ.

**Kết quả và kết luận.** Dùng TensorFlow/Keras bản CPU.

**Ảnh hưởng tới bước sau.** Máy chỉ có CPU nên các mô hình đóng băng phải tính trước đặc trưng một lần rồi lưu lại.

## Bước 0.3 — Chốt quy trình đánh giá (12/11/2026)

**Mục tiêu.** Có một quy trình đánh giá dùng chung để mọi phương pháp so sánh được với nhau.

**Đã làm.** Viết mục 5 của kế hoạch thực nghiệm: chia stratified 70/15/15 cố định, tập test chỉ dùng một lần ở cuối, chọn siêu tham số bằng 5-fold cross-validation (nhóm truyền thống) hoặc tập val (CNN), chỉ số chính là macro-F1.

**Lý do chọn cách này.** Chia theo tỉ lệ lớp (stratified) giữ nguyên phân bố 4 lớp ở cả ba tập. Tách riêng tập test để con số cuối cùng không bị ảnh hưởng bởi việc chọn siêu tham số.

**Phương án khác và vì sao không chọn.**
- Chỉ chia train/test: không chọn, vì khi đó siêu tham số (ví dụ k của KNN) sẽ được chọn trên chính tập test và kết quả báo cáo lạc quan hơn thực tế. **[lập luận]**
- Cross-validation cho cả CNN: không chọn, vì huấn luyện lặp 5 lần mỗi cấu hình trên CPU tốn quá nhiều thời gian. **[lập luận]**
- Accuracy làm chỉ số chính: không chọn, vì dữ liệu mất cân bằng. **[lập luận]**, sẽ kiểm chứng bằng thí nghiệm I1.

**Bằng chứng.** Chưa có số liệu; các lập luận trên sẽ được kiểm chứng ở D1 và I1.

**Kết quả và kết luận.** Quy trình đã chốt ở mục 5 của kế hoạch.

**Ảnh hưởng tới bước sau.** Bước 1 tạo `data/splits/` theo đúng quy trình này.

## Bước 1 — Chia dữ liệu, ảnh gần trùng, thống kê và thí nghiệm D1 (12/11/2026)

**Mục tiêu.** Có ba tập train/val/test cố định; biết trong dữ liệu có bao nhiêu ảnh gần trùng nằm vắt qua các tập; kiểm chứng bằng số liệu lý do không dùng bản "Augmented" có sẵn (giả thuyết D1).

**Đã làm.**
- `src/config.py`, `src/data.py`: đường dẫn, tên lớp, seed; đọc ảnh nhận cả `.jpg` lẫn `.JPG`, luôn resize về 256×256; cache toàn bộ ảnh vào `data/cache/images_256.npy`.
- `python -m src.split_data`: chia stratified 70/15/15 với `seed=42`, ghi `data/splits/{train,val,test}.csv`; tính perceptual hash (pHash 64 bit) cho 2.471 ảnh, đếm số cặp ảnh có khoảng cách Hamming nhỏ theo từng ngưỡng; vẽ thống kê.
- `python -m src.env_info`: ghi phiên bản thư viện vào `results/environment.txt`.
- `src/features/` (màu, cạnh, HOG tự cài), `src/transforms.py`, `src/classical.py` (5-fold CV trên train+val), `src/metrics.py`: khung dùng chung cho các bước sau. HOG tự cài (để có phiên bản bỏ chuẩn hóa block cho C2) được đối chiếu với `skimage.feature.hog`: `python -m src.features.hog` cho sai khác tuyệt đối lớn nhất 2,2·10⁻⁷ trên 20 ảnh, cùng 3.780 chiều.
- `python -m src.exp_d1`: thí nghiệm D1 và phần phụ D1b.

**Lý do chọn cách này.**
- D1 chạy hoàn toàn trong tập phát triển (train+val, 2.100 ảnh), tập "test" của D1 là 20% tách ra từ tập phát triển, lặp 5 lần với 5 seed. Làm vậy vì mục 5 của kế hoạch quy định tập test chính thức chỉ dùng một lần ở cuối. **[lập luận]**
- Cùng lý do, mọi thí nghiệm "biến đổi ảnh test" ở các bước sau (M1, M2, M6, C1, C2, C4) được chấm trên fold giữ lại của cross-validation (nhóm truyền thống) hoặc trên tập val (CNN), không phải tập test chính thức. **[lập luận]**
- D1 thêm cách (c) ngoài hai cách (a), (b) của kế hoạch: chia theo ảnh gốc rồi tăng cường cả hai phía. Tập chấm của (a) có cả ảnh lật/xoay còn của (b) chỉ có ảnh gốc, nên (a)−(b) lẫn hai hiệu ứng: rò rỉ và độ khó của tập chấm. (c) có cùng thành phần tập chấm với (a) nên (a)−(c) là phần thổi phồng thuần do rò rỉ. **[đo]** `results/D1/table.md`: với HOG + Logistic Regression, (b) cao hơn (a) 0,008 dù (a) có rò rỉ, vì tập chấm của (b) dễ hơn; (a)−(c) = +0,000.
- Bản tăng cường tự tạo: mỗi ảnh gốc thêm 1 bản lật ngang và 1 bản xoay góc ngẫu nhiên trong ±[10°, 30°], tức gấp 3 số ảnh, cùng tỉ lệ với bản "Augmented" của tác giả (7.413 = 3 × 2.471). **[lập luận]**

**Phương án khác và vì sao không chọn.**
- Ngưỡng pHash để gọi là "gần trùng": xem hình các cặp theo từng mức Hamming (`results/figures/near_duplicates_by_distance.png`). Hamming ≤ 2: cùng một quả chụp lại gần như y hệt. Hamming = 4 trở lên: phần lớn là hai quả khác nhau có bố cục giống nhau (quả nhỏ nằm giữa nền trơn, nhiều nhất ở Sabri Kola). Chọn ≤ 2 làm ngưỡng "gần trùng chắc chắn", báo cáo thêm ≤ 4 làm cận trên. **[đo]** bảng dưới.
- Chia lại dữ liệu theo nhóm ảnh gần trùng (group split) thay cho chia stratified ngẫu nhiên: không chọn, vì đo thấy ảnh gần trùng không làm đổi kết quả. **[đo]** `results/D1/table_near_duplicates.md`: chênh macro-F1 giữa StratifiedKFold và StratifiedGroupKFold nằm trong ±0,007, nhỏ hơn độ lệch chuẩn giữa các fold ở cả 5 mô hình.
- Dùng tập test chính thức cho D1 như câu chữ của kế hoạch ("độ chính xác trên test"): không chọn, lý do ở trên. **[lập luận]**
- Chỉ dùng một mô hình cho D1: không chọn; chạy 5 mô hình (mỗi mô hình vài giây) vì mức rò rỉ phụ thuộc đặc trưng có bất biến với phép lật/xoay hay không. **[đo]** bảng D1.

**Bằng chứng.**

Số ảnh mỗi tập (`results/data/data_stats.json`, hình `results/figures/data_split_counts.png`, ảnh mẫu `results/figures/data_samples.png`):

| Tập | Bangla Kola | Champa Kola | Sabri Kola | Sagor Kola | Tổng |
|---|---|---|---|---|---|
| train | 311 | 695 | 356 | 367 | 1.729 |
| val | 66 | 150 | 76 | 79 | 371 |
| test | 67 | 149 | 77 | 78 | 371 |

Ảnh gần trùng theo pHash (`results/data/near_duplicates.json`; hình `results/figures/near_duplicates_closest_pairs.png`):

| Ngưỡng Hamming | Số cặp | Cặp khác lớp | Cặp train–test | Cặp train–val | Cặp val–test | Ảnh test có "bản gần trùng" trong train |
|---|---|---|---|---|---|---|
| = 0 | 3 | 0 | 0 | 1 | 0 | 0 |
| ≤ 2 | 42 | 1 | 7 | 8 | 2 | 7 / 371 |
| ≤ 4 | 383 | 10 | 83 | 86 | 15 | 44 / 371 |
| ≤ 6 | 1.527 | 56 | 336 | 355 | 62 | 103 / 371 |

D1 — Accuracy trung bình ± độ lệch chuẩn qua 5 lần chia 80/20 trong tập phát triển (`results/D1/table.md`, `results/D1/metrics.json`, hình `results/figures/D1_leakage.png`):

| Mô hình | (a) tăng cường rồi chia | (b) chia rồi tăng cường train | (c) chia theo ảnh gốc, tăng cường 2 phía | (a)−(b) | (a)−(c) |
|---|---|---|---|---|---|
| HSV hist + KNN (k=1) | 0,999 ± 0,001 | 0,978 ± 0,010 | 0,978 ± 0,008 | +0,022 ± 0,010 | +0,021 ± 0,009 |
| HSV hist + LogReg | 0,999 ± 0,001 | 0,991 ± 0,002 | 0,990 ± 0,002 | +0,008 ± 0,002 | +0,009 ± 0,002 |
| HOG + KNN (k=1) | 0,974 ± 0,003 | 0,974 ± 0,003 | 0,970 ± 0,002 | +0,000 ± 0,003 | +0,005 ± 0,004 |
| HOG + LogReg | 0,973 ± 0,006 | 0,981 ± 0,003 | 0,973 ± 0,003 | −0,008 ± 0,004 | +0,000 ± 0,004 |
| Edge + KNN (k=1) | 0,896 ± 0,012 | 0,889 ± 0,014 | 0,862 ± 0,012 | +0,007 ± 0,020 | +0,034 ± 0,020 |

D1b — macro-F1 5-fold, chia thường so với chia giữ nhóm ảnh gần trùng (`results/D1/table_near_duplicates.md`):

| Mô hình | StratifiedKFold | GroupKFold (≤ 2) | GroupKFold (≤ 4) |
|---|---|---|---|
| HSV hist + KNN (k=1) | 0,975 ± 0,005 | 0,975 ± 0,003 | 0,973 ± 0,008 |
| HSV hist + LogReg | 0,991 ± 0,005 | 0,991 ± 0,003 | 0,993 ± 0,003 |
| HOG + KNN (k=1) | 0,967 ± 0,011 | 0,968 ± 0,004 | 0,970 ± 0,005 |
| HOG + LogReg | 0,985 ± 0,004 | 0,986 ± 0,006 | 0,986 ± 0,005 |
| Edge + KNN (k=1) | 0,902 ± 0,019 | 0,902 ± 0,015 | 0,909 ± 0,007 |

**Kết quả và kết luận.**
- Chia xong: 1.729 / 371 / 371 ảnh, tỉ lệ lớp giữ nguyên ở cả ba tập.
- Ảnh gần trùng: có. Ở ngưỡng chắc chắn (Hamming ≤ 2) có 42 cặp, 7 ảnh test có bản gần trùng trong train. Ảnh hưởng tới kết quả không đo được (D1b). pHash không bắt được trường hợp cùng một quả chụp ở tư thế khác, nên con số này là cận dưới; ghi vào phần hạn chế.
- D1 khớp dự đoán với đặc trưng màu: histogram HSV gần như không đổi khi lật/xoay ảnh, nên bản tăng cường của ảnh test nằm trong train là "đáp án có sẵn"; accuracy bị thổi phồng +0,021 (KNN k=1) và +0,009 (LogReg), lớn hơn độ lệch chuẩn. Cả hai đều lên 0,999.
- D1 **trái dự đoán** với HOG: mức thổi phồng (a)−(c) chỉ +0,005 ± 0,004 (KNN) và +0,000 ± 0,004 (LogReg), chưa đủ cơ sở phân biệt với 0. Giải thích: HOG mã hóa hướng cạnh theo vị trí, nên bản lật hay xoay của một ảnh cho vector khác hẳn ảnh gốc; bản sao trong train không còn là láng giềng gần của ảnh test. Rò rỉ chỉ gây hại khi đặc trưng bất biến với phép tăng cường. Với CNN có tăng cường ảnh (học bất biến lật/xoay) thì dự kiến bị ảnh hưởng như histogram; chưa đo. **[lập luận]**
- Với đặc trưng cạnh (vector tổng theo hàng/cột): (a)−(c) = +0,034 ± 0,020, lớn hơn độ lệch chuẩn.

**Ảnh hưởng tới bước sau.**
- Giữ quyết định dùng ảnh gốc và chỉ tăng cường trên phần huấn luyện.
- Số liệu sơ bộ cho thấy bài toán dễ với đặc trưng màu (HSV hist + LogReg đạt macro-F1 0,991 trên 5-fold). Cần kiểm tra ở S2 xem mô hình có dựa vào màu nền không.
- Độ lệch chuẩn giữa các fold khoảng 0,004–0,02; các so sánh ở bước sau lấy đó làm ngưỡng "đủ cơ sở phân biệt".

---

## Bảng quyết định

| Quyết định | Phương án thay thế | Bằng chứng | Nhãn |
|---|---|---|---|
| Dùng ảnh gốc Version 2 | Bản Augmented; Version 1 | Bước 0.1; chờ D1 | [lập luận] |
| TensorFlow/Keras cho CNN | PyTorch | Bước 0.2 | [lập luận] |
| Chia 70/15/15, test dùng một lần | Chỉ train/test; cross-validation cho CNN | Bước 0.3 | [lập luận] |
| macro-F1 là chỉ số chính | Accuracy | Bước 0.3; chờ I1 | [lập luận] |
| Không dùng bản Augmented; chỉ tăng cường phần huấn luyện | Tăng cường rồi chia | Bước 1, `results/D1/table.md`: thổi phồng +0,021 (HSV + KNN), +0,009 (HSV + LogReg); ≈ 0 với HOG | [đo] |
| D1 và các phép biến đổi kiểm chứng chấm trên fold giữ lại / val, không dùng test | Dùng tập test chính thức | Bước 1 | [lập luận] |
| Giữ cách chia stratified ngẫu nhiên | Chia theo nhóm ảnh gần trùng | Bước 1, `results/D1/table_near_duplicates.md`: chênh ≤ 0,007, nhỏ hơn độ lệch chuẩn | [đo] |
| Ngưỡng gần trùng pHash: Hamming ≤ 2 | ≤ 4, ≤ 6 | Bước 1, `results/figures/near_duplicates_by_distance.png` | [đo] |
| HOG tự cài đặt (có tùy chọn bỏ chuẩn hóa block) | `skimage.feature.hog` | Bước 1: sai khác lớn nhất 2,2·10⁻⁷ so với skimage | [đo] |
