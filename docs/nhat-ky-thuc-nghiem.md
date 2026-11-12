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

## Bước 2 — Đặc trưng màu, KNN và bộ phân lớp tuyến tính: M1–M6, P1–P6 (12/11/2026)

**Mục tiêu.** Kiểm chứng các giả thuyết về đặc trưng màu (mục 6.2) và về bộ phân lớp (mục 6.4) bằng 5-fold stratified cross-validation trên train+val (2.100 ảnh).

**Đã làm.**
- `src/features/color.py`: histogram RGB từng kênh, histogram HSV 3 chiều, histogram H-S, moment màu (9 chiều), màu chủ đạo k-means (k=3, 12 chiều), CCV và histogram cùng mức lượng tử hóa (64 màu).
- `src/transforms.py`: xáo trộn lưới 4×4, nhân kênh V, chuyển xám, nhiễu Gauss, che 25%, xoay. Mỗi ảnh có bộ sinh số ngẫu nhiên riêng theo (seed, tên phép, chỉ số ảnh).
- `python -m src.exp_color` (M1–M6) và `python -m src.exp_classifier` (P1–P6). Mỗi thí nghiệm ghi `results/<mã>/metrics.json`, `results/<mã>/table.md` và hình `results/figures/<mã>_*.png`.
- `scripts/run_all.sh`: chạy lại toàn bộ các bước theo thứ tự (`bash scripts/run_all.sh`, hoặc `bash scripts/run_all.sh 2` cho riêng bước này).

**Lý do chọn cách này.**
- Mô hình luôn học trên ảnh sạch; phép biến đổi chỉ áp lên fold giữ lại. Độ sụt tính theo từng fold rồi lấy trung bình ± độ lệch chuẩn, nên so được với nhiễu giữa các fold. **[lập luận]**
- Mỗi đặc trưng chấm bằng cả KNN (k=5, Euclidean, z-score) và Logistic Regression (C=1, z-score) với tham số mặc định cố định, để kết luận về đặc trưng không phụ thuộc một bộ phân lớp. Việc tinh chỉnh siêu tham số làm riêng ở P1–P3 và ở bước chốt cấu hình. **[đo]** `results/M4/table.md`: thứ hạng đặc trưng đổi theo bộ phân lớp (moment màu đứng thứ hai với KNN nhưng gần cuối với LogReg).
- M6 chạy hai chế độ: "học màu, chấm xám" (đo độ bền) và "học xám, chấm xám" (đo lượng thông tin còn lại khi không có màu). Chế độ đầu của kế hoạch chỉ cho biết histogram màu sập khi phân bố đầu vào đổi; chế độ sau mới trả lời câu hỏi "giống chuối khác nhau ở màu hay hình dạng". **[đo]** `results/M6/table.md`.
- M3 quét histogram RGB từng kênh đủ 6 mức bin của kế hoạch; với histogram HSV 3 chiều chỉ quét 2, 4, 8, 16 bin mỗi kênh vì 256 bin mỗi kênh cho 16,7 triệu chiều. **[lập luận]**
- P4 thêm phần phụ: giữ đặc trưng 9 chiều, nhân tập tham chiếu lên 5, 20, 60 lần (bản sao cộng nhiễu nhỏ, chỉ để đo thời gian) vì kết quả chính trái dự đoán. **[đo]** `results/P4/table_scaling.md`.

**Phương án khác và vì sao không chọn.**
- Cài moment màu bằng phép tính float64 trên toàn ảnh: bản đầu mất 17,9 ms/ảnh; viết lại bằng float32 còn khoảng 4 ms/ảnh, giá trị lệch tối đa 8,7·10⁻⁷. Dùng bản sau. **[đo]** `results/M4/table.md` (cột thời gian).
- Dùng histogram HSV đủ 3 kênh làm đặc trưng "bền với độ sáng": không chọn cho mục đích đó. **[đo]** `results/M2/table.md`: khi nhân V với 0,6, histogram HSV (8,8,8) sụt 0,66–0,67 macro-F1, gần bằng histogram RGB; chỉ histogram H-S giữ được (sụt 0,000–0,027).
- Quy tắc k = số lớp + 1 = 5: không tốt bằng k chọn bằng cross-validation. **[đo]** `results/P1/table.md`: k=1 tốt nhất ở cả ba đặc trưng; chênh +0,006 (moment, nhỏ hơn độ lệch chuẩn), +0,011 (hist HSV, nhỏ hơn độ lệch chuẩn), +0,020 (HOG, lớn hơn độ lệch chuẩn).
- K-D Tree / Ball Tree thay cho Brute Force: không chọn với cỡ dữ liệu này. **[đo]** `results/P4/table.md`: với 1.680 ảnh tham chiếu, brute nhanh nhất ở mọi số chiều.
- Hellinger cho đặc trưng histogram: 0,976 ± 0,015 so với Euclidean + z-score 0,965 ± 0,014 trên hist HSV; chênh nhỏ hơn độ lệch chuẩn, chưa đủ cơ sở phân biệt. **[đo]** `results/P2/table.md`.

**Bằng chứng.** (macro-F1 trung bình ± độ lệch chuẩn qua 5 fold)

M1 — xáo trộn lưới 4×4 (`results/M1/table.md`, hình `results/figures/M1_shuffle.png`):

| Đặc trưng | Bộ phân lớp | Macro-F1 ảnh gốc | Macro-F1 ảnh xáo trộn 4×4 | Độ sụt |
|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0,964 ± 0,012 | 0,964 ± 0,012 | +0,000 ± 0,000 |
| Hist RGB (3×32) | LogReg | 0,976 ± 0,013 | 0,976 ± 0,013 | +0,000 ± 0,000 |
| Hist HSV (8,8,8) | KNN (k=5) | 0,965 ± 0,014 | 0,965 ± 0,014 | +0,000 ± 0,000 |
| Hist HSV (8,8,8) | LogReg | 0,991 ± 0,005 | 0,991 ± 0,005 | +0,000 ± 0,000 |
| Moment màu | KNN (k=5) | 0,972 ± 0,008 | 0,972 ± 0,008 | +0,000 ± 0,000 |
| Moment màu | LogReg | 0,908 ± 0,009 | 0,908 ± 0,009 | +0,000 ± 0,000 |
| CCV (64 màu) | KNN (k=5) | 0,946 ± 0,011 | 0,864 ± 0,018 | +0,082 ± 0,015 |
| CCV (64 màu) | LogReg | 0,975 ± 0,006 | 0,922 ± 0,011 | +0,053 ± 0,009 |
| Cạnh Sobel | KNN (k=5) | 0,890 ± 0,019 | 0,231 ± 0,021 | +0,658 ± 0,028 |
| Cạnh Sobel | LogReg | 0,894 ± 0,008 | 0,201 ± 0,040 | +0,692 ± 0,042 |
| HOG 64×128 | KNN (k=5) | 0,946 ± 0,007 | 0,260 ± 0,017 | +0,687 ± 0,011 |
| HOG 64×128 | LogReg | 0,985 ± 0,004 | 0,287 ± 0,020 | +0,699 ± 0,024 |

M2 — nhân kênh V (`results/M2/table.md`, hình `results/figures/M2_brightness.png`):

| Đặc trưng | Bộ phân lớp | Ảnh gốc | V×0,6 | Sụt (V×0,6) | V×1,4 | Sụt (V×1,4) |
|---|---|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0,964 ± 0,012 | 0,092 ± 0,002 | +0,873 ± 0,013 | 0,205 ± 0,019 | +0,760 ± 0,024 |
| Hist RGB (3×32) | LogReg | 0,976 ± 0,013 | 0,212 ± 0,054 | +0,763 ± 0,064 | 0,279 ± 0,038 | +0,696 ± 0,036 |
| Hist HSV (8,8,8) | KNN (k=5) | 0,965 ± 0,014 | 0,302 ± 0,042 | +0,663 ± 0,043 | 0,589 ± 0,020 | +0,376 ± 0,025 |
| Hist HSV (8,8,8) | LogReg | 0,991 ± 0,005 | 0,318 ± 0,035 | +0,673 ± 0,035 | 0,662 ± 0,004 | +0,329 ± 0,009 |
| Hist H-S (8,8) | KNN (k=5) | 0,976 ± 0,006 | 0,975 ± 0,009 | +0,000 ± 0,008 | 0,975 ± 0,006 | +0,001 ± 0,002 |
| Hist H-S (8,8) | LogReg | 0,985 ± 0,008 | 0,959 ± 0,014 | +0,027 ± 0,011 | 0,974 ± 0,007 | +0,011 ± 0,004 |

M3 — số bin (`results/M3/table.md`, hình `results/figures/M3_bins.png`):

| Đặc trưng | Số bin mỗi kênh | Số chiều | KNN (k=5) | LogReg | Tỉ lệ phần tử bằng 0 |
|---|---|---|---|---|---|
| Hist RGB từng kênh | 4 | 12 | 0,936 ± 0,015 | 0,914 ± 0,015 | 0,00 |
| Hist RGB từng kênh | 8 | 24 | 0,966 ± 0,010 | 0,968 ± 0,007 | 0,03 |
| Hist RGB từng kênh | 16 | 48 | 0,969 ± 0,011 | 0,979 ± 0,012 | 0,06 |
| Hist RGB từng kênh | 32 | 96 | 0,964 ± 0,012 | 0,976 ± 0,013 | 0,08 |
| Hist RGB từng kênh | 64 | 192 | 0,963 ± 0,015 | 0,977 ± 0,012 | 0,09 |
| Hist RGB từng kênh | 256 | 768 | 0,962 ± 0,015 | 0,975 ± 0,011 | 0,12 |
| Hist HSV 3 chiều | 2 | 8 | 0,932 ± 0,011 | 0,885 ± 0,012 | 0,20 |
| Hist HSV 3 chiều | 4 | 64 | 0,963 ± 0,009 | 0,965 ± 0,012 | 0,55 |
| Hist HSV 3 chiều | 8 | 512 | 0,965 ± 0,014 | 0,991 ± 0,005 | 0,72 |
| Hist HSV 3 chiều | 16 | 4096 | 0,917 ± 0,015 | 0,992 ± 0,004 | 0,87 |

M4 — các đặc trưng màu (`results/M4/table.md`, hình `results/figures/M4_color_features.png`):

| Đặc trưng | Số chiều | Rút đặc trưng (ms/ảnh) | KNN (k=5) | Huấn luyện KNN (k=5) (s) | Dự đoán KNN (k=5) (ms/ảnh) | LogReg | Huấn luyện LogReg (s) | Dự đoán LogReg (ms/ảnh) | Linear SVM |
|---|---|---|---|---|---|---|---|---|---|
| Hist RGB (3×32) | 96 | 0,16 | 0,964 ± 0,012 | 0,003 | 0,009 | 0,976 ± 0,013 | 0,037 | 0,002 | 0,960 ± 0,012 |
| Hist HSV (8,8,8) | 512 | 0,21 | 0,965 ± 0,014 | 0,008 | 0,025 | 0,991 ± 0,005 | 0,735 | 0,008 | 0,989 ± 0,008 |
| Hist H-S (8,8) | 64 | 0,21 | 0,976 ± 0,006 | 0,002 | 0,009 | 0,985 ± 0,008 | 0,016 | 0,001 | 0,958 ± 0,007 |
| Moment màu | 9 | 4,25 | 0,972 ± 0,008 | 0,001 | 0,007 | 0,908 ± 0,009 | 0,014 | 0,001 | 0,838 ± 0,011 |
| Màu chủ đạo (k=3) | 12 | 1,58 | 0,945 ± 0,005 | 0,002 | 0,006 | 0,899 ± 0,009 | 0,017 | 0,001 | 0,826 ± 0,013 |
| CCV (64 màu) | 128 | 2,03 | 0,946 ± 0,011 | 0,003 | 0,013 | 0,975 ± 0,006 | 0,017 | 0,001 | 0,965 ± 0,014 |
| Hist 64 màu | 64 | 0,53 | 0,966 ± 0,010 | 0,002 | 0,008 | 0,968 ± 0,008 | 0,014 | 0,001 | 0,945 ± 0,020 |

M5 — CCV (`results/M5/table.md`, hình `results/figures/M5_ccv.png`):

| Đặc trưng | Số chiều | Bộ phân lớp | Ảnh gốc | Xáo trộn 4×4 | Độ sụt |
|---|---|---|---|---|---|
| Hist 64 màu | 64 | KNN (k=5) | 0,966 ± 0,010 | 0,962 ± 0,008 | +0,003 ± 0,004 |
| Hist 64 màu | 64 | LogReg | 0,968 ± 0,008 | 0,966 ± 0,009 | +0,002 ± 0,004 |
| CCV (64 màu) | 128 | KNN (k=5) | 0,946 ± 0,011 | 0,864 ± 0,018 | +0,082 ± 0,015 |
| CCV (64 màu) | 128 | LogReg | 0,975 ± 0,006 | 0,922 ± 0,011 | +0,053 ± 0,009 |

M6 — ảnh xám (`results/M6/table.md`, hình `results/figures/M6_grayscale.png`):

| Đặc trưng | Bộ phân lớp | Học màu, chấm màu | Học màu, chấm xám | Học xám, chấm xám | Sụt khi bỏ màu (học lại) |
|---|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0,964 ± 0,012 | 0,631 ± 0,026 | 0,927 ± 0,015 | +0,038 |
| Hist RGB (3×32) | LogReg | 0,976 ± 0,013 | 0,456 ± 0,030 | 0,865 ± 0,012 | +0,111 |
| Hist HSV (8,8,8) | KNN (k=5) | 0,965 ± 0,014 | 0,121 ± 0,021 | 0,912 ± 0,020 | +0,052 |
| Hist HSV (8,8,8) | LogReg | 0,991 ± 0,005 | 0,154 ± 0,022 | 0,804 ± 0,010 | +0,187 |
| Moment màu | KNN (k=5) | 0,972 ± 0,008 | 0,120 ± 0,041 | 0,748 ± 0,038 | +0,223 |
| Moment màu | LogReg | 0,908 ± 0,009 | 0,320 ± 0,063 | 0,676 ± 0,018 | +0,232 |
| CCV (64 màu) | KNN (k=5) | 0,946 ± 0,011 | 0,403 ± 0,014 | 0,782 ± 0,028 | +0,164 |
| CCV (64 màu) | LogReg | 0,975 ± 0,006 | 0,292 ± 0,035 | 0,762 ± 0,020 | +0,213 |
| Cạnh Sobel | KNN (k=5) | 0,890 ± 0,019 | 0,890 ± 0,019 | 0,890 ± 0,019 | +0,000 |
| Cạnh Sobel | LogReg | 0,894 ± 0,008 | 0,894 ± 0,008 | 0,894 ± 0,008 | +0,000 |
| HOG 64×128 | KNN (k=5) | 0,946 ± 0,007 | 0,946 ± 0,012 | 0,945 ± 0,012 | +0,001 |
| HOG 64×128 | LogReg | 0,985 ± 0,004 | 0,987 ± 0,006 | 0,986 ± 0,005 | -0,001 |
| HOG 64×128 (màu, kênh gradient lớn nhất) | KNN (k=5) | 0,944 ± 0,013 | 0,943 ± 0,012 | 0,945 ± 0,012 | -0,001 |
| HOG 64×128 (màu, kênh gradient lớn nhất) | LogReg | 0,982 ± 0,005 | 0,984 ± 0,005 | 0,986 ± 0,005 | -0,005 |

P1 — chọn k (`results/P1/table.md`, đủ 16 giá trị k ở `results/P1/table_all_k.md`, hình `results/figures/P1_knn_k.png`):

| Đặc trưng | k = số lớp + 1 = 5 | k tốt nhất theo CV | Macro-F1 tại k tốt nhất | Chênh so với k=5 | k=1 | k=31 |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0,972 ± 0,008 | 1 | 0,977 ± 0,003 | +0,006 | 0,977 ± 0,003 | 0,937 ± 0,012 |
| Hist HSV (512 chiều) | 0,965 ± 0,014 | 1 | 0,975 ± 0,005 | +0,011 | 0,975 ± 0,005 | 0,867 ± 0,026 |
| HOG 64×128 (3780 chiều) | 0,946 ± 0,007 | 1 | 0,967 ± 0,011 | +0,020 | 0,967 ± 0,011 | 0,871 ± 0,012 |

P2 — độ đo và chuẩn hóa (`results/P2/table.md`, `results/P2/table_rank_agreement.md`, hình `results/figures/P2_metrics.png`):

| Độ đo | Chuẩn hóa | Moment màu (9 chiều) | Hist HSV (512 chiều) | HOG 64×128 (3780 chiều) |
|---|---|---|---|---|
| Euclidean | không | 0,977 ± 0,005 | 0,953 ± 0,006 | 0,954 ± 0,008 |
| Euclidean | vector đơn vị (L2) | 0,970 ± 0,005 | 0,936 ± 0,011 | 0,954 ± 0,008 |
| Euclidean | z-score | 0,972 ± 0,008 | 0,965 ± 0,014 | 0,946 ± 0,007 |
| Manhattan | không | 0,977 ± 0,004 | 0,964 ± 0,009 | 0,978 ± 0,007 |
| Manhattan | vector đơn vị (L2) | 0,967 ± 0,006 | 0,956 ± 0,013 | 0,978 ± 0,007 |
| Manhattan | z-score | 0,974 ± 0,008 | 0,961 ± 0,009 | 0,975 ± 0,010 |
| Cosine | không | 0,970 ± 0,005 | 0,936 ± 0,011 | 0,954 ± 0,008 |
| Cosine | vector đơn vị (L2) | 0,970 ± 0,005 | 0,936 ± 0,011 | 0,954 ± 0,008 |
| Hellinger | L1 rồi lấy căn | không áp dụng (có giá trị âm) | 0,976 ± 0,015 | 0,957 ± 0,011 |

| Đặc trưng | Đã chuẩn hóa đơn vị: cùng 10 láng giềng, cùng thứ tự | Đã chuẩn hóa đơn vị: cùng tập 10 láng giềng | Chưa chuẩn hóa: cùng tập 10 láng giềng |
|---|---|---|---|
| Moment màu (9 chiều) | 99,5% | 100,0% | 11,2% |
| Hist HSV (512 chiều) | 100,0% | 100,0% | 18,1% |
| HOG 64×128 (3780 chiều) | 100,0% | 100,0% | 99,3% |

P3 — bình chọn đa số so với ưu tiên điểm gần (`results/P3/table.md`, hình `results/figures/P3_weights.png`):

| Đặc trưng | k | Macro-F1 uniform | Macro-F1 distance | Recall Bangla Kola (lớp ít ảnh nhất) uniform | Recall Bangla Kola distance | Recall thấp nhất trong 4 lớp uniform → distance |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 5 | 0,972 ± 0,008 | 0,972 ± 0,008 | 0,984 ± 0,013 | 0,984 ± 0,013 | 0,922 → 0,924 |
| Moment màu (9 chiều) | 15 | 0,955 ± 0,008 | 0,961 ± 0,006 | 0,942 ± 0,026 | 0,968 ± 0,018 | 0,888 → 0,888 |
| Hist HSV (512 chiều) | 5 | 0,965 ± 0,014 | 0,966 ± 0,013 | 0,981 ± 0,007 | 0,981 ± 0,007 | 0,879 → 0,888 |
| Hist HSV (512 chiều) | 15 | 0,931 ± 0,013 | 0,933 ± 0,014 | 0,952 ± 0,018 | 0,955 ± 0,022 | 0,756 → 0,765 |
| HOG 64×128 (3780 chiều) | 5 | 0,946 ± 0,007 | 0,949 ± 0,008 | 0,955 ± 0,021 | 0,952 ± 0,026 | 0,816 → 0,830 |
| HOG 64×128 (3780 chiều) | 15 | 0,912 ± 0,011 | 0,914 ± 0,010 | 0,920 ± 0,033 | 0,923 ± 0,036 | 0,697 → 0,706 |

P4 — thuật toán tìm láng giềng (`results/P4/table.md`, `results/P4/table_scaling.md`, hình `results/figures/P4_knn_algorithms.png`):

| Đặc trưng | brute: dựng (ms) | brute: dự đoán 420 ảnh (ms) | kd_tree: dựng (ms) | kd_tree: dự đoán 420 ảnh (ms) | kd_tree: kết quả trùng brute | ball_tree: dựng (ms) | ball_tree: dự đoán 420 ảnh (ms) | ball_tree: kết quả trùng brute |
|---|---|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0,6 | 1,6 | 1,4 | 3,2 | có | 1,1 | 4,5 | có |
| Hist HSV (512 chiều) | 1,3 | 9,1 | 38,7 | 361,1 | có | 29,3 | 244,3 | có |
| HOG 64×128 (3780 chiều) | 8,1 | 73,9 | 367,7 | 4533,7 | có | 223,5 | 2231,8 | có |

| Số điểm tham chiếu | brute: dự đoán 420 ảnh (ms) | kd_tree: dự đoán 420 ảnh (ms) | ball_tree: dự đoán 420 ảnh (ms) |
|---|---|---|---|
| 1680 | 2,0 | 3,7 | 5,0 |
| 8400 | 22,3 | 6,1 | 9,4 |
| 33600 | 120,3 | 11,6 | 20,6 |
| 100800 | 412,5 | 18,6 | 67,0 |

P5 — thời gian (`results/P5/table.md`, hình `results/figures/P5_time.png`):

| Đặc trưng | Bộ phân lớp | Huấn luyện 1680 ảnh (ms) | Dự đoán 420 ảnh (ms) | Dự đoán (ms/ảnh) | Macro-F1 |
|---|---|---|---|---|---|
| Moment màu (9 chiều) | KNN (k=5) | 1,6 | 3,0 | 0,0072 | 0,972 ± 0,008 |
| Moment màu (9 chiều) | Logistic Regression | 15,2 | 0,5 | 0,0011 | 0,908 ± 0,009 |
| Moment màu (9 chiều) | Linear SVM | 5,3 | 0,5 | 0,0012 | 0,838 ± 0,011 |
| Hist HSV (512 chiều) | KNN (k=5) | 8,8 | 19,4 | 0,0461 | 0,965 ± 0,014 |
| Hist HSV (512 chiều) | Logistic Regression | 528,9 | 1,5 | 0,0035 | 0,991 ± 0,005 |
| Hist HSV (512 chiều) | Linear SVM | 491,3 | 2,1 | 0,0049 | 0,989 ± 0,008 |
| HOG 64×128 (3780 chiều) | KNN (k=5) | 65,0 | 92,5 | 0,2203 | 0,946 ± 0,007 |
| HOG 64×128 (3780 chiều) | Logistic Regression | 1225,1 | 10,3 | 0,0245 | 0,985 ± 0,004 |
| HOG 64×128 (3780 chiều) | Linear SVM | 2568,0 | 9,5 | 0,0225 | 0,982 ± 0,004 |

P6 — tuyến tính so với KNN (`results/P6/table.md`, hình `results/figures/P6_linear_vs_knn.png`):

| Đặc trưng | KNN (k=5) | Logistic Regression | Linear SVM | LogReg − KNN |
|---|---|---|---|---|
| Moment màu (9 chiều) | 0,972 ± 0,008 | 0,908 ± 0,009 | 0,838 ± 0,011 | -0,064 |
| Hist HSV (512 chiều) | 0,965 ± 0,014 | 0,991 ± 0,005 | 0,989 ± 0,008 | +0,027 |
| HOG 64×128 (3780 chiều) | 0,946 ± 0,007 | 0,985 ± 0,004 | 0,982 ± 0,004 | +0,039 |

**Kết quả và kết luận.**

| # | Dự đoán | Kết quả đo | Kết luận |
|---|---|---|---|
| M1 | Histogram giữ nguyên khi xáo trộn; HOG giảm | Histogram RGB, HSV và moment: sụt đúng 0,000. HOG sụt 0,69–0,70; cạnh sụt 0,66–0,69 (còn 0,20–0,29, mức đoán mò với 4 lớp) | Khớp |
| M2 | HSV bền với độ sáng hơn RGB; H-S bền nhất | H-S sụt ≤ 0,027. HSV đủ 3 kênh vẫn sụt 0,33–0,67; RGB sụt 0,70–0,87 | Khớp một phần: tách V ra chưa đủ, phải bỏ hẳn V khỏi đặc trưng |
| M3 | Ít bin kém; nhiều bin thưa và nhạy nhiễu | 4 bin/kênh kém rõ (0,914–0,936). Hist RGB từ 8 đến 256 bin gần như ngang nhau (chênh < độ lệch chuẩn). Hist HSV 16³ = 4.096 chiều: KNN giảm còn 0,917, LogReg vẫn 0,992 | Khớp vế "ít bin". Vế "nhiều bin" chỉ thấy ở KNN trên histogram 3 chiều (87% phần tử bằng 0), không thấy ở histogram từng kênh hay ở LogReg |
| M4 | Moment gọn nhưng thô | 9 chiều; KNN 0,972 ngang histogram, nhưng LogReg chỉ 0,908 và Linear SVM 0,838. Rút đặc trưng chậm hơn histogram (≈4 ms so với 0,2 ms mỗi ảnh) | Khớp với bộ phân lớp tuyến tính; trái dự đoán với KNN (không kém histogram) và về thời gian rút |
| M5 | CCV thêm thông tin không gian nên tốt hơn histogram, và nhạy với bố cục | Ảnh gốc: LogReg 0,975 so với 0,968 (chênh < độ lệch chuẩn); KNN 0,946 so với 0,966 (CCV kém hơn). Xáo trộn: CCV sụt 0,053–0,082, histogram sụt ≤ 0,003 | Vế "nhạy với bố cục" khớp. Vế "tốt hơn histogram" không được xác nhận |
| M6 | Bỏ màu: đặc trưng màu mất tác dụng, cạnh và HOG gần như không đổi | Học lại trên ảnh xám: đặc trưng màu sụt 0,04–0,23; cạnh và HOG đổi ≤ 0,005. HOG trên ảnh xám vẫn đạt 0,986 | Khớp. Bốn giống phân biệt được bằng hình dạng (HOG xám 0,986) lẫn bằng màu (hist HSV 0,991) |
| P1 | k nhỏ nhạy nhiễu, k lớn mất chi tiết cục bộ | Đường cong giảm đều theo k; k=1 tốt nhất, k=31 thấp nhất (0,867–0,937) | Vế "k lớn" khớp. Vế "k nhỏ nhạy nhiễu" không xuất hiện trên dữ liệu này |
| P2 | KNN phụ thuộc độ đo và thang đo; sau chuẩn hóa đơn vị, Euclidean và cosine cùng thứ hạng | Cùng tập 10 láng giềng ở 100% ảnh truy vấn (trước chuẩn hóa: 11% với moment, 18% với hist HSV). Macro-F1 của Euclidean-đơn vị và cosine trùng nhau từng chữ số. Manhattan tốt nhất trên HOG (0,978 so với 0,954) | Khớp |
| P3 | Ưu tiên điểm gần giúp lớp ít ảnh | Chênh macro-F1 ≤ 0,006; recall Bangla Kola đổi trong khoảng −0,003 đến +0,026, đều nhỏ hơn hoặc xấp xỉ độ lệch chuẩn | Chưa đủ cơ sở phân biệt |
| P4 | Cây chỉ tăng tốc, không đổi kết quả, kém khi nhiều chiều | Kết quả trùng brute ở mọi trường hợp. 3.780 chiều: K-D Tree chậm hơn brute khoảng 60 lần. 9 chiều với 1.680 điểm: cây vẫn chậm hơn brute (3,2 ms so với 1,6 ms); từ 8.400 điểm trở lên K-D Tree mới nhanh hơn (6,1 ms so với 22,3 ms) | Khớp, kèm điều kiện: cây chỉ có lợi khi ít chiều *và* tập tham chiếu đủ lớn |
| P5 | KNN không tốn huấn luyện nhưng dự đoán chậm; tuyến tính ngược lại | HOG: KNN huấn luyện 0,07 s, dự đoán 0,22 ms/ảnh; LogReg huấn luyện 1,2 s, dự đoán 0,02 ms/ảnh | Khớp |
| P6 | Biên tuyến tính yếu ở đặc trưng ít chiều | Moment (9 chiều): LogReg kém KNN 0,064. Hist HSV (512): hơn 0,027. HOG (3.780): hơn 0,039 | Khớp |

Giải thích các điểm trái dự đoán:
- P1 (k=1 tốt nhất): các lớp tách nhau rõ trong không gian đặc trưng và mỗi quả thường được chụp nhiều lần ở tư thế gần giống nhau, nên láng giềng gần nhất hầu như luôn cùng lớp; tăng k chỉ kéo thêm điểm ở xa, phần nhiều thuộc lớp đông nhất (Champa Kola). **[lập luận]**; phù hợp với việc recall thấp nhất trong 4 lớp tụt mạnh khi k tăng (`results/P3/table.md`: HOG 0,816 ở k=5 xuống 0,697 ở k=15).
- M5 (CCV không hơn histogram): trên ảnh một quả chuối trên nền trơn, hầu hết điểm ảnh cùng màu đều nằm trong vùng liên thông lớn, nên phần "không liên kết" gần như rỗng và CCV mang ít thông tin mới, trong khi số chiều tăng gấp đôi. **[lập luận]**
- M4 (moment với KNN ngang histogram): 9 con số tóm tắt đủ tách 4 lớp nếu biên quyết định được phép cong (KNN), nhưng không tách tuyến tính được (LogReg). **[đo]** P6.

**Ảnh hưởng tới bước sau.**
- Đặc trưng màu tốt nhất: histogram HSV (8,8,8) + LogReg, 0,991 ± 0,005. Histogram H-S (8,8) gần bằng (0,985 ± 0,008), bền với độ sáng và chỉ 64 chiều; giữ cả hai làm ứng viên ghép với HOG ở C5.
- KNN ở bước chốt cấu hình sẽ quét k ∈ {1, 3, 5}, độ đo Euclidean/Manhattan/Hellinger thay vì cố định k=5.
- Hai đặc trưng màu và hình dạng đều đạt trên 0,98 một mình; cần S2 để biết phần của nền ảnh trong con số này.

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
| Chấm mỗi đặc trưng bằng cả KNN và LogReg | Chỉ một bộ phân lớp | Bước 2, `results/M4/table.md`: thứ hạng đặc trưng đổi theo bộ phân lớp | [đo] |
| Đặc trưng bền với độ sáng: histogram H-S (bỏ V) | Histogram HSV đủ 3 kênh | Bước 2, `results/M2/table.md`: sụt ≤ 0,027 so với 0,33–0,67 | [đo] |
| Histogram HSV dùng (8,8,8) bin | 2, 4, 16 bin mỗi kênh | Bước 2, `results/M3/table.md`: 8 bin đạt 0,991 (LogReg); 16 bin ngang nhưng 4.096 chiều và KNN giảm còn 0,917 | [đo] |
| k của KNN chọn bằng cross-validation | Quy tắc k = số lớp + 1 | Bước 2, `results/P1/table.md`: k=1 hơn k=5 từ 0,006 đến 0,020 | [đo] |
| KNN dùng Brute Force | K-D Tree, Ball Tree | Bước 2, `results/P4/table.md`: brute nhanh nhất với 1.680 ảnh tham chiếu, kết quả trùng nhau | [đo] |
| Đặc trưng nhiều chiều dùng bộ phân lớp tuyến tính, đặc trưng ít chiều dùng KNN | Một bộ phân lớp cho mọi đặc trưng | Bước 2, `results/P6/table.md` | [đo] |
