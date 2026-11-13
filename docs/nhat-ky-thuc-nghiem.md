# Nhật ký thực nghiệm

Ghi lại từng bước của đồ án: đã làm gì, vì sao, đã cân nhắc phương án nào và bằng chứng nào dẫn tới lựa chọn. Quy tắc ghi nằm ở mục 7 của [kế hoạch thực nghiệm](ke-hoach-thuc-nghiem.md).

Nhãn dùng trong nhật ký:

- **[đo]**: có số liệu so sánh, kèm đường dẫn tới file kết quả.
- **[lập luận]**: dựa trên tính chất phương pháp hoặc giới hạn tài nguyên, chưa đo.

## Mẫu cho mỗi bước

```
## Bước <số>: <tên bước> (<ngày>)

**Mục tiêu.**
**Đã làm.** (file code, lệnh chạy)
**Lý do chọn cách này.**
**Phương án khác và vì sao không chọn.** (mỗi phương án gắn nhãn [đo] hoặc [lập luận])
**Bằng chứng.** (bảng số liệu, đường dẫn results/, hình)
**Kết quả và kết luận.**
**Ảnh hưởng tới bước sau.**
```

---

## Bước 0.1: Chọn và chuẩn bị dữ liệu (12/11/2026)

**Mục tiêu.** Có bộ ảnh gốc của 4 giống chuối, kiểm tra được tính toàn vẹn, sắp xếp sẵn theo lớp.

**Đã làm.**
- Tải file `Banana Classification Dataset.zip` (12162003 byte) của bộ BananaImageBD, Version 2, từ Mendeley Data.
- Kiểm tra SHA-256 của file tải về, khớp với giá trị Mendeley công bố (`df5c7cea…abbf1f`).
- Giải nén, đổi tên thư mục lớp sang dạng `snake_case`, đặt vào `data/raw/`.
- Đếm ảnh từng lớp, đếm đuôi file, kiểm tra kích thước ảnh và tìm file trùng nhau tuyệt đối bằng md5.
- Viết `scripts/download_data.sh` để tải lại và kiểm tra đúng như trên.

**Lý do chọn cách này.** Dùng ảnh gốc từ nguồn phát hành chính thức để số lượng ảnh đối chiếu được với mô tả của tác giả, và giữ bằng chứng toàn vẹn (mã băm) để ai cũng tải lại được đúng bộ dữ liệu.

**Phương án khác và vì sao không chọn.**
- Bản "Augmented" có sẵn (7413 ảnh): không dùng, vì ảnh tăng cường sinh từ cùng ảnh gốc sẽ nằm ở cả train lẫn test khi chia ngẫu nhiên. **[lập luận]**, sẽ kiểm chứng bằng thí nghiệm D1.
- Version 1 của bộ dữ liệu (2512 ảnh gốc, Champa Kola 1035): không dùng, vì Version 2 là bản tác giả cập nhật sau và bỏ bớt 41 ảnh Champa Kola. **[lập luận]**

**Bằng chứng.**

| Kiểm tra | Kết quả |
|---|---|
| Tổng số ảnh | 2471 (khớp mô tả Version 2) |
| Bangla Kola / Champa Kola / Sabri Kola / Sagor Kola | 444 / 994 / 509 / 524 |
| Đuôi file | 2378 `.jpg`, 93 `.JPG` |
| File trùng nhau tuyệt đối (md5) | 0 |
| File không đọc được | 0 |
| Chế độ màu | cả 2471 ảnh là RGB |
| Kích thước ảnh | 2469 ảnh 256×256; 2 ảnh khác: `bangla_kola/Bangla_Kola_55.jpg` (237×244), `champa_kola/Champa_Kola_324.jpg` (218×256) |

**Kết quả và kết luận.** Dữ liệu đầy đủ và khớp nguồn về số lượng. Mô tả của tác giả ghi ảnh 256×256 nhưng có 2 ảnh lệch kích thước. Lớp lớn nhất (Champa Kola) gấp khoảng 2.2 lần lớp nhỏ nhất (Bangla Kola), nên cần xử lý mất cân bằng và dùng macro-F1 làm chỉ số chính.

**Ảnh hưởng tới bước sau.** Code đọc ảnh phải nhận cả `.jpg` và `.JPG`, và luôn resize về kích thước chuẩn trước khi rút đặc trưng (không giả định ảnh nào cũng 256×256). Bước 1 cần tìm ảnh gần trùng, vì md5 chỉ bắt được file giống hệt nhau.

## Bước 0.2: Chọn thư viện (12/11/2026)

**Mục tiêu.** Chốt công cụ lập trình cho cả phần truyền thống và phần CNN.

**Đã làm.** Lập `requirements.txt`: OpenCV, scikit-image, scikit-learn, imbalanced-learn cho phần truyền thống; TensorFlow/Keras (bản CPU) cho CNN. Đã kiểm tra các thư viện trong danh sách cài được cùng nhau trên Python 3.13.

**Lý do chọn cách này.** Các ví dụ trong bài giảng dùng OpenCV, scikit-learn và Keras, nên code của đồ án đối chiếu trực tiếp được với nội dung đã học.

**Phương án khác và vì sao không chọn.**
- PyTorch cho phần CNN: không chọn, vì ví dụ CNN và transfer learning trong bài giảng viết bằng Keras; dùng cùng thư viện thì cách khai báo lớp và cách đếm tham số khớp với bài học. **[lập luận]**

**Bằng chứng.** Chưa có số liệu; đây là lựa chọn công cụ.

**Kết quả và kết luận.** Dùng TensorFlow/Keras bản CPU.

**Ảnh hưởng tới bước sau.** Máy chỉ có CPU nên các mô hình đóng băng phải tính trước đặc trưng một lần rồi lưu lại.

## Bước 0.3: Chốt quy trình đánh giá (12/11/2026)

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

## Bước 1: Chia dữ liệu, ảnh gần trùng, thống kê và thí nghiệm D1 (12/11/2026)

**Mục tiêu.** Có ba tập train/val/test cố định; biết trong dữ liệu có bao nhiêu ảnh gần trùng nằm vắt qua các tập; kiểm chứng bằng số liệu lý do không dùng bản "Augmented" có sẵn (giả thuyết D1).

**Đã làm.**
- `src/config.py`, `src/data.py`: đường dẫn, tên lớp, seed; đọc ảnh nhận cả `.jpg` lẫn `.JPG`, luôn resize về 256×256; cache toàn bộ ảnh vào `data/cache/images_256.npy`.
- `python -m src.split_data`: chia stratified 70/15/15 với `seed=42`, ghi `data/splits/{train,val,test}.csv`; tính perceptual hash (pHash 64 bit) cho 2471 ảnh, đếm số cặp ảnh có khoảng cách Hamming nhỏ theo từng ngưỡng; vẽ thống kê.
- `python -m src.env_info`: ghi phiên bản thư viện vào `results/environment.txt`.
- `src/features/` (màu, cạnh, HOG tự cài), `src/transforms.py`, `src/classical.py` (5-fold CV trên train+val), `src/metrics.py`: khung dùng chung cho các bước sau. HOG tự cài (để có phiên bản bỏ chuẩn hóa block cho C2) được đối chiếu với `skimage.feature.hog`: `python -m src.features.hog` cho sai khác tuyệt đối lớn nhất 2.2·10⁻⁷ trên 20 ảnh, cùng 3780 chiều.
- `python -m src.exp_d1`: thí nghiệm D1 và phần phụ D1b.

**Lý do chọn cách này.**
- D1 chạy hoàn toàn trong tập phát triển (train+val, 2100 ảnh), tập "test" của D1 là 20% tách ra từ tập phát triển, lặp 5 lần với 5 seed. Làm vậy vì mục 5 của kế hoạch quy định tập test chính thức chỉ dùng một lần ở cuối. **[lập luận]**
- Cùng lý do, mọi thí nghiệm "biến đổi ảnh test" ở các bước sau (M1, M2, M6, C1, C2, C4) được chấm trên fold giữ lại của cross-validation (nhóm truyền thống) hoặc trên tập val (CNN), không phải tập test chính thức. **[lập luận]**
- D1 thêm cách (c) ngoài hai cách (a), (b) của kế hoạch: chia theo ảnh gốc rồi tăng cường cả hai phía. Tập chấm của (a) có cả ảnh lật/xoay còn của (b) chỉ có ảnh gốc, nên (a)-(b) lẫn hai hiệu ứng: rò rỉ và độ khó của tập chấm. (c) có cùng thành phần tập chấm với (a) nên (a)-(c) là phần thổi phồng thuần do rò rỉ. **[đo]** `results/D1/table.md`: với HOG + Logistic Regression, (b) cao hơn (a) 0.008 dù (a) có rò rỉ, vì tập chấm của (b) dễ hơn; (a)-(c) = +0.000.
- Bản tăng cường tự tạo: mỗi ảnh gốc thêm 1 bản lật ngang và 1 bản xoay góc ngẫu nhiên trong ±[10°, 30°], tức gấp 3 số ảnh, cùng tỉ lệ với bản "Augmented" của tác giả (7413 = 3 × 2471). **[lập luận]**

**Phương án khác và vì sao không chọn.**
- Ngưỡng pHash để gọi là "gần trùng": xem hình các cặp theo từng mức Hamming (`results/figures/near_duplicates_by_distance.png`). Hamming ≤ 2: cùng một quả chụp lại gần như y hệt. Hamming = 4 trở lên: phần lớn là hai quả khác nhau có bố cục giống nhau (quả nhỏ nằm giữa nền trơn, nhiều nhất ở Sabri Kola). Chọn ≤ 2 làm ngưỡng "gần trùng chắc chắn", báo cáo thêm ≤ 4 làm cận trên. **[đo]** bảng dưới.
- Chia lại dữ liệu theo nhóm ảnh gần trùng (group split) thay cho chia stratified ngẫu nhiên: không chọn, vì đo thấy ảnh gần trùng không làm đổi kết quả. **[đo]** `results/D1/table_near_duplicates.md`: chênh macro-F1 giữa StratifiedKFold và StratifiedGroupKFold nằm trong ±0.007, nhỏ hơn độ lệch chuẩn giữa các fold ở cả 5 mô hình.
- Dùng tập test chính thức cho D1 như câu chữ của kế hoạch ("độ chính xác trên test"): không chọn, lý do ở trên. **[lập luận]**
- Chỉ dùng một mô hình cho D1: không chọn; chạy 5 mô hình (mỗi mô hình vài giây) vì mức rò rỉ phụ thuộc đặc trưng có bất biến với phép lật/xoay hay không. **[đo]** bảng D1.

**Bằng chứng.**

Số ảnh mỗi tập (`results/data/data_stats.json`, hình `results/figures/data_split_counts.png`, ảnh mẫu `results/figures/data_samples.png`):

| Tập | Bangla Kola | Champa Kola | Sabri Kola | Sagor Kola | Tổng |
|---|---|---|---|---|---|
| train | 311 | 695 | 356 | 367 | 1729 |
| val | 66 | 150 | 76 | 79 | 371 |
| test | 67 | 149 | 77 | 78 | 371 |

Ảnh gần trùng theo pHash (`results/data/near_duplicates.json`; hình `results/figures/near_duplicates_closest_pairs.png`):

| Ngưỡng Hamming | Số cặp | Cặp khác lớp | Cặp train-test | Cặp train-val | Cặp val-test | Ảnh test có "bản gần trùng" trong train |
|---|---|---|---|---|---|---|
| = 0 | 3 | 0 | 0 | 1 | 0 | 0 |
| ≤ 2 | 42 | 1 | 7 | 8 | 2 | 7 / 371 |
| ≤ 4 | 383 | 10 | 83 | 86 | 15 | 44 / 371 |
| ≤ 6 | 1527 | 56 | 336 | 355 | 62 | 103 / 371 |

D1: Accuracy trung bình ± độ lệch chuẩn qua 5 lần chia 80/20 trong tập phát triển (`results/D1/table.md`, `results/D1/metrics.json`, hình `results/figures/D1_leakage.png`):

| Mô hình | (a) tăng cường rồi chia | (b) chia rồi tăng cường train | (c) chia theo ảnh gốc, tăng cường 2 phía | (a)-(b) | (a)-(c) |
|---|---|---|---|---|---|
| HSV hist + KNN (k=1) | 0.999 ± 0.001 | 0.978 ± 0.010 | 0.978 ± 0.008 | +0.022 ± 0.010 | +0.021 ± 0.009 |
| HSV hist + LogReg | 0.999 ± 0.001 | 0.991 ± 0.002 | 0.990 ± 0.002 | +0.008 ± 0.002 | +0.009 ± 0.002 |
| HOG + KNN (k=1) | 0.974 ± 0.003 | 0.974 ± 0.003 | 0.970 ± 0.002 | +0.000 ± 0.003 | +0.005 ± 0.004 |
| HOG + LogReg | 0.973 ± 0.006 | 0.981 ± 0.003 | 0.973 ± 0.003 | -0.008 ± 0.004 | +0.000 ± 0.004 |
| Edge + KNN (k=1) | 0.896 ± 0.012 | 0.889 ± 0.014 | 0.862 ± 0.012 | +0.007 ± 0.020 | +0.034 ± 0.020 |

D1b: macro-F1 5-fold, chia thường so với chia giữ nhóm ảnh gần trùng (`results/D1/table_near_duplicates.md`):

| Mô hình | StratifiedKFold | GroupKFold (≤ 2) | GroupKFold (≤ 4) |
|---|---|---|---|
| HSV hist + KNN (k=1) | 0.975 ± 0.005 | 0.975 ± 0.003 | 0.973 ± 0.008 |
| HSV hist + LogReg | 0.991 ± 0.005 | 0.991 ± 0.003 | 0.993 ± 0.003 |
| HOG + KNN (k=1) | 0.967 ± 0.011 | 0.968 ± 0.004 | 0.970 ± 0.005 |
| HOG + LogReg | 0.985 ± 0.004 | 0.986 ± 0.006 | 0.986 ± 0.005 |
| Edge + KNN (k=1) | 0.902 ± 0.019 | 0.902 ± 0.015 | 0.909 ± 0.007 |

**Kết quả và kết luận.**
- Chia xong: 1729 / 371 / 371 ảnh, tỉ lệ lớp giữ nguyên ở cả ba tập.
- Ảnh gần trùng: có. Ở ngưỡng chắc chắn (Hamming ≤ 2) có 42 cặp, 7 ảnh test có bản gần trùng trong train. Ảnh hưởng tới kết quả không đo được (D1b). pHash không bắt được trường hợp cùng một quả chụp ở tư thế khác, nên con số này là cận dưới; ghi vào phần hạn chế.
- D1 khớp dự đoán với đặc trưng màu: histogram HSV gần như không đổi khi lật/xoay ảnh, nên bản tăng cường của ảnh test nằm trong train là "đáp án có sẵn"; accuracy bị thổi phồng +0.021 (KNN k=1) và +0.009 (LogReg), lớn hơn độ lệch chuẩn. Cả hai đều lên 0.999.
- D1 **trái dự đoán** với HOG: mức thổi phồng (a)-(c) chỉ +0.005 ± 0.004 (KNN) và +0.000 ± 0.004 (LogReg), chưa đủ cơ sở phân biệt với 0. Giải thích: HOG mã hóa hướng cạnh theo vị trí, nên bản lật hay xoay của một ảnh cho vector khác hẳn ảnh gốc; bản sao trong train không còn là láng giềng gần của ảnh test. Rò rỉ chỉ gây hại khi đặc trưng bất biến với phép tăng cường. Với CNN có tăng cường ảnh (học bất biến lật/xoay) thì dự kiến bị ảnh hưởng như histogram; chưa đo. **[lập luận]**
- Với đặc trưng cạnh (vector tổng theo hàng/cột): (a)-(c) = +0.034 ± 0.020, lớn hơn độ lệch chuẩn.

**Ảnh hưởng tới bước sau.**
- Giữ quyết định dùng ảnh gốc và chỉ tăng cường trên phần huấn luyện.
- Số liệu sơ bộ cho thấy bài toán dễ với đặc trưng màu (HSV hist + LogReg đạt macro-F1 0.991 trên 5-fold). Cần kiểm tra ở S2 xem mô hình có dựa vào màu nền không.
- Độ lệch chuẩn giữa các fold khoảng 0.004 đến 0.02; các so sánh ở bước sau lấy đó làm ngưỡng "đủ cơ sở phân biệt".

## Bước 2: Đặc trưng màu, KNN và bộ phân lớp tuyến tính: M1-M6, P1-P6 (12/11/2026)

**Mục tiêu.** Kiểm chứng các giả thuyết về đặc trưng màu (mục 6.2) và về bộ phân lớp (mục 6.4) bằng 5-fold stratified cross-validation trên train+val (2100 ảnh).

**Đã làm.**
- `src/features/color.py`: histogram RGB từng kênh, histogram HSV 3 chiều, histogram H-S, moment màu (9 chiều), màu chủ đạo k-means (k=3, 12 chiều), CCV và histogram cùng mức lượng tử hóa (64 màu).
- `src/transforms.py`: xáo trộn lưới 4×4, nhân kênh V, chuyển xám, nhiễu Gauss, che 25%, xoay. Mỗi ảnh có bộ sinh số ngẫu nhiên riêng theo (seed, tên phép, chỉ số ảnh).
- `python -m src.exp_color` (M1-M6) và `python -m src.exp_classifier` (P1-P6). Mỗi thí nghiệm ghi `results/<mã>/metrics.json`, `results/<mã>/table.md` và hình `results/figures/<mã>_*.png`.
- `scripts/run_all.sh`: chạy lại toàn bộ các bước theo thứ tự (`bash scripts/run_all.sh`, hoặc `bash scripts/run_all.sh 2` cho riêng bước này).

**Lý do chọn cách này.**
- Mô hình luôn học trên ảnh sạch; phép biến đổi chỉ áp lên fold giữ lại. Độ sụt tính theo từng fold rồi lấy trung bình ± độ lệch chuẩn, nên so được với nhiễu giữa các fold. **[lập luận]**
- Mỗi đặc trưng chấm bằng cả KNN (k=5, Euclidean, z-score) và Logistic Regression (C=1, z-score) với tham số mặc định cố định, để kết luận về đặc trưng không phụ thuộc một bộ phân lớp. Việc tinh chỉnh siêu tham số làm riêng ở P1-P3 và ở bước chốt cấu hình. **[đo]** `results/M4/table.md`: thứ hạng đặc trưng đổi theo bộ phân lớp (moment màu đứng thứ hai với KNN nhưng gần cuối với LogReg).
- M6 chạy hai chế độ: "học màu, chấm xám" (đo độ bền) và "học xám, chấm xám" (đo lượng thông tin còn lại khi không có màu). Chế độ đầu của kế hoạch chỉ cho biết histogram màu sập khi phân bố đầu vào đổi; chế độ sau mới trả lời câu hỏi "giống chuối khác nhau ở màu hay hình dạng". **[đo]** `results/M6/table.md`.
- M3 quét histogram RGB từng kênh đủ 6 mức bin của kế hoạch; với histogram HSV 3 chiều chỉ quét 2, 4, 8, 16 bin mỗi kênh vì 256 bin mỗi kênh cho 16.7 triệu chiều. **[lập luận]**
- P4 thêm phần phụ: giữ đặc trưng 9 chiều, nhân tập tham chiếu lên 5, 20, 60 lần (bản sao cộng nhiễu nhỏ, chỉ để đo thời gian) vì kết quả chính trái dự đoán. **[đo]** `results/P4/table_scaling.md`.

**Phương án khác và vì sao không chọn.**
- Cài moment màu bằng phép tính float64 trên toàn ảnh: bản đầu mất 17.9 ms/ảnh; viết lại bằng float32 còn khoảng 4 ms/ảnh, giá trị lệch tối đa 8.7·10⁻⁷. Dùng bản sau. **[đo]** `results/M4/table.md` (cột thời gian).
- Dùng histogram HSV đủ 3 kênh làm đặc trưng "bền với độ sáng": không chọn cho mục đích đó. **[đo]** `results/M2/table.md`: khi nhân V với 0.6, histogram HSV (8,8,8) sụt 0.66 đến 0.67 macro-F1, gần bằng histogram RGB; chỉ histogram H-S giữ được (sụt 0.000 đến 0.027).
- Quy tắc k = số lớp + 1 = 5: không tốt bằng k chọn bằng cross-validation. **[đo]** `results/P1/table.md`: k=1 tốt nhất ở cả ba đặc trưng; chênh +0.006 (moment, nhỏ hơn độ lệch chuẩn), +0.011 (hist HSV, nhỏ hơn độ lệch chuẩn), +0.020 (HOG, lớn hơn độ lệch chuẩn).
- K-D Tree / Ball Tree thay cho Brute Force: không chọn với cỡ dữ liệu này. **[đo]** `results/P4/table.md`: với 1680 ảnh tham chiếu, brute nhanh nhất ở mọi số chiều.
- Hellinger cho đặc trưng histogram: 0.976 ± 0.015 so với Euclidean + z-score 0.965 ± 0.014 trên hist HSV; chênh nhỏ hơn độ lệch chuẩn, chưa đủ cơ sở phân biệt. **[đo]** `results/P2/table.md`.

**Bằng chứng.** (macro-F1 trung bình ± độ lệch chuẩn qua 5 fold)

M1: xáo trộn lưới 4×4 (`results/M1/table.md`, hình `results/figures/M1_shuffle.png`):

| Đặc trưng | Bộ phân lớp | Macro-F1 ảnh gốc | Macro-F1 ảnh xáo trộn 4×4 | Độ sụt |
|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0.964 ± 0.012 | 0.964 ± 0.012 | +0.000 ± 0.000 |
| Hist RGB (3×32) | LogReg | 0.976 ± 0.013 | 0.976 ± 0.013 | +0.000 ± 0.000 |
| Hist HSV (8,8,8) | KNN (k=5) | 0.965 ± 0.014 | 0.965 ± 0.014 | +0.000 ± 0.000 |
| Hist HSV (8,8,8) | LogReg | 0.991 ± 0.005 | 0.991 ± 0.005 | +0.000 ± 0.000 |
| Moment màu | KNN (k=5) | 0.972 ± 0.008 | 0.972 ± 0.008 | +0.000 ± 0.000 |
| Moment màu | LogReg | 0.908 ± 0.009 | 0.908 ± 0.009 | +0.000 ± 0.000 |
| CCV (64 màu) | KNN (k=5) | 0.946 ± 0.011 | 0.864 ± 0.018 | +0.082 ± 0.015 |
| CCV (64 màu) | LogReg | 0.975 ± 0.006 | 0.922 ± 0.011 | +0.053 ± 0.009 |
| Cạnh Sobel | KNN (k=5) | 0.890 ± 0.019 | 0.231 ± 0.021 | +0.658 ± 0.028 |
| Cạnh Sobel | LogReg | 0.894 ± 0.008 | 0.201 ± 0.040 | +0.692 ± 0.042 |
| HOG 64×128 | KNN (k=5) | 0.946 ± 0.007 | 0.260 ± 0.017 | +0.687 ± 0.011 |
| HOG 64×128 | LogReg | 0.985 ± 0.004 | 0.287 ± 0.020 | +0.699 ± 0.024 |

M2: nhân kênh V (`results/M2/table.md`, hình `results/figures/M2_brightness.png`):

| Đặc trưng | Bộ phân lớp | Ảnh gốc | V×0.6 | Sụt (V×0.6) | V×1.4 | Sụt (V×1.4) |
|---|---|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0.964 ± 0.012 | 0.092 ± 0.002 | +0.873 ± 0.013 | 0.205 ± 0.019 | +0.760 ± 0.024 |
| Hist RGB (3×32) | LogReg | 0.976 ± 0.013 | 0.212 ± 0.054 | +0.763 ± 0.064 | 0.279 ± 0.038 | +0.696 ± 0.036 |
| Hist HSV (8,8,8) | KNN (k=5) | 0.965 ± 0.014 | 0.302 ± 0.042 | +0.663 ± 0.043 | 0.589 ± 0.020 | +0.376 ± 0.025 |
| Hist HSV (8,8,8) | LogReg | 0.991 ± 0.005 | 0.318 ± 0.035 | +0.673 ± 0.035 | 0.662 ± 0.004 | +0.329 ± 0.009 |
| Hist H-S (8,8) | KNN (k=5) | 0.976 ± 0.006 | 0.975 ± 0.009 | +0.000 ± 0.008 | 0.975 ± 0.006 | +0.001 ± 0.002 |
| Hist H-S (8,8) | LogReg | 0.985 ± 0.008 | 0.959 ± 0.014 | +0.027 ± 0.011 | 0.974 ± 0.007 | +0.011 ± 0.004 |

M3: số bin (`results/M3/table.md`, hình `results/figures/M3_bins.png`):

| Đặc trưng | Số bin mỗi kênh | Số chiều | KNN (k=5) | LogReg | Tỉ lệ phần tử bằng 0 |
|---|---|---|---|---|---|
| Hist RGB từng kênh | 4 | 12 | 0.936 ± 0.015 | 0.914 ± 0.015 | 0.00 |
| Hist RGB từng kênh | 8 | 24 | 0.966 ± 0.010 | 0.968 ± 0.007 | 0.03 |
| Hist RGB từng kênh | 16 | 48 | 0.969 ± 0.011 | 0.979 ± 0.012 | 0.06 |
| Hist RGB từng kênh | 32 | 96 | 0.964 ± 0.012 | 0.976 ± 0.013 | 0.08 |
| Hist RGB từng kênh | 64 | 192 | 0.963 ± 0.015 | 0.977 ± 0.012 | 0.09 |
| Hist RGB từng kênh | 256 | 768 | 0.962 ± 0.015 | 0.975 ± 0.011 | 0.12 |
| Hist HSV 3 chiều | 2 | 8 | 0.932 ± 0.011 | 0.885 ± 0.012 | 0.20 |
| Hist HSV 3 chiều | 4 | 64 | 0.963 ± 0.009 | 0.965 ± 0.012 | 0.55 |
| Hist HSV 3 chiều | 8 | 512 | 0.965 ± 0.014 | 0.991 ± 0.005 | 0.72 |
| Hist HSV 3 chiều | 16 | 4096 | 0.917 ± 0.015 | 0.992 ± 0.004 | 0.87 |

M4: các đặc trưng màu (`results/M4/table.md`, hình `results/figures/M4_color_features.png`):

| Đặc trưng | Số chiều | Rút đặc trưng (ms/ảnh) | KNN (k=5) | Huấn luyện KNN (k=5) (s) | Dự đoán KNN (k=5) (ms/ảnh) | LogReg | Huấn luyện LogReg (s) | Dự đoán LogReg (ms/ảnh) | Linear SVM |
|---|---|---|---|---|---|---|---|---|---|
| Hist RGB (3×32) | 96 | 0.16 | 0.964 ± 0.012 | 0.003 | 0.009 | 0.976 ± 0.013 | 0.037 | 0.002 | 0.960 ± 0.012 |
| Hist HSV (8,8,8) | 512 | 0.21 | 0.965 ± 0.014 | 0.008 | 0.025 | 0.991 ± 0.005 | 0.735 | 0.008 | 0.989 ± 0.008 |
| Hist H-S (8,8) | 64 | 0.21 | 0.976 ± 0.006 | 0.002 | 0.009 | 0.985 ± 0.008 | 0.016 | 0.001 | 0.958 ± 0.007 |
| Moment màu | 9 | 4.25 | 0.972 ± 0.008 | 0.001 | 0.007 | 0.908 ± 0.009 | 0.014 | 0.001 | 0.838 ± 0.011 |
| Màu chủ đạo (k=3) | 12 | 1.58 | 0.945 ± 0.005 | 0.002 | 0.006 | 0.899 ± 0.009 | 0.017 | 0.001 | 0.826 ± 0.013 |
| CCV (64 màu) | 128 | 2.03 | 0.946 ± 0.011 | 0.003 | 0.013 | 0.975 ± 0.006 | 0.017 | 0.001 | 0.965 ± 0.014 |
| Hist 64 màu | 64 | 0.53 | 0.966 ± 0.010 | 0.002 | 0.008 | 0.968 ± 0.008 | 0.014 | 0.001 | 0.945 ± 0.020 |

M5: CCV (`results/M5/table.md`, hình `results/figures/M5_ccv.png`):

| Đặc trưng | Số chiều | Bộ phân lớp | Ảnh gốc | Xáo trộn 4×4 | Độ sụt |
|---|---|---|---|---|---|
| Hist 64 màu | 64 | KNN (k=5) | 0.966 ± 0.010 | 0.962 ± 0.008 | +0.003 ± 0.004 |
| Hist 64 màu | 64 | LogReg | 0.968 ± 0.008 | 0.966 ± 0.009 | +0.002 ± 0.004 |
| CCV (64 màu) | 128 | KNN (k=5) | 0.946 ± 0.011 | 0.864 ± 0.018 | +0.082 ± 0.015 |
| CCV (64 màu) | 128 | LogReg | 0.975 ± 0.006 | 0.922 ± 0.011 | +0.053 ± 0.009 |

M6: ảnh xám (`results/M6/table.md`, hình `results/figures/M6_grayscale.png`):

| Đặc trưng | Bộ phân lớp | Học màu, chấm màu | Học màu, chấm xám | Học xám, chấm xám | Sụt khi bỏ màu (học lại) |
|---|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0.964 ± 0.012 | 0.631 ± 0.026 | 0.927 ± 0.015 | +0.038 |
| Hist RGB (3×32) | LogReg | 0.976 ± 0.013 | 0.456 ± 0.030 | 0.865 ± 0.012 | +0.111 |
| Hist HSV (8,8,8) | KNN (k=5) | 0.965 ± 0.014 | 0.121 ± 0.021 | 0.912 ± 0.020 | +0.052 |
| Hist HSV (8,8,8) | LogReg | 0.991 ± 0.005 | 0.154 ± 0.022 | 0.804 ± 0.010 | +0.187 |
| Moment màu | KNN (k=5) | 0.972 ± 0.008 | 0.120 ± 0.041 | 0.748 ± 0.038 | +0.223 |
| Moment màu | LogReg | 0.908 ± 0.009 | 0.320 ± 0.063 | 0.676 ± 0.018 | +0.232 |
| CCV (64 màu) | KNN (k=5) | 0.946 ± 0.011 | 0.403 ± 0.014 | 0.782 ± 0.028 | +0.164 |
| CCV (64 màu) | LogReg | 0.975 ± 0.006 | 0.292 ± 0.035 | 0.762 ± 0.020 | +0.213 |
| Cạnh Sobel | KNN (k=5) | 0.890 ± 0.019 | 0.890 ± 0.019 | 0.890 ± 0.019 | +0.000 |
| Cạnh Sobel | LogReg | 0.894 ± 0.008 | 0.894 ± 0.008 | 0.894 ± 0.008 | +0.000 |
| HOG 64×128 | KNN (k=5) | 0.946 ± 0.007 | 0.946 ± 0.012 | 0.945 ± 0.012 | +0.001 |
| HOG 64×128 | LogReg | 0.985 ± 0.004 | 0.987 ± 0.006 | 0.986 ± 0.005 | -0.001 |
| HOG 64×128 (màu, kênh gradient lớn nhất) | KNN (k=5) | 0.944 ± 0.013 | 0.943 ± 0.012 | 0.945 ± 0.012 | -0.001 |
| HOG 64×128 (màu, kênh gradient lớn nhất) | LogReg | 0.982 ± 0.005 | 0.984 ± 0.005 | 0.986 ± 0.005 | -0.005 |

P1: chọn k (`results/P1/table.md`, đủ 16 giá trị k ở `results/P1/table_all_k.md`, hình `results/figures/P1_knn_k.png`):

| Đặc trưng | k = số lớp + 1 = 5 | k tốt nhất theo CV | Macro-F1 tại k tốt nhất | Chênh so với k=5 | k=1 | k=31 |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0.972 ± 0.008 | 1 | 0.977 ± 0.003 | +0.006 | 0.977 ± 0.003 | 0.937 ± 0.012 |
| Hist HSV (512 chiều) | 0.965 ± 0.014 | 1 | 0.975 ± 0.005 | +0.011 | 0.975 ± 0.005 | 0.867 ± 0.026 |
| HOG 64×128 (3780 chiều) | 0.946 ± 0.007 | 1 | 0.967 ± 0.011 | +0.020 | 0.967 ± 0.011 | 0.871 ± 0.012 |

P2: độ đo và chuẩn hóa (`results/P2/table.md`, `results/P2/table_rank_agreement.md`, hình `results/figures/P2_metrics.png`):

| Độ đo | Chuẩn hóa | Moment màu (9 chiều) | Hist HSV (512 chiều) | HOG 64×128 (3780 chiều) |
|---|---|---|---|---|
| Euclidean | không | 0.977 ± 0.005 | 0.953 ± 0.006 | 0.954 ± 0.008 |
| Euclidean | vector đơn vị (L2) | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Euclidean | z-score | 0.972 ± 0.008 | 0.965 ± 0.014 | 0.946 ± 0.007 |
| Manhattan | không | 0.977 ± 0.004 | 0.964 ± 0.009 | 0.978 ± 0.007 |
| Manhattan | vector đơn vị (L2) | 0.967 ± 0.006 | 0.956 ± 0.013 | 0.978 ± 0.007 |
| Manhattan | z-score | 0.974 ± 0.008 | 0.961 ± 0.009 | 0.975 ± 0.010 |
| Cosine | không | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Cosine | vector đơn vị (L2) | 0.970 ± 0.005 | 0.936 ± 0.011 | 0.954 ± 0.008 |
| Hellinger | L1 rồi lấy căn | không áp dụng (có giá trị âm) | 0.976 ± 0.015 | 0.957 ± 0.011 |

| Đặc trưng | Đã chuẩn hóa đơn vị: cùng 10 láng giềng, cùng thứ tự | Đã chuẩn hóa đơn vị: cùng tập 10 láng giềng | Chưa chuẩn hóa: cùng tập 10 láng giềng |
|---|---|---|---|
| Moment màu (9 chiều) | 99.5% | 100.0% | 11.2% |
| Hist HSV (512 chiều) | 100.0% | 100.0% | 18.1% |
| HOG 64×128 (3780 chiều) | 100.0% | 100.0% | 99.3% |

P3: bình chọn đa số so với ưu tiên điểm gần (`results/P3/table.md`, hình `results/figures/P3_weights.png`):

| Đặc trưng | k | Macro-F1 uniform | Macro-F1 distance | Recall Bangla Kola (lớp ít ảnh nhất) uniform | Recall Bangla Kola distance | Recall thấp nhất trong 4 lớp uniform → distance |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 5 | 0.972 ± 0.008 | 0.972 ± 0.008 | 0.984 ± 0.013 | 0.984 ± 0.013 | 0.922 → 0.924 |
| Moment màu (9 chiều) | 15 | 0.955 ± 0.008 | 0.961 ± 0.006 | 0.942 ± 0.026 | 0.968 ± 0.018 | 0.888 → 0.888 |
| Hist HSV (512 chiều) | 5 | 0.965 ± 0.014 | 0.966 ± 0.013 | 0.981 ± 0.007 | 0.981 ± 0.007 | 0.879 → 0.888 |
| Hist HSV (512 chiều) | 15 | 0.931 ± 0.013 | 0.933 ± 0.014 | 0.952 ± 0.018 | 0.955 ± 0.022 | 0.756 → 0.765 |
| HOG 64×128 (3780 chiều) | 5 | 0.946 ± 0.007 | 0.949 ± 0.008 | 0.955 ± 0.021 | 0.952 ± 0.026 | 0.816 → 0.830 |
| HOG 64×128 (3780 chiều) | 15 | 0.912 ± 0.011 | 0.914 ± 0.010 | 0.920 ± 0.033 | 0.923 ± 0.036 | 0.697 → 0.706 |

P4: thuật toán tìm láng giềng (`results/P4/table.md`, `results/P4/table_scaling.md`, hình `results/figures/P4_knn_algorithms.png`):

| Đặc trưng | brute: dựng (ms) | brute: dự đoán 420 ảnh (ms) | kd_tree: dựng (ms) | kd_tree: dự đoán 420 ảnh (ms) | kd_tree: kết quả trùng brute | ball_tree: dựng (ms) | ball_tree: dự đoán 420 ảnh (ms) | ball_tree: kết quả trùng brute |
|---|---|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0.6 | 1.6 | 1.4 | 3.2 | có | 1.1 | 4.5 | có |
| Hist HSV (512 chiều) | 1.3 | 9.1 | 38.7 | 361.1 | có | 29.3 | 244.3 | có |
| HOG 64×128 (3780 chiều) | 8.1 | 73.9 | 367.7 | 4533.7 | có | 223.5 | 2231.8 | có |

| Số điểm tham chiếu | brute: dự đoán 420 ảnh (ms) | kd_tree: dự đoán 420 ảnh (ms) | ball_tree: dự đoán 420 ảnh (ms) |
|---|---|---|---|
| 1680 | 2.0 | 3.7 | 5.0 |
| 8400 | 22.3 | 6.1 | 9.4 |
| 33600 | 120.3 | 11.6 | 20.6 |
| 100800 | 412.5 | 18.6 | 67.0 |

P5: thời gian (`results/P5/table.md`, hình `results/figures/P5_time.png`):

| Đặc trưng | Bộ phân lớp | Huấn luyện 1680 ảnh (ms) | Dự đoán 420 ảnh (ms) | Dự đoán (ms/ảnh) | Macro-F1 |
|---|---|---|---|---|---|
| Moment màu (9 chiều) | KNN (k=5) | 1.6 | 3.0 | 0.0072 | 0.972 ± 0.008 |
| Moment màu (9 chiều) | Logistic Regression | 15.2 | 0.5 | 0.0011 | 0.908 ± 0.009 |
| Moment màu (9 chiều) | Linear SVM | 5.3 | 0.5 | 0.0012 | 0.838 ± 0.011 |
| Hist HSV (512 chiều) | KNN (k=5) | 8.8 | 19.4 | 0.0461 | 0.965 ± 0.014 |
| Hist HSV (512 chiều) | Logistic Regression | 528.9 | 1.5 | 0.0035 | 0.991 ± 0.005 |
| Hist HSV (512 chiều) | Linear SVM | 491.3 | 2.1 | 0.0049 | 0.989 ± 0.008 |
| HOG 64×128 (3780 chiều) | KNN (k=5) | 65.0 | 92.5 | 0.2203 | 0.946 ± 0.007 |
| HOG 64×128 (3780 chiều) | Logistic Regression | 1225.1 | 10.3 | 0.0245 | 0.985 ± 0.004 |
| HOG 64×128 (3780 chiều) | Linear SVM | 2568.0 | 9.5 | 0.0225 | 0.982 ± 0.004 |

P6: tuyến tính so với KNN (`results/P6/table.md`, hình `results/figures/P6_linear_vs_knn.png`):

| Đặc trưng | KNN (k=5) | Logistic Regression | Linear SVM | LogReg - KNN |
|---|---|---|---|---|
| Moment màu (9 chiều) | 0.972 ± 0.008 | 0.908 ± 0.009 | 0.838 ± 0.011 | -0.064 |
| Hist HSV (512 chiều) | 0.965 ± 0.014 | 0.991 ± 0.005 | 0.989 ± 0.008 | +0.027 |
| HOG 64×128 (3780 chiều) | 0.946 ± 0.007 | 0.985 ± 0.004 | 0.982 ± 0.004 | +0.039 |

**Kết quả và kết luận.**

| # | Dự đoán | Kết quả đo | Kết luận |
|---|---|---|---|
| M1 | Histogram giữ nguyên khi xáo trộn; HOG giảm | Histogram RGB, HSV và moment: sụt đúng 0.000. HOG sụt 0.69 đến 0.70; cạnh sụt 0.66 đến 0.69 (còn 0.20 đến 0.29, mức đoán mò với 4 lớp) | Khớp |
| M2 | HSV bền với độ sáng hơn RGB; H-S bền nhất | H-S sụt ≤ 0.027. HSV đủ 3 kênh vẫn sụt 0.33 đến 0.67; RGB sụt 0.70 đến 0.87 | Khớp một phần: tách V ra chưa đủ, phải bỏ hẳn V khỏi đặc trưng |
| M3 | Ít bin kém; nhiều bin thưa và nhạy nhiễu | 4 bin/kênh kém rõ (0.914 đến 0.936). Hist RGB từ 8 đến 256 bin gần như ngang nhau (chênh < độ lệch chuẩn). Hist HSV 16³ = 4096 chiều: KNN giảm còn 0.917, LogReg vẫn 0.992 | Khớp vế "ít bin". Vế "nhiều bin" chỉ thấy ở KNN trên histogram 3 chiều (87% phần tử bằng 0), không thấy ở histogram từng kênh hay ở LogReg |
| M4 | Moment gọn nhưng thô | 9 chiều; KNN 0.972 ngang histogram, nhưng LogReg chỉ 0.908 và Linear SVM 0.838. Rút đặc trưng chậm hơn histogram (≈4 ms so với 0.2 ms mỗi ảnh) | Khớp với bộ phân lớp tuyến tính; trái dự đoán với KNN (không kém histogram) và về thời gian rút |
| M5 | CCV thêm thông tin không gian nên tốt hơn histogram, và nhạy với bố cục | Ảnh gốc: LogReg 0.975 so với 0.968 (chênh < độ lệch chuẩn); KNN 0.946 so với 0.966 (CCV kém hơn). Xáo trộn: CCV sụt 0.053 đến 0.082, histogram sụt ≤ 0.003 | Vế "nhạy với bố cục" khớp. Vế "tốt hơn histogram" không được xác nhận |
| M6 | Bỏ màu: đặc trưng màu mất tác dụng, cạnh và HOG gần như không đổi | Học lại trên ảnh xám: đặc trưng màu sụt 0.04 đến 0.23; cạnh và HOG đổi ≤ 0.005. HOG trên ảnh xám vẫn đạt 0.986 | Khớp. Bốn giống phân biệt được bằng hình dạng (HOG xám 0.986) lẫn bằng màu (hist HSV 0.991) |
| P1 | k nhỏ nhạy nhiễu, k lớn mất chi tiết cục bộ | Đường cong giảm đều theo k; k=1 tốt nhất, k=31 thấp nhất (0.867 đến 0.937) | Vế "k lớn" khớp. Vế "k nhỏ nhạy nhiễu" không xuất hiện trên dữ liệu này |
| P2 | KNN phụ thuộc độ đo và thang đo; sau chuẩn hóa đơn vị, Euclidean và cosine cùng thứ hạng | Cùng tập 10 láng giềng ở 100% ảnh truy vấn (trước chuẩn hóa: 11% với moment, 18% với hist HSV). Macro-F1 của Euclidean-đơn vị và cosine trùng nhau từng chữ số. Manhattan tốt nhất trên HOG (0.978 so với 0.954) | Khớp |
| P3 | Ưu tiên điểm gần giúp lớp ít ảnh | Chênh macro-F1 ≤ 0.006; recall Bangla Kola đổi trong khoảng -0.003 đến +0.026, đều nhỏ hơn hoặc xấp xỉ độ lệch chuẩn | Chưa đủ cơ sở phân biệt |
| P4 | Cây chỉ tăng tốc, không đổi kết quả, kém khi nhiều chiều | Kết quả trùng brute ở mọi trường hợp. 3780 chiều: K-D Tree chậm hơn brute khoảng 60 lần. 9 chiều với 1680 điểm: cây vẫn chậm hơn brute (3.2 ms so với 1.6 ms); từ 8400 điểm trở lên K-D Tree mới nhanh hơn (6.1 ms so với 22.3 ms) | Khớp, kèm điều kiện: cây chỉ có lợi khi ít chiều *và* tập tham chiếu đủ lớn |
| P5 | KNN không tốn huấn luyện nhưng dự đoán chậm; tuyến tính ngược lại | HOG: KNN huấn luyện 0.07 s, dự đoán 0.22 ms/ảnh; LogReg huấn luyện 1.2 s, dự đoán 0.02 ms/ảnh | Khớp |
| P6 | Biên tuyến tính yếu ở đặc trưng ít chiều | Moment (9 chiều): LogReg kém KNN 0.064. Hist HSV (512): hơn 0.027. HOG (3780): hơn 0.039 | Khớp |

Giải thích các điểm trái dự đoán:
- P1 (k=1 tốt nhất): các lớp tách nhau rõ trong không gian đặc trưng và mỗi quả thường được chụp nhiều lần ở tư thế gần giống nhau, nên láng giềng gần nhất hầu như luôn cùng lớp; tăng k chỉ kéo thêm điểm ở xa, phần nhiều thuộc lớp đông nhất (Champa Kola). **[lập luận]**; phù hợp với việc recall thấp nhất trong 4 lớp tụt mạnh khi k tăng (`results/P3/table.md`: HOG 0.816 ở k=5 xuống 0.697 ở k=15).
- M5 (CCV không hơn histogram): trên ảnh một quả chuối trên nền trơn, hầu hết điểm ảnh cùng màu đều nằm trong vùng liên thông lớn, nên phần "không liên kết" gần như rỗng và CCV mang ít thông tin mới, trong khi số chiều tăng gấp đôi. **[lập luận]**
- M4 (moment với KNN ngang histogram): 9 con số tóm tắt đủ tách 4 lớp nếu biên quyết định được phép cong (KNN), nhưng không tách tuyến tính được (LogReg). **[đo]** P6.

**Ảnh hưởng tới bước sau.**
- Đặc trưng màu tốt nhất: histogram HSV (8,8,8) + LogReg, 0.991 ± 0.005. Histogram H-S (8,8) gần bằng (0.985 ± 0.008), bền với độ sáng và chỉ 64 chiều; giữ cả hai làm ứng viên ghép với HOG ở C5.
- KNN ở bước chốt cấu hình sẽ quét k ∈ {1, 3, 5}, độ đo Euclidean/Manhattan/Hellinger thay vì cố định k=5.
- Hai đặc trưng màu và hình dạng đều đạt trên 0.98 một mình; cần S2 để biết phần của nền ảnh trong con số này.

## Bước 3: Đặc trưng cạnh, HOG, ghép đặc trưng (C1-C5) và chốt cấu hình truyền thống (12/11/2026)

**Mục tiêu.** Kiểm chứng các giả thuyết ở mục 6.3; chọn siêu tham số và chốt cấu hình tốt nhất của nhóm truyền thống cho từng họ đặc trưng (màu, cạnh, HOG, ghép).

**Đã làm.**
- `src/features/edge.py`: Roberts, Prewitt, Sobel, Laplacian; vector 512 chiều (tổng độ lớn gradient theo 256 hàng và 256 cột).
- `src/features/hog.py`: HOG theo các bước của bài giảng, có tùy chọn bỏ chuẩn hóa block, phiếu cứng/chia phiếu, ảnh xám/ảnh màu.
- `python -m src.exp_shape` (C1-C5): `results/C1` … `results/C5`, hình `results/figures/C*_*.png`.
- `python -m src.exp_select`: quét k ∈ {1, 3, 5} và độ đo (Euclidean, Manhattan, Hellinger) cho KNN; C cho Logistic Regression và Linear SVM; ghi `results/select_traditional/` (bảng, `selected.json`, confusion matrix out-of-fold của 4 cấu hình đã chốt).
- `src/trad_config.py`: một cấu hình truyền thống được mô tả bằng dict (đặc trưng, cách ghép, bộ phân lớp) để các bước sau và demo dựng lại đúng mô hình đã chốt.

**Lý do chọn cách này.**
- C1 thêm mức nhiễu σ = 2 ngoài ba mức 5, 10, 20 của kế hoạch. **[đo]** `results/C1/table.md`: từ σ = 10 mọi toán tử đều rơi về mức đoán mò (macro-F1 ≈ 0.09), nên ba mức gốc không phân biệt được các toán tử; ở σ = 2 mới thấy khác biệt (Prewitt/Sobel sụt 0.02 đến 0.23, Roberts 0.46 đến 0.67, Laplacian 0.60 đến 0.83).
- C1 đo thêm mức thay đổi tương đối của chính vector đặc trưng, không qua bộ phân lớp, để tách "độ nhạy của toán tử" khỏi "độ nhạy của bộ phân lớp". **[đo]** `results/C1/table_feature_change.md`.
- Ghép đặc trưng "chuẩn hóa từng phần": z-score từng chiều rồi chia mỗi phần cho căn bậc hai số chiều của phần đó, để phần màu (64-512 chiều) không bị phần HOG (3780 chiều) lấn át trong khoảng cách. **[đo]** `results/C5/table.md`: với KNN, Hist H-S + HOG tăng từ 0.948 (z-score chung) lên 0.977; Hist HSV + moment + HOG từ 0.945 lên 0.985.
- HOG giữ cấu hình của bài giảng (64×128, cell 8×8, 9 bin, 3780 chiều). **[đo]** `results/C3/table.md`: với LogReg, 9 trong 12 cấu hình nằm trong khoảng 0.982 đến 0.986, chênh nhỏ hơn độ lệch chuẩn (0.004 đến 0.007); không có cơ sở để chọn cấu hình khác.

**Phương án khác và vì sao không chọn.**
- HOG 64×128 với cell 16×16 (504-1008 chiều): không chọn, LogReg chỉ 0.959 đến 0.960 so với 0.985 của cell 8×8, chênh lớn hơn độ lệch chuẩn. **[đo]** `results/C3/table.md`.
- HOG 128×128 (8100 chiều): không chọn, 0.986 ± 0.004 so với 0.985 ± 0.004 của 64×128, chưa đủ cơ sở phân biệt, mà vector dài gấp 2.1 lần và rút đặc trưng lâu gấp 2.2 lần. **[đo]** `results/C3/table.md`.
- HOG trên ảnh màu (lấy kênh có gradient lớn nhất): 0.982 ± 0.005 so với 0.985 ± 0.004 trên ảnh xám; chưa đủ cơ sở phân biệt, giữ ảnh xám cho đơn giản. **[đo]** `results/C3/table_variants.md`.
- HOG phiếu cứng như skimage: LogReg ngang nhau (0.985), KNN kém hơn (0.928 so với 0.946); giữ cách chia phiếu cho hai bin gần nhất như bài giảng. **[đo]** `results/C3/table_variants.md`.
- Ghép thẳng rồi z-score chung: với LogReg ngang cách chuẩn hóa từng phần (0.989 đến 0.993 so với 0.991 đến 0.993), nhưng với KNN kém hơn rõ (xem trên). **[đo]** `results/C5/table.md`.
- Linear SVM với C cố định 0.01 trên đặc trưng ghép đã chuẩn hóa từng phần: tụt còn 0.959 đến 0.966. Nguyên nhân là phép chia cho căn số chiều làm đặc trưng nhỏ đi nên cùng một C thành điều chuẩn mạnh hơn; khi quét C, Linear SVM C=1 đạt 0.996. **[đo]** `results/select_traditional/table.md`.
- Chọn đặc trưng cạnh làm đại diện hình dạng thay cho HOG: không chọn; cấu hình cạnh tốt nhất 0.955 ± 0.015 so với HOG 0.986 ± 0.007, và sập hoàn toàn với nhiễu σ ≥ 5. **[đo]** `results/select_traditional/table_selected.md`, `results/C1/table.md`.

**Bằng chứng.** (macro-F1 trung bình ± độ lệch chuẩn qua 5 fold)

C1: toán tử cạnh và nhiễu (`results/C1/table.md`, `results/C1/table_feature_change.md`, hình `results/figures/C1_edge_noise.png`):

| Toán tử | Bộ phân lớp | Ảnh gốc | Sụt σ=2 | Sụt σ=5 | Sụt σ=10 | Sụt σ=20 |
|---|---|---|---|---|---|---|
| Roberts (2×2) | KNN (k=5) | 0.901 ± 0.019 | +0.461 ± 0.017 | +0.789 ± 0.019 | +0.813 ± 0.019 | +0.813 ± 0.019 |
| Roberts (2×2) | LogReg | 0.932 ± 0.011 | +0.669 ± 0.015 | +0.839 ± 0.007 | +0.844 ± 0.010 | +0.844 ± 0.010 |
| Prewitt (3×3) | KNN (k=5) | 0.890 ± 0.019 | +0.026 ± 0.020 | +0.523 ± 0.025 | +0.799 ± 0.014 | +0.803 ± 0.019 |
| Prewitt (3×3) | LogReg | 0.892 ± 0.007 | +0.198 ± 0.019 | +0.768 ± 0.013 | +0.805 ± 0.007 | +0.805 ± 0.007 |
| Sobel (3×3) | KNN (k=5) | 0.890 ± 0.019 | +0.023 ± 0.022 | +0.535 ± 0.021 | +0.800 ± 0.017 | +0.802 ± 0.019 |
| Sobel (3×3) | LogReg | 0.894 ± 0.008 | +0.226 ± 0.020 | +0.776 ± 0.013 | +0.806 ± 0.007 | +0.806 ± 0.007 |
| Laplacian (bậc hai) | KNN (k=5) | 0.886 ± 0.014 | +0.601 ± 0.008 | +0.797 ± 0.014 | +0.798 ± 0.014 | +0.798 ± 0.014 |
| Laplacian (bậc hai) | LogReg | 0.914 ± 0.015 | +0.826 ± 0.015 | +0.827 ± 0.015 | +0.827 ± 0.015 | +0.827 ± 0.015 |

| Toán tử | σ=2 | σ=5 | σ=10 | σ=20 |
|---|---|---|---|---|
| Roberts (2×2) | 0.35 | 1.00 | 2.14 | 4.47 |
| Prewitt (3×3) | 0.14 | 0.43 | 0.98 | 2.11 |
| Sobel (3×3) | 0.15 | 0.46 | 1.03 | 2.22 |
| Laplacian (bậc hai) | 0.78 | 2.16 | 4.55 | 9.37 |

C2: HOG và độ sáng (`results/C2/table.md`, `results/C2/table_feature_change.md`, `results/C2/table_saturation.md`, hình `results/figures/C2_hog_brightness.png`):

| Đặc trưng | Số chiều | Bộ phân lớp | Ảnh gốc | Sụt (V×0.6) | Sụt (V×1.4) |
|---|---|---|---|---|---|
| HOG, chuẩn hóa block L2 | 3780 | KNN (k=5) | 0.946 ± 0.007 | +0.009 ± 0.007 | +0.163 ± 0.008 |
| HOG, chuẩn hóa block L2 | 3780 | LogReg | 0.985 ± 0.004 | +0.004 ± 0.004 | +0.215 ± 0.033 |
| HOG, không chuẩn hóa block | 1152 | KNN (k=5) | 0.845 ± 0.015 | +0.394 ± 0.023 | -0.032 ± 0.011 |
| HOG, không chuẩn hóa block | 1152 | LogReg | 0.960 ± 0.005 | +0.080 ± 0.009 | +0.057 ± 0.006 |
| Hist RGB (3×32) | 96 | KNN (k=5) | 0.964 ± 0.012 | +0.873 ± 0.013 | +0.760 ± 0.024 |
| Hist RGB (3×32) | 96 | LogReg | 0.976 ± 0.013 | +0.763 ± 0.064 | +0.696 ± 0.036 |
| Hist HSV (8,8,8) | 512 | KNN (k=5) | 0.965 ± 0.014 | +0.663 ± 0.043 | +0.376 ± 0.025 |
| Hist HSV (8,8,8) | 512 | LogReg | 0.991 ± 0.005 | +0.673 ± 0.035 | +0.329 ± 0.009 |
| Hist H-S (8,8) | 64 | KNN (k=5) | 0.976 ± 0.006 | +0.000 ± 0.008 | +0.001 ± 0.002 |
| Hist H-S (8,8) | 64 | LogReg | 0.985 ± 0.008 | +0.027 ± 0.011 | +0.011 ± 0.004 |

| Đặc trưng | Thay đổi tương đối của vector (V×0.6) | Thay đổi tương đối của vector (V×1.4) |
|---|---|---|
| HOG, chuẩn hóa block L2 | 0.21 | 0.60 |
| HOG, không chuẩn hóa block | 0.40 | 0.28 |

| Ảnh | Tỉ lệ cell gần như không có gradient | Tỉ lệ điểm ảnh bão hòa (kênh lớn nhất = 255) |
|---|---|---|
| Ảnh gốc | 0.002 | 0.005 |
| V×0.6 | 0.003 | 0.000 |
| V×1.4 | 0.141 | 0.687 |

C3: tham số HOG (`results/C3/table.md`, `results/C3/table_variants.md`, hình `results/figures/C3_hog_params.png`):

| Kích thước ảnh (rộng×cao) | Cell | Số bin | Số chiều | Rút đặc trưng (ms/ảnh) | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|---|---|---|---|
| 64×128 | 8×8 | 6 | 2520 | 0.90 | 0.954 ± 0.007 | 0.982 ± 0.007 | 0.977 ± 0.003 |
| 64×128 | 8×8 | 9 | 3780 | 1.03 | 0.946 ± 0.007 | 0.985 ± 0.004 | 0.982 ± 0.004 |
| 64×128 | 8×8 | 12 | 5040 | 1.36 | 0.939 ± 0.011 | 0.985 ± 0.006 | 0.979 ± 0.006 |
| 64×128 | 16×16 | 6 | 504 | 0.83 | 0.920 ± 0.010 | 0.959 ± 0.006 | 0.959 ± 0.008 |
| 64×128 | 16×16 | 9 | 756 | 0.92 | 0.911 ± 0.010 | 0.959 ± 0.005 | 0.958 ± 0.009 |
| 64×128 | 16×16 | 12 | 1008 | 1.05 | 0.910 ± 0.008 | 0.960 ± 0.006 | 0.959 ± 0.009 |
| 128×128 | 8×8 | 6 | 5400 | 1.86 | 0.952 ± 0.013 | 0.986 ± 0.005 | 0.978 ± 0.006 |
| 128×128 | 8×8 | 9 | 8100 | 2.29 | 0.942 ± 0.017 | 0.986 ± 0.004 | 0.982 ± 0.007 |
| 128×128 | 8×8 | 12 | 10800 | 2.39 | 0.941 ± 0.013 | 0.985 ± 0.005 | 0.978 ± 0.010 |
| 128×128 | 16×16 | 6 | 1176 | 1.47 | 0.961 ± 0.007 | 0.983 ± 0.007 | 0.981 ± 0.005 |
| 128×128 | 16×16 | 9 | 1764 | 1.82 | 0.951 ± 0.014 | 0.982 ± 0.007 | 0.981 ± 0.011 |
| 128×128 | 16×16 | 12 | 2352 | 2.18 | 0.954 ± 0.011 | 0.984 ± 0.006 | 0.981 ± 0.010 |

| Biến thể | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|
| 64×128, cell 8, 9 bin, phiếu cứng (như skimage) | 0.928 ± 0.009 | 0.985 ± 0.004 | 0.976 ± 0.003 |
| 64×128, cell 8, 9 bin, ảnh màu (kênh gradient lớn nhất) | 0.944 ± 0.013 | 0.982 ± 0.005 | 0.977 ± 0.006 |

C4: che 25% diện tích (`results/C4/table.md`, `results/C4/table_hist_scaling.md`, hình `results/figures/C4_occlusion.png`):

| Đặc trưng | Loại | Bộ phân lớp | Ảnh gốc | Che 25% | Độ sụt |
|---|---|---|---|---|---|
| Hist RGB (3×32) | toàn cục | KNN (k=5) | 0.964 ± 0.012 | 0.085 ± 0.000 | +0.879 ± 0.012 |
| Hist RGB (3×32) | toàn cục | LogReg | 0.976 ± 0.013 | 0.085 ± 0.000 | +0.890 ± 0.014 |
| Hist HSV (8,8,8) | toàn cục | KNN (k=5) | 0.965 ± 0.014 | 0.077 ± 0.003 | +0.887 ± 0.013 |
| Hist HSV (8,8,8) | toàn cục | LogReg | 0.991 ± 0.005 | 0.222 ± 0.113 | +0.769 ± 0.117 |
| Hist H-S (8,8) | toàn cục | KNN (k=5) | 0.976 ± 0.006 | 0.843 ± 0.030 | +0.133 ± 0.029 |
| Hist H-S (8,8) | toàn cục | LogReg | 0.985 ± 0.008 | 0.762 ± 0.028 | +0.224 ± 0.021 |
| Moment màu | toàn cục | KNN (k=5) | 0.972 ± 0.008 | 0.312 ± 0.042 | +0.659 ± 0.043 |
| Moment màu | toàn cục | LogReg | 0.908 ± 0.009 | 0.227 ± 0.006 | +0.681 ± 0.010 |
| HOG 64×128 | cục bộ ghép lại | KNN (k=5) | 0.946 ± 0.007 | 0.784 ± 0.030 | +0.162 ± 0.028 |
| HOG 64×128 | cục bộ ghép lại | LogReg | 0.985 ± 0.004 | 0.794 ± 0.027 | +0.192 ± 0.026 |
| Cạnh Sobel | cục bộ ghép lại | KNN (k=5) | 0.890 ± 0.019 | 0.322 ± 0.006 | +0.568 ± 0.023 |
| Cạnh Sobel | cục bộ ghép lại | LogReg | 0.894 ± 0.008 | 0.272 ± 0.014 | +0.621 ± 0.017 |

| Đặc trưng | KNN (k=5), chuẩn hóa | Ảnh gốc | Che 25% | Độ sụt |
|---|---|---|---|---|
| Hist RGB (3×32) | z-score | 0.964 ± 0.012 | 0.085 ± 0.000 | +0.879 ± 0.012 |
| Hist RGB (3×32) | không chuẩn hóa | 0.952 ± 0.013 | 0.859 ± 0.024 | +0.093 ± 0.027 |
| Hist RGB (3×32) | Hellinger | 0.974 ± 0.011 | 0.870 ± 0.028 | +0.104 ± 0.026 |
| Hist HSV (8,8,8) | z-score | 0.965 ± 0.014 | 0.077 ± 0.003 | +0.887 ± 0.013 |
| Hist HSV (8,8,8) | không chuẩn hóa | 0.953 ± 0.006 | 0.920 ± 0.021 | +0.032 ± 0.019 |
| Hist HSV (8,8,8) | Hellinger | 0.976 ± 0.015 | 0.964 ± 0.013 | +0.012 ± 0.002 |
| Hist H-S (8,8) | z-score | 0.976 ± 0.006 | 0.843 ± 0.030 | +0.133 ± 0.029 |
| Hist H-S (8,8) | không chuẩn hóa | 0.947 ± 0.011 | 0.556 ± 0.103 | +0.391 ± 0.107 |
| Hist H-S (8,8) | Hellinger | 0.979 ± 0.005 | 0.605 ± 0.136 | +0.374 ± 0.138 |

C5: ghép đặc trưng (`results/C5/table.md`, hình `results/figures/C5_fusion.png`):

| Đặc trưng | Cách ghép | Số chiều | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|---|---|
| Hist HSV (8,8,8) | - | 512 | 0.965 ± 0.014 | 0.991 ± 0.005 | 0.989 ± 0.008 |
| Hist H-S (8,8) | - | 64 | 0.976 ± 0.006 | 0.985 ± 0.008 | 0.958 ± 0.007 |
| HOG 64×128 | - | 3780 | 0.946 ± 0.007 | 0.985 ± 0.004 | 0.982 ± 0.004 |
| Hist HSV + HOG | ghép thẳng, z-score chung | 4292 | 0.943 ± 0.008 | 0.993 ± 0.003 | 0.991 ± 0.003 |
| Hist HSV + HOG | chuẩn hóa từng phần rồi ghép | 4292 | 0.956 ± 0.012 | 0.993 ± 0.004 | 0.961 ± 0.007 |
| Hist H-S + HOG | ghép thẳng, z-score chung | 3844 | 0.948 ± 0.009 | 0.989 ± 0.003 | 0.989 ± 0.003 |
| Hist H-S + HOG | chuẩn hóa từng phần rồi ghép | 3844 | 0.977 ± 0.005 | 0.992 ± 0.003 | 0.966 ± 0.014 |
| Hist HSV + moment + HOG | ghép thẳng, z-score chung | 4301 | 0.945 ± 0.008 | 0.993 ± 0.003 | 0.992 ± 0.003 |
| Hist HSV + moment + HOG | chuẩn hóa từng phần rồi ghép | 4301 | 0.985 ± 0.009 | 0.991 ± 0.004 | 0.959 ± 0.007 |

Chốt cấu hình (`results/select_traditional/table.md`, `results/select_traditional/table_selected.md`):

| Họ | Đặc trưng | Cấu hình tốt nhất của bộ phân lớp | Số chiều | Macro-F1 | Accuracy | Huấn luyện (s) | Dự đoán (ms/ảnh) |
|---|---|---|---|---|---|---|---|
| màu | Hist HSV (8,8,8) | KNN k=1, Hellinger, ưu tiên điểm gần | 512 | 0.987 ± 0.006 | 0.989 ± 0.006 | 0.01 | 0.031 |
| màu | Hist HSV (8,8,8) | LogReg C=0.1 | 512 | 0.991 ± 0.005 | 0.992 ± 0.005 | 0.72 | 0.004 |
| màu | Hist HSV (8,8,8) | Linear SVM C=0.1 | 512 | 0.990 ± 0.004 | 0.991 ± 0.004 | 2.95 | 0.006 |
| màu | Hist H-S (8,8) | KNN k=1, Manhattan, ưu tiên điểm gần | 64 | 0.986 ± 0.004 | 0.986 ± 0.004 | 0.00 | 0.039 |
| màu | Hist H-S (8,8) | LogReg C=1 | 64 | 0.985 ± 0.008 | 0.986 ± 0.007 | 0.02 | 0.001 |
| màu | Hist H-S (8,8) | Linear SVM C=0.1 | 64 | 0.973 ± 0.004 | 0.977 ± 0.003 | 0.04 | 0.003 |
| màu | Moment màu | KNN k=1, Manhattan, ưu tiên điểm gần | 9 | 0.978 ± 0.004 | 0.981 ± 0.003 | 0.00 | 0.011 |
| màu | Moment màu | LogReg C=10 | 9 | 0.910 ± 0.011 | 0.926 ± 0.008 | 0.03 | 0.001 |
| màu | Moment màu | Linear SVM C=1 | 9 | 0.874 ± 0.007 | 0.894 ± 0.004 | 0.01 | 0.001 |
| cạnh | Cạnh Roberts | KNN k=1, Manhattan, ưu tiên điểm gần | 512 | 0.955 ± 0.015 | 0.959 ± 0.013 | 0.01 | 0.219 |
| cạnh | Cạnh Roberts | LogReg C=1 | 512 | 0.932 ± 0.011 | 0.940 ± 0.009 | 5.02 | 0.004 |
| cạnh | Cạnh Roberts | Linear SVM C=0.01 | 512 | 0.925 ± 0.014 | 0.937 ± 0.012 | 0.55 | 0.005 |
| cạnh | Cạnh Sobel | KNN k=1, Manhattan, ưu tiên điểm gần | 512 | 0.952 ± 0.014 | 0.956 ± 0.012 | 0.01 | 0.235 |
| cạnh | Cạnh Sobel | LogReg C=1 | 512 | 0.894 ± 0.008 | 0.900 ± 0.006 | 6.86 | 0.004 |
| cạnh | Cạnh Sobel | Linear SVM C=0.1 | 512 | 0.883 ± 0.014 | 0.895 ± 0.013 | 1.01 | 0.005 |
| HOG | HOG 64×128 | KNN k=1, Manhattan, ưu tiên điểm gần | 3780 | 0.983 ± 0.004 | 0.984 ± 0.003 | 0.07 | 1.892 |
| HOG | HOG 64×128 | LogReg C=0.1 | 3780 | 0.986 ± 0.007 | 0.989 ± 0.006 | 2.66 | 0.041 |
| HOG | HOG 64×128 | Linear SVM C=0.001 | 3780 | 0.983 ± 0.003 | 0.985 ± 0.002 | 1.98 | 0.023 |
| ghép | Hist HSV + HOG | KNN k=1, Manhattan, ưu tiên điểm gần, chuẩn hóa từng phần | 4292 | 0.989 ± 0.003 | 0.989 ± 0.002 | 0.12 | 2.204 |
| ghép | Hist HSV + HOG | LogReg C=10, chuẩn hóa từng phần | 4292 | 0.996 ± 0.003 | 0.997 ± 0.002 | 1.58 | 0.055 |
| ghép | Hist HSV + HOG | Linear SVM C=1, chuẩn hóa từng phần | 4292 | 0.996 ± 0.004 | 0.996 ± 0.003 | 1.18 | 0.027 |
| ghép | Hist H-S + HOG | KNN k=1, Manhattan, ưu tiên điểm gần, chuẩn hóa từng phần | 3844 | 0.990 ± 0.003 | 0.990 ± 0.003 | 0.07 | 1.918 |
| ghép | Hist H-S + HOG | LogReg C=100, chuẩn hóa từng phần | 3844 | 0.996 ± 0.003 | 0.997 ± 0.003 | 1.29 | 0.047 |
| ghép | Hist H-S + HOG | Linear SVM C=1, chuẩn hóa từng phần | 3844 | 0.996 ± 0.003 | 0.997 ± 0.003 | 1.03 | 0.024 |

| Vai trò | Cấu hình | Macro-F1 (5-fold) | Accuracy |
|---|---|---|---|
| trad_color | Hist HSV (8,8,8) + LogReg C=0.1 | 0.991 ± 0.005 | 0.992 |
| trad_edge | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.955 ± 0.015 | 0.959 |
| trad_hog | HOG 64×128 + LogReg C=0.1 | 0.986 ± 0.007 | 0.989 |
| trad_fusion | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.996 ± 0.003 | 0.997 |

**Kết quả và kết luận.**

| # | Dự đoán | Kết quả đo | Kết luận |
|---|---|---|---|
| C1 | Roberts nhạy nhiễu; Prewitt, Sobel có làm trơn nên bền hơn; Laplacian nhạy nhất | Thay đổi tương đối của vector ở σ = 5: Prewitt 0.43; Sobel 0.46; Roberts 1.00; Laplacian 2.16. Ở σ = 2, độ sụt macro-F1 theo đúng thứ tự đó | Khớp. Ngoài dự đoán: vector "tổng theo hàng/cột" rất kém bền, mọi toán tử đều sập từ σ = 10 |
| C2 | HOG chuẩn hóa block bền với thay đổi độ sáng; bỏ chuẩn hóa thì kém bền | V×0.6: có chuẩn hóa sụt 0.004 đến 0.009, không chuẩn hóa sụt 0.08 đến 0.39. V×1.4: có chuẩn hóa sụt 0.16 đến 0.22, không chuẩn hóa chỉ -0.03 đến +0.06 | Khớp khi làm tối. **Trái dự đoán** khi làm sáng |
| C3 | Tham số HOG quyết định mức chi tiết | Cell 16×16 trên ảnh 64×128 kém rõ (0.959). Các cấu hình còn lại ngang nhau với LogReg. Số bin (6, 9, 12) không tạo khác biệt vượt độ lệch chuẩn | Khớp một phần: chỉ kích thước cell so với kích thước ảnh là có tác dụng đo được |
| C4 | Histogram toàn cục sụt nhiều hơn HOG khi bị che | Với z-score: Hist RGB/HSV rơi về 0.08 đến 0.22; HOG còn 0.78 đến 0.79; Hist H-S còn 0.76 đến 0.84. Với Hellinger: Hist HSV chỉ sụt 0.012 | Khớp theo cấu hình mặc định, nhưng nguyên nhân chính là bước z-score chứ không phải tính "toàn cục" |
| C5 | Màu và hình dạng bổ sung cho nhau | Sau khi quét siêu tham số: ghép 0.996 ± 0.003; màu riêng 0.991 ± 0.005; HOG riêng 0.986 ± 0.007 | Ghép hơn HOG riêng 0.010 (lớn hơn độ lệch chuẩn); hơn màu riêng 0.005 (bằng độ lệch chuẩn, chưa đủ cơ sở phân biệt) |

Giải thích các điểm trái dự đoán:
- C2, V×1.4: phép nhân làm 68.7% điểm ảnh bão hòa ở 255 (ảnh gốc 0.5%) và 14.1% số cell mất hẳn gradient (ảnh gốc 0.2%). Đây không còn là phép đổi độ sáng tuyến tính mà chuẩn hóa block khử được. Chuẩn hóa L2 khuếch đại các block nền có gradient yếu lên độ dài đơn vị; khi nền bão hòa, các block đó về 0 nên vector đã chuẩn hóa đổi nhiều (thay đổi tương đối 0.60) hơn vector chưa chuẩn hóa (0.28). **[đo]** `results/C2/table_saturation.md`, `results/C2/table_feature_change.md`.
- C4: hình vuông đen dồn 25% khối lượng histogram vào một bin gần như luôn bằng 0 trên ảnh huấn luyện; z-score chia cho độ lệch chuẩn rất nhỏ của bin đó nên khoảng cách bị một chiều chi phối. Bỏ z-score hoặc dùng Hellinger thì Hist HSV chỉ sụt 0.03 và 0.012. Hist H-S thì ngược lại (z-score sụt 0.13, Hellinger 0.37) vì điểm đen có H = 0, S = 0 rơi vào cùng bin với nền xám trắng vốn đã lớn. **[đo]** `results/C4/table_hist_scaling.md`.

**Ảnh hưởng tới bước sau.**
- Bốn cấu hình truyền thống đã chốt (`results/select_traditional/selected.json`); tốt nhất là **Hist H-S + HOG + LogReg (C=100, chuẩn hóa từng phần)**, macro-F1 0.996 ± 0.003. Cấu hình này dùng cho I2, S2, so sánh ở N7, N8 và đánh giá cuối.
- Các cấu hình tốt nhất cách nhau ít hơn độ lệch chuẩn; trên tập test 371 ảnh, một ảnh sai tương ứng 0.27% accuracy, nên so sánh cuối cùng phải đọc kèm khoảng sai số.

## Bước 4-5: CNN hướng A, hướng B và so sánh hai hướng: N1-N8 (12/11/2026)

**Mục tiêu.** Huấn luyện ba mô hình hướng A theo bài giảng (`a_lenet`, `a_dropout`, `a_vgg16`) và hai mô hình hướng B lớn hơn (`b_mobilenetv3`, `b_resnet50`); kiểm chứng các giả thuyết N1-N8 (mục 6.5); chốt một biến thể cho mỗi mô hình để đánh giá cuối.

**Đã làm.**
- `src/cnn/models.py`: năm kiến trúc. Hướng A đúng mô tả của kế hoạch. Hướng B: thân ImageNet → GlobalAveragePooling → Dropout(0.2) → Softmax(4). Mọi mô hình nhận ảnh RGB 0-255; phần chuẩn hóa nằm trong mô hình. VGG16 và ResNet50 dùng lớp `Normalization` trừ trung bình ImageNet và đảo trục kênh của kernel tích chập đầu tiên (trọng số gốc học trên ảnh BGR). Cách này tương đương `preprocess_input`: sai khác lớn nhất ở đầu ra thân mạng là 1.5·10⁻⁴ (VGG16, giá trị lớn nhất 228.7) và 8.7·10⁻⁵ (ResNet50).
- `src/cnn/train.py`: Adam, cross-entropy, dừng sớm theo macro-F1 trên val (giữ trọng số tốt nhất), checkpoint mỗi epoch trong `checkpoints/<run>/`, `--resume`. Mô hình có thân đóng băng (`a_vgg16`, giai đoạn 1 của hướng B) tính trước đầu ra thân mạng một lần (`data/cache/cnnfeat_*.npy`) rồi chỉ huấn luyện phần đầu. Giai đoạn 2 của hướng B nạp đầu phân lớp của giai đoạn 1 rồi mở băng thân mạng.
- Các lệnh huấn luyện nằm ở bước 4 và 5 của `scripts/run_all.sh` (`bash scripts/run_all.sh 4 5`); kết quả mỗi lần ở `results/cnn/<run>/` (metrics.json, confusion matrix, history.json, curves.png). Bảng tổng hợp: `results/cnn/table_runs.md`.
- `python -m src.exp_cnn`: các phép đo N1-N8 (`results/N1` … `results/N8`, hình `results/figures/N*_*.png`).

**Lý do chọn cách này.**
- `a_lenet` dùng ảnh vào 64×64. **[đo]** `results/cnn/table_runs.md`: 32×32 cho 0.954; 64×64 cho 0.973; 128×128 cho 0.956 (macro-F1 trên val).
- `a_vgg16` dùng learning rate 1e-3 cho phần đầu. **[đo]** 0.990 so với 0.984 khi dùng 1e-4. Chênh 0.006 tương ứng khoảng 2 ảnh val, nên đây là lựa chọn theo val chứ chưa phải khác biệt chắc chắn.
- Khi fine-tune (giai đoạn 2), các lớp BatchNormalization luôn để đóng băng. **[đo]** Lần chạy đầu để BN mở: accuracy train ở epoch 1 chỉ 0.356 trong khi accuracy val 0.976. Thống kê theo lô làm lệch đầu ra thân mạng, và đầu phân lớp học ở giai đoạn 1 không còn khớp. Phép đo lặp lại được ở `results/N4/table_batchnorm.md`:

| Thân mạng | Cách mở băng | Thay đổi tương đối của embedding (training=True so với suy luận) | Accuracy đầu phân lớp GĐ1, embedding chế độ suy luận | Accuracy đầu phân lớp GĐ1, embedding chế độ huấn luyện |
|---|---|---|---|---|
| b_mobilenetv3 | BN mở (trainable=True) | 0.820 | 1.000 | 0.047 |
| b_mobilenetv3 | BN đóng băng (cách đồ án dùng) | 0.000 | 1.000 | 1.000 |
| b_resnet50 | BN mở (trainable=True) | 0.838 | 1.000 | 0.125 |
| b_resnet50 | BN đóng băng (cách đồ án dùng) | 0.000 | 1.000 | 1.000 |

- Biến thể chốt cho đánh giá cuối được chọn theo macro-F1 trên val: `a_lenet`, `a_dropout_aug`, `a_vgg16_lr1e-3`, `b_mobilenetv3_lr1e-4`, `b_resnet50` (`src/exp_cnn.py`, hàm `final_run`). **[đo]** `results/cnn/table_runs.md`. Lưu ý: val chỉ có 371 ảnh và các mô hình tốt nhất đều trên 0.99, nên con số trên val của biến thể được chọn là lạc quan; con số không thiên lệch là kết quả trên test ở bước 8.
- Thời gian dự đoán trong N6 đo bằng hàm đã biên dịch (`tf.function`) và đo khi máy không chạy tác vụ khác. **[đo]** Lần đo đầu gọi mô hình ở chế độ eager và đo trong lúc máy đang huấn luyện song song: MobileNetV3 ra 137 ms mỗi ảnh, gần bằng VGG16 (159 ms). Đó là chi phí điều phối từng lớp nhỏ chứ không phải chi phí tính toán; đo lại bằng hàm đã biên dịch còn 12.5 ms (`results/N6/metrics.json`).
- `b_resnet50` chạy được giai đoạn 2 (mở băng khối conv5, 15 triệu tham số) trên CPU: khoảng 61 s mỗi epoch khi máy rảnh, 8 epoch. **[đo]** `results/N6/table.md`.

**Phương án khác và vì sao không chọn.**
- Mở băng cả BatchNormalization khi fine-tune: không chọn (xem bảng trên). **[đo]**
- Giai đoạn 2 với lr 1e-3: không chọn. Macro-F1 tốt nhất 0.993, nhưng trong quá trình huấn luyện tụt xuống 0.917 và dao động mạnh. **[đo]** `results/N4/table.md`.
- Lưu `a_vgg16` vào `models/` cho demo: không làm được, file 370 MB vượt giới hạn 50 MB của kế hoạch. **[đo]** `results/N6/table.md`.
- Cross-validation cho CNN: không làm, đã quyết định ở bước 0.3. Thay vào đó, N3 lặp 3 seed khởi tạo để ước lượng nhiễu giữa các lần huấn luyện: độ lệch chuẩn 0.005 đến 0.006 trên val. **[đo]** `results/N3/table_seeds.md`.

**Bằng chứng.**

N1: số tham số (`results/N1/table.md`, `results/N1/table_totals.md`, hình `results/figures/N1_params.png`):

| Mô hình | Lớp | Công thức | Tính tay | model.summary() | Khớp | Ghi chú |
|---|---|---|---|---|---|---|
| a_lenet | conv1 | (5×5×3+1)×6 | 456 | 456 | có |  |
| a_lenet | conv2 | (5×5×6+1)×16 | 2416 | 2416 | có |  |
| a_lenet | fc1 | (2704+1)×120 | 324600 | 324600 | có | Flatten: 13×13×16 = 2704 |
| a_lenet | fc2 | (120+1)×84 | 10164 | 10164 | có |  |
| a_lenet | softmax | (84+1)×4 | 340 | 340 | có |  |
| a_lenet | Tổng |  | 337976 | 337976 | có | đầu vào 64×64×3 |
| a_dropout | conv1 | (5×5×3+1)×32 | 2432 | 2432 | có |  |
| a_dropout | conv2 | (5×5×32+1)×16 | 12816 | 12816 | có |  |
| a_dropout | fc1 | (13456+1)×128 | 1722496 | 1722496 | có | Flatten: 29×29×16 = 13456 |
| a_dropout | fc2 | (128+1)×64 | 8256 | 8256 | có |  |
| a_dropout | softmax | (64+1)×4 | 260 | 260 | có |  |
| a_dropout | Tổng |  | 1746260 | 1746260 | có | đầu vào 128×128×3 |

| Mô hình | Tổng số tham số | Tham số được huấn luyện |
|---|---|---|
| a_lenet | 337976 | 337976 |
| a_dropout | 1746260 | 1746260 |
| a_vgg16 | 40669252 | 25954564 |
| b_mobilenetv3 | 941428 | giai đoạn 1: 2308; giai đoạn 2: 917204 |
| b_resnet50 | 23595908 | giai đoạn 1: 8196; giai đoạn 2: 14961668 |

Tất cả các lần huấn luyện (`results/cnn/table_runs.md`):

| Run | Mô hình | Ảnh vào | Số ảnh train | lr | Tăng cường | Tham số huấn luyện | Epoch chọn / tổng | Macro-F1 val | Accuracy val | Accuracy train | Giây / epoch | Thời gian huấn luyện (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_dropout | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 25 / 40 | 0.940 | 0.946 | 0.999 | 4.2 | 168 |
| a_dropout_aug | a_dropout | 128×128 | 1729 | 0.001 | có | 1746260 | 25 / 40 | 0.948 | 0.954 | 0.967 | 4.8 | 194 |
| a_dropout_nodrop | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 33 / 40 | 0.951 | 0.957 | 1.000 | 3.8 | 150 |
| a_dropout_nodrop_seed43 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 18 / 40 | 0.960 | 0.968 | 1.000 | 5.8 | 234 |
| a_dropout_nodrop_seed44 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 37 / 40 | 0.947 | 0.951 | 1.000 | 6.0 | 242 |
| a_dropout_seed43 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 22 / 40 | 0.947 | 0.954 | 0.997 | 6.3 | 253 |
| a_dropout_seed44 | a_dropout | 128×128 | 1729 | 0.001 | không | 1746260 | 39 / 40 | 0.935 | 0.943 | 0.999 | 5.8 | 233 |
| a_lenet | a_lenet | 64×64 | 1729 | 0.001 | không | 337976 | 40 / 50 | 0.973 | 0.978 | 1.000 | 0.6 | 31 |
| a_lenet_frac25 | a_lenet | 64×64 | 432 | 0.001 | không | 337976 | 22 / 32 | 0.924 | 0.930 | 1.000 | 0.2 | 7 |
| a_lenet_frac50 | a_lenet | 64×64 | 864 | 0.001 | không | 337976 | 13 / 23 | 0.950 | 0.957 | 0.995 | 0.4 | 8 |
| a_lenet_s128 | a_lenet | 128×128 | 1729 | 0.001 | không | 1628216 | 13 / 23 | 0.956 | 0.962 | 1.000 | 1.7 | 40 |
| a_lenet_s32 | a_lenet | 32×32 | 1729 | 0.001 | không | 61496 | 20 / 30 | 0.954 | 0.960 | 0.984 | 0.3 | 9 |
| a_vgg16 | a_vgg16 | 224×224 | 1729 | 0.0001 | không | 25954564 | 7 / 15 | 0.984 | 0.987 | 1.000 | 3.9 | 59 |
| a_vgg16_frac25 | a_vgg16 | 224×224 | 432 | 0.0001 | không | 25954564 | 3 / 11 | 0.967 | 0.973 | 0.995 | 1.2 | 13 |
| a_vgg16_frac50 | a_vgg16 | 224×224 | 864 | 0.0001 | không | 25954564 | 3 / 11 | 0.963 | 0.968 | 1.000 | 2.1 | 23 |
| a_vgg16_lr1e-3 | a_vgg16 | 224×224 | 1729 | 0.001 | không | 25954564 | 16 / 24 | 0.990 | 0.992 | 0.999 | 3.9 | 94 |
| a_vgg16_lr1e-3_cw | a_vgg16 | 224×224 | 1729 | 0.001 | không | 25954564 | 8 / 16 | 0.987 | 0.989 | 0.999 | 4.1 | 66 |
| a_vgg16_lr1e-3_minaug | a_vgg16 | 224×224 | 2780 | 0.001 | không | 25954564 | 9 / 17 | 0.994 | 0.995 | 0.999 | 6.1 | 104 |
| a_vgg16_lr1e-3_os | a_vgg16 | 224×224 | 2780 | 0.001 | không | 25954564 | 4 / 12 | 0.987 | 0.989 | 0.997 | 6.1 | 74 |
| b_mobilenetv3 | b_mobilenetv3 | 224×224 | 1729 | 1e-05 | có | 917204 | 12 / 17 | 0.993 | 0.995 | 0.997 | 20.1 | 341 |
| b_mobilenetv3_frac25 | b_mobilenetv3 | 224×224 | 432 | 1e-05 | có | 917204 | 12 / 17 | 0.975 | 0.978 | 0.998 | 7.3 | 124 |
| b_mobilenetv3_frac50 | b_mobilenetv3 | 224×224 | 864 | 1e-05 | có | 917204 | 2 / 7 | 0.974 | 0.978 | 0.994 | 16.2 | 114 |
| b_mobilenetv3_lr1e-3 | b_mobilenetv3 | 224×224 | 1729 | 0.001 | có | 917204 | 4 / 9 | 0.993 | 0.995 | 0.998 | 26.3 | 236 |
| b_mobilenetv3_lr1e-4 | b_mobilenetv3 | 224×224 | 1729 | 0.0001 | có | 917204 | 7 / 12 | 1.000 | 1.000 | 1.000 | 26.9 | 323 |
| b_mobilenetv3_lr1e-4_cw | b_mobilenetv3 | 224×224 | 1729 | 0.0001 | có | 917204 | 8 / 13 | 1.000 | 1.000 | 1.000 | 16.3 | 212 |
| b_mobilenetv3_lr1e-4_minaug | b_mobilenetv3 | 224×224 | 2780 | 0.0001 | có | 917204 | 5 / 10 | 1.000 | 1.000 | 0.999 | 25.0 | 250 |
| b_mobilenetv3_lr1e-4_os | b_mobilenetv3 | 224×224 | 2780 | 0.0001 | có | 917204 | 3 / 8 | 1.000 | 1.000 | 0.998 | 28.6 | 229 |
| b_mobilenetv3_noaug | b_mobilenetv3 | 224×224 | 1729 | 1e-05 | không | 917204 | 4 / 9 | 0.997 | 0.997 | 0.999 | 23.1 | 208 |
| b_mobilenetv3_s1 | b_mobilenetv3 | 224×224 | 1729 | 0.001 | không | 2308 | 21 / 31 | 0.987 | 0.989 | 0.997 | 0.1 | 4 |
| b_mobilenetv3_s1_frac25 | b_mobilenetv3 | 224×224 | 432 | 0.001 | không | 2308 | 41 / 51 | 0.974 | 0.978 | 1.000 | 0.1 | 3 |
| b_mobilenetv3_s1_frac50 | b_mobilenetv3 | 224×224 | 864 | 0.001 | không | 2308 | 17 / 27 | 0.980 | 0.984 | 0.993 | 0.1 | 2 |
| b_resnet50 | b_resnet50 | 224×224 | 1729 | 1e-05 | có | 14961668 | 3 / 8 | 0.998 | 0.997 | 1.000 | 91.2 | 730 |
| b_resnet50_s1 | b_resnet50 | 224×224 | 1729 | 0.001 | không | 8196 | 10 / 20 | 0.996 | 0.997 | 1.000 | 0.2 | 4 |

N2: lượng dữ liệu (`results/N2/table.md`, hình `results/figures/N2_learning_curve.png`):

| Mô hình | 25% train (432 ảnh) | 50% train (864 ảnh) | 100% train (1729 ảnh) | Tăng từ 25% lên 100% |
|---|---|---|---|---|
| a_lenet (LeNet, từ đầu) | 0.924 | 0.950 | 0.973 | +0.050 |
| a_vgg16 (đóng băng) | 0.967 | 0.963 | 0.984 | +0.016 |
| b_mobilenetv3 (fine-tune) | 0.975 | 0.974 | 0.993 | +0.018 |

N3: Dropout (`results/N3/table.md`, `results/N3/table_seeds.md`, hình `results/figures/N3_dropout.png`):

| Cấu hình | Epoch chọn / tổng | Macro-F1 val | Accuracy train (epoch chọn) | Accuracy val (epoch chọn) | Loss train (epoch cuối) | Loss val (epoch cuối) | Khoảng cách loss val - train (epoch cuối) | Loss val thấp nhất |
|---|---|---|---|---|---|---|---|---|
| a_dropout có Dropout(0.5) | 25 / 40 | 0.940 | 0.999 | 0.946 | 0.0022 | 0.5444 | 0.5422 | 0.2713 |
| a_dropout không Dropout | 33 / 40 | 0.951 | 1.000 | 0.957 | 0.0000 | 0.3486 | 0.3486 | 0.1584 |

| Cấu hình | Số lần chạy | Macro-F1 val từng seed | Macro-F1 val (TB ± ĐLC) | Khoảng cách loss val - train ở epoch 40 (TB ± ĐLC) | Loss val thấp nhất (TB ± ĐLC) |
|---|---|---|---|---|---|
| Có Dropout(0.5) | 3 | 0.940, 0.947, 0.935 | 0.941 ± 0.005 | 0.456 ± 0.061 | 0.248 ± 0.021 |
| Không Dropout | 3 | 0.951, 0.960, 0.947 | 0.953 ± 0.006 | 0.376 ± 0.081 | 0.202 ± 0.055 |

N4: đóng băng và fine-tune (`results/N4/table.md`, hình `results/figures/N4_finetune_lr.png`):

| Cấu hình | Tham số được huấn luyện | Macro-F1 val | Accuracy val | Loss val (epoch chọn) | Macro-F1 val sau epoch 1 | Macro-F1 val thấp nhất | Epoch chọn / tổng | Thời gian mỗi epoch (s) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small, giai đoạn 1 (đóng băng thân) | 2308 | 0.987 | 0.989 | 0.0505 | 0.872 | 0.872 | 21 / 31 | 0.1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-5 | 917204 | 0.993 | 0.995 | 0.0203 | 0.981 | 0.978 | 12 / 17 | 20.1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-4 | 917204 | 1.000 | 1.000 | 0.0045 | 0.974 | 0.974 | 7 / 12 | 26.9 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-3 | 917204 | 0.993 | 0.995 | 0.0184 | 0.964 | 0.917 | 4 / 9 | 26.3 |
| ResNet50, giai đoạn 1 (đóng băng thân) | 8196 | 0.996 | 0.997 | 0.0269 | 0.942 | 0.942 | 10 / 20 | 0.2 |
| ResNet50, giai đoạn 2 (mở băng conv5), lr = 1e-5 | 14961668 | 0.998 | 0.997 | 0.0140 | 0.984 | 0.984 | 3 / 8 | 91.2 |

N5: tăng cường ảnh (`results/N5/table.md`, hình `results/figures/N5_augment_a_dropout.png`, `results/figures/N5_augment_b_mobilenetv3.png`):

| Cấu hình | Epoch chọn / tổng | Macro-F1 val | Accuracy train (epoch chọn) | Accuracy val (epoch chọn) | Loss train (epoch cuối) | Loss val (epoch cuối) | Khoảng cách loss val - train (epoch cuối) | Loss val thấp nhất |
|---|---|---|---|---|---|---|---|---|
| a_dropout, không tăng cường | 25 / 40 | 0.940 | 0.999 | 0.946 | 0.0022 | 0.5444 | 0.5422 | 0.2713 |
| a_dropout, có tăng cường | 25 / 40 | 0.948 | 0.967 | 0.954 | 0.2613 | 0.3150 | 0.0536 | 0.1286 |
| b_mobilenetv3 giai đoạn 2, không tăng cường | 4 / 9 | 0.997 | 0.999 | 0.997 | 0.0045 | 0.0107 | 0.0062 | 0.0107 |
| b_mobilenetv3 giai đoạn 2, có tăng cường | 12 / 17 | 0.993 | 0.997 | 0.995 | 0.0125 | 0.0183 | 0.0058 | 0.0183 |

N6: hướng A so với hướng B (`results/N6/table.md`, hình `results/figures/N6_a_vs_b.png`):

| Hướng | Mô hình | Biến thể được chốt | Macro-F1 val | Accuracy val | Số tham số | File mô hình (MB) | Thời gian huấn luyện (s) | Dự đoán 1 ảnh (ms, CPU) | Dự đoán theo lô 32 (ms/ảnh) | Cách tính thời gian huấn luyện |
|---|---|---|---|---|---|---|---|---|---|---|
| A | a_lenet | a_lenet | 0.973 | 0.978 | 337976 | 4.1 | 30 | 1.4 | 0.2 | 50 epoch × 0.6 s |
| A | a_dropout | a_dropout_aug | 0.948 | 0.954 | 1746260 | 21.0 | 192 | 2.5 | 0.7 | 40 epoch × 4.8 s |
| A | a_vgg16 | a_vgg16_lr1e-3 | 0.990 | 0.992 | 40669252 | 370.4 | 226 | 130.4 | 67.5 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 129 s; đầu phân lớp 24 epoch × 4.0 s |
| B | b_mobilenetv3 | b_mobilenetv3_lr1e-4 | 1.000 | 1.000 | 941428 | 11.7 | 216 | 12.5 | 2.6 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 25 s; giai đoạn 1: 31 epoch, 4 s; giai đoạn 2: 12 epoch × 15.6 s |
| B | b_resnet50 | b_resnet50 | 0.998 | 0.997 | 23595908 | 214.8 | 591 | 61.1 | 28.0 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 98 s; giai đoạn 1: 20 epoch, 4 s; giai đoạn 2: 8 epoch × 61.2 s |

N7: embedding (`results/N7/table.md`, hình `results/figures/N7_embedding.png`):

| Đặc trưng | Loại | Số chiều | KNN (k=5) | LogReg |
|---|---|---|---|---|
| Hist HSV (8,8,8) | thủ công | 512 | 0.982 | 0.997 |
| HOG 64×128 | thủ công | 3780 | 0.945 | 0.987 |
| Hist H-S + HOG (ghép, chuẩn hóa từng phần) | thủ công | 3844 | 0.990 | 0.998 |
| VGG16 ImageNet, trung bình 7×7 (chưa học trên chuối) | embedding | 512 | 0.932 | 0.986 |
| a_vgg16, lớp Dense(256) áp chót | embedding | 256 | 0.986 | 0.986 |
| MobileNetV3 ImageNet, sau GAP (chưa fine-tune) | embedding | 576 | 0.981 | 0.993 |
| b_mobilenetv3_lr1e-4, sau GAP (đã fine-tune) | embedding | 576 | 1.000 | 0.996 |

N8: độ bền (`results/N8/table.md`, hình `results/figures/N8_robustness.png`):

| Mô hình | Ảnh gốc | Xáo trộn 4×4 | V×0.6 | V×1.4 | Ảnh xám | Nhiễu σ=5 | Nhiễu σ=10 | Nhiễu σ=20 | Che 25% | Xoay -30° | Xoay -15° | Xoay +15° | Xoay +30° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống: Hist HSV (8,8,8) | 0.997 | 0.997 | 0.280 | 0.705 | 0.145 | 0.451 | 0.085 | 0.085 | 0.076 | 0.987 | 0.993 | 0.992 | 0.994 |
| Truyền thống: HOG 64×128 | 0.983 | 0.302 | 0.984 | 0.800 | 0.981 | 0.537 | 0.461 | 0.318 | 0.801 | 0.679 | 0.929 | 0.886 | 0.550 |
| Truyền thống: Hist H-S + HOG | 0.998 | 0.898 | 0.995 | 0.888 | 0.770 | 0.706 | 0.180 | 0.101 | 0.918 | 0.978 | 0.987 | 0.988 | 0.974 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | 0.990 | 0.097 | 0.984 | 0.969 | 0.953 | 0.981 | 0.976 | 0.932 | 0.850 | 0.888 | 0.973 | 0.980 | 0.906 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | 1.000 | 0.850 | 0.913 | 0.907 | 0.963 | 0.997 | 0.987 | 0.953 | 0.659 | 0.993 | 1.000 | 0.997 | 0.996 |
| Hướng B, lr 1e-5, có tăng cường (N5): b_mobilenetv3 | 0.993 | 0.775 | 0.899 | 0.936 | 0.914 | 0.980 | 0.972 | 0.949 | 0.506 | 0.987 | 0.990 | 0.993 | 0.993 |
| Hướng B, lr 1e-5, không tăng cường (N5): b_mobilenetv3_noaug | 0.997 | 0.717 | 0.952 | 0.911 | 0.959 | 0.977 | 0.955 | 0.908 | 0.555 | 0.958 | 0.977 | 0.993 | 0.978 |

**Kết quả và kết luận.**

| # | Dự đoán | Kết quả đo | Kết luận |
|---|---|---|---|
| N1 | Số tham số tính được từ kiến trúc | 10/10 lớp của `a_lenet` và `a_dropout` khớp đúng `model.summary()`. Lớp Dense đầu tiên chiếm 96% (`a_lenet`: 324600/337976) và 99% (`a_dropout`: 1722496/1746260) số tham số | Khớp |
| N2 | Mô hình học từ đầu cần nhiều dữ liệu hơn mô hình dùng transfer learning | Từ 25% lên 100% train: `a_lenet` tăng 0.050; `a_vgg16` tăng 0.016; `b_mobilenetv3` tăng 0.018. Với 25% train, hai mô hình pretrained (0.967; 0.975) đã hơn `a_lenet` dùng 100% train (0.973) hoặc ngang | Khớp. Đường của hai mô hình pretrained không đơn điệu (50% không hơn 25%); chênh 0.004 ≈ 1-2 ảnh val, nằm trong nhiễu |
| N3 | Dropout giảm quá khớp | Qua 3 seed: có Dropout 0.941 ± 0.005, không Dropout 0.953 ± 0.006; khoảng cách loss val - train ở epoch 40: 0.456 ± 0.061 so với 0.376 ± 0.081 | **Trái dự đoán**: Dropout(0.5) làm kém đi 0.012, lớn hơn độ lệch chuẩn giữa các seed |
| N4 | Fine-tune thích nghi tốt hơn đóng băng; lr lớn phá trọng số đã học | MobileNetV3: giai đoạn 1 đạt 0.987; giai đoạn 2 đạt 0.993 (lr 1e-5), 1.000 (lr 1e-4), 0.993 (lr 1e-3). Với lr 1e-3, macro-F1 tụt xuống 0.917 ở epoch 2 và dao động tới epoch cuối; lr 1e-5 không xuống dưới 0.978. ResNet50: 0.996 → 0.998 | Khớp vế đầu (mức tăng nhỏ vì giai đoạn 1 đã cao). Vế lr: lr 1e-3 làm quá trình huấn luyện mất ổn định, nhưng nhờ dừng sớm theo val nên kết quả cuối không kém |
| N5 | Tăng cường ảnh giảm quá khớp | `a_dropout`: khoảng cách loss val - train giảm từ 0.542 xuống 0.054, macro-F1 0.940 → 0.948. `b_mobilenetv3` (lr 1e-5): 0.997 không tăng cường so với 0.993 có tăng cường. Trên ảnh xoay ±30° (bảng N8), bản có tăng cường giữ 0.987 đến 0.993, bản không tăng cường 0.958 đến 0.978 | Khớp với mô hình học từ đầu. Với mô hình pretrained, quá khớp vốn đã nhỏ (khoảng cách 0.006) nên tăng cường không tăng điểm trên val, nhưng làm mô hình bền hơn với phép xoay |
| N6 | Mạng lớn, pretrained tốt hơn nhưng tốn tài nguyên hơn | Hướng B: 1.000 (MobileNetV3) và 0.998 (ResNet50); hướng A: 0.948 đến 0.990. Dự đoán 1 ảnh: `a_lenet` 1.4 ms, MobileNetV3 12.5 ms, ResNet50 61.1 ms, `a_vgg16` 130.4 ms | Vế "tốt hơn" khớp. Vế "tốn tài nguyên" chỉ đúng với ResNet50: MobileNetV3 (0.94 triệu tham số, 11.7 MB) nhỏ hơn và dự đoán nhanh hơn `a_vgg16` (40.7 triệu tham số, 370 MB) gấp khoảng 10 lần |
| N7 | Embedding do mạng học tốt hơn đặc trưng thủ công | KNN trên embedding MobileNetV3 đã fine-tune: 1.000; LogReg: 0.996. Đặc trưng thủ công tốt nhất (H-S + HOG) + LogReg: 0.998. Embedding ImageNet chưa học trên chuối (VGG16, trung bình 7×7) + KNN: 0.932, kém hist HSV + KNN (0.982) | Khớp với KNN. Với LogReg, embedding và đặc trưng thủ công tốt nhất ngang nhau. Embedding chỉ vượt trội sau khi được fine-tune trên chính dữ liệu này |
| N8 | CNN bền với biến đổi hơn đặc trưng thủ công | Xem bảng N8 | Khớp với nhiễu, xoay, đổi độ sáng. **Trái dự đoán** với che 25% (MobileNetV3 còn 0.659, kém HOG 0.801 và H-S + HOG 0.918) và với xáo trộn 4×4 (`a_vgg16` còn 0.097) |

Giải thích các điểm trái dự đoán:
- N3: lớp Dropout đặt sau Dense(128), còn 1.72 triệu tham số của lớp Dense ngay sau Flatten (99% mô hình) vẫn khớp tập train đến accuracy 0.999 ở cả hai cấu hình. Dropout chỉ làm nhiễu 128 đầu ra của lớp đó, nên không chặn được quá khớp ở nơi nó xảy ra mà lại làm quá trình tối ưu ồn hơn. **[lập luận]** Tăng cường ảnh (N5) tác động vào đầu vào nên giảm quá khớp rõ hơn nhiều: khoảng cách 0.054 so với 0.542. **[đo]**
- N8, che 25%: hình vuông đen là thứ chưa từng xuất hiện trong ảnh huấn luyện, và mạng không có cơ chế chuẩn hóa cục bộ như block của HOG. Bộ ghép H-S + HOG có hai phần độc lập nên khi phần màu bị lệch thì phần HOG còn giữ được. **[lập luận]**
- N8, xáo trộn 4×4: `a_vgg16` dùng Flatten trên bản đồ 7×7×512 nên mỗi trọng số gắn với một vị trí cố định; xáo trộn ô làm sai toàn bộ vị trí. MobileNetV3 dùng GlobalAveragePooling (lấy trung bình theo vị trí) nên ít nhạy hơn: 0.850 so với 0.097. **[lập luận]**, khớp số liệu đo.

**Ảnh hưởng tới bước sau.**
- Năm biến thể chốt dùng cho đánh giá cuối: `a_lenet`, `a_dropout_aug`, `a_vgg16_lr1e-3`, `b_mobilenetv3_lr1e-4`, `b_resnet50`.
- I2 dùng `a_vgg16_lr1e-3` (hướng A) và `b_mobilenetv3_lr1e-4` (hướng B) làm mô hình gốc.
- Demo dùng `b_mobilenetv3` (11.7 MB) thay cho `a_vgg16` (370 MB).

## Bước 6-7: Mất cân bằng dữ liệu (I1-I2) và phân đoạn tách nền (S1-S2) (12/11/2026)

**Mục tiêu.** Kiểm chứng I1, I2 (mục 6.6) và S1, S2 (mục 6.7): accuracy có che khuất lớp ít ảnh không; các cách xử lý mất cân bằng tác động thế nào; mô hình dựa vào quả chuối hay vào nền ảnh.

**Đã làm.**
- `python -m src.exp_imbalance`: I1 trên các mô hình chưa xử lý mất cân bằng (out-of-fold 5-fold cho nhóm truyền thống, val cho CNN). I2 thử bốn cách: không xử lý, class weight, oversampling (`RandomOverSampler`, lặp ảnh lớp ít), tăng cường ảnh cho lớp ít (lật/xoay ảnh của các lớp chưa bằng lớp đông nhất). Áp dụng trên mô hình tốt nhất của nhóm truyền thống (trong từng fold, chỉ lấy mẫu lại phần huấn luyện), của hướng A (`a_vgg16_lr1e-3`) và của hướng B (`b_mobilenetv3_lr1e-4`), mỗi CNN huấn luyện lại 3 lần (`--class-weight`, `--oversample`, `--minority-aug`; lệnh ở bước 6 của `scripts/run_all.sh`).
- `src/segmentation.py`: ngưỡng toàn cục (T = 127), Otsu trên ảnh xám, ngưỡng thích nghi (Gauss, cửa sổ 51), k-means màu Lab (k = 2), Otsu trên khoảng cách màu tới màu nền ở viền ảnh, và hậu xử lý (đóng/mở hình thái, giữ vùng liên thông lớn nhất, lấp lỗ).
- `python -m src.exp_segment`: S1 (hình định tính và các chỉ số không cần nhãn mask trên 400 ảnh), S2 (đặc trưng trên toàn ảnh / chỉ vùng chuối / chỉ vùng nền, các phép thử chỉ dùng nền, và CNN dự đoán trên ảnh đã bỏ quả chuối).

**Lý do chọn cách này.**
- I1 thêm các mô hình yếu (KNN k = 31, moment màu, cạnh) và một bộ phân lớp "luôn đoán lớp đông nhất". Các mô hình tốt nhất có accuracy trên 0.99 nên không thể hiện được hiện tượng accuracy che khuất lớp kém. **[đo]** `results/I1/table.md`.
- I2 thêm một cấu hình truyền thống yếu hơn (cạnh Sobel + LogReg) để minh họa. Ở mô hình tốt nhất, mọi recall đã ≥ 0.995 nên không còn chỗ cho các cách xử lý tác động. **[đo]** `results/I2/table.md`.
- S1 chọn **Otsu trên khoảng cách màu tới nền + hậu xử lý** làm mask cho S2. **[đo]** `results/S1/table.md`: so với các phương pháp không hậu xử lý, đây là cách cho ít mảnh rời nhất (trung vị 1) và ít lỗ thủng nhất (0.010). Ngưỡng toàn cục và ngưỡng thích nghi tách cả vết thâm trên vỏ thành nền (lỗ thủng 0.085 và 0.282). Trên hình `results/figures/S1_segmentation.png`, phương pháp này giữ được cả quả chuối chín sẫm, chỗ mà Otsu trên ảnh xám bị thủng.
- Phiên bản "chỉ vùng nền" lấy nền **bên ngoài mask đã nới rộng 12 điểm ảnh**, rồi lấp vùng chuối bằng inpainting (Telea). **[đo]** Lần chạy đầu tô vùng chuối bằng một màu phẳng; trên hình `results/figures/S2_versions.png` vẫn thấy rõ đường viền hình bóng quả, và với một ảnh Bangla Kola sẫm màu, mask chưa phủ hết quả. Sau khi sửa, gradient dọc viền vùng đã lấp chỉ bằng 1.18 lần (trung vị) gradient của vùng nền xa quả (`results/S2/table_leak.md`).
- S2 thêm phép thử **chỉ dùng 4 góc ảnh 64×64**, không phụ thuộc mask, để loại hẳn nghi vấn còn sót hình bóng. **[đo]** `results/S2/table_corners.md`.

**Phương án khác và vì sao không chọn.**
- K-means màu (k = 2) làm mask cho S2: không chọn; còn lỗ thủng (0.027) và mảnh rời ở quả có vết thâm lớn. **[đo]** `results/S1/table.md`, `results/figures/S1_segmentation.png`.
- Tính IoU/Dice với nhãn mask: chưa làm được, vì dữ liệu không có nhãn mask và nhóm chưa tự gán. Cột "IoU với phương pháp được chọn" trong `results/S1/table.md` chỉ đo mức giống nhau giữa các phương pháp, không phải độ chính xác. **[lập luận]**
- Chọn một cách xử lý mất cân bằng cho cấu hình cuối: không chọn cách nào. Không cách nào cải thiện vượt độ lệch chuẩn (truyền thống) hoặc vượt 1-2 ảnh val (CNN); giữ cấu hình "không xử lý". **[đo]** `results/I2/table.md`.

**Bằng chứng.**

I1 (`results/I1/table.md`, hình `results/figures/I1_accuracy_vs_macro_f1.png`):

| Mô hình (chưa xử lý mất cân bằng) | Chấm trên | Accuracy | Macro-F1 | Accuracy - Macro-F1 | Recall Bangla | Recall Champa | Recall Sabri | Recall Sagor | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Luôn đoán lớp đông nhất (Champa Kola) | train+val | 0.402 | 0.143 | +0.259 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 (Bangla) |
| HOG 64×128 + KNN k=31 | out-of-fold 5-fold | 0.884 | 0.871 | +0.014 | 0.899 | 1.000 | 0.988 | 0.552 | 0.552 (Sagor) |
| Hist HSV + KNN k=31 | out-of-fold 5-fold | 0.883 | 0.868 | +0.015 | 0.870 | 0.998 | 0.986 | 0.576 | 0.576 (Sagor) |
| Moment màu + LogReg | out-of-fold 5-fold | 0.925 | 0.908 | +0.017 | 0.870 | 0.992 | 0.986 | 0.785 | 0.785 (Sagor) |
| Cạnh Sobel + LogReg | out-of-fold 5-fold | 0.900 | 0.894 | +0.007 | 0.822 | 0.920 | 0.977 | 0.857 | 0.822 (Bangla) |
| Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | out-of-fold 5-fold | 0.959 | 0.955 | +0.004 | 0.950 | 0.995 | 0.988 | 0.870 | 0.870 (Sagor) |
| HOG 64×128 + LogReg C=0.1 | out-of-fold 5-fold | 0.989 | 0.986 | +0.002 | 0.968 | 0.998 | 0.993 | 0.984 | 0.968 (Bangla) |
| Hist HSV (8,8,8) + LogReg C=0.1 | out-of-fold 5-fold | 0.992 | 0.991 | +0.001 | 0.992 | 0.998 | 0.998 | 0.978 | 0.978 (Sagor) |
| Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | out-of-fold 5-fold | 0.997 | 0.996 | +0.000 | 0.995 | 0.996 | 1.000 | 0.996 | 0.995 (Bangla) |
| CNN a_lenet_frac25 | val | 0.930 | 0.924 | +0.006 | 0.879 | 0.953 | 0.961 | 0.899 | 0.879 (Bangla) |
| CNN a_lenet | val | 0.978 | 0.973 | +0.005 | 1.000 | 0.993 | 0.987 | 0.924 | 0.924 (Sagor) |
| CNN a_dropout | val | 0.946 | 0.940 | +0.006 | 0.924 | 0.987 | 0.987 | 0.848 | 0.848 (Sagor) |
| CNN a_vgg16_lr1e-3 | val | 0.992 | 0.990 | +0.002 | 0.985 | 1.000 | 0.987 | 0.987 | 0.985 (Bangla) |
| CNN b_mobilenetv3_s1 | val | 0.989 | 0.987 | +0.002 | 0.970 | 1.000 | 1.000 | 0.975 | 0.970 (Bangla) |
| CNN b_mobilenetv3_lr1e-4 | val | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| CNN b_resnet50_s1 | val | 0.997 | 0.996 | +0.001 | 0.985 | 1.000 | 1.000 | 1.000 | 0.985 (Bangla) |
| CNN b_resnet50 | val | 0.997 | 0.998 | -0.000 | 1.000 | 1.000 | 1.000 | 0.987 | 0.987 (Sagor) |

I2 (`results/I2/table.md`, precision/recall đủ 4 lớp ở `results/I2/table_per_class.md`, hình `results/figures/I2_imbalance.png`):

| Mô hình | Cách xử lý | Chấm | Macro-F1 | Accuracy | Precision Bangla (ít nhất) | Recall Bangla (ít nhất) | Precision Champa (đông nhất) | Recall Champa (đông nhất) | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Không xử lý | 5-fold CV | 0.996 ± 0.003 | 0.997 | 0.995 | 0.995 | 0.999 | 0.996 | 0.995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Class weight | 5-fold CV | 0.996 ± 0.003 | 0.997 | 0.995 | 0.995 | 0.999 | 0.996 | 0.995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Oversampling | 5-fold CV | 0.995 ± 0.005 | 0.996 | 0.992 | 0.992 | 0.999 | 0.996 | 0.992 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Tăng cường lớp ít | 5-fold CV | 0.996 ± 0.002 | 0.996 | 0.992 | 0.997 | 0.999 | 0.995 | 0.993 (Sagor) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Không xử lý | 5-fold CV | 0.894 ± 0.008 | 0.900 | 0.849 | 0.822 | 0.921 | 0.920 | 0.822 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Class weight | 5-fold CV | 0.888 ± 0.007 | 0.895 | 0.823 | 0.828 | 0.925 | 0.902 | 0.828 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Oversampling | 5-fold CV | 0.888 ± 0.005 | 0.895 | 0.821 | 0.825 | 0.926 | 0.905 | 0.825 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Tăng cường lớp ít | 5-fold CV | 0.884 ± 0.011 | 0.892 | 0.776 | 0.857 | 0.957 | 0.892 | 0.852 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Không xử lý | val | 0.990 | 0.992 | 0.970 | 0.985 | 1.000 | 1.000 | 0.985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Class weight | val | 0.987 | 0.989 | 0.970 | 0.985 | 0.993 | 1.000 | 0.975 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Oversampling | val | 0.987 | 0.989 | 0.956 | 0.985 | 1.000 | 0.993 | 0.985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Tăng cường lớp ít | val | 0.994 | 0.995 | 1.000 | 1.000 | 0.993 | 1.000 | 0.987 (Sabri) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Không xử lý | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Class weight | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Oversampling | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Tăng cường lớp ít | val | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (Bangla) |

S1 (`results/S1/table.md`, hình `results/figures/S1_segmentation.png`):

| Phương pháp | Số mảnh rời (trung bình) | Số mảnh rời (trung vị) | Tỉ lệ lỗ thủng trong vật | Tỉ lệ diện tích vật | IoU với phương pháp được chọn |
|---|---|---|---|---|---|
| Ngưỡng toàn cục (T=127) | 11.6 | 9 | 0.085 | 0.105 | 0.689 |
| Otsu (ảnh xám) | 2.9 | 2 | 0.052 | 0.149 | 0.799 |
| Ngưỡng thích nghi | 16.9 | 14 | 0.282 | 0.081 | 0.561 |
| K-means màu (k=2) | 2.5 | 1 | 0.027 | 0.140 | 0.859 |
| Otsu trên khoảng cách màu tới nền | 2.5 | 1 | 0.010 | 0.145 | 0.971 |
| Otsu khoảng cách màu + hậu xử lý (dùng cho S2) | 1.0 | 1 | 0.000 | 0.144 | 1.000 |

S2: đặc trưng thủ công, 5-fold CV trên train+val (`results/S2/table.md`, hình `results/figures/S2_background_bias.png`, `results/figures/S2_versions.png`):

| Đặc trưng + bộ phân lớp | Phiên bản ảnh | Macro-F1 | Accuracy |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Toàn ảnh | 0.991 ± 0.005 | 0.992 ± 0.005 |
| HOG 64×128 + LogReg C=0.1 | Toàn ảnh | 0.986 ± 0.007 | 0.989 ± 0.006 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Toàn ảnh | 0.996 ± 0.003 | 0.997 ± 0.003 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng chuối | 0.964 ± 0.010 | 0.967 ± 0.008 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng chuối | 0.884 ± 0.010 | 0.888 ± 0.009 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng chuối | 0.958 ± 0.005 | 0.961 ± 0.005 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng nền | 0.933 ± 0.008 | 0.943 ± 0.008 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng nền | 0.980 ± 0.009 | 0.983 ± 0.008 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng nền | 0.986 ± 0.005 | 0.988 ± 0.005 |

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| HOG của 4 góc ảnh (128×128) + LogReg C=0.1 | 8100 | 0.968 ± 0.008 | 0.973 ± 0.007 |
| Hist HSV (8,8,8) của 4 góc ảnh + LogReg C=0.1 | 512 | 0.939 ± 0.009 | 0.948 ± 0.008 |

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| Màu Lab trung bình của nền (3 số) + KNN k=5 | 3 | 0.937 ± 0.011 | 0.948 ± 0.011 |
| Màu Lab trung bình của nền (3 số) + LogReg | 3 | 0.618 ± 0.027 | 0.693 ± 0.018 |
| Tỉ lệ diện tích quả chuối trong ảnh (1 số) + KNN k=5 | 1 | 0.619 ± 0.022 | 0.642 ± 0.016 |
| HOG của mask nhị phân (chỉ hình bóng) + LogReg C=0.1 | 3780 | 0.857 ± 0.017 | 0.861 ± 0.014 |

| Lớp | Nền: L trung bình ± std | Nền: a | Nền: b | Tỉ lệ diện tích quả chuối |
|---|---|---|---|---|
| Bangla Kola | 201.1 ± 13.7 | 128.1 ± 1.1 | 128.3 ± 2.1 | 0.136 |
| Champa Kola | 190.2 ± 11.7 | 127.2 ± 0.4 | 123.6 ± 0.9 | 0.164 |
| Sabri Kola | 207.4 ± 12.4 | 127.3 ± 1.0 | 119.9 ± 10.4 | 0.054 |
| Sagor Kola | 196.1 ± 20.1 | 128.2 ± 0.7 | 128.2 ± 2.3 | 0.220 |

S2: CNN (`results/S2/table_cnn.md`):

| Mô hình (học trên ảnh đầy đủ) | Toàn ảnh | Chỉ vùng chuối | Chỉ vùng nền |
|---|---|---|---|
| a_vgg16_lr1e-3 | 0.990 | 0.952 | 0.792 |
| b_mobilenetv3_lr1e-4 | 1.000 | 0.927 | 0.683 |
| b_resnet50 | 0.998 | 0.926 | 0.725 |

**Kết quả và kết luận.**

| # | Dự đoán | Kết quả đo | Kết luận |
|---|---|---|---|
| I1 | Accuracy có thể cao trong khi lớp ít ảnh bị nhận kém | Đoán toàn lớp đông nhất: accuracy 0.402, macro-F1 0.143, recall 3 lớp còn lại bằng 0. KNN k = 31 trên HOG: accuracy 0.884 nhưng recall Sagor Kola chỉ 0.552. Ở mô hình tốt, accuracy - macro-F1 ≤ 0.003 | Khớp. Lớp bị nhận kém nhất thường là **Sagor Kola** (524 ảnh) chứ không phải lớp ít ảnh nhất (Bangla Kola, 444 ảnh); độ khó của lớp quan trọng hơn số lượng |
| I2 | Mỗi cách xử lý tác động khác nhau lên precision và recall của lớp ít | Truyền thống tốt nhất: cả bốn cách trong khoảng 0.995 đến 0.996 ± 0.002 đến 0.005. Cạnh Sobel: tăng cường lớp ít nâng recall Bangla 0.822 → 0.857 nhưng hạ precision Bangla 0.849 → 0.776, macro-F1 giảm 0.010. `a_vgg16`: tăng cường lớp ít 0.994 so với 0.990 (khoảng 1 ảnh val). `b_mobilenetv3`: cả bốn cách đều 1.000 trên val | Khớp ở mô hình yếu: các cách xử lý đổi precision lấy recall của lớp ít. Ở các mô hình mạnh không đo được tác động. Với tỉ lệ mất cân bằng 2.2 lần, mất cân bằng không phải vấn đề chính của bộ dữ liệu này |
| S1 | Ngưỡng toàn cục chỉ tốt khi nền và vật tách biệt về mức xám; Otsu tự chọn ngưỡng; k-means gom theo màu | Ngưỡng toàn cục: 11.6 mảnh rời mỗi ảnh, thủng ở vết thâm. Otsu xám: 2.9 mảnh, vẫn thủng ở quả sẫm. Ngưỡng thích nghi: chỉ bắt được viền và vân, 16.9 mảnh. K-means màu và Otsu trên khoảng cách màu: 2.5 mảnh, ít lỗ thủng nhất | Khớp. Nền trơn sáng nên các cách dựa trên màu tốt hơn các cách dựa trên mức xám |
| S2 | Nếu "chỉ nền" vẫn cho kết quả cao thì dữ liệu thiên lệch theo bối cảnh chụp | Chỉ nền: Hist HSV 0.933, HOG 0.980, H-S + HOG 0.986. Chỉ 4 góc ảnh: HOG 0.968, hist HSV 0.939. Chỉ 3 con số màu nền + KNN: 0.937. CNN học trên ảnh đầy đủ, đoán trên ảnh đã xóa quả: 0.683 đến 0.792 (mức đoán mò ≈ 0.25) | **Có thiên lệch theo bối cảnh chụp.** Mỗi lớp được chụp trong điều kiện nền, ánh sáng và khoảng cách riêng: nền Sabri Kola xanh hơn (b = 119.9 so với 123.6 đến 128.3), quả Sabri chiếm 5.4% diện tích ảnh so với 22.0% ở Sagor. Đây là hạn chế lớn nhất của đồ án |

Ý nghĩa của S2 đối với các kết quả trước:
- Con số macro-F1 0.99 đến 1.00 của các mô hình tốt nhất là độ chính xác **trên bộ dữ liệu này**, không phải độ chính xác nhận diện giống chuối nói chung. Khi chỉ nhìn quả chuối (nền tô màu phẳng), cấu hình truyền thống tốt nhất còn 0.958 và CNN còn 0.926 đến 0.952. Các con số này gần với khả năng phân biệt giống bằng chính quả chuối hơn, nhưng vẫn chưa loại hết thiên lệch: kích thước quả trong ảnh còn phụ thuộc khoảng cách chụp, và kích thước một mình đã đạt 0.619.
- Giải thích lại một số kết quả trước: HOG trên ảnh xám vẫn đạt 0.986 (M6) một phần nhờ kết cấu và độ sáng của nền; histogram H-S bền với độ sáng (M2) nhưng vẫn mang màu nền. **[lập luận]**, dựa trên số liệu S2.
- Muốn đánh giá đúng khả năng nhận diện giống thì cần ảnh chụp trong cùng điều kiện cho cả 4 giống, hoặc một tập test chụp ở bối cảnh khác. Ghi vào phần hạn chế và hướng phát triển.

**Ảnh hưởng tới bước sau.**
- Cấu hình cuối không thêm bước xử lý mất cân bằng.
- Đánh giá cuối báo cáo kèm cảnh báo về thiên lệch nền; phần phân tích lỗi xem các ảnh sai có liên quan tới nền hoặc kích thước quả không.
- Demo ghi rõ mô hình chỉ đáng tin với ảnh chụp kiểu bộ dữ liệu (một quả, nền trơn).

## Bước 8: Đánh giá cuối trên tập test, phân tích lỗi, tổng hợp (12/11/2026)

**Mục tiêu.** Dùng tập test (371 ảnh) đúng một lần cho các cấu hình đã chốt; phân tích ảnh bị đoán sai; gom mọi kết quả vào `results/summary.csv` và `results/summary.md`; xuất mô hình cho demo.

**Đã làm.**
- `python -m src.final_eval`: 4 cấu hình truyền thống (`results/select_traditional/selected.json`, học lại trên train+val) và 5 CNN (biến thể chốt ở bước 4-5, học trên train) dự đoán trên test. Mỗi mô hình có `results/final/<tên>/` (metrics.json, classification_report.txt, confusion matrix) kèm khoảng tin cậy 95% bằng bootstrap (2000 lần lấy mẫu lại ảnh test). Dự đoán lưu ở `results/final/test_predictions.npz`.
- `python -m src.error_analysis`: phân tích lỗi từ dự đoán đã lưu, không chạy lại mô hình.
- `python -m src.summarize`: `results/summary.csv` (718 dòng, mọi cấu hình và điều kiện của 30 thí nghiệm) và `results/summary.md`.
- `python -m src.export_demo`: `models/b_mobilenetv3.keras` (11.7 MB), `models/a_lenet.keras` (4.1 MB), `models/trad_best.joblib` (0.15 MB) và `models/models.json`. Demo `app/app.py` chạy được cả trên dòng lệnh lẫn giao diện web (ảnh chụp `results/figures/demo_screenshot.png`).

**Lý do chọn cách này.**
- Mô hình đại diện mỗi nhóm (dùng để so sánh từng cặp và phân tích lỗi) được chọn theo điểm lúc phát triển (5-fold CV hoặc val), không theo điểm test. **[lập luận]**: chọn theo test thì phần so sánh sẽ thiên lệch. Đại diện: `trad_fusion`, `a_vgg16`, `b_mobilenetv3`.
- Báo cáo khoảng tin cậy bootstrap vì test chỉ có 371 ảnh: một ảnh sai ứng với 0.27% accuracy. **[lập luận]**

**Phương án khác và vì sao không chọn.**
- Chạy lại mọi phép biến đổi kiểm chứng (M1, M2, N8…) trên test: không làm; các thí nghiệm đó đã chấm trên fold giữ lại và val như quyết định ở bước 1, nên test chỉ dùng cho kết quả chính. **[lập luận]**

**Bằng chứng.**

Kết quả chính trên test (`results/final/table.md`, hình `results/figures/final_test_macro_f1.png`):

| Nhóm | Mô hình | Macro-F1 test | KTC 95% (bootstrap) | Accuracy test | Số ảnh sai / 371 | Recall thấp nhất | Macro-F1 lúc chọn (5-fold CV hoặc val) |
|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.994 | [0.984; 1.000] | 0.995 | 2 | 0.985 (Bangla) | 0.991 ± 0.005 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.964 | [0.944; 0.982] | 0.968 | 12 | 0.885 (Sagor) | 0.955 ± 0.015 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.990 | [0.977; 1.000] | 0.992 | 3 | 0.985 (Bangla) | 0.986 ± 0.007 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 1.000 | [1.000; 1.000] | 1.000 | 0 | 1.000 (Bangla) | 0.996 ± 0.003 |
| CNN hướng A | a_lenet (biến thể: a_lenet) | 0.978 | [0.960; 0.993] | 0.981 | 7 | 0.955 (Bangla) | 0.973 |
| CNN hướng A | a_dropout (biến thể: a_dropout_aug) | 0.942 | [0.917; 0.965] | 0.943 | 21 | 0.872 (Sagor) | 0.948 |
| CNN hướng A | a_vgg16 (biến thể: a_vgg16_lr1e-3) | 0.986 | [0.971; 0.997] | 0.989 | 4 | 0.970 (Bangla) | 0.990 |
| CNN hướng B | b_mobilenetv3 (biến thể: b_mobilenetv3_lr1e-4) | 1.000 | [1.000; 1.000] | 1.000 | 0 | 1.000 (Bangla) | 1.000 |
| CNN hướng B | b_resnet50 (biến thể: b_resnet50) | 0.997 | [0.988; 1.000] | 0.997 | 1 | 0.987 (Sabri) | 0.998 |

So sánh từng cặp giữa đại diện các nhóm (`results/final/table_pairs.md`):

| Mô hình 1 | Mô hình 2 | Cả hai sai | Chỉ mô hình 1 sai | Chỉ mô hình 2 sai | Cả hai đúng |
|---|---|---|---|---|---|
| trad_fusion | a_vgg16 | 0 | 0 | 4 | 367 |
| trad_fusion | b_mobilenetv3 | 0 | 0 | 0 | 371 |
| a_vgg16 | b_mobilenetv3 | 0 | 4 | 0 | 367 |

Ảnh gần trùng và kết quả trên test (`results/final/table_near_duplicates.md`):

| Ngưỡng Hamming | Mô hình | Ảnh test có ảnh gần trùng trong train+val | Accuracy trên nhóm đó | Ảnh test không có | Accuracy trên nhóm không có |
|---|---|---|---|---|---|
| ≤ 2 | trad_fusion | 9 | 1.000 | 362 | 1.000 |
| ≤ 2 | a_vgg16 | 9 | 1.000 | 362 | 0.989 |
| ≤ 2 | b_mobilenetv3 | 9 | 1.000 | 362 | 1.000 |
| ≤ 4 | trad_fusion | 50 | 1.000 | 321 | 1.000 |
| ≤ 4 | a_vgg16 | 50 | 1.000 | 321 | 0.988 |
| ≤ 4 | b_mobilenetv3 | 50 | 1.000 | 321 | 1.000 |

Phân tích lỗi (`results/final/table_error_pairs.md`, `results/final/table_error_context.md`, `results/final/table_hard_images.md`, `results/final/errors.csv`, hình `results/figures/final_errors_all.png`, `results/figures/final_errors_a_vgg16.png`):

| Thực tế | Đoán thành | Số lượt sai (cộng trên 9 mô hình) |
|---|---|---|
| Sagor Kola | Champa Kola | 16 |
| Bangla Kola | Sagor Kola | 8 |
| Sagor Kola | Bangla Kola | 7 |
| Champa Kola | Sagor Kola | 5 |
| Bangla Kola | Sabri Kola | 4 |
| Sabri Kola | Bangla Kola | 4 |
| Sabri Kola | Sagor Kola | 3 |
| Champa Kola | Bangla Kola | 2 |
| Bangla Kola | Champa Kola | 1 |

| Lớp | Số ảnh test | Ảnh bị ≥ 1 mô hình đoán sai | Diện tích quả TB (ảnh đúng hết) | Diện tích quả TB (ảnh bị sai) |
|---|---|---|---|---|
| Bangla Kola | 67 | 9 | 0.127 | 0.142 |
| Champa Kola | 149 | 7 | 0.166 | 0.153 |
| Sabri Kola | 77 | 2 | 0.054 | 0.099 |
| Sagor Kola | 78 | 23 | 0.244 | 0.181 |

| Ảnh | Lớp đúng | Số mô hình sai / 9 | Các mô hình sai → lớp đoán |
|---|---|---|---|
| data/raw/sabri_kola/Sabri_Kola_5.jpg | Sabri Kola | 6 | trad_edge→Sagor; trad_hog→Sagor; a_lenet→Bangla; a_dropout→Sagor; a_vgg16→Bangla; b_resnet50→Bangla |
| data/raw/bangla_kola/Bangla_Kola_189.jpg | Bangla Kola | 3 | a_lenet→Sagor; a_dropout→Sagor; a_vgg16→Sagor |
| data/raw/bangla_kola/Bangla_Kola_422.jpg | Bangla Kola | 2 | trad_hog→Sagor; a_dropout→Sagor |
| data/raw/bangla_kola/Bangla_Kola_195.jpg | Bangla Kola | 2 | a_lenet→Sagor; a_dropout→Sagor |

**Kết quả và kết luận.**
- Hai cấu hình đạt 371/371 ảnh test: **Hist H-S + HOG + LogReg** (truyền thống) và **`b_mobilenetv3`** (hướng B). Khoảng tin cậy bootstrap của cả hai là [1.000; 1.000]. Theo quy tắc ba, tỉ lệ lỗi thật khi 0/371 ảnh sai có cận trên 95% xấp xỉ 3/371 ≈ 0.8%. **[lập luận]**
- Hai mô hình này không phân biệt được bằng tập test. Khoảng tin cậy của `b_resnet50` (0.997), Hist HSV + LogReg (0.994), HOG + LogReg (0.990) và `a_vgg16` (0.986) cũng chồng lên nhau. Chỉ kết luận được rằng `a_dropout` (0.942), cạnh Roberts + KNN (0.964) và `a_lenet` (0.978) kém hơn nhóm dẫn đầu.
- Điểm trên test bằng hoặc cao hơn một chút so với lúc chọn (CV/val) ở hầu hết cấu hình: không thấy dấu hiệu quá khớp vào tập chọn siêu tham số. Ngoại lệ là `a_dropout` (0.948 → 0.942), `a_vgg16` (0.990 → 0.986) và `b_resnet50` (0.998 → 0.997), cả ba đều giảm trong khoảng 1-2 ảnh.
- Ảnh gần trùng không làm phồng kết quả: trên 321 ảnh test không có bản gần trùng (Hamming ≤ 4) trong train+val, hai mô hình dẫn đầu vẫn đúng 100%.
- Lỗi tập trung ở **Sagor Kola** (23/78 ảnh test bị ít nhất 1 mô hình đoán sai) và chủ yếu bị nhầm thành Champa Kola. Ảnh Sagor bị sai có quả nhỏ hơn trung bình của lớp (diện tích 0.181 so với 0.244). Ảnh bị nhiều mô hình sai nhất (`Sabri_Kola_5.jpg`, 6/9 mô hình) là một quả Sabri to trên nền trắng, không giống ảnh Sabri điển hình (quả nhỏ, nền xanh nhạt). Cả hai quan sát khớp với thiên lệch theo kích thước quả và nền đã đo ở S2.

**Ảnh hưởng tới bước sau.**
- Báo cáo, slide và README dùng `results/final/table.md` làm bảng kết quả chính, luôn kèm cảnh báo của S2: điểm gần tuyệt đối có một phần đến từ bối cảnh chụp.
- Demo dùng `b_mobilenetv3` làm mặc định; có thêm `a_lenet` và cấu hình truyền thống tốt nhất để so sánh.

## Bước 9: Demo, báo cáo, slide, tài liệu hỏi đáp, README và kiểm tra tái lập (12/11/2026)

**Mục tiêu.** Hoàn thiện các sản phẩm phải nộp theo mục 3 và mục 8 của kế hoạch; bảo đảm người khác chạy lại được toàn bộ mà không phải làm thêm gì.

**Đã làm.**
- `report/report_template.md` → `python -m src.build_report` → `report/report.md` và `report/report.docx`. Báo cáo viết lại từ nhật ký theo đúng 7 mục của kế hoạch; mỗi thí nghiệm trình bày theo trình tự tính chất → dự đoán → kết quả đo → kết luận. Bảng số liệu không gõ tay: chỗ `{{T:đường_dẫn}}` trong file mẫu được thay bằng bảng trong `results/`. Bảng thành viên để trống.
- `python -m src.hand_calc`: số liệu cho ba ví dụ tính tay (Precision/Recall từ Confusion Matrix, khoảng cách giữa hai vector moment màu, số tham số lớp tích chập) trong `results/hand_calc/`.
- `python -m src.make_slides` → `slides/bao-cao-do-an.pptx`: 28 slide, cả 28 slide có ghi chú thuyết trình; bảng trên slide đọc trực tiếp từ `results/`.
- `docs/hoi-dap-bao-ve.md`: 36 câu hỏi, mỗi câu dẫn tới bảng hoặc hình cụ thể.
- `README.md`: cách cài đặt, cách chạy lại (cả 9 bước hoặc từng bước), demo, bảng kết quả chính, cảnh báo về thiên lệch nền.
- `requirements.txt`: ghim đúng phiên bản thư viện đã dùng để sinh `results/`.
- Thuộc tính tác giả của file docx và pptx: "Nhóm đồ án CS231".

**Lý do chọn cách này.**
- Báo cáo và slide được sinh bằng code từ `results/` thay vì chép số bằng tay, để khi chạy lại thí nghiệm thì số trong báo cáo tự cập nhật và không lệch với kết quả. **[lập luận]**
- Ghim phiên bản thư viện. **[đo]** `pip install --dry-run -r requirements.txt` giải được đầy đủ trên Python 3.13.

**Phương án khác và vì sao không chọn.**
- Xuất báo cáo PDF: không làm; kế hoạch cho phép pdf hoặc docx, nhóm chọn docx để còn điền bảng thành viên. **[lập luận]**

**Bằng chứng.** Kiểm tra tái lập: chạy lại `bash scripts/run_all.sh 1 2 3` từ đầu rồi so từng dòng `summary_rows.json` với bản đã commit.

| Phép kiểm tra | Kết quả |
|---|---|
| Số dòng cấu hình được so (macro-F1, độ lệch chuẩn, accuracy) | 718 |
| Số dòng khác biệt | 0 |
| `data/splits/`, `results/select_traditional/selected.json` | không đổi |
| Phần thay đổi sau khi chạy lại | chỉ các cột thời gian (huấn luyện, dự đoán, rút đặc trưng) |
| Demo dòng lệnh (`python app/app.py --image …`) với 3 mô hình | chạy được, đoán đúng ảnh ví dụ |
| Demo web (`python app/app.py`) | HTTP 200; ảnh chụp `results/figures/demo_screenshot.png` |

**Kết quả và kết luận.** Các sản phẩm theo mục 3 đã đủ. Phần truyền thống tái lập chính xác từng con số. Phần CNN cố định seed và bật chế độ tính toán tất định của TensorFlow, nhưng chưa chạy lại toàn bộ để đối chiếu (mất khoảng 2 giờ); trên máy khác hoặc phiên bản thư viện khác, số liệu CNN có thể lệch nhẹ.

**Ảnh hưởng tới bước sau.** Khi thêm thí nghiệm mới, sửa `report/report_template.md` và `src/make_slides.py` rồi chạy lại bước 9 của `scripts/run_all.sh`.

## Bước 10: Chuẩn bị phần mở rộng: ba nhánh độc lập, bộ kiểm tra khó, nhánh G (12/11/2026)

**Mục tiêu.** Thêm ba nhánh so sánh độc lập với pipeline gốc (mục 12 của kế hoạch): X: xử lý dữ liệu do nhóm thiết kế; T: bản Augmented của tác giả bộ dữ liệu; B: phương pháp của bài báo đã công bố trên bộ dữ liệu này. Bước này dựng phần dùng chung để bốn nhánh (kể cả nhánh gốc G) so được với nhau.

**Đã làm.**
- Sửa kế hoạch trước khi viết code: thêm mục 12 (ba nhánh, quy tắc chung, bộ kiểm tra khó) và sửa mục 8 (slide chính là Beamer LaTeX style UIT).
- `src/stress.py`: bộ kiểm tra khó gồm 6 phiên bản biến đổi của tập chấm: nền hoán đổi, V×0.6, V×1.4, nhiễu Gauss σ = 10, xoay +30°, che 25%. Hình minh họa: `results/figures/E_stress_examples.png` (`python -m src.stress`).
- `src/pipeline.py`: chạy 9 cấu hình đã chốt của bước 8 trên một bộ dữ liệu bất kỳ (ảnh train/val/test dạng mảng), cùng siêu tham số; ghi `results/<nhánh>/<mô hình>/` và `table.md`. `python -m src.pipeline` chấm lại nhánh G trên test và bộ kiểm tra khó (dùng lại các CNN đã huấn luyện ở bước 4-5).
- `src/segmentation.py`: thêm mask theo **mô hình nền bậc hai** (`banana_mask_v2`): khớp một mặt bậc hai theo (x, y) cho từng kênh Lab của nền, bắt đầu từ vành ngoài ảnh, rồi tách ngưỡng Otsu trên khoảng cách màu tới mặt đó.
- `scripts/download_data.sh augmented`: tải bản Augmented của tác giả (kiểm tra SHA-256), không commit vào repo (`data/aug_author/` nằm trong `.gitignore`).

**Lý do chọn cách này.**
- Cần bộ kiểm tra khó vì trên ảnh gốc, hai cấu hình của nhánh G đã đúng 371/371 ảnh test; điểm trên ảnh gốc không còn chỗ để một nhánh khác thể hiện tốt hơn hay kém hơn. **[đo]** `results/final/table.md`.
- Phép thử "nền hoán đổi" (ghép quả chuối lên nền lấy từ ảnh của một lớp khác) đo trực tiếp điều S2 đã phát hiện: mô hình dựa vào nền tới mức nào. **[đo]** bảng dưới: mọi cấu hình của nhánh G đều tụt, cấu hình truyền thống tốt nhất từ 1.000 xuống 0.410.
- Mỗi nhánh chỉ đổi một thứ so với G và chấm trên cùng 371 ảnh test gốc, để chênh lệch đo được quy về đúng thứ đã đổi. **[lập luận]**
- Phải thay mask của bước S1. **[đo]** Mask "Otsu trên khoảng cách màu tới một màu nền" tràn ra nền ở những ảnh có vệt sáng của đèn quanh quả: ở 129/2471 ảnh (5.2%), diện tích mask cũ lớn hơn mask mới từ 1.5 lần (`results/E1_xu_ly/table_cleaning.md`, đo ở bước 13). Hai chỉ số không cần nhãn của S1 (số mảnh rời, tỉ lệ lỗ thủng) không phát hiện được lỗi này, vì vùng tràn là một mảnh liền, không thủng.

**Phương án khác và vì sao không chọn.**
- So sánh các nhánh chỉ bằng điểm trên ảnh test gốc: không chọn, lý do ở trên. **[đo]**
- Tạo tập test mới ngoài bối cảnh chụp của bộ dữ liệu: không làm được, đồ án không có ảnh như vậy; bộ kiểm tra khó là phép thay thế bằng biến đổi tổng hợp và không thay được ảnh thật. **[lập luận]**
- Gộp ba yêu cầu thành một pipeline chồng lên nhau (ví dụ dùng bản Augmented của tác giả rồi mới xử lý): không chọn; khi đó không tách được đóng góp của từng thay đổi. **[lập luận]**

**Bằng chứng.** Nhánh G trên test gốc và bộ kiểm tra khó (`results/so_sanh_nhanh/G/table.md`; điểm bền = trung bình 6 cột phép thử):

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 371 | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.997 | 0.994 | [0.984; 1.000] | 0.995 | 2 | 0.726 | 0.304 | 0.673 | 0.086 | 0.983 | 0.150 | 0.487 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.957 | 0.964 | [0.944; 0.982] | 0.968 | 12 | 0.759 | 0.691 | 0.901 | 0.085 | 0.730 | 0.755 | 0.654 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.983 | 0.990 | [0.977; 1.000] | 0.992 | 3 | 0.145 | 0.987 | 0.776 | 0.484 | 0.505 | 0.805 | 0.617 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.998 | 1.000 | [1.000; 1.000] | 1.000 | 0 | 0.410 | 0.997 | 0.856 | 0.172 | 0.972 | 0.959 | 0.728 |
| CNN hướng A | a_lenet | 0.973 | 0.978 | [0.960; 0.993] | 0.981 | 7 | 0.744 | 0.979 | 0.977 | 0.978 | 0.874 | 0.469 | 0.837 |
| CNN hướng A | a_dropout | 0.948 | 0.942 | [0.917; 0.965] | 0.943 | 21 | 0.855 | 0.864 | 0.932 | 0.940 | 0.925 | 0.100 | 0.769 |
| CNN hướng A | a_vgg16 | 0.990 | 0.986 | [0.971; 0.997] | 0.989 | 4 | 0.657 | 0.979 | 0.963 | 0.984 | 0.887 | 0.843 | 0.886 |
| CNN hướng B | b_mobilenetv3 | 1.000 | 1.000 | [1.000; 1.000] | 1.000 | 0 | 0.794 | 0.919 | 0.890 | 0.989 | 1.000 | 0.659 | 0.875 |
| CNN hướng B | b_resnet50 | 0.998 | 0.997 | [0.988; 1.000] | 0.997 | 1 | 0.792 | 0.916 | 0.966 | 0.990 | 0.993 | 0.714 | 0.895 |

**Kết quả và kết luận.**
- Điểm bền của nhánh G nằm trong khoảng 0.49 đến 0.90, trong khi macro-F1 trên ảnh gốc là 0.94 đến 1.00. Thứ hạng cũng đổi: `b_resnet50` (0.895) và `a_vgg16` (0.886) bền hơn `b_mobilenetv3` (0.875), dù `b_mobilenetv3` đúng 371/371 ảnh gốc.
- Nền hoán đổi làm mọi cấu hình tụt: HOG còn 0.145, Hist H-S + HOG còn 0.410, các CNN còn 0.66 đến 0.86. HOG tụt sâu nhất vì nó mô tả cả kết cấu và độ sáng không đều của nền. Đây là xác nhận thứ hai, bằng một phép đo khác, cho kết luận của S2.
- Nhóm truyền thống gần như mất hết với nhiễu σ = 10 (0.09 đến 0.48), còn CNN giữ 0.94 đến 0.99; khớp N8.

**Ảnh hưởng tới bước sau.** Ba nhánh X, T, B dùng chung `src/pipeline.py`, bộ kiểm tra khó và mask mới. Kết quả từng nhánh nằm ở `results/E1_xu_ly/`, `results/E2_aug_tac_gia/`, `results/E3_bai_bao/`; bảng so sánh ở `results/so_sanh_nhanh/`.

## Bước 11: Nhánh T: chạy pipeline với bản Augmented của tác giả bộ dữ liệu (12/11/2026)

**Mục tiêu.** Chạy 9 cấu hình của pipeline gốc với bản "Augmented Banana Classification Dataset" (7413 ảnh) do tác giả bộ dữ liệu công bố, và so với nhánh G.

**Đã làm.**
- Tải bản Augmented từ Mendeley Data (SHA-256 `e56e2f87…240ea2`). Đây là bản xuất từ Roboflow: 7413 file, tên dạng `<tên ảnh gốc>_jpg.rf.<mã>.jpg`. Ánh xạ được 7413/7413 file về ảnh gốc; mỗi ảnh gốc có đúng 3 file: 1 bản gần như nguyên gốc và 2 bản tăng cường (lật, xoay bội số 90°, xoay nhỏ có góc đen, đổi sáng, làm mờ, nhiễu muối tiêu).
- `python -m src.exp_arm_t`:
  - **T-kr (không rò rỉ):** train = 3 file của mỗi ảnh gốc thuộc tập train của G (5187 ảnh); val và test = ảnh gốc của G. Chạy đủ 9 cấu hình, chấm trên test gốc và bộ kiểm tra khó.
  - **T-tl (theo thông lệ):** gộp 7413 file, chia ngẫu nhiên phân tầng 70/15/15 (seed 42) không quan tâm ảnh gốc: 5189 / 1112 / 1112. Chạy 7 cấu hình.

**Lý do chọn cách này.**
- Chạy cả hai cách dùng. T-kr trả lời "bản tăng cường của tác giả có giúp mô hình không" trên cùng tập test với G. T-tl là cách dùng trực tiếp nhất (lấy bộ 7413 ảnh rồi chia), cũng là cách bài báo của Ergün dùng (bước 12), nên cần biết nó cho ra con số gì. **[lập luận]**
- T-tl bỏ `a_dropout` và `b_resnet50` (hai mô hình huấn luyện lâu nhất trên 5189 ảnh). T-tl chỉ để đo mức rò rỉ, không dùng cho bảng so sánh bốn nhánh. **[lập luận]** (giới hạn thời gian trên CPU; lệnh chạy đủ có trong `python -m src.exp_arm_t --help`).

**Phương án khác và vì sao không chọn.**
- Chấm T-tl trên tập test gốc của G để so trực tiếp: không làm được một cách có nghĩa, vì tập train của T-tl chứa bản tăng cường (và một phần bản gốc) của chính các ảnh test đó. **[đo]** bảng mức trùng nguồn bên dưới.
- Commit bản Augmented vào repo: không; 67 MB, tải lại được bằng một lệnh có kiểm tra mã băm. **[lập luận]**

**Bằng chứng.**

T-kr: 9 cấu hình, chấm trên test gốc (`results/E2_aug_tac_gia/khong_ro_ri/table.md`):

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 371 | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.993 | 0.997 | [0.989; 1.000] | 0.997 | 1 | 0.637 | 0.350 | 0.628 | 0.086 | 0.994 | 0.637 | 0.555 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.933 | 0.948 | [0.924; 0.969] | 0.951 | 18 | 0.825 | 0.659 | 0.890 | 0.340 | 0.736 | 0.682 | 0.689 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.950 | 0.972 | [0.952; 0.990] | 0.978 | 8 | 0.254 | 0.950 | 0.731 | 0.492 | 0.688 | 0.606 | 0.620 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.993 | 0.989 | [0.976; 0.998] | 0.989 | 4 | 0.255 | 0.989 | 0.971 | 0.313 | 0.982 | 0.963 | 0.745 |
| CNN hướng A | a_lenet | 0.959 | 0.966 | [0.944; 0.986] | 0.973 | 10 | 0.616 | 0.693 | 0.962 | 0.964 | 0.944 | 0.501 | 0.780 |
| CNN hướng A | a_dropout | 0.982 | 0.963 | [0.941; 0.982] | 0.965 | 13 | 0.910 | 0.750 | 0.721 | 0.963 | 0.949 | 0.483 | 0.796 |
| CNN hướng A | a_vgg16 | 0.994 | 0.985 | [0.970; 0.997] | 0.987 | 5 | 0.807 | 0.979 | 0.959 | 0.982 | 0.942 | 0.762 | 0.905 |
| CNN hướng B | b_mobilenetv3 | 1.000 | 0.998 | [0.991; 1.000] | 0.997 | 1 | 0.867 | 0.954 | 0.904 | 0.993 | 1.000 | 0.736 | 0.909 |
| CNN hướng B | b_resnet50 | 1.000 | 0.997 | [0.989; 1.000] | 0.997 | 1 | 0.817 | 0.925 | 0.971 | 0.991 | 0.997 | 0.844 | 0.924 |

So với nhánh G trên cùng tập test (`results/so_sanh_nhanh/table_models.md`, cột G và T):

| Cấu hình | G: macro-F1 test | G: điểm bền | T-kr: macro-F1 test | T-kr: điểm bền |
|---|---|---|---|---|
| Hist HSV + LogReg | 0.994 | 0.487 | 0.997 | 0.555 |
| Cạnh Roberts + KNN | 0.964 | 0.654 | 0.948 | 0.689 |
| HOG + LogReg | 0.990 | 0.617 | 0.972 | 0.620 |
| Hist H-S + HOG + LogReg | 1.000 | 0.728 | 0.989 | 0.745 |
| a_lenet | 0.978 | 0.837 | 0.966 | 0.780 |
| a_dropout | 0.942 | 0.769 | 0.963 | 0.796 |
| a_vgg16 | 0.986 | 0.886 | 0.985 | 0.905 |
| b_mobilenetv3 | 1.000 | 0.875 | 0.998 | 0.909 |
| b_resnet50 | 0.997 | 0.895 | 0.997 | 0.924 |
| Trung bình 9 cấu hình | 0.983 | 0.750 | 0.979 | 0.769 |

T-tl: chia ngẫu nhiên theo thông lệ, chấm trên tập test của chính cách chia này (`results/E2_aug_tac_gia/thong_le/table.md`):

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 1112 |
|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.980 | 0.991 | [0.984; 0.997] | 0.993 | 8 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.590 | 0.620 | [0.589; 0.650] | 0.640 | 400 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.817 | 0.847 | [0.825; 0.868] | 0.854 | 162 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.978 | 0.976 | [0.966; 0.985] | 0.978 | 24 |
| CNN hướng A | a_lenet | 0.964 | 0.952 | [0.937; 0.965] | 0.960 | 44 |
| CNN hướng A | a_vgg16 | 0.971 | 0.967 | [0.956; 0.978] | 0.970 | 33 |
| CNN hướng B | b_mobilenetv3 | 0.999 | 0.995 | [0.990; 0.999] | 0.996 | 5 |

| Chỉ số | Giá trị |
|---|---|
| Số ảnh test | 1112 |
| Ảnh test có ít nhất 1 file cùng ảnh gốc trong train | 1008 (90.6%) |
| … có đúng 2 file cùng ảnh gốc trong train | 546 (49.1%) |
| Ảnh test không có file cùng ảnh gốc trong train | 104 |

| Mô hình | Accuracy (ảnh test có anh em trong train) | Số ảnh | Accuracy (ảnh test không có) | Số ảnh  |
|---|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 | 0.993 | 1008 | 0.990 | 104 |
| Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.640 | 1008 | 0.644 | 104 |
| HOG 64×128 + LogReg C=0.1 | 0.859 | 1008 | 0.808 | 104 |
| Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.976 | 1008 | 1.000 | 104 |
| a_lenet | 0.959 | 1008 | 0.971 | 104 |
| a_vgg16 | 0.970 | 1008 | 0.971 | 104 |
| b_mobilenetv3 | 0.997 | 1008 | 0.981 | 104 |

**Kết quả và kết luận.**
- **Trên ảnh gốc, bản Augmented của tác giả không cải thiện kết quả:** trung bình 9 cấu hình 0.979 so với 0.983 của G. Khác biệt lớn nhất là ở các mô hình nhạy với hướng và vị trí: HOG + LogReg giảm từ 0.990 xuống 0.972 (8 ảnh sai so với 3), Hist H-S + HOG từ 1.000 xuống 0.989, `a_lenet` từ 0.978 xuống 0.966. Ảnh xoay 90° và lật làm tăng độ đa dạng mà các mô hình này không biểu diễn được, trong khi ảnh test gốc không bị xoay. `a_dropout` tăng (0.942 → 0.963), khớp N5: mô hình học từ đầu quá khớp thì được lợi từ tăng cường. Hai mô hình hướng B gần như không đổi (0.998 và 0.997; chênh 1 ảnh).
- **Trên bộ kiểm tra khó, bản Augmented giúp một ít:** điểm bền trung bình 0.769 so với 0.750; các mô hình pretrained tăng 0.02 đến 0.03 (`b_resnet50` 0.924 so với 0.895). Phần tăng đến đúng từ những phép có trong công thức tăng cường: HOG với xoay +30° từ 0.505 lên 0.688, nhiễu σ = 10 ở cạnh Roberts từ 0.085 lên 0.340. Phép không có trong công thức thì không cải thiện: nền hoán đổi của Hist H-S + HOG còn 0.255 (G: 0.410).
- **T-tl trái với suy đoán thông thường:** điểm của cách chia ngẫu nhiên không cao hơn mà **thấp hơn** T-kr ở phần lớn mô hình (HOG 0.847; cạnh 0.620; `a_vgg16` 0.967). Lý do là tập test của T-tl gồm cả ảnh đã xoay, lật, thêm nhiễu nên khó hơn tập test gốc; hai con số không cùng thước đo. Mức rò rỉ phải đọc ở bảng tách nhóm: 90.6% ảnh test có file cùng ảnh gốc trong train. Trên nhóm đó, HOG đúng 0.859 so với 0.808 ở nhóm không có (+0.051) và `b_mobilenetv3` đúng 0.997 so với 0.981; histogram màu gần như không chênh (0.993 so với 0.990); `a_lenet` và Hist H-S + HOG thì ngược chiều. Nhóm "không có" chỉ có 104 ảnh nên các chênh lệch này có sai số lớn (1 ảnh ≈ 1%).
- Kết luận về rò rỉ khớp D1 ở chỗ mức thổi phồng phụ thuộc mô hình, nhưng nhỏ hơn dự đoán. Bước 12 đo lại bằng 5-fold trên cùng bộ ảnh (có và không giữ nhóm), là phép đo chặt hơn.

**Ảnh hưởng tới bước sau.** Nhánh T trong bảng so sánh bốn nhánh là T-kr. Bước 12 dùng lại cách ánh xạ file → ảnh gốc để chia fold theo nhóm.

## Bước 12: Nhánh B: phương pháp của bài báo trên bộ BananaImageBD (12/11/2026)

**Mục tiêu.** Chạy phương pháp của các bài báo đã công bố trên chính bộ dữ liệu này và so với pipeline của đồ án, theo hai hướng: B1 chỉ dùng tới mức CNN (trong phạm vi môn học), B2 đúng như bài báo.

**Đã làm.**
- Đọc hai bài báo:
  - Ferdaus và cộng sự (2025), *Data in Brief* 58, 111239: bài báo mô tả dữ liệu. Chuối được đặt trên giấy trắng dưới đèn LED trắng, **mỗi quả chụp từ bốn mặt** bằng iPhone 13 Pro Max; ảnh được loại trùng, lọc ảnh kém, resize 256×256, làm nét ảnh mờ; bản Augmented gồm ảnh gốc và 2 bản tăng cường mỗi ảnh. **Bài báo không huấn luyện mô hình nào và không công bố kết quả phân loại**; phần xử lý dữ liệu của bài báo đã được kiểm chứng ở nhánh T.
  - Ergün (2025), *Scientific Reports* 15, 10366: rút đặc trưng bằng Vision Transformer (ViT) pretrained rồi phân lớp bằng SVM; so với đặc trưng ResNet, SqueezeNet và các bộ phân lớp GBM, MLP, CNN. Dùng bản Augmented (7413 ảnh), ảnh 224×224, 5-fold cross-validation. Kết quả công bố tốt nhất trên BananaImageBD: ViT + SVM, accuracy 99.70%, F1 0.9965.
  - Cả hai bài báo đều **không dùng YOLO**.
- `python -m src.exp_papers` (cần PyTorch cho ViT và SqueezeNet):
  - **B1: trong phạm vi môn học:** bộ rút đặc trưng là CNN pretrained ImageNet: ResNet50 (2048 chiều), MobileNetV3-Small (576 chiều), SqueezeNet 1.1 (512 chiều); bộ phân lớp: SVM (RBF và tuyến tính), Logistic Regression, KNN, MLP (một lớp ẩn 256).
  - **B2: đúng như bài báo:** ViT-B/16 pretrained ImageNet, vector 768 chiều của token phân lớp; bộ phân lớp: SVM (RBF và tuyến tính), MLP, GBM.
  - Ba giao thức chấm: (i) như bài báo: bản Augmented, 5-fold phân tầng; (ii) cùng dữ liệu, 5-fold giữ 3 file của cùng ảnh gốc trong một fold; (iii) giao thức của đồ án: học trên train+val gốc (2100 ảnh), chấm trên test gốc (371 ảnh) và bộ kiểm tra khó.

**Lý do chọn cách này.**
- Bài báo của Ergün không nêu tên checkpoint ViT, kernel và siêu tham số của SVM, biến thể ResNet/SqueezeNet, tham số GBM/MLP. Đồ án chọn: ViT-B/16 của torchvision (trọng số ImageNet-1k), SVM RBF với C = 1 và gamma mặc định, ResNet50, SqueezeNet 1.1, `HistGradientBoostingClassifier` 100 vòng, MLP một lớp ẩn 256. **[lập luận]**: đây là các lựa chọn mặc định phổ biến nhất; kết quả tái hiện lệch bài báo 0.21 điểm phần trăm ở cấu hình chính (bảng dưới). **[đo]**
- Thêm giao thức (ii) vì bài báo dùng bản Augmented với 5-fold mà không nói có giữ nhóm theo ảnh gốc hay không; ở bước 11 đã đo được 90.6% ảnh test có file cùng ảnh gốc trong train khi chia ngẫu nhiên. **[đo]**
- Thêm giao thức (iii) để so được với nhánh G trên cùng tập test và cùng bộ kiểm tra khó. **[lập luận]**
- Hướng B1 giữ nguyên ý tưởng của bài báo (mạng pretrained làm bộ rút đặc trưng, bộ phân lớp riêng) nhưng thay ViT bằng CNN, là kiến thức của môn học; ResNet và SqueezeNet cũng chính là hai CNN đối chứng trong bài báo. **[lập luận]**

**Phương án khác và vì sao không chọn.**
- Tái hiện bộ phân lớp "CNN" trên vector đặc trưng của bài báo: không làm, bài báo không mô tả kiến trúc. **[lập luận]**
- GBM cho các bộ rút đặc trưng CNN: không chạy; ViT + GBM đã mất khoảng 10 phút và kém SVM rõ (97.71% so với 99.49%). **[đo]**
- Chạy thêm YOLO: không làm, vì không bài báo nào trong hai bài dùng YOLO. **[lập luận]**

**Bằng chứng.**

Giao thức (i) và (ii) trên bản Augmented (`results/E3_bai_bao/table_protocols.md`, hình `results/figures/E3_papers_protocols.png`):

| Hướng | Bộ rút đặc trưng | Bộ phân lớp | Bài báo công bố: accuracy (%) | (i) Như bài báo: accuracy (%) | (i) Macro-F1 | (ii) Giữ nhóm theo ảnh gốc: accuracy (%) | (ii) Macro-F1 | Thổi phồng (i) - (ii), điểm % |
|---|---|---|---|---|---|---|---|---|
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | SVM (RBF, C=1) | 96.69 | 99.16 ± 0.15 | 0.9900 | 98.97 ± 0.38 | 0.9881 | +0.19 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | SVM tuyến tính (C=0.01) | - | 99.15 ± 0.16 | 0.9901 | 98.85 ± 0.33 | 0.9864 | +0.30 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | MLP (1 lớp ẩn 256) | 97.76 | 98.97 ± 0.28 | 0.9881 | 98.87 ± 0.20 | 0.9868 | +0.11 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | Logistic Regression | - | 99.12 ± 0.15 | 0.9894 | 98.96 ± 0.29 | 0.9876 | +0.16 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | KNN (k=5) | - | 93.96 ± 0.47 | 0.9356 | 93.05 ± 0.75 | 0.9258 | +0.90 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | SVM (RBF, C=1) | - | 99.27 ± 0.21 | 0.9912 | 99.08 ± 0.31 | 0.9891 | +0.19 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | SVM tuyến tính (C=0.01) | - | 98.80 ± 0.34 | 0.9858 | 98.60 ± 0.39 | 0.9836 | +0.20 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | MLP (1 lớp ẩn 256) | - | 99.22 ± 0.11 | 0.9909 | 98.91 ± 0.32 | 0.9875 | +0.31 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | Logistic Regression | - | 98.68 ± 0.22 | 0.9844 | 98.50 ± 0.36 | 0.9828 | +0.18 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | KNN (k=5) | - | 96.07 ± 0.62 | 0.9572 | 95.55 ± 0.81 | 0.9516 | +0.53 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | SVM (RBF, C=1) | 97.63 | 98.58 ± 0.42 | 0.9836 | 98.12 ± 0.29 | 0.9783 | +0.46 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | SVM tuyến tính (C=0.01) | - | 98.03 ± 0.43 | 0.9772 | 97.65 ± 0.14 | 0.9726 | +0.38 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | MLP (1 lớp ẩn 256) | 95.88 | 98.57 ± 0.20 | 0.9837 | 98.35 ± 0.24 | 0.9815 | +0.22 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | Logistic Regression | - | 97.59 ± 0.41 | 0.9727 | 97.33 ± 0.40 | 0.9693 | +0.26 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | KNN (k=5) | - | 91.73 ± 0.33 | 0.9079 | 91.61 ± 0.70 | 0.9064 | +0.12 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | SVM (RBF, C=1) | 99.70 | 99.49 ± 0.15 | 0.9937 | 99.20 ± 0.33 | 0.9905 | +0.28 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | SVM tuyến tính (C=0.01) | - | 99.41 ± 0.19 | 0.9929 | 99.12 ± 0.37 | 0.9895 | +0.28 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | MLP (1 lớp ẩn 256) | 99.29 | 99.50 ± 0.21 | 0.9942 | 99.03 ± 0.38 | 0.9886 | +0.47 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | GBM (HistGradientBoosting) | 97.39 | 97.71 ± 0.36 | 0.9734 | 97.14 ± 0.60 | 0.9666 | +0.57 |

Giao thức (iii), chấm trên test gốc và bộ kiểm tra khó (`results/E3_bai_bao/table_test.md`):

| Hướng | Bộ rút đặc trưng | Bộ phân lớp | Macro-F1 5-fold (train+val) | Macro-F1 test | KTC 95% | Số ảnh sai / 371 | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | ResNet50 (ImageNet) | SVM (RBF, C=1) | 0.993 ± 0.003 | 0.997 | [0.988; 1.000] | 1 | 0.824 | 0.961 | 0.928 | 0.901 | 0.964 | 0.087 | 0.778 |
| B1 | ResNet50 (ImageNet) | SVM tuyến tính (C=0.01) | 0.994 ± 0.002 | 0.997 | [0.989; 1.000] | 1 | 0.718 | 0.969 | 0.908 | 0.977 | 0.972 | 0.502 | 0.841 |
| B1 | ResNet50 (ImageNet) | MLP (1 lớp ẩn 256) | 0.990 ± 0.004 | 0.997 | [0.988; 1.000] | 1 | 0.849 | 0.966 | 0.938 | 0.979 | 0.968 | 0.534 | 0.873 |
| B1 | ResNet50 (ImageNet) | Logistic Regression | 0.995 ± 0.003 | 0.997 | [0.988; 1.000] | 1 | 0.744 | 0.988 | 0.897 | 0.986 | 0.983 | 0.513 | 0.852 |
| B1 | ResNet50 (ImageNet) | KNN (k=5) | 0.971 ± 0.008 | 0.982 | [0.966; 0.995] | 6 | 0.921 | 0.933 | 0.978 | 0.977 | 0.956 | 0.572 | 0.890 |
| B1 | MobileNetV3-Small (ImageNet) | SVM (RBF, C=1) | 0.994 ± 0.006 | 0.990 | [0.976; 1.000] | 3 | 0.805 | 0.984 | 0.898 | 0.945 | 0.946 | 0.130 | 0.785 |
| B1 | MobileNetV3-Small (ImageNet) | SVM tuyến tính (C=0.01) | 0.996 ± 0.003 | 0.993 | [0.982; 1.000] | 2 | 0.742 | 0.965 | 0.856 | 0.897 | 0.981 | 0.575 | 0.836 |
| B1 | MobileNetV3-Small (ImageNet) | MLP (1 lớp ẩn 256) | 0.994 ± 0.003 | 0.993 | [0.982; 1.000] | 2 | 0.831 | 0.974 | 0.905 | 0.966 | 0.952 | 0.451 | 0.846 |
| B1 | MobileNetV3-Small (ImageNet) | Logistic Regression | 0.995 ± 0.001 | 0.997 | [0.989; 1.000] | 1 | 0.746 | 0.969 | 0.847 | 0.933 | 0.964 | 0.367 | 0.805 |
| B1 | MobileNetV3-Small (ImageNet) | KNN (k=5) | 0.975 ± 0.008 | 0.973 | [0.953; 0.990] | 9 | 0.887 | 0.950 | 0.956 | 0.962 | 0.958 | 0.569 | 0.880 |
| B1 | SqueezeNet 1.1 (ImageNet) | SVM (RBF, C=1) | 0.990 ± 0.004 | 0.997 | [0.988; 1.000] | 1 | 0.680 | 0.861 | 0.885 | 0.279 | 0.984 | 0.087 | 0.629 |
| B1 | SqueezeNet 1.1 (ImageNet) | SVM tuyến tính (C=0.01) | 0.993 ± 0.003 | 0.997 | [0.988; 1.000] | 1 | 0.647 | 0.939 | 0.827 | 0.906 | 0.986 | 0.409 | 0.786 |
| B1 | SqueezeNet 1.1 (ImageNet) | MLP (1 lớp ẩn 256) | 0.994 ± 0.005 | 0.997 | [0.988; 1.000] | 1 | 0.727 | 0.942 | 0.880 | 0.816 | 0.980 | 0.553 | 0.816 |
| B1 | SqueezeNet 1.1 (ImageNet) | Logistic Regression | 0.995 ± 0.003 | 0.997 | [0.988; 1.000] | 1 | 0.637 | 0.932 | 0.852 | 0.783 | 0.975 | 0.428 | 0.768 |
| B1 | SqueezeNet 1.1 (ImageNet) | KNN (k=5) | 0.962 ± 0.006 | 0.979 | [0.962; 0.992] | 8 | 0.713 | 0.939 | 0.923 | 0.835 | 0.921 | 0.405 | 0.789 |
| B2 | ViT-B/16 (ImageNet) | SVM (RBF, C=1) | 0.994 ± 0.004 | 0.997 | [0.989; 1.000] | 1 | 0.857 | 0.990 | 0.978 | 0.986 | 0.978 | 0.728 | 0.919 |
| B2 | ViT-B/16 (ImageNet) | SVM tuyến tính (C=0.01) | 0.996 ± 0.004 | 1.000 | [1.000; 1.000] | 0 | 0.827 | 0.994 | 0.977 | 0.990 | 0.983 | 0.854 | 0.937 |
| B2 | ViT-B/16 (ImageNet) | MLP (1 lớp ẩn 256) | 0.995 ± 0.004 | 0.994 | [0.984; 1.000] | 2 | 0.864 | 0.988 | 0.958 | 0.988 | 0.981 | 0.751 | 0.922 |
| B2 | ViT-B/16 (ImageNet) | GBM (HistGradientBoosting) | 0.975 ± 0.005 | 0.972 | [0.952; 0.988] | 9 | 0.849 | 0.958 | 0.942 | 0.939 | 0.943 | 0.658 | 0.881 |

Thời gian rút đặc trưng mỗi ảnh trên CPU (`results/E3_bai_bao/summary.json`; đo trong lúc máy còn chạy tác vụ khác nên chỉ có ý nghĩa tương đối): MobileNetV3-Small 4.5 ms; SqueezeNet 4.8 ms; ResNet50 27.5 ms; ViT-B/16 105.7 ms.

**Kết quả và kết luận.**
- **Tái hiện được kết quả chính của bài báo:** ViT + SVM đạt 99.49 ± 0.15% theo giao thức (i), bài báo công bố 99.70%. Thứ hạng bộ phân lớp trên đặc trưng ViT cũng khớp: SVM ≈ MLP (99.50%) > GBM (97.71%; bài báo 97.39%).
- **Không tái hiện được khoảng cách giữa ViT và CNN mà bài báo công bố.** Bài báo: ViT + SVM hơn ResNet + SVM 3.01 điểm (99.70 so với 96.69) và hơn SqueezeNet + SVM 2.07 điểm. Đo được: ResNet50 + SVM 99.16%, MobileNetV3 + SVM 99.27%, SqueezeNet + SVM 98.58%; ViT chỉ hơn 0.22 đến 0.90 điểm. Có thể bài báo dùng biến thể ResNet khác hoặc lấy đặc trưng ở lớp khác; bài báo không nêu. **[lập luận]**
- **Giao thức của bài báo có rò rỉ, mức thổi phồng nhỏ nhưng nhất quán:** (i) cao hơn (ii) ở cả 19/19 tổ hợp, từ +0.11 đến +0.90 điểm phần trăm; với ViT + SVM là +0.28 (99.49 → 99.20). Mức này xấp xỉ độ lệch chuẩn giữa các fold (0.15 đến 0.4), nên từng tổ hợp riêng lẻ chưa đủ cơ sở phân biệt; dấu nhất quán ở 19/19 tổ hợp mới là bằng chứng. Thổi phồng lớn nhất ở KNN trên đặc trưng ResNet50 (+0.90), khớp D1: KNN hưởng lợi nhiều nhất từ bản sao trong train.
- **Theo giao thức của đồ án, phương pháp của bài báo không hơn pipeline gốc trên ảnh gốc nhưng bền hơn:** ViT + SVM (RBF) sai 1/371 ảnh (macro-F1 0.997), ViT + SVM tuyến tính đúng 371/371; nhánh G có hai cấu hình đúng 371/371. Trên bộ kiểm tra khó, đặc trưng ViT đạt điểm bền 0.919 đến 0.937, cao hơn mọi cấu hình của nhánh G (cao nhất 0.895) dù ViT không được fine-tune. Đổi lại ViT-B/16 có 86 triệu tham số và rút đặc trưng chậm hơn MobileNetV3 khoảng 23 lần.
- **Hướng B1 (CNN làm bộ rút đặc trưng) ngang ViT trên ảnh gốc nhưng kém bền hơn:** macro-F1 test 0.990 đến 0.997 với SVM, LogReg, MLP; điểm bền 0.63 đến 0.89. So với fine-tune cùng mạng ở nhánh G: MobileNetV3 đóng băng + SVM tuyến tính có điểm bền 0.836, MobileNetV3 fine-tune có tăng cường 0.875; fine-tune kèm tăng cường làm mô hình bền hơn.
- **Ngoài dự đoán: SVM kernel RBF sập khi ảnh bị che 25%** trên đặc trưng CNN (0.087 với ResNet50 và SqueezeNet; 0.130 với MobileNetV3), trong khi SVM tuyến tính trên cùng đặc trưng giữ 0.41 đến 0.58. Giải thích: với ảnh lệch xa phân bố huấn luyện, giá trị kernel RBF tới mọi vector hỗ trợ đều gần 0, nên quyết định chỉ còn phụ thuộc hệ số chặn và mọi ảnh rơi vào một lớp (macro-F1 0.087 đúng bằng mức "đoán một lớp" đã gặp ở C4). **[lập luận]**

**Ảnh hưởng tới bước sau.** Đại diện của nhánh B trong bảng so sánh bốn nhánh: B1 là MobileNetV3 + SVM tuyến tính (chọn theo 5-fold trên train+val), B2 là ViT + SVM RBF (cấu hình chính của bài báo). Thông tin "mỗi quả chụp bốn mặt" của bài báo dữ liệu được kiểm tra ở bước 13 (ảnh mang số thứ tự liền kề).

## Bước 13: Nhánh X: xử lý dữ liệu do nhóm thiết kế, rồi chạy lại pipeline (12/11/2026)

**Mục tiêu.** Trả lời câu hỏi "pipeline gốc đã xử lý và tăng cường dữ liệu chưa", thiết kế phần làm sạch, tiền xử lý và tăng cường, chọn tổ hợp tốt nhất bằng số liệu, rồi chạy lại 9 cấu hình của pipeline gốc trên dữ liệu đã xử lý.

**Hiện trạng của pipeline gốc.** Chỉ resize ảnh về 256×256 và chuẩn hóa giá trị điểm ảnh bên trong mô hình. Không làm sạch, không cân bằng trắng, không tách nền. Tăng cường ảnh chỉ bật lúc huấn luyện cho `a_dropout` và hai mô hình hướng B; nhóm truyền thống không có (trừ thí nghiệm I2).

**Đã làm.**
- `python -m src.exp_arm_x --only X1`: kiểm tra và làm sạch dữ liệu; kiểm tra dấu vết "mỗi quả chụp bốn mặt".
- `src/preprocess.py`: năm mức tiền xử lý: không xử lý; cân bằng trắng theo nền; nền phẳng; cân bằng trắng + nền phẳng; thêm cắt theo quả. `src/augment.py`: ba mức tăng cường chỉ trên train: không; hình học + quang học (2 bản mỗi ảnh); thêm thay nền ngẫu nhiên.
- `python -m src.exp_arm_x --only X2`: chấm 12 tổ hợp trên val bằng hai mô hình đại diện, 7 phép thử, bootstrap 1000 lần cho sai số của điểm chọn.
- `python -m src.exp_arm_x --only RUN`: chạy lại pipeline cho hai biến thể: **X-a** (điểm chọn cao nhất) đủ 9 cấu hình; **X-b** (tốt nhất trong nhóm không xử lý ảnh lúc suy luận) cho 6 cấu hình đại diện.
- `python -m src.exp_arm_x --only X3`: phép đo riêng về cân bằng trắng. `python -m src.exp_segment --only S2V2`: chạy lại S2 với mask mới.

**Lý do chọn cách này.**
- Tiêu chí chọn là điểm trên 7 phép thử (ảnh gốc và 6 phép khó) chứ không riêng ảnh gốc. **[đo]** Trên ảnh gốc, tổ hợp "không xử lý" đã đạt 1.000 / 0.993 trên val; nếu chỉ nhìn ảnh gốc thì mọi bước xử lý đều "làm kém đi" và không bước nào được chọn (`results/E1_xu_ly/table_selection.md`).
- Dùng hai mô hình đại diện rẻ (cấu hình truyền thống tốt nhất; embedding MobileNetV3 đóng băng + LogReg) để chấm 12 tổ hợp trong khoảng 12 phút; huấn luyện đủ 9 cấu hình cho từng tổ hợp sẽ mất khoảng 1 giờ mỗi tổ hợp. **[lập luận]**
- Tăng cường được áp lên ảnh thô rồi mới qua tiền xử lý như mọi ảnh khác, để ảnh train và ảnh cần dự đoán đi qua cùng một đường. **[lập luận]**

**Phương án khác và vì sao không chọn (kể cả những thứ đã thử và bỏ).**
- **Lưới tổ hợp đầu tiên thiếu mức "nền phẳng không cân bằng trắng".** Lần chạy đầu chỉ có 10 tổ hợp, nền phẳng luôn đi kèm cân bằng trắng. Tổ hợp đứng đầu khi đó là cân bằng trắng + nền phẳng (0.843), và ba tổ hợp kế tiếp cách nó dưới 0.01. Khi chạy thử pipeline với tổ hợp này, histogram HSV + LogReg tụt mạnh ở phép nền hoán đổi dù nền đã phẳng. Từ đó nhóm tìm ra nguyên nhân (cân bằng trắng, xem X3), thêm mức "nền phẳng" vào lưới và chạy lại toàn bộ phần chọn. Kết quả của lần chạy đầu không được dùng. **[đo]** `results/E1_xu_ly/table_wb_leak.md`, `results/E1_xu_ly/table_selection.md`.
- **Cân bằng trắng theo nền:** không chọn. Hệ số cân bằng tính từ màu nền rồi nhân vào cả quả chuối; khi nền đổi, màu quả lệch trung bình 12.9 đơn vị Lab (không cân bằng trắng: 0.0), và histogram HSV + LogReg sụt 0.351 ở phép nền hoán đổi (chỉ nền phẳng: 0.023). Bước này không khử thông tin của nền mà chuyển nó vào màu quả. **[đo]** `results/E1_xu_ly/table_wb_leak.md`.
- **Cắt theo quả:** không chọn; điểm chọn 0.804 đến 0.813, và phép che 25% còn 0.20 đến 0.30 vì ô đen bị coi là vật thể nên khung cắt sai. **[đo]** `results/E1_xu_ly/table_selection.md`.
- **Chia lại tập theo quả/đợt chụp:** chưa làm. X1 cho thấy chia ngẫu nhiên lạc quan hơn chia theo khối ảnh liên tiếp 0.024 đến 0.069, nhưng dữ liệu không có nhãn "quả nào", và đổi cách chia sẽ làm mọi kết quả trước không so được với nhau. Ghi vào hạn chế. **[đo]** `results/E1_xu_ly/table_blocked_cv.md`.
- **Chạy đủ 9 cấu hình cho X-b:** chưa chạy `a_lenet`, `a_dropout`, `b_resnet50` (giới hạn thời gian trên CPU; lệnh: `python -m src.exp_arm_x --only RUN --xb-all`). **[lập luận]**
- **Loại ảnh mờ, sửa nhãn:** không làm. 18 ảnh có độ nét dưới 1/4 trung vị được liệt kê ở `results/E1_xu_ly/low_sharpness.csv` nhưng giữ lại; không ảnh nào bị cả hai mô hình kiểm tra cùng đoán sai nên không có ứng viên sai nhãn. **[đo]** `results/E1_xu_ly/table_cleaning.md`.

**Bằng chứng.**

X1: làm sạch (`results/E1_xu_ly/table_cleaning.md`, danh sách ảnh bị loại: `results/E1_xu_ly/train_dropped.csv`):

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
| Mask của bước S1 lớn hơn mask mới từ 1.5 lần (mask cũ tràn ra nền) | 129 / 2471 (5.2%) |

X1: dấu vết "mỗi quả chụp bốn mặt" (`results/E1_xu_ly/table_adjacent.md`, `results/E1_xu_ly/table_blocked_cv.md`):

| Chỉ số | Đo được | Nếu ngẫu nhiên |
|---|---|---|
| Ảnh có láng giềng gần nhất cùng lớp (embedding MobileNetV3) mang số thứ tự cách ≤ 3 | 20.0% | 1.0% |
| … cách đúng 1 | 10.1% | 0.3% |

| Mô hình | 5-fold ngẫu nhiên phân tầng | 5-fold theo khối ảnh liên tiếp | Chênh |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 | 0.991 ± 0.005 | 0.922 ± 0.042 | +0.069 |
| HOG 64×128 + LogReg C=0.1 | 0.986 ± 0.007 | 0.933 ± 0.062 | +0.053 |
| Hist H-S + HOG + LogReg C=100 | 0.996 ± 0.003 | 0.973 ± 0.017 | +0.024 |
| Embedding MobileNetV3 (ImageNet) + LogReg | 0.995 ± 0.005 | 0.964 ± 0.025 | +0.031 |

X2-X4: chọn tổ hợp trên val (`results/E1_xu_ly/table_selection.md`, hình `results/figures/E1_selection.png`, ảnh minh họa `results/figures/E1_preprocess_examples.png`):

| Tiền xử lý | Tăng cường (train) | Số ảnh train | Ảnh gốc (TT / CNN) | Nền hoán đổi (TT / CNN) | V×0.6 (TT / CNN) | V×1.4 (TT / CNN) | Nhiễu σ=10 (TT / CNN) | Xoay +30° (TT / CNN) | Che 25% (TT / CNN) | TB 7 phép thử (TT) | TB 7 phép thử (CNN) | Điểm chọn | Kém tổ hợp đứng đầu (KTC 95%) | Phân biệt được với tổ hợp đứng đầu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Không xử lý | Không tăng cường | 1694 | 1.000 / 0.993 | 0.354 / 0.788 | 0.995 / 0.955 | 0.900 / 0.906 | 0.190 / 0.930 | 0.970 / 0.961 | 0.947 / 0.372 | 0.765 | 0.843 | 0.804 ± 0.006 | 0.085 [+0.073; +0.098] | có |
| Không xử lý | Hình học + quang học | 5082 | 0.997 / 0.986 | 0.479 / 0.768 | 0.994 / 0.896 | 0.992 / 0.881 | 0.245 / 0.924 | 0.963 / 0.971 | 0.965 / 0.502 | 0.805 | 0.847 | 0.826 ± 0.006 | 0.064 [+0.052; +0.075] | có |
| Không xử lý | Hình học + quang học + thay nền ngẫu nhiên | 5082 | 0.998 / 0.987 | 0.859 / 0.960 | 0.988 / 0.952 | 0.945 / 0.916 | 0.141 / 0.945 | 0.932 / 0.954 | 0.834 / 0.284 | 0.814 | 0.857 | 0.835 ± 0.006 | 0.054 [+0.043; +0.065] | có |
| Cân bằng trắng theo nền | Không tăng cường | 1694 | 0.988 / 0.987 | 0.661 / 0.788 | 0.982 / 0.985 | 0.874 / 0.948 | 0.172 / 0.926 | 0.906 / 0.932 | 0.770 / 0.304 | 0.765 | 0.838 | 0.802 ± 0.007 | 0.088 [+0.074; +0.102] | có |
| Cân bằng trắng theo nền | Hình học + quang học | 5082 | 0.976 / 0.980 | 0.727 / 0.855 | 0.973 / 0.985 | 0.973 / 0.944 | 0.221 / 0.942 | 0.912 / 0.960 | 0.836 / 0.379 | 0.802 | 0.863 | 0.833 ± 0.006 | 0.056 [+0.045; +0.068] | có |
| Cân bằng trắng theo nền | Hình học + quang học + thay nền ngẫu nhiên | 5082 | 0.967 / 0.970 | 0.836 / 0.944 | 0.974 / 0.978 | 0.939 / 0.943 | 0.165 / 0.927 | 0.888 / 0.924 | 0.802 / 0.522 | 0.796 | 0.887 | 0.841 ± 0.007 | 0.048 [+0.036; +0.059] | có |
| Nền phẳng (không cân bằng trắng) | Không tăng cường | 1694 | 0.985 / 0.978 | 0.962 / 0.973 | 0.985 / 0.923 | 0.974 / 0.945 | 0.678 / 0.955 | 0.964 / 0.962 | 0.744 / 0.271 | 0.899 | 0.858 | 0.878 ± 0.006 | 0.011 [+0.003; +0.020] | có |
| Nền phẳng (không cân bằng trắng) | Hình học + quang học | 5082 | 0.984 / 0.971 | 0.933 / 0.974 | 0.988 / 0.930 | 0.982 / 0.949 | 0.633 / 0.955 | 0.965 / 0.967 | 0.822 / 0.398 | 0.901 | 0.878 | 0.889 ± 0.006 | - | - |
| Cân bằng trắng + nền phẳng | Không tăng cường | 1694 | 0.957 / 0.981 | 0.844 / 0.974 | 0.953 / 0.981 | 0.946 / 0.974 | 0.624 / 0.967 | 0.882 / 0.970 | 0.513 / 0.231 | 0.817 | 0.868 | 0.843 ± 0.006 | 0.047 [+0.037; +0.057] | có |
| Cân bằng trắng + nền phẳng | Hình học + quang học | 5082 | 0.963 / 0.965 | 0.758 / 0.952 | 0.961 / 0.965 | 0.936 / 0.961 | 0.505 / 0.948 | 0.911 / 0.961 | 0.540 / 0.394 | 0.796 | 0.878 | 0.837 ± 0.007 | 0.052 [+0.042; +0.061] | có |
| Cân bằng trắng + nền phẳng + cắt theo quả | Không tăng cường | 1694 | 0.950 / 0.977 | 0.759 / 0.968 | 0.956 / 0.977 | 0.938 / 0.983 | 0.635 / 0.974 | 0.788 / 0.952 | 0.289 / 0.113 | 0.759 | 0.849 | 0.804 ± 0.005 | 0.085 [+0.071; +0.099] | có |
| Cân bằng trắng + nền phẳng + cắt theo quả | Hình học + quang học | 5082 | 0.937 / 0.961 | 0.752 / 0.940 | 0.952 / 0.963 | 0.929 / 0.970 | 0.652 / 0.930 | 0.839 / 0.951 | 0.287 / 0.316 | 0.764 | 0.862 | 0.813 ± 0.007 | 0.077 [+0.063; +0.091] | có |

X3: cân bằng trắng theo nền (`results/E1_xu_ly/table_wb_leak.md`):

| Tiền xử lý | Hist HSV + LogReg: val gốc | Hist HSV + LogReg: val nền hoán đổi | Sụt | Màu quả chuối lệch đi khi đổi nền (khoảng cách Lab trung bình) |
|---|---|---|---|---|
| Không xử lý | 0.997 | 0.718 | +0.278 | 0.0 |
| Nền phẳng (không cân bằng trắng) | 0.997 | 0.973 | +0.023 | 0.0 |
| Cân bằng trắng theo nền | 0.977 | 0.750 | +0.226 | 12.9 |
| Cân bằng trắng + nền phẳng | 0.983 | 0.632 | +0.351 | 12.9 |

X-a: nền phẳng + tăng cường hình học/quang học, 9 cấu hình (`results/E1_xu_ly/pipeline/table.md`):

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 371 | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.990 | 0.993 | [0.981; 1.000] | 0.995 | 2 | 0.966 | 0.834 | 0.784 | 0.354 | 0.993 | 0.169 | 0.683 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.814 | 0.842 | [0.802; 0.878] | 0.849 | 56 | 0.838 | 0.799 | 0.820 | 0.792 | 0.686 | 0.381 | 0.719 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.836 | 0.828 | [0.790; 0.864] | 0.830 | 63 | 0.836 | 0.810 | 0.831 | 0.822 | 0.629 | 0.294 | 0.704 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.984 | 0.982 | [0.965; 0.996] | 0.987 | 5 | 0.951 | 0.978 | 0.979 | 0.651 | 0.974 | 0.805 | 0.890 |
| CNN hướng A | a_lenet | 0.947 | 0.922 | [0.892; 0.950] | 0.930 | 26 | 0.925 | 0.861 | 0.846 | 0.922 | 0.797 | 0.317 | 0.778 |
| CNN hướng A | a_dropout | 0.979 | 0.959 | [0.936; 0.978] | 0.960 | 15 | 0.959 | 0.880 | 0.942 | 0.962 | 0.946 | 0.207 | 0.816 |
| CNN hướng A | a_vgg16 | 0.982 | 0.958 | [0.936; 0.977] | 0.957 | 16 | 0.951 | 0.933 | 0.954 | 0.964 | 0.926 | 0.413 | 0.857 |
| CNN hướng B | b_mobilenetv3 | 1.000 | 0.995 | [0.987; 1.000] | 0.995 | 2 | 0.995 | 0.943 | 0.915 | 0.988 | 0.998 | 0.450 | 0.881 |
| CNN hướng B | b_resnet50 | 0.987 | 0.997 | [0.989; 1.000] | 0.997 | 1 | 0.997 | 0.983 | 0.956 | 0.985 | 0.990 | 0.397 | 0.885 |

X-b: không xử lý ảnh lúc suy luận, tăng cường có thay nền ngẫu nhiên, 6 cấu hình (`results/E1_xu_ly/pipeline_b/table.md`):

| Nhóm | Mô hình | Macro-F1 val | Macro-F1 test | KTC 95% | Accuracy test | Số ảnh sai / 371 | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0.1 | 0.980 | 0.988 | [0.974; 0.998] | 0.989 | 4 | 0.963 | 0.459 | 0.714 | 0.184 | 0.982 | 0.143 | 0.574 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0.934 | 0.960 | [0.938; 0.979] | 0.962 | 14 | 0.799 | 0.691 | 0.892 | 0.179 | 0.708 | 0.617 | 0.648 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0.1 | 0.946 | 0.953 | [0.931; 0.974] | 0.957 | 16 | 0.495 | 0.922 | 0.720 | 0.662 | 0.615 | 0.571 | 0.664 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0.998 | 0.984 | [0.969; 0.995] | 0.984 | 6 | 0.853 | 0.977 | 0.930 | 0.128 | 0.908 | 0.842 | 0.773 |
| CNN hướng A | a_vgg16 | 0.987 | 0.977 | [0.960; 0.991] | 0.978 | 8 | 0.907 | 0.960 | 0.953 | 0.965 | 0.943 | 0.516 | 0.874 |
| CNN hướng B | b_mobilenetv3 | 1.000 | 1.000 | [1.000; 1.000] | 1.000 | 0 | 0.975 | 0.954 | 0.935 | 0.995 | 1.000 | 0.441 | 0.883 |

Phép che 25% tách theo mức quả chuối bị che (`results/so_sanh_nhanh/table_occlusion_split.md`):

| Nhánh | Cấu hình | Quả bị che ≥ 90% (40 ảnh) | Quả bị che 30 đến 90% (286 ảnh) | Quả bị che < 30% (45 ảnh) |
|---|---|---|---|---|
| G | b_mobilenetv3 | 0.175 | 0.699 | 0.911 |
| G | Hist H-S + HOG + LogReg | 0.950 | 0.965 | 0.978 |
| X-a | b_mobilenetv3 | 0.125 | 0.451 | 0.667 |
| X-a | Hist H-S + HOG + LogReg | 0.750 | 0.850 | 0.844 |
| X-b | b_mobilenetv3 | 0.125 | 0.448 | 0.822 |
| X-b | Hist H-S + HOG + LogReg | 0.975 | 0.815 | 0.933 |
| T | b_mobilenetv3 | 0.550 | 0.717 | 1.000 |
| T | Hist H-S + HOG + LogReg | 0.975 | 0.962 | 1.000 |

S2 chạy lại với mask mới (`results/S2/table_mask_v2.md`, `results/S2/table_cnn_mask_v2.md`):

| Đặc trưng + bộ phân lớp | Phiên bản ảnh | Macro-F1 | Accuracy |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Toàn ảnh | 0.991 ± 0.005 | 0.992 ± 0.005 |
| HOG 64×128 + LogReg C=0.1 | Toàn ảnh | 0.986 ± 0.007 | 0.989 ± 0.006 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Toàn ảnh | 0.996 ± 0.003 | 0.997 ± 0.003 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng chuối | 0.964 ± 0.007 | 0.967 ± 0.005 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng chuối | 0.883 ± 0.005 | 0.886 ± 0.006 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng chuối | 0.966 ± 0.007 | 0.969 ± 0.007 |
| Hist HSV (8,8,8) + LogReg C=0.1 [màu tốt nhất] | Chỉ vùng nền | 0.933 ± 0.012 | 0.943 ± 0.011 |
| HOG 64×128 + LogReg C=0.1 | Chỉ vùng nền | 0.976 ± 0.005 | 0.980 ± 0.004 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng nền | 0.983 ± 0.006 | 0.986 ± 0.006 |

| Mô hình (học trên ảnh đầy đủ) | Toàn ảnh | Chỉ vùng chuối | Chỉ vùng nền |
|---|---|---|---|
| a_vgg16_lr1e-3 | 0.990 | 0.949 | 0.799 |
| b_mobilenetv3_lr1e-4 | 1.000 | 0.967 | 0.676 |
| b_resnet50 | 0.998 | 0.924 | 0.721 |

**Kết quả và kết luận.**
- **X1.** Làm sạch chỉ loại 35/1729 ảnh train. Phát hiện quan trọng hơn: 20.0% số ảnh có láng giềng gần nhất cùng lớp mang số thứ tự cách không quá 3 (ngẫu nhiên: 1.0%), và cross-validation chia theo khối ảnh liên tiếp thấp hơn chia ngẫu nhiên 0.024 đến 0.069 ở cả bốn mô hình. Khớp với mô tả của bài báo dữ liệu: mỗi quả được chụp bốn mặt. Ảnh của cùng một quả hoặc cùng một đợt chụp nằm ở cả hai phía khi chia ngẫu nhiên, nên mọi con số chấm bằng cách chia ngẫu nhiên, kể cả 1.000 trên test, lạc quan hơn so với khi gặp quả chuối chưa từng thấy.
- **X2-X4.** Tách nền là bước có tác dụng lớn nhất: điểm chọn tăng từ 0.804 (không xử lý) lên 0.878 (nền phẳng) và 0.889 ± 0.006 khi thêm tăng cường hình học/quang học; tổ hợp này phân biệt được với mọi tổ hợp khác theo bootstrap. Phần tăng đến từ phép nền hoán đổi (0.57 → 0.95, trung bình hai mô hình) và nhiễu (0.56 → 0.79). Cái giá là điểm trên ảnh gốc giảm (0.996 → 0.978) và phép che 25% kém đi (0.66 → 0.61).
- **X-a (9 cấu hình).** Sau khi tách nền, phép nền hoán đổi gần như không còn tác dụng: điểm ở cột "nền hoán đổi" xấp xỉ điểm trên ảnh gốc ở cả 9 cấu hình (`b_mobilenetv3` 0.995 và 0.995; nhánh G: 1.000 và 0.794). Điểm bền trung bình tăng từ 0.750 lên 0.801. Đổi lại, điểm trên ảnh gốc giảm ở những mô hình trước đó dựa nhiều vào nền: HOG 0.990 → 0.828, cạnh 0.964 → 0.842, `a_lenet` 0.978 → 0.922, `a_vgg16` 0.986 → 0.958; trung bình 9 cấu hình 0.983 → 0.942. Các mô hình mạnh nhất gần như giữ nguyên: `b_resnet50` 0.997, `b_mobilenetv3` 0.995 (2 ảnh sai), Hist HSV 0.993, Hist H-S + HOG 0.982. Điểm yếu: phép che 25% (0.17 đến 0.45, riêng Hist H-S + HOG 0.805), vì ô đen làm hỏng bước phân đoạn.
- **X-b (6 cấu hình).** Chỉ thêm ảnh train có nền thay ngẫu nhiên, không đụng tới ảnh lúc suy luận: `b_mobilenetv3` vẫn đúng 371/371 ảnh gốc và đạt 0.975 ở phép nền hoán đổi (G: 0.794); Hist H-S + HOG đạt 0.984 và 0.853 (G: 1.000 và 0.410); `a_vgg16` 0.977 và 0.907 (G: 0.986 và 0.657). Trung bình 6 cấu hình: ảnh gốc 0.977 (G 0.989; X-a 0.933), điểm bền 0.736 (G 0.708; X-a 0.789).
- **Phép che 25%** là chỗ cả hai biến thể X kém nhánh G ở mô hình CNN (0.44 đến 0.45 so với 0.66). Bảng tách theo mức bị che cho thấy một phần lý do với cấu hình truyền thống: ở 40 ảnh mà quả bị che gần hết (35 ảnh là Sabri Kola), cấu hình truyền thống của nhánh G vẫn đúng 0.950, tức nhận ra Sabri từ nền; sau khi tách nền (X-a) còn 0.750. Với `b_mobilenetv3` thì không phải vậy: ở nhóm đó các nhánh G, X-a, X-b đều sai gần hết (0.125 đến 0.175), còn chênh lệch nằm ở nhóm bị che một phần (0.699 so với 0.45); nguyên nhân chưa tách được.
- **Kiểm tra lại S2 với mask mới.** Kết luận của S2 không đổi. Với đặc trưng thủ công, chỉ vùng nền vẫn cho 0.933 đến 0.983 (mask cũ: 0.933 đến 0.986) và chỉ vùng chuối cho 0.883 đến 0.966 (mask cũ: 0.884 đến 0.964). CNN học trên ảnh đầy đủ rồi đoán trên ảnh chỉ còn nền đạt 0.676 đến 0.799 (mask cũ: 0.683 đến 0.792). Khác biệt đáng kể duy nhất: `b_mobilenetv3` trên ảnh chỉ còn quả chuối đạt 0.967 thay vì 0.927, vì mask cũ đôi khi giữ lại cả một mảng nền quanh quả.
- **Trả lời câu hỏi "xử lý nào là tối ưu".** Không có tổ hợp nào tốt hơn ở mọi mặt. Trên ảnh gốc, không bước xử lý nào cải thiện được pipeline gốc, vì nó đã đúng 371/371 ảnh. Nếu mục tiêu là mô hình nhìn quả chuối thay vì nền, thì với mô hình tốt nhất (`b_mobilenetv3`) biến thể X-b là lựa chọn cân bằng nhất: không mất điểm trên ảnh gốc, nền hoán đổi từ 0.794 lên 0.975, và không cần bước phân đoạn lúc dự đoán. X-a khử nền triệt để hơn (0.995) nhưng phụ thuộc vào phân đoạn và làm các mô hình yếu mất điểm.

**Ảnh hưởng tới bước sau.** Bảng so sánh bốn nhánh dùng cả X-a và X-b. Mô hình `b_mobilenetv3` của X-b được thêm vào demo như một lựa chọn bền hơn với nền ảnh. Hạn chế "mỗi quả chụp nhiều mặt" được đưa vào báo cáo.

## Bước 14: So sánh bốn nhánh; cập nhật báo cáo, slide Beamer, hỏi đáp, README (12/11/2026)

**Mục tiêu.** Đặt bốn nhánh G, X, T, B cạnh nhau trên cùng tập test và bộ kiểm tra khó; đưa kết quả phần mở rộng vào các sản phẩm phải nộp; chuyển slide sang Beamer LaTeX theo style UIT.

**Đã làm.**
- `python -m src.compare_arms`: `results/so_sanh_nhanh/table_models.md` (từng cấu hình qua các nhánh), `table_arms.md` và `table_stress.md` (cấu hình đại diện mỗi nhánh), `table_occlusion_split.md`; hình `results/figures/E_compare_arms.png`, `E_compare_stress.png`.
- Báo cáo: thêm mục 4.4 và mục 5.13 (bốn nhánh), cập nhật hạn chế, hướng phát triển và tài liệu tham khảo; `src/md2docx.py` đưa bảng từ 11 cột trở lên sang trang ngang.
- Slide: `src/beamer_builder.py` và `slides/beamerthemeUIT.sty` (màu xanh và cam lấy từ logo chính thức của trường, trang bìa có logo, mục lục, chân trang tên trường). `python -m src.make_slides` sinh từ một danh sách slide duy nhất: `slides/bao-cao-do-an.tex` và `.pdf` (39 slide), `bao-cao-do-an-ghi-chu.pdf` (kèm trang ghi chú thuyết trình sau mỗi slide) và bản `.pptx` cùng nội dung.
- `docs/hoi-dap-bao-ve.md`: thêm 15 câu về phần mở rộng (tổng 51 câu). `README.md`: phần mở rộng, các bước 10-14, cách cài PyTorch và LaTeX; bảng trong README được làm mới từ `results/` khi chạy `python -m src.build_report`.
- Demo: thêm mô hình `b_mobilenetv3_thay_nen` (nhánh X-b) vào `models/`.
- `scripts/run_all.sh`: các bước 10-14; bước 9 (demo, báo cáo, slide) chạy sau cùng.

**Lý do chọn cách này.**
- Cấu hình đại diện của mỗi nhánh chọn theo điểm lúc phát triển (macro-F1 trên val, hoặc 5-fold trên train+val với nhánh B), không theo test, như ở bước 8. **[lập luận]** Kết quả là cả bốn dòng dữ liệu (G, X-a, X-b, T) đều có đại diện là `b_mobilenetv3`, nên so được cùng một kiến trúc qua các nhánh.
- Slide Beamer và pptx sinh từ cùng một nguồn để hai bản không lệch nhau; số trên slide đọc từ `results/`. **[lập luận]**
- Mô hình mặc định của demo vẫn là `b_mobilenetv3` của pipeline gốc; bản X-b là lựa chọn thêm. **[đo]** Hai bản ngang nhau trên ảnh gốc (0 ảnh sai), bản X-b bền hơn với nền (0.975 so với 0.794) nhưng kém hơn khi bị che (0.441 so với 0.659); chưa có ảnh thật ngoài bối cảnh để quyết định thay hẳn (`results/so_sanh_nhanh/table_stress.md`).

**Phương án khác và vì sao không chọn.**
- Xếp hạng các nhánh bằng một con số duy nhất (điểm bền): không chọn làm kết luận chính. **[đo]** X-a, X-b và G có điểm bền gần như bằng nhau (0.881; 0.883; 0.875) dù khác nhau rõ ở từng phép thử: nền hoán đổi 0.995 / 0.975 / 0.794, che 25% 0.450 / 0.441 / 0.659 (`results/so_sanh_nhanh/table_stress.md`).
- Chỉ giữ bản slide pptx như kế hoạch ban đầu: không; bản chính đổi sang Beamer style UIT, kế hoạch mục 8 đã sửa. **[lập luận]**

**Bằng chứng.**

Từng cấu hình qua các nhánh dữ liệu (`results/so_sanh_nhanh/table_models.md`, hình `results/figures/E_compare_arms.png`):

| Cấu hình | G: macro-F1 test | G: số ảnh sai | G: điểm bền | X-a: macro-F1 test | X-a: số ảnh sai | X-a: điểm bền | X-b: macro-F1 test | X-b: số ảnh sai | X-b: điểm bền | T: macro-F1 test | T: số ảnh sai | T: điểm bền |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hist HSV + LogReg | 0.994 | 2 | 0.487 | 0.993 | 2 | 0.683 | 0.988 | 4 | 0.574 | 0.997 | 1 | 0.555 |
| Cạnh Roberts + KNN | 0.964 | 12 | 0.654 | 0.842 | 56 | 0.719 | 0.960 | 14 | 0.648 | 0.948 | 18 | 0.689 |
| HOG + LogReg | 0.990 | 3 | 0.617 | 0.828 | 63 | 0.704 | 0.953 | 16 | 0.664 | 0.972 | 8 | 0.620 |
| Hist H-S + HOG + LogReg | 1.000 | 0 | 0.728 | 0.982 | 5 | 0.890 | 0.984 | 6 | 0.773 | 0.989 | 4 | 0.745 |
| a_lenet | 0.978 | 7 | 0.837 | 0.922 | 26 | 0.778 | - | - | - | 0.966 | 10 | 0.780 |
| a_dropout | 0.942 | 21 | 0.769 | 0.959 | 15 | 0.816 | - | - | - | 0.963 | 13 | 0.796 |
| a_vgg16 | 0.986 | 4 | 0.886 | 0.958 | 16 | 0.857 | 0.977 | 8 | 0.874 | 0.985 | 5 | 0.905 |
| b_mobilenetv3 | 1.000 | 0 | 0.875 | 0.995 | 2 | 0.881 | 1.000 | 0 | 0.883 | 0.998 | 1 | 0.909 |
| b_resnet50 | 0.997 | 1 | 0.895 | 0.997 | 1 | 0.885 | - | - | - | 0.997 | 1 | 0.924 |
| **Trung bình 9 cấu hình** | **0.983** | **5.6** | **0.750** | **0.942** | **20.7** | **0.801** | - | - | - | **0.979** | **6.8** | **0.769** |
| **Trung bình 6 cấu hình có ở mọi nhánh** | **0.989** | **3.5** | **0.708** | **0.933** | **24.0** | **0.789** | **0.977** | **8.0** | **0.736** | **0.981** | **6.2** | **0.737** |

Cấu hình đại diện của mỗi nhánh (`results/so_sanh_nhanh/table_arms.md`, `results/so_sanh_nhanh/table_stress.md`, hình `results/figures/E_compare_stress.png`):

| Nhánh | Cấu hình đại diện | Điểm lúc chọn | Macro-F1 test | KTC 95% | Số ảnh sai / 371 | Điểm bền |
|---|---|---|---|---|---|---|
| G: pipeline gốc | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 1.000 | [1.000; 1.000] | 0 | 0.875 |
| X-a: xử lý dữ liệu: tiền xử lý lúc suy luận | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 0.995 | [0.987; 1.000] | 2 | 0.881 |
| X-b: xử lý dữ liệu: tăng cường có thay nền | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 1.000 | [1.000; 1.000] | 0 | 0.883 |
| T: Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 1.000 (macro-F1 trên val) | 0.998 | [0.991; 1.000] | 1 | 0.909 |
| B1: bài báo, trong phạm vi môn học (CNN + bộ phân lớp) | MobileNetV3-Small + SVM tuyến tính | 0.996 (5-fold trên train+val) | 0.993 | [0.982; 1.000] | 2 | 0.836 |
| B2: bài báo, đúng phương pháp (ViT + SVM) | ViT-B/16 + SVM | 0.994 (5-fold trên train+val) | 0.997 | [0.989; 1.000] | 1 | 0.919 |

| Nhánh | Cấu hình đại diện | Ảnh gốc | Nền hoán đổi | V×0.6 | V×1.4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|
| G: pipeline gốc | b_mobilenetv3 | 1.000 | 0.794 | 0.919 | 0.890 | 0.989 | 1.000 | 0.659 | 0.875 |
| X-a: xử lý dữ liệu: tiền xử lý lúc suy luận | b_mobilenetv3 | 0.995 | 0.995 | 0.943 | 0.915 | 0.988 | 0.998 | 0.450 | 0.881 |
| X-b: xử lý dữ liệu: tăng cường có thay nền | b_mobilenetv3 | 1.000 | 0.975 | 0.954 | 0.935 | 0.995 | 1.000 | 0.441 | 0.883 |
| T: Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 0.998 | 0.867 | 0.954 | 0.904 | 0.993 | 1.000 | 0.736 | 0.909 |
| B1: bài báo, trong phạm vi môn học (CNN + bộ phân lớp) | MobileNetV3-Small + SVM tuyến tính | 0.993 | 0.742 | 0.965 | 0.856 | 0.897 | 0.981 | 0.575 | 0.836 |
| B2: bài báo, đúng phương pháp (ViT + SVM) | ViT-B/16 + SVM | 0.997 | 0.857 | 0.990 | 0.978 | 0.986 | 0.978 | 0.728 | 0.919 |

**Kết quả và kết luận.**
1. **Trên ảnh gốc, không nhánh nào hơn pipeline gốc.** Cấu hình đại diện của sáu dòng sai từ 0 đến 2 ảnh trên 371; khoảng tin cậy chồng lên nhau. Pipeline gốc đã ở mức trần của tập test này. Tính trung bình 9 cấu hình, G cao nhất (0.983), T gần bằng (0.979), X-a thấp hơn (0.942) vì các mô hình yếu mất phần thông tin từ nền.
2. **Khác biệt nằm ở bộ kiểm tra khó, mỗi nhánh mạnh ở một chỗ.** Nhánh X gần như xóa được sự phụ thuộc vào nền (nền hoán đổi 0.975 đến 0.995 so với 0.794 của G). Nhánh T tăng độ bền đều hơn ở các phép có trong công thức tăng cường của tác giả (điểm bền 0.909). Nhánh B2 (ViT + SVM) có điểm bền cao nhất (0.919) mà không cần huấn luyện mạng, đổi lại mô hình nặng và nằm ngoài môn học. Nhánh B1 (CNN đóng băng + SVM tuyến tính) kém bền nhất (0.836).
3. **Tính trung bình 9 cấu hình, X-a có điểm bền cao nhất** (0.801; T 0.769; G 0.750), nhờ các cấu hình truyền thống được lợi nhiều từ việc tách nền (Hist H-S + HOG: 0.728 → 0.890).
4. **Tăng cường dữ liệu chỉ có ích khi nhắm đúng loại biến đổi cần bền.** Bản Augmented của tác giả không có phép nào liên quan tới nền nên không giảm được thiên lệch nền; phép thay nền ngẫu nhiên của nhánh X thì có.
5. **Hai nguồn lạc quan áp dụng cho mọi nhánh và cho cả số của bài báo:** thiên lệch nền (S2, bước 10) và ảnh của cùng một quả nằm ở cả hai phía khi chia ngẫu nhiên (bước 13). Riêng giao thức của bài báo còn thêm rò rỉ do chia fold trên bản Augmented (khoảng 0.3 điểm phần trăm, bước 12).

**Ảnh hưởng tới bước sau.** Đồ án dừng ở đây. Việc còn lại thuộc về nhóm: điền bảng thành viên, kiểm tra nội dung trước khi nộp, và nếu có điều kiện thì chụp một tập ảnh ngoài bối cảnh để kiểm tra các kết luận về độ bền.

---

## Bước 15: Đưa trọng số CNN, bản Augmented và log vào repo; đo lại dung lượng file mô hình (13/11/2026)

**Mục tiêu.** Người clone repo có sẵn mọi thứ để xem, kiểm tra và chạy lại kết quả mà không phải huấn luyện lại CNN: trọng số của 54 lần huấn luyện, bản Augmented của tác giả, log của các lần chạy. Trước bước này ba thứ đó chỉ nằm trên máy chạy thí nghiệm.

**Đã làm.**
- `python -m src.cnn.checkpoints --slim`: lưu lại `checkpoints/<run>/best.keras` của 54 run ở dạng chỉ gồm kiến trúc và trọng số, bỏ trạng thái của bộ tối ưu Adam; so từng mảng trọng số với bản cũ (và với `best.weights.h5`) rồi mới xóa bản cũ. Bảng: `results/cnn/table_slim.md`.
- `src/cnn/train.py`: huấn luyện xong thì `fit_loop` lưu `best.keras` ở dạng trên (`save_inference`); run đã xong thì `--resume` nạp `best.keras`, không huấn luyện lại. `src/cnn/predict.py`, `src/exp_cnn.py`: đầu phân lớp giai đoạn 1 đọc từ `best.keras`.
- `python -m src.cnn.checkpoints`: nạp từng `best.keras`, dự đoán lại và so với kết quả đã ghi. Bảng: `results/cnn/table_checkpoints.md`.
- Đo lại dung lượng file mô hình trên file mới: cột "File mô hình (MB)" của N6 (`results/N6/table.md`, `results/figures/N6_a_vs_b.png`), trường `best_model_file_mb` của 33 run trong `results/cnn/`, mô hình demo trong `models/` (`python -m src.export_demo`). `src/exp_cnn.py` tách `full_model_file` và `n6_render` để bảng và hình N6 dựng lại được từ `results/N6/metrics.json`; các số đo thời gian của N6 giữ nguyên, không đo lại.
- `src/pipeline.py`: các nhánh mở rộng mặc định tính lại mọi cấu hình, CNN dùng lại trọng số trong `checkpoints/` và không chuẩn bị dữ liệu train khi run đã xong; `SKIP_DONE=1` bỏ qua cấu hình đã có kết quả. `scripts/run_all.sh`: thêm `RETRAIN=1` (xóa `checkpoints/` để huấn luyện lại từ đầu).
- `.gitignore`: commit `checkpoints/<run>/best.keras` và `state.json`, `data/aug_author/` (7413 ảnh), `logs/` (64 file). `data/cache/` vẫn không commit.
- Sửa con số dung lượng trong báo cáo, slide, hỏi đáp, README; README thêm mục "Những gì có sẵn trong repo"; kế hoạch mục 8 thêm `checkpoints/`, `data/aug_author/`, `logs/`.

**Lý do chọn cách này.**
- GitHub từ chối file trên 100 MiB (104.86 MB). **[đo]** Trước khi chuyển có 14 file `best.keras` vượt mức này (`a_vgg16` 311.49 MB × 11 run, `b_resnet50` 214.77 MB × 3 run); sau khi chuyển, file lớn nhất là 103.84 MB (`results/cnn/table_slim.md`).
- Bỏ trạng thái Adam không làm đổi dự đoán. **[đo]** 0/3920 mảng trọng số khác bản cũ khi so từng bit (`results/cnn/table_slim.md`); 40/40 run được chấm lại cho đúng macro-F1 và số ảnh sai đã ghi (`results/cnn/table_checkpoints.md`).
- Dung lượng file ghi ở N6 trước đây gồm cả trạng thái Adam (hai mảng moment cho mỗi trọng số huấn luyện được), nên lớn hơn mô hình dùng để dự đoán 2.3-3 lần. **[đo]** File mới xấp xỉ 4 byte × số tham số (bảng dưới). Con số "11.7 MB" của `b_mobilenetv3` trong báo cáo, slide và README là của file cũ; nay là 4.3 MB.
- Các nhánh mở rộng trước đây bỏ qua cấu hình đã có `metrics.json`, nên chạy lại bước 11 hoặc 13 trên repo đã có kết quả thì không tính lại gì. **[lập luận]** Chạy lại phải tính lại được kết quả từ dữ liệu và trọng số; việc bỏ qua chỉ cần khi chạy tiếp một lần chạy bị ngắt, nên chuyển thành tùy chọn `SKIP_DONE=1`.

**Phương án khác và vì sao không chọn.**
- Giữ file kèm trạng thái Adam và dùng Git LFS: không chọn. **[đo]** Tổng 4458 MB so với 1563 MB sau khi chuyển. **[lập luận]** LFS có hạn mức dung lượng và băng thông theo tài khoản; người clone không cài `git-lfs` chỉ nhận được file con trỏ. Trạng thái Adam chỉ cần khi huấn luyện tiếp một run đã dừng, việc đồ án không làm.
- Cắt file lớn thành nhiều phần dưới 100 MB: không chọn. **[lập luận]** Người chạy lại phải ghép file trước khi dùng; file sau khi bỏ trạng thái Adam đã dưới giới hạn.
- Lưu trọng số dạng float16 để giảm thêm một nửa: không chọn. **[lập luận]** Cách này đổi giá trị trọng số, không còn bảo đảm dự đoán trùng từng ảnh với kết quả đã ghi.
- Để trọng số ở một nơi lưu trữ ngoài repo rồi dẫn link: không chọn. **[lập luận]** Người chạy lại phải tải thêm từ nguồn thứ hai và link có thể hết hạn; để trong repo thì một lệnh clone là đủ.
- Commit cả `data/cache/`: không chọn. **[đo]** Thư mục nặng 8.3 GiB, có 15 file trên 100 MiB, file lớn nhất 1.36 GiB (`du -sh data/cache`, `find data/cache -size +100M`). **[lập luận]** Đây là đặc trưng tính lại được từ ảnh và code, không cần huấn luyện.
- Đo lại toàn bộ N6 (kể cả thời gian): không chọn. **[lập luận]** Dung lượng file là đại lượng tất định, còn thời gian phụ thuộc tải của máy lúc đo; đo lại sẽ làm đổi các số thời gian đã dùng trong báo cáo mà không có lý do.

**Bằng chứng.**

Dung lượng `best.keras` trước và sau khi bỏ trạng thái Adam, gộp theo kiến trúc (`results/cnn/table_slim.md` liệt kê từng run):

| Kiến trúc | Số run | Số tham số | Mỗi file trước (MB) | Mỗi file sau (MB) | Tổng trước (MB) | Tổng sau (MB) | Số mảng trọng số | Số mảng khác (so từng bit) |
|---|---|---|---|---|---|---|---|---|
| a_lenet (64×64) | 6 | 337976 | 4.10 | 1.39 | 25 | 8 | 60 | 0 |
| a_lenet (32×32) | 1 | 61496 | 0.79 | 0.28 | 1 | 0 | 10 | 0 |
| a_lenet (128×128) | 1 | 1628216 | 19.59 | 6.55 | 20 | 7 | 10 | 0 |
| a_dropout | 9 | 1746260 | 21.02 | 7.04 | 189 | 63 | 90 | 0 |
| a_vgg16 (phần đầu phân lớp) | 11 | 25954564 | 311.49 | 103.84 | 3426 | 1142 | 66 | 0 |
| b_mobilenetv3 | 13 | 941428 | 11.74 | 4.34 | 153 | 56 | 2704 | 0 |
| b_resnet50 | 3 | 23595908 | 214.77 | 95.05 | 644 | 285 | 960 | 0 |
| Đầu phân lớp giai đoạn 1 (hướng B) | 10 | 2308-8196 | 0.05 đến 0.12 | 0.03 đến 0.05 | 1 | 0 | 20 | 0 |
| **Tổng** | **54** | | | | **4458** | **1563** | **3920** | **0** |

Chấm lại bằng trọng số trong repo (`results/cnn/table_checkpoints.md`):

| Nhóm run | Tập chấm | Số run chấm lại | Số run trùng macro-F1 và accuracy đã ghi |
|---|---|---|---|
| Bước 4-6 (có kết quả trong `results/cnn/`) | val, 371 ảnh | 33 | 33 |
| Nhánh T-kr và X-b (chấm trên ảnh test gốc) | test, 371 ảnh | 7 | 7 |
| Nhánh X-a, T-tl và đầu phân lớp giai đoạn 1 của các nhánh | không chấm lại bằng lệnh này | 14 |: (trọng số đã so từng bit) |

Dung lượng file mô hình đầy đủ ở N6 (`results/N6/table.md`; bản cũ nằm trong lịch sử git của file này):

| Mô hình | Số tham số | File cũ, kèm trạng thái Adam (MB) | File mới (MB) | 4 byte × số tham số (MB) |
|---|---|---|---|---|
| a_lenet | 337976 | 4.1 | 1.4 | 1.4 |
| a_dropout | 1746260 | 21.0 | 7.0 | 7.0 |
| a_vgg16 | 40669252 | 370.4 | 162.8 | 162.7 |
| b_mobilenetv3 | 941428 | 11.7 | 4.3 | 3.8 |
| b_resnet50 | 23595908 | 214.8 | 95.1 | 94.4 |

Dựng lại bảng N6 từ `results/N6/metrics.json` bằng `n6_render` trước khi đổi dung lượng cho ra đúng bảng đã commit (`git diff` rỗng), nên các cột khác không bị ảnh hưởng.

Chạy lại lệnh huấn luyện khi đã có trọng số (5 lệnh của bước 4-5 với `--resume`: `a_lenet`, `a_vgg16_lr1e-3`, `b_mobilenetv3_s1`, `b_mobilenetv3_lr1e-4`, `b_resnet50`): cả 5 in "run đã hoàn tất … dùng lại trọng số đã lưu", tổng 88 giây. So với bản đã ghi: `history.json`, `classification_report.txt`, `confusion_matrix.png`, `curves.png` trùng từng byte; trong `metrics.json` chỉ khác trường thời gian `predict_ms_per_image_batched_val`.

Chạy lại cả nhánh T-kr (`python -m src.exp_arm_t --only kr`, 9 cấu hình, 5 CNN dùng lại trọng số): 401 giây; `git status results/E2_aug_tac_gia` không có file nào thay đổi, tức `metrics.json`, `predictions.npz`, bảng và hình của cả 9 cấu hình trùng từng byte với bản đã commit.

Clone mới từ GitHub sau khi push (`git clone`, nhánh `dev`): 80 giây, tải 1.5 GB, chiếm 3.2 GB trên đĩa; có đủ 54 file `best.keras`, 7413 ảnh trong `data/aug_author/`, 64 file log; mã băm SHA-256 của `checkpoints/b_mobilenetv3_lr1e-4/best.keras` trùng file trên máy chạy thí nghiệm. Clone không kèm `checkpoints/` và `data/aug_author/` (lệnh ở README, mục "Tải repo"): 6 giây, 123 MB.

**Kết quả và kết luận.**
- Repo có sẵn trọng số của cả 54 lần huấn luyện (1563 MB), bản Augmented của tác giả (7413 ảnh) và log. Chạy lại bất kỳ bước nào cũng không phải huấn luyện lại; thời gian huấn luyện đã ghi trong `state.json` của 54 run cộng lại là 3.3 giờ.
- Các con số về độ chính xác tái lập được từ trọng số trong repo: 40/40 run chấm lại trùng kết quả đã ghi, nhánh T-kr tính lại trùng từng byte.
- Con số dung lượng file mô hình trong các sản phẩm trước bước này lớn hơn thực tế vì tính cả trạng thái Adam. Sau khi đo lại, nhận định ở N6 không đổi (`b_mobilenetv3` nhỏ và nhanh hơn `a_vgg16`), chỉ đổi giá trị: 4.3 MB so với 162.8 MB.
- Hai thứ không tái lập được trùng từng số: các số đo thời gian (phụ thuộc máy) và việc huấn luyện lại từ đầu trên máy hoặc phiên bản thư viện khác.

**Ảnh hưởng tới bước sau.** Từ nay mỗi run chỉ để lại `best.keras` và `state.json`; `best.weights.h5` và `last.keras` chỉ tồn tại trong lúc huấn luyện. `data/cache/` không có trong repo nên lần chạy lại đầu tiên phải rút lại đặc trưng.

---

## Bước 15 (bổ sung): Chạy lại toàn bộ trong một bản clone mới, không huấn luyện lại (13/11/2026)

**Mục tiêu.** Kiểm tra lời hứa "clone về là chạy lại được": lấy repo từ GitHub về một thư mục trống (không có `data/cache/`), chạy `python -m src.cnn.checkpoints` rồi `bash scripts/run_all.sh` cho bước 1-8 và 10-14, đo thời gian từng bước và so mọi file sinh ra với bản đã commit.

**Đã làm.**
- Clone `dev` (commit `5889e1b`) vào một thư mục riêng; chạy từng bước, ghi thời gian và log của mỗi bước.
- Máy chạy thí nghiệm bị khởi động lại giữa bước 13 (sau khi xong nhánh X1, X2); bước 13 và 14 được chạy lại từ đầu. Thời gian bước 13 vì thế được đo khi một phần cache của X1-X2 đã có sẵn.
- So từng file trong `results/`, `models/`, `report/`, `slides/` với bản đã commit: file JSON so theo từng trường, bảng `.md` và `summary.csv` so theo từng cột, mô hình `.keras` so từng mảng trọng số, hình `.png` so từng điểm ảnh.
- Sửa hai chỗ chưa nhất quán mà lần so này tìm ra (xem Kết quả).

**Lý do chọn cách này.** Chạy trong bản clone mới thì không có file nào "dính" từ máy cũ (cache, checkpoint tạm, kết quả cũ), nên đây là phép thử gần nhất với việc người khác tải repo về chạy.

**Phương án khác và vì sao không chọn.**
- Chạy lại ngay trong thư mục làm việc: cache và file trung gian còn sẵn, không phát hiện được file thiếu trong repo. **[lập luận]**
- Chỉ chạy lại một nhánh (đã làm ở Bước 15 với T-kr, 401 giây): không bao phủ bước 1-10 và 12-14. **[đo]**

**Bằng chứng.** **[đo]**

Thời gian (CPU 4 nhân, không huấn luyện lại):

| Bước | Thời gian |
|---|---|
| Kiểm tra trọng số (`src.cnn.checkpoints`) | 267 s |
| 1 | 92 s |
| 2 | 303 s |
| 3 | 1129 s |
| 4 | 196 s |
| 5 | 856 s |
| 6 | 822 s |
| 7 | 418 s |
| 8 | 84 s |
| 10 | 347 s |
| 11 | 757 s |
| 12 | 1390 s |
| 13 | 2470 s |
| 14 (gồm bước 9: mô hình demo, báo cáo, slide) | 45 s |
| **Tổng** | **9176 s ≈ 2.5 giờ** |

Khác biệt so với bản đã commit (mọi bước đều kết thúc với mã thoát 0):

| Loại file | Kết quả so sánh |
|---|---|
| JSON trong `results/` (81 file) | Chỉ khác ở các trường thời gian (`predict_ms_per_image`, `fit_time_s`, `predict_time_s`, `extract_ms_per_image`, `train_time_s` của N6, …), trừ hai chỗ nêu dưới |
| `results/summary.csv` (968 dòng × 14 cột) | Chỉ khác 3 cột: `extract_ms_per_image`, `fit_time_s`, `predict_ms_per_image` |
| Bảng `.md` trong `results/` (8 file) | Chỉ khác ở cột thời gian (C3, M4, N6, P4, P5, chọn cấu hình truyền thống) |
| `models/*.keras` (3 file) | 0 mảng trọng số khác (10 + 208 + 208 mảng); cùng dung lượng, chỉ khác dấu thời gian trong file zip |
| Báo cáo, slide | Chỉ khác ở các bảng thời gian lấy từ `results/` |
| Hình | 4 hình có trục hoặc nhãn ghi thời gian; `E_stress_examples.png` khác 56/1255254 điểm ảnh ở 2 hàng sát mép dưới, cả ở cột ảnh gốc, tức do khâu vẽ chứ không do dữ liệu |
| `results/environment.txt` | Khác phiên bản nhân Linux của máy |

Hai chỗ chưa nhất quán được sửa:

| Chỗ | Trước | Sau |
|---|---|---|
| `best_model_file_mb` trong `results/N3`, `N4`, `N5/metrics.json` (12 giá trị) | Vẫn là dung lượng file kèm trạng thái Adam (ví dụ `b_mobilenetv3` 11.74 MB) vì Bước 15 chỉ sửa `results/cnn/` | Dung lượng `best.keras` hiện tại (4.34 MB) |
| Trường `seed`, `minority_aug` trong `metrics.json` của 15 run cũ và bản tổng hợp N3-N5 | Không có (các run này chạy trước khi thêm hai tùy chọn) | `seed: 42`, `minority_aug: false`, đúng cấu hình lúc huấn luyện |

Bảng `.md`, báo cáo và slide không dùng hai trường này nên không đổi.

**Kết quả và kết luận.** Bản clone từ GitHub chạy lại được toàn bộ thí nghiệm trong khoảng 2.5 giờ mà không huấn luyện lại; không có macro-F1, accuracy, ma trận nhầm lẫn hay lựa chọn mô hình nào khác bản đã commit. Sau khi sửa hai chỗ trên, lần chạy lại tiếp theo chỉ còn khác ở số đo thời gian.

**Ảnh hưởng tới bước sau.** README có thêm cột "Chạy lại, dùng trọng số có sẵn (đo)" bên cạnh cột thời gian huấn luyện lại từ đầu.

## Bước 16: Demo nhiều tab: dự đoán, so sánh, Grad-CAM, thử độ bền, nhiều ảnh, đánh giá theo metric, kết quả (13/11/2026)

**Mục tiêu.** Mở rộng demo (Bước 9 chỉ có một màn hình tải ảnh và dự đoán) để khi bảo vệ trình bày được cả cách mô hình ra quyết định, độ bền của nó và các metric cơ bản của môn học, bằng chính dữ liệu và mô hình của đồ án.

**Đã làm.**
- `app/core.py`: phần xử lý không phụ thuộc giao diện: nạp mô hình trong `models/`, dự đoán một ảnh hoặc cả lô, Grad-CAM, áp các phép biến đổi của bộ kiểm tra khó lên một ảnh, đọc tập val/test từ `data/raw` theo `data/splits/`, sinh lại 6 phiên bản của bộ kiểm tra khó, tính accuracy, precision, recall, F1, TP/FP/FN và ma trận nhầm lẫn.
- `app/app.py`: giao diện Gradio 7 tab (Dự đoán; So sánh mô hình; Giải thích (Grad-CAM); Thử độ bền; Nhiều ảnh; Đánh giá theo metric; Kết quả thực nghiệm). Chế độ dòng lệnh `--image` giữ nguyên.
- Ảnh chụp giao diện: `results/figures/demo_screenshot.png`, `demo_gradcam.png`, `demo_metric.png`; cập nhật README, báo cáo (mục 5.12), slide (thêm một trang Grad-CAM), tài liệu hỏi đáp.
- Kiểm tra bằng trình duyệt: mở từng tab, tải ảnh, bấm từng nút, đọc kết quả hiển thị.

**Lý do chọn cách này.**
- Giữ Gradio như Bước 9: đã có trong `requirements.txt`, có sẵn webcam, dán ảnh, thanh xác suất; lệnh chạy không đổi. **[lập luận]**
- Tab metric tính lại từ ảnh và mô hình thay vì chép số từ `results/`, để người xem tự kiểm được rằng con số trong báo cáo ra từ mô hình đang chạy. **[lập luận]**
- Grad-CAM dùng điểm lớp trước softmax: với MobileNetV3 xác suất thường bằng 1.000 nên gradient của xác suất gần 0 và bản đồ nhiệt không có nghĩa. **[lập luận]**
- Hàm dự đoán một ảnh được biên dịch thành đồ thị (`tf.function`) thay vì gọi mô hình trực tiếp. **[đo]**, xem bảng dưới.

**Phương án khác và vì sao không chọn.**
- Streamlit: phải thêm thư viện và đổi lệnh chạy. **[lập luận]**
- Trang web tĩnh chạy TF.js: không chạy được mô hình truyền thống (HOG + histogram màu, scikit-learn), Grad-CAM và tab metric khó làm hơn nhiều. **[lập luận]**
- Tab metric đọc sẵn kết quả từ `results/` thay vì tính lại: nhanh hơn nhưng không chứng minh được số liệu ra từ mô hình. **[lập luận]**

**Bằng chứng.** **[đo]**

Tab "Đánh giá theo metric" so với kết quả đã ghi (test 371 ảnh, ảnh gốc):

| Mô hình | Tính lại trong demo | Đã ghi (`models/models.json`) |
|---|---|---|
| `b_mobilenetv3` | accuracy 1.0000, macro-F1 1.0000, sai 0 | 1.0000 / 1.0000 |
| `a_lenet` | accuracy 0.9811, macro-F1 0.9777, sai 7 | 0.9811 / 0.9777 |
| `b_mobilenetv3_thay_nen` | accuracy 1.0000, macro-F1 1.0000, sai 0 | 1.0000 / 1.0000 |
| `trad_best` | accuracy 1.0000, macro-F1 1.0000, sai 0 | 1.0000 / 1.0000 |

| Kiểm tra | Kết quả |
|---|---|
| 6 phiên bản bộ kiểm tra khó sinh trong demo so với `data/cache/stress_test.npz` của thí nghiệm | 6/6 giống hệt từng điểm ảnh |
| `b_mobilenetv3`, nền hoán đổi, tính trong demo | macro-F1 0.7939; `results/so_sanh_nhanh/table_stress.md` ghi 0.794 |
| `b_mobilenetv3`, val, ảnh gốc | macro-F1 1.0000, khớp `results/cnn/b_mobilenetv3_lr1e-4/metrics.json` |
| Xác suất tính theo đường Grad-CAM so với dự đoán của mô hình (3 CNN) | chênh lệch lớn nhất 0.0 |
| Lưới Grad-CAM | MobileNetV3: 7×7; LeNet: 26×26 |
| Kiểm tra bằng trình duyệt 7 tab | mọi nút chạy, không có lỗi trên trang |

Thời gian dự đoán một ảnh (CPU, trung bình 10 lần):

| Mô hình | Gọi trực tiếp (ms) | Hàm đã biên dịch (ms) |
|---|---|---|
| `b_mobilenetv3` | 135.6 | 12.2 |
| `a_lenet` | 9.7 | 1.6 |
| `b_mobilenetv3_thay_nen` | 132.8 | 11.9 |

**Kết quả và kết luận.** Demo có 7 tab; mọi số liệu trong demo hoặc tính trực tiếp từ ảnh và mô hình của đồ án, hoặc đọc từ `results/`, và các số tính lại khớp với số đã ghi. Tab Thử độ bền cho thấy trực tiếp thiên lệch theo nền đã đo ở S2 và ở bộ kiểm tra khó. Ngưỡng nhắc "mô hình không chắc chắn" (xác suất cao nhất dưới 0.6) chỉ là quy ước của demo, không rút ra từ thí nghiệm, và được ghi rõ như vậy trên giao diện.

**Ảnh hưởng tới bước sau.** Không đổi kết quả thí nghiệm. Báo cáo, slide, README và tài liệu hỏi đáp mô tả demo mới.

## Bước 17: Báo cáo, slide và kịch bản thuyết trình theo yêu cầu trình bày của môn học (13/11/2026)

**Mục tiêu.** Trình bày lại báo cáo và slide theo phiếu đánh giá đồ án và theo hướng dẫn báo cáo của giảng viên trên lớp, soạn kịch bản thuyết trình. Không làm thêm thí nghiệm, không đổi số liệu.

**Đã làm.**
- Báo cáo (`report/report_template.md`, dựng bằng `python -m src.build_report`): thêm trang bìa có tên đề tài và ba thành viên; mục 2 thêm cơ sở thực hiện và mục tiêu; mục 3 phát biểu bài toán theo dạng "Cho, Tìm", có Input, Output, ràng buộc và hình minh họa; mục 4 viết lại theo kiểu mỗi phương pháp có sẵn gồm hai ý "là gì, đề xuất ở đâu" và "nhóm dùng như thế nào", thêm bảng điểm khác biệt giữa các phương pháp; mục 5 thêm cách đọc ma trận nhầm lẫn theo tỉ lệ hàng và bảng "thực nghiệm khác với tài liệu tham khảo"; mục 6 thêm nhận xét và kết luận; mục 7 tăng từ 20 lên 40 tài liệu, thêm nguồn của dữ liệu, hình, mô hình huấn luyện sẵn và thư viện.
- `src/build_report.py`, `src/md2docx.py`: giữ dấu chấm thập phân của `results/`, trang bìa, số trang, đánh số hình, xuất thêm `report/report.pdf` bằng LibreOffice.
- `src/fig_overview.py`: hai hình mới `results/figures/problem_io.png` (ảnh vào, mô hình, xác suất và nhãn ra) và `method_overview.png` (sơ đồ hai hướng giải quyết).
- Slide (`src/make_slides.py`, `src/beamer_builder.py`): 26 slide chính theo thứ tự của phiếu đánh giá; các slide chi tiết của từng thí nghiệm chuyển sang `src/slides_detail.py` làm phụ lục.
- `docs/kich-ban-thuyet-trinh.md` (sinh từ cùng danh sách slide) và `docs/kich-ban-phan-co-dinh.md`: lời thuyết trình cho từng slide, kịch bản demo, câu hỏi dự trù, những điểm cần nhớ.
- Quy ước viết số và dấu câu: số thập phân dùng dấu chấm, số hàng nghìn viết liền, không dùng gạch dài. Áp dụng cho báo cáo, slide, README, kế hoạch, nhật ký, tài liệu hỏi đáp, nhãn trong demo, và cho tiêu đề bảng, tiêu đề hình trong code (25 file trong `src/`, `app/`, `scripts/`).

**Lý do chọn cách này.**
- Thứ tự slide và nội dung từng phần bám theo các mục chấm điểm của phiếu đánh giá (lý do, phát biểu bài toán, hình minh họa, phương pháp, dataset, độ đo, kết quả, demo, nhận xét, slide, hỏi đáp), để mỗi mục chấm có đúng một chỗ trên slide. **[lập luận]**
- Phương pháp trình bày trước, dataset và thực nghiệm trình bày sau, theo hướng dẫn trên lớp. **[lập luận]**
- Với phương pháp có sẵn, slide chỉ nói ngắn nó là gì kèm link tài liệu gốc, phần chính là nhóm dùng nó như thế nào (phiên bản, số chiều đặc trưng, sửa gì cho bài toán 4 lớp). **[lập luận]**
- Số thập phân dùng dấu chấm cho thống nhất với bảng số liệu do code sinh ra và với cách viết số trong môn học. **[lập luận]**
- Tiêu đề hình và bảng sửa ngay trong code chứ không sửa trên file kết quả, để lần chạy lại sau cho ra đúng tiêu đề mới. **[lập luận]**

**Phương án khác và vì sao không chọn.**
- Giữ bộ slide cũ 45 trang đi theo từng thí nghiệm: quá dài cho một buổi báo cáo và không theo thứ tự của phiếu đánh giá. Nội dung đó vẫn giữ ở phụ lục để dùng khi hỏi đáp. **[lập luận]**
- Viết một báo cáo thứ hai ngắn hơn: báo cáo hiện có đã theo đúng 7 mục được yêu cầu, nên chỉ cần chỉnh cách trình bày; hai báo cáo dễ lệch số liệu với nhau. **[lập luận]**
- Chạy lại toàn bộ thí nghiệm trong thư mục chính để có hình với tiêu đề mới: số đo thời gian sẽ đổi theo, kéo theo phải sửa mọi chỗ đã dẫn số đo thời gian. Thay vào đó, nhãn trong các file bảng và số liệu được sửa bằng thay chuỗi đúng như code mới sinh ra, hình được sinh lại trong một bản clone riêng rồi chép sang (bước 17, phần tiếp theo). **[lập luận]**

**Bằng chứng.** **[đo]**

| Sản phẩm | Kết quả |
|---|---|
| `report/report.pdf` | 76 trang, 36 hình, 76 bảng, 40 tài liệu tham khảo |
| `report/report.md` | 0 gạch dài, 0 số thập phân viết bằng dấu phẩy |
| `slides/bao-cao-do-an.pdf` | 54 trang: 26 slide chính, 28 slide phụ lục; mọi trang có số trang |
| `slides/bao-cao-do-an-ghi-chu.pdf` | 108 trang (mỗi slide kèm một trang lời thuyết trình) |
| `docs/kich-ban-thuyet-trinh.md` | 26 slide, tổng thời gian dự kiến khoảng 18 phút, 22 câu hỏi dự trù |
| Code trong `src/`, `app/`, `scripts/` | 0 chuỗi còn gạch dài |
| Bảng và số liệu trong `results/` | 83 file được sửa nhãn (tiêu đề bảng, nhãn V×0.6, V×1.4, ô trống); không số liệu nào đổi |

Đối chiếu nhãn đã sửa với kết quả do code mới sinh ra, trên các bước đã chạy lại trong bản clone tính tới lúc ghi mục này (bước 1 và 2: D1, M1 đến M6, P1 đến P6): mọi file bảng và số liệu trùng nhau, chỉ khác số đo thời gian.

Bốn hình có ghi số đo thời gian (`M4_color_features.png`, `P4_knn_algorithms.png`, `P5_time.png`, `N6_a_vs_b.png`) được vẽ lại từ `metrics.json` đã commit, nên số đo thời gian trên hình không đổi. Ba ảnh chụp giao diện demo được chụp lại sau khi sửa nhãn mô hình.

**Kết quả và kết luận.** Báo cáo, slide và kịch bản đã theo cấu trúc được yêu cầu, số liệu giữ nguyên. Các hình còn lại trong `results/figures/` vẫn mang tiêu đề cũ cho tới khi bản chạy lại trong clone hoàn tất và được chép sang.

**Ảnh hưởng tới bước sau.** Bước 9 của `scripts/run_all.sh` giờ chạy thêm `src.fig_overview` và sinh thêm `report/report.pdf`, `docs/kich-ban-thuyet-trinh.md`. Trong báo cáo, hai cột "Công việc phụ trách", "Mức độ hoàn thành" ở mục 1 và mục 7.3 để nhóm tự điền. Tên người trình bày từng phần điền ở biến `PRESENTERS` trong `src/make_slides.py`.

## Bảng quyết định

| Quyết định | Phương án thay thế | Bằng chứng | Nhãn |
|---|---|---|---|
| Dùng ảnh gốc Version 2 | Bản Augmented; Version 1 | Bước 0.1; chờ D1 | [lập luận] |
| TensorFlow/Keras cho CNN | PyTorch | Bước 0.2 | [lập luận] |
| Chia 70/15/15, test dùng một lần | Chỉ train/test; cross-validation cho CNN | Bước 0.3 | [lập luận] |
| macro-F1 là chỉ số chính | Accuracy | Bước 0.3; chờ I1 | [lập luận] |
| Không dùng bản Augmented; chỉ tăng cường phần huấn luyện | Tăng cường rồi chia | Bước 1, `results/D1/table.md`: thổi phồng +0.021 (HSV + KNN), +0.009 (HSV + LogReg); ≈ 0 với HOG | [đo] |
| D1 và các phép biến đổi kiểm chứng chấm trên fold giữ lại / val, không dùng test | Dùng tập test chính thức | Bước 1 | [lập luận] |
| Giữ cách chia stratified ngẫu nhiên | Chia theo nhóm ảnh gần trùng | Bước 1, `results/D1/table_near_duplicates.md`: chênh ≤ 0.007, nhỏ hơn độ lệch chuẩn | [đo] |
| Ngưỡng gần trùng pHash: Hamming ≤ 2 | ≤ 4, ≤ 6 | Bước 1, `results/figures/near_duplicates_by_distance.png` | [đo] |
| HOG tự cài đặt (có tùy chọn bỏ chuẩn hóa block) | `skimage.feature.hog` | Bước 1: sai khác lớn nhất 2.2·10⁻⁷ so với skimage | [đo] |
| Chấm mỗi đặc trưng bằng cả KNN và LogReg | Chỉ một bộ phân lớp | Bước 2, `results/M4/table.md`: thứ hạng đặc trưng đổi theo bộ phân lớp | [đo] |
| Đặc trưng bền với độ sáng: histogram H-S (bỏ V) | Histogram HSV đủ 3 kênh | Bước 2, `results/M2/table.md`: sụt ≤ 0.027 so với 0.33 đến 0.67 | [đo] |
| Histogram HSV dùng (8,8,8) bin | 2, 4, 16 bin mỗi kênh | Bước 2, `results/M3/table.md`: 8 bin đạt 0.991 (LogReg); 16 bin ngang nhưng 4096 chiều và KNN giảm còn 0.917 | [đo] |
| k của KNN chọn bằng cross-validation | Quy tắc k = số lớp + 1 | Bước 2, `results/P1/table.md`: k=1 hơn k=5 từ 0.006 đến 0.020 | [đo] |
| KNN dùng Brute Force | K-D Tree, Ball Tree | Bước 2, `results/P4/table.md`: brute nhanh nhất với 1680 ảnh tham chiếu, kết quả trùng nhau | [đo] |
| Đặc trưng nhiều chiều dùng bộ phân lớp tuyến tính, đặc trưng ít chiều dùng KNN | Một bộ phân lớp cho mọi đặc trưng | Bước 2, `results/P6/table.md` | [đo] |
| HOG 64×128, cell 8×8, 9 bin, ảnh xám, chia phiếu hai bin | 11 cấu hình kích thước/cell/bin khác; ảnh màu; phiếu cứng | Bước 3, `results/C3/table.md`: 9/12 cấu hình trong 0.982 đến 0.986 (LogReg), chênh < độ lệch chuẩn; cell 16×16 trên 64×128 kém (0.959) | [đo] |
| Ghép đặc trưng: chuẩn hóa từng phần rồi ghép | Ghép thẳng, z-score chung | Bước 3, `results/C5/table.md`: KNN 0.977 so với 0.948; LogReg ngang nhau | [đo] |
| Cấu hình truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100 | Màu riêng (0.991), HOG riêng (0.986), cạnh (0.955); KNN, Linear SVM | Bước 3, `results/select_traditional/table.md` | [đo] |
| Thêm mức nhiễu σ = 2 cho C1 | Chỉ σ = 5, 10, 20 | Bước 3, `results/C1/table.md`: từ σ = 10 mọi toán tử đều về mức đoán mò | [đo] |
| `a_lenet` dùng ảnh vào 64×64 | 32×32, 128×128 | Bước 4-5, `results/cnn/table_runs.md`: 0.973 so với 0.954 và 0.956 | [đo] |
| Fine-tune với BatchNormalization đóng băng | Mở băng cả BN | Bước 4-5, `results/N4/table_batchnorm.md` | [đo] |
| Biến thể CNN chốt theo macro-F1 trên val | Cấu hình mặc định | Bước 4-5, `results/cnn/table_runs.md` | [đo] |
| Giai đoạn 2 hướng B: lr 1e-4 | 1e-5, 1e-3 | Bước 4-5, `results/N4/table.md`: 1.000 / 0.993 / 0.993; lr 1e-3 dao động xuống 0.917 | [đo] |
| Demo dùng `b_mobilenetv3` | `a_vgg16` | Bước 4-5, `results/N6/table.md`: 11.7 MB so với 370 MB (giới hạn 50 MB) | [đo] |
| Không thêm bước xử lý mất cân bằng vào cấu hình cuối | Class weight, oversampling, tăng cường lớp ít | Bước 6-7, `results/I2/table.md`: không cách nào cải thiện vượt độ lệch chuẩn ở mô hình mạnh | [đo] |
| Mask cho S2: Otsu trên khoảng cách màu tới nền + hậu xử lý | Ngưỡng toàn cục, Otsu xám, ngưỡng thích nghi, k-means | Bước 6-7, `results/S1/table.md` | [đo] |
| Bản "chỉ nền" dùng mask nới rộng 12 px + inpainting | Tô vùng chuối bằng màu phẳng | Bước 6-7, `results/S2/table_leak.md`, `results/figures/S2_versions.png` | [đo] |
| Ghi thiên lệch bối cảnh chụp là hạn chế chính | - | Bước 6-7, `results/S2/table_corners.md`, `results/S2/table_cnn.md` | [đo] |
| Đại diện mỗi nhóm chọn theo điểm CV/val | Chọn theo điểm test | Bước 8 | [lập luận] |
| Báo cáo KTC 95% bootstrap trên test | Chỉ báo cáo điểm | Bước 8, `results/final/table.md` | [lập luận] |
| So sánh các nhánh bằng cả ảnh gốc lẫn bộ kiểm tra khó (điểm bền) | Chỉ dùng điểm trên ảnh test gốc | Bước 10, `results/so_sanh_nhanh/G/table.md`: hai cấu hình đã đúng 371/371 ảnh gốc | [đo] |
| Mask theo mô hình nền bậc hai cho phần mở rộng | Mask của bước S1 | Bước 10 và 13, `results/E1_xu_ly/table_cleaning.md`: mask cũ tràn nền ở 5.2% ảnh | [đo] |
| Ba nhánh mở rộng độc lập, mỗi nhánh đổi một thứ so với G | Chồng các thay đổi lên nhau | Bước 10 | [lập luận] |
| Nhánh T dùng cách chia không rò rỉ (T-kr) làm kết quả chính | Chia ngẫu nhiên bản Augmented (T-tl) | Bước 11, `results/E2_aug_tac_gia/table_leak_stats.md`: 90.6% ảnh test của T-tl có file cùng ảnh gốc trong train | [đo] |
| Không thay tập train gốc bằng bản Augmented của tác giả | Dùng bản Augmented làm tập train | Bước 11, `results/so_sanh_nhanh/table_models.md`: macro-F1 trung bình 0.979 so với 0.983; điểm bền 0.769 so với 0.750 | [đo] |
| Chấm phương pháp của bài báo theo cả ba giao thức (như bài báo; giữ nhóm; giao thức của đồ án) | Chỉ dùng giao thức của bài báo | Bước 12, `results/E3_bai_bao/table_protocols.md`: giao thức của bài báo cao hơn 0.11 đến 0.90 điểm ở 19/19 tổ hợp | [đo] |
| Không thay pipeline của đồ án bằng ViT + SVM | Dùng ViT làm mô hình chính | Bước 12, `results/E3_bai_bao/table_test.md`: ngang nhau trên ảnh gốc; ViT bền hơn (0.919 đến 0.937 so với 0.895) nhưng nằm ngoài môn học và chậm hơn ~23 lần | [đo] |
| Tiêu chí chọn cách xử lý dữ liệu: trung bình 7 phép thử (ảnh gốc + 6 phép khó) trên val | Chỉ dùng điểm trên ảnh gốc | Bước 13, `results/E1_xu_ly/table_selection.md`: trên ảnh gốc "không xử lý" đã đạt 1.000 / 0.993 | [đo] |
| Tiền xử lý của X-a: nền phẳng, giữ nguyên màu quả | Cân bằng trắng theo nền; cắt theo quả | Bước 13, `results/E1_xu_ly/table_wb_leak.md` (cân bằng trắng làm màu quả lệch 12.9 Lab khi đổi nền), `table_selection.md` (0.889 so với 0.843 và 0.813) | [đo] |
| Chạy thêm biến thể X-b (tăng cường có thay nền, không xử lý lúc suy luận) | Chỉ chạy tổ hợp điểm cao nhất | Bước 13, `results/E1_xu_ly/pipeline_b/table.md`: `b_mobilenetv3` giữ 371/371 ảnh gốc, nền hoán đổi 0.975 | [đo] |
| Giữ cách chia ngẫu nhiên, ghi "mỗi quả chụp nhiều mặt" là hạn chế | Chia lại theo khối ảnh liên tiếp | Bước 13, `results/E1_xu_ly/table_blocked_cv.md`: chia ngẫu nhiên cao hơn 0.024 đến 0.069 | [đo] |
| Kết luận về các nhánh theo từng phép thử, không theo một điểm bền duy nhất | Xếp hạng bằng điểm bền | Bước 14, `results/so_sanh_nhanh/table_stress.md`: G, X-a, X-b có điểm bền 0.875 đến 0.883 nhưng khác nhau rõ ở nền hoán đổi và che 25% | [đo] |
| Slide chính: Beamer LaTeX style UIT, sinh cùng nguồn với bản pptx | Chỉ pptx | Bước 14; kế hoạch mục 8 | [lập luận] |
| Demo giữ `b_mobilenetv3` của pipeline gốc làm mặc định, thêm bản X-b | Thay hẳn bằng bản X-b | Bước 14, `results/so_sanh_nhanh/table_stress.md` | [đo] |
| `best.keras` chỉ gồm kiến trúc và trọng số, commit vào repo | Giữ trạng thái Adam và dùng Git LFS; cắt file; float16; lưu ngoài repo | Bước 15, `results/cnn/table_slim.md`: 4458 → 1563 MB, 0/3920 mảng trọng số khác; `results/cnn/table_checkpoints.md`: 40/40 run trùng kết quả đã ghi | [đo] |
| Dung lượng file mô hình đo trên file không kèm trạng thái Adam | Giữ số đo trên file kèm trạng thái Adam | Bước 15, `results/N6/table.md`: file mới xấp xỉ 4 byte × số tham số | [đo] |
| Không commit `data/cache/` | Commit cả cache | Bước 15: 8.3 GiB, 15 file trên 100 MiB; tính lại được, không cần huấn luyện | [đo] |
| Nhánh mở rộng mặc định tính lại, dùng lại trọng số; `SKIP_DONE=1` để bỏ qua cấu hình đã xong | Luôn bỏ qua cấu hình đã có kết quả | Bước 15: nhánh T-kr tính lại trùng từng byte trong 401 giây | [đo] |
| Kiểm chứng tái lập bằng bản clone mới từ GitHub | Chạy lại trong thư mục làm việc | Bước 15 (bổ sung): 2.5 giờ, mọi khác biệt chỉ ở số đo thời gian | [đo] |
| Demo 7 tab; tab metric tính lại từ ảnh và mô hình | Chỉ dự đoán một ảnh; đọc sẵn số từ `results/` | Bước 16: số tính lại khớp số đã ghi cho 4 mô hình; bộ kiểm tra khó sinh lại giống hệt | [đo] |
| Slide chính theo thứ tự phiếu đánh giá; slide chi tiết từng thí nghiệm chuyển xuống phụ lục | Giữ bộ slide đi theo từng thí nghiệm | Bước 17 | [lập luận] |
| Số thập phân dùng dấu chấm, không dùng gạch dài, kể cả tiêu đề hình và bảng do code sinh | Giữ dấu phẩy thập phân | Bước 17: 0 gạch dài trong `report/report.md` và trong code | [đo] |
