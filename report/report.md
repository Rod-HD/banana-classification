# Nhận diện giống chuối từ ảnh — Báo cáo đồ án CS231

- **Môn học:** CS231 — Nhập môn Thị giác máy tính
- **Mã nguồn:** https://github.com/Rod-HD/banana-classification
- **Tài liệu kèm theo:** kế hoạch thực nghiệm (`docs/ke-hoach-thuc-nghiem.md`), nhật ký thực nghiệm (`docs/nhat-ky-thuc-nghiem.md`), bảng tổng hợp mọi kết quả (`results/summary.md`, `results/summary.csv`).

Mọi con số trong báo cáo lấy từ các file trong `results/` do code của đồ án sinh ra; mỗi bảng và hình đều ghi đường dẫn tới file gốc. Cách chạy lại toàn bộ: `bash scripts/run_all.sh` (xem README).

## 1. Thông tin thành viên và mức độ hoàn thành

| STT | Họ và tên | MSSV | Công việc phụ trách | Mức độ hoàn thành |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |

Mức độ hoàn thành chung của đồ án:

## 2. Lý do chọn đề tài và tầm quan trọng của bài toán

Bangladesh có nhiều giống chuối với giá bán, cách dùng và vùng trồng khác nhau. Ở chợ đầu mối và cửa hàng, việc phân loại giống hiện vẫn làm bằng mắt. Bộ dữ liệu BananaImageBD được nhóm tác giả của East West University thu thập từ chính các chợ đầu mối và cửa hàng trái cây ở Bangladesh, nhắm tới các ứng dụng nông nghiệp chính xác, chế biến thực phẩm và quản lý chuỗi cung ứng [1].

Với môn học, bài toán này phù hợp để so sánh các phương pháp đã học vì:

- bốn giống khác nhau cả về màu vỏ lẫn hình dạng, nên có thể kiểm chứng câu hỏi "đặc trưng màu hay đặc trưng hình dạng quan trọng hơn" bằng thực nghiệm;
- dữ liệu nhỏ (2.471 ảnh) và mất cân bằng (lớp lớn nhất gấp 2,2 lần lớp nhỏ nhất), đúng hoàn cảnh mà các tính chất như quá khớp, transfer learning, xử lý mất cân bằng thể hiện rõ;
- ảnh chụp một quả trên nền trơn, nên áp dụng được cả các kỹ thuật phân đoạn tách nền đã học.

## 3. Phát biểu bài toán

- **Input:** một ảnh màu chụp một quả hoặc nải chuối.
- **Output:** tên giống chuối, là một trong 4 lớp Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola, kèm xác suất của từng lớp.
- **Ngoài phạm vi:** không xác định hàm lượng dinh dưỡng, không nhận diện giống ngoài 4 lớp trên, không phân loại độ chín.

## 4. Phương pháp giải quyết

Đồ án thử **tất cả các phương pháp đã học áp dụng được cho phân loại ảnh**, so sánh chúng bằng thực nghiệm, và giải thích kết quả bằng tính chất của từng phương pháp. Mỗi nhận định đi theo ba bước: tính chất của phương pháp → dự đoán → thí nghiệm kiểm chứng. Tổng cộng có 30 thí nghiệm (mục 5.4 đến 5.10).

![Ảnh mẫu của 4 giống chuối](../results/figures/data_samples.png)

### 4.1 Hướng truyền thống: rút đặc trưng + bộ phân lớp

**Đặc trưng màu** (`src/features/color.py`):

- Histogram RGB từng kênh (mặc định 32 bin mỗi kênh, 96 chiều); histogram HSV 3 chiều (8, 8, 8) = 512 chiều; histogram chỉ dùng H và S (8, 8) = 64 chiều. Mỗi histogram chuẩn hóa tổng bằng 1.
- Moment màu trên HSV: trung bình, độ lệch chuẩn, độ lệch (skewness) của từng kênh, 9 chiều.
- Màu chủ đạo: k-means (k = 3) trên điểm ảnh RGB, mỗi cụm cho (tỉ lệ, R, G, B), sắp theo tỉ lệ giảm dần, 12 chiều.
- Color Coherence Vector (CCV): lượng tử hóa 64 màu; với mỗi màu, đếm số điểm ảnh thuộc vùng liên thông có diện tích ≥ 1% ảnh ("liên kết") và số còn lại ("không liên kết"), 128 chiều.

**Đặc trưng cạnh** (`src/features/edge.py`): gradient bằng các toán tử Roberts (2×2), Prewitt, Sobel (3×3) và Laplacian; vector đặc trưng là tổng độ lớn gradient theo 256 hàng và 256 cột của ảnh xám 256×256, 512 chiều.

**HOG** (`src/features/hog.py`), cài đặt theo đúng các bước của bài giảng: resize về 64×128 → gradient bằng mặt nạ [−1, 0, 1] → histogram 9 bin hướng không dấu trên cell 8×8 (chia phiếu cho hai bin gần nhất) → chuẩn hóa L2 trên block 16×16 (2×2 cell, trượt 1 cell) → ghép thành vector 7 × 15 × 36 = 3.780 chiều. Nhóm tự cài đặt để có tùy chọn bỏ bước chuẩn hóa block (thí nghiệm C2). Kết quả đối chiếu với `skimage.feature.hog` lệch tối đa 2,2·10⁻⁷.

**Ghép đặc trưng:** mỗi phần được chuẩn hóa z-score rồi chia cho căn bậc hai số chiều của phần đó, để phần màu (64 chiều) và phần HOG (3.780 chiều) đóng góp ngang nhau vào khoảng cách.

**Độ đo khoảng cách:** Euclidean, Manhattan, cosine và Hellinger (Euclidean trên căn bậc hai của histogram đã chuẩn hóa L1); có và không chuẩn hóa về vector đơn vị.

**Bộ phân lớp:**

- KNN: chọn k, độ đo, bình chọn đa số (`uniform`) hoặc ưu tiên điểm gần (`distance`), thuật toán tìm láng giềng Brute Force, K-D Tree, Ball Tree.
- Logistic Regression (bộ phân lớp tuyến tính).
- Linear SVM: môn học chỉ nhắc tới như bộ phân lớp đi kèm HOG ("HOG + SVM"), nên đồ án dùng nó như **bộ phân lớp bổ sung** để đối chiếu.

### 4.2 Hướng học sâu: CNN, hai hướng tách riêng

**Hướng A — CNN theo bài giảng** (`src/cnn/models.py`):

| Mô hình | Kiến trúc | Ảnh vào | Huấn luyện |
|---|---|---|---|
| `a_lenet` | Conv(6, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(120) → Dense(84) → Softmax(4) | 64×64×3 | từ đầu |
| `a_dropout` | Conv(32, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(128) → Dropout(0,5) → Dense(64) → Softmax(4) | 128×128×3 | từ đầu |
| `a_vgg16` | VGG16 ImageNet đóng băng phần tích chập → Flatten → Dense(1024) → Dense(256) → Softmax(4) | 224×224×3 | transfer learning (đúng ví dụ bài giảng) |

**Hướng B — CNN lớn hơn, fine-tune:** `b_mobilenetv3` (MobileNetV3-Small ImageNet) và `b_resnet50` (ResNet50 ImageNet, kiến trúc đã gặp ở bài xác định vị trí). Phần đầu: GlobalAveragePooling → Dropout(0,2) → Softmax(4). Giai đoạn 1: đóng băng thân mạng, chỉ huấn luyện phần đầu. Giai đoạn 2: mở băng (toàn mạng với MobileNetV3, khối conv5 với ResNet50), fine-tune với learning rate nhỏ, có tăng cường ảnh (lật, xoay ±30°, phóng to/thu nhỏ ±10%). Hướng B dùng kiến trúc nằm ngoài ví dụ của bài giảng, nhưng vẫn là khái niệm đã học: transfer learning, tức tái sử dụng kiến trúc và trọng số đã học.

**Cấu hình chung:** Adam, cross-entropy, lô 32, dừng sớm theo macro-F1 trên tập val (giữ trọng số tốt nhất), seed cố định. Khi fine-tune, các lớp BatchNormalization giữ đóng băng (lý do ở mục 5.8, N4).

### 4.3 Phân đoạn tách nền

Ngưỡng toàn cục, Otsu, ngưỡng thích nghi và gom cụm k-means (`src/segmentation.py`), cộng thêm Otsu trên khoảng cách màu tới màu nền ở viền ảnh. Phân đoạn dùng để kiểm tra mô hình dựa vào quả chuối hay vào nền ảnh (S2).

### 4.4 Quy trình đánh giá

- Chia một lần, cố định: stratified 70/15/15 (train/val/test), `seed = 42` (`data/splits/`).
- **Tập test chỉ dùng một lần ở cuối** cho các cấu hình đã chốt (mục 5.11).
- Nhóm truyền thống: gộp train+val (2.100 ảnh), chọn siêu tham số bằng 5-fold stratified cross-validation, báo cáo trung bình ± độ lệch chuẩn qua 5 fold. Chênh lệch nhỏ hơn độ lệch chuẩn được kết luận là "chưa đủ cơ sở phân biệt".
- Nhóm CNN: huấn luyện trên train, chọn epoch và siêu tham số theo val.
- Các thí nghiệm "biến đổi ảnh" (xáo trộn, đổi độ sáng, nhiễu, che, xoay) chấm trên fold giữ lại (truyền thống) hoặc val (CNN); mô hình luôn học trên ảnh sạch.
- Chỉ số chính: **macro-F1** (vì dữ liệu mất cân bằng), báo cáo kèm Accuracy, Precision và Recall từng lớp, Confusion Matrix.

## 5. Kết quả thực nghiệm

### 5.1 Dataset

Bộ BananaImageBD, Version 2, phần ảnh gốc phân loại giống [1]: 2.471 ảnh JPEG, giấy phép CC BY 4.0. 2.469 ảnh có kích thước 256×256; 2 ảnh lệch kích thước nên code luôn resize. Có 93 file đuôi `.JPG` viết hoa. Không có file trùng tuyệt đối (md5).

| Tập | Bangla Kola | Champa Kola | Sabri Kola | Sagor Kola | Tổng |
|---|---|---|---|---|---|
| train | 311 | 695 | 356 | 367 | 1.729 |
| val | 66 | 150 | 76 | 79 | 371 |
| test | 67 | 149 | 77 | 78 | 371 |
| Tổng | 444 | 994 | 509 | 524 | 2.471 |

**Ảnh gần trùng.** Nhóm dùng perceptual hash (pHash 64 bit). Có 42 cặp ảnh cách nhau Hamming ≤ 2, tức cùng một quả chụp lại gần như y hệt; 7 ảnh test có bản gần trùng trong train. Chia lại dữ liệu sao cho các ảnh gần trùng nằm cùng một fold không làm đổi macro-F1 quá ±0,007, nhỏ hơn độ lệch chuẩn giữa các fold (`results/D1/table_near_duplicates.md`). Trên tập test, hai mô hình tốt nhất cũng đúng 100% ở các ảnh không có bản gần trùng (`results/final/table_near_duplicates.md`).

### 5.2 Độ đo

Với lớp c: Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2·P·R / (P + R). Macro-F1 là trung bình F1 của 4 lớp, nên mỗi lớp có trọng số như nhau dù số ảnh khác nhau.

**Ví dụ tính tay 1 — Precision và Recall từ Confusion Matrix.** Confusion Matrix của `a_vgg16` trên tập test (`results/hand_calc/table_cm_a_vgg16.md`):

| Thực tế \ Dự đoán | Bangla Kola | Champa Kola | Sabri Kola | Sagor Kola |
|---|---|---|---|---|
| Bangla Kola | 65 | 0 | 0 | 2 |
| Champa Kola | 0 | 149 | 0 | 0 |
| Sabri Kola | 2 | 0 | 75 | 0 |
| Sagor Kola | 0 | 0 | 0 | 78 |

Với lớp Bangla Kola: TP = 65 (đường chéo); FP = tổng cột Bangla trừ TP = 65 + 0 + 2 + 0 − 65 = 2 (hai ảnh Sabri bị đoán thành Bangla); FN = tổng hàng Bangla trừ TP = 65 + 0 + 0 + 2 − 65 = 2 (hai ảnh Bangla bị đoán thành Sagor). Precision = 65 / (65 + 2) = 0,970; Recall = 65 / (65 + 2) = 0,970; F1 = 0,970. Kết quả khớp với `classification_report` trong `results/final/a_vgg16/`.

**Ví dụ tính tay 2 — khoảng cách giữa hai vector đặc trưng.** Vector moment màu 9 chiều (HSV đã chia về [0, 1]) của ảnh x = `Bangla_Kola_1.jpg` và y = `Champa_Kola_1.jpg` (`results/hand_calc/`):

| Thành phần | x (Bangla_Kola_1.jpg) | y (Champa_Kola_1.jpg) | x − y | (x − y)² | Trị tuyệt đối của x − y |
|---|---|---|---|---|---|
| mean H | 0,5101 | 0,5277 | -0,0177 | 0,000312 | 0,0177 |
| mean S | 0,0807 | 0,1471 | -0,0664 | 0,004414 | 0,0664 |
| mean V | 0,7770 | 0,6726 | +0,1044 | 0,010891 | 0,1044 |
| std H | 0,3993 | 0,1485 | +0,2508 | 0,062896 | 0,2508 |
| std S | 0,1833 | 0,1824 | +0,0009 | 0,000001 | 0,0009 |
| std V | 0,1433 | 0,1116 | +0,0317 | 0,001006 | 0,0317 |
| skew H | 0,0747 | -0,1999 | +0,2746 | 0,075392 | 0,2746 |
| skew S | 0,2600 | 0,2628 | -0,0028 | 0,000008 | 0,0028 |
| skew V | -0,2072 | -0,1558 | -0,0514 | 0,002643 | 0,0514 |

- Euclidean: d = √(Σ(x − y)²) = √0,157561 = 0,3969.
- Manhattan: d = Σ|x − y| = 0,8006.
- Cosine: x·y = 0,99805; ‖x‖ = 1,09546; ‖y‖ = 0,97654; cos = 0,99805 / (1,09546 × 0,97654) = 0,9330.
- Sau khi chuẩn hóa về vector đơn vị, khoảng cách Euclidean bằng √(2 − 2·cos) = √(2 − 2 × 0,9330) = 0,3662, khớp với kết quả tính trực tiếp (0,3662). Vì Euclidean trên vector đơn vị là hàm đơn điệu của cosine, hai độ đo cho cùng thứ hạng láng giềng; thí nghiệm P2 kiểm chứng điều này trên toàn bộ dữ liệu.

Ví dụ tính tay 3 (số tham số một lớp CNN) nằm ở mục 5.8, thí nghiệm N1.

### 5.3 Lập trình

- Python 3.13, OpenCV 5.0, scikit-image 0.26, scikit-learn 1.9, imbalanced-learn 0.14, TensorFlow 2.21 / Keras 3.15 (bản CPU), Gradio cho demo. Phiên bản đầy đủ: `results/environment.txt`.
- Máy chạy: CPU 4 nhân, 16 GB RAM, không có GPU. Đặc trưng thủ công và đầu ra của các thân mạng đóng băng chỉ tính một lần rồi lưu vào `data/cache/`.
- Mỗi bước là một module chạy được từ dòng lệnh (`python -m src.<module> --help`); `scripts/run_all.sh` chạy lại toàn bộ theo thứ tự. Huấn luyện CNN lưu checkpoint mỗi epoch và hỗ trợ `--resume`.
- Kết quả: `results/<mã thí nghiệm>/` (bảng `table*.md`, `metrics.json`), hình trong `results/figures/`, tổng hợp ở `results/summary.csv` (718 dòng).

### 5.4 Thí nghiệm dữ liệu: D1 — rò rỉ do tăng cường ảnh trước khi chia tập

**Tính chất.** Tăng cường ảnh trước khi chia tập làm bản sao của cùng một ảnh rơi vào cả train lẫn test.
**Dự đoán.** Kết quả trên test bị thổi phồng; đây là lý do không dùng bản "Augmented" có sẵn của bộ dữ liệu.
**Thí nghiệm.** Trong tập train+val, tạo 2 bản tăng cường cho mỗi ảnh (lật ngang, xoay ±10–30°). So ba cách: (a) tăng cường rồi chia ngẫu nhiên 80/20; (b) chia trước, chỉ tăng cường phần huấn luyện; (c) chia theo ảnh gốc rồi tăng cường cả hai phía. (c) có tập chấm cùng thành phần với (a), nên (a) − (c) là phần thổi phồng thuần do rò rỉ. Lặp 5 lần.

| Mô hình | (a) tăng cường rồi chia | (b) chia rồi tăng cường train | (c) chia theo ảnh gốc, tăng cường 2 phía | Thổi phồng (a)−(b) | Thổi phồng (a)−(c) | Macro-F1 (a) | Macro-F1 (b) |
|---|---|---|---|---|---|---|---|
| hsv_hist + knn (k=1) | 0,999 ± 0,001 | 0,978 ± 0,010 | 0,978 ± 0,008 | +0,022 ± 0,010 | +0,021 ± 0,009 | 0,999 ± 0,001 | 0,974 ± 0,011 |
| hsv_hist + logreg | 0,999 ± 0,001 | 0,991 ± 0,002 | 0,990 ± 0,002 | +0,008 ± 0,002 | +0,009 ± 0,002 | 0,999 ± 0,001 | 0,989 ± 0,003 |
| hog + knn (k=1) | 0,974 ± 0,003 | 0,974 ± 0,003 | 0,970 ± 0,002 | +0,000 ± 0,003 | +0,005 ± 0,004 | 0,972 ± 0,003 | 0,974 ± 0,004 |
| hog + logreg | 0,973 ± 0,006 | 0,981 ± 0,003 | 0,973 ± 0,003 | -0,008 ± 0,004 | +0,000 ± 0,004 | 0,969 ± 0,006 | 0,978 ± 0,004 |
| edge + knn (k=1) | 0,896 ± 0,012 | 0,889 ± 0,014 | 0,862 ± 0,012 | +0,007 ± 0,020 | +0,034 ± 0,020 | 0,895 ± 0,013 | 0,885 ± 0,016 |

![D1 — Accuracy theo ba cách chia](../results/figures/D1_leakage.png)

**Kết luận.** Khớp dự đoán với đặc trưng màu: histogram HSV gần như không đổi khi lật hay xoay, nên bản sao trong train là "đáp án có sẵn". Accuracy bị đẩy lên 0,999, thổi phồng +0,021 (KNN k = 1) và +0,009 (LogReg). **Trái dự đoán với HOG:** mức thổi phồng ≈ 0 (+0,005 ± 0,004 và +0,000 ± 0,004), vì HOG mã hóa hướng cạnh theo vị trí nên bản lật hay xoay có vector khác hẳn ảnh gốc. Rò rỉ chỉ gây hại khi đặc trưng bất biến với phép tăng cường.

### 5.5 Thí nghiệm đặc trưng màu: M1–M6

Mọi số liệu dưới đây là macro-F1, 5-fold CV trên train+val; phép biến đổi chỉ áp lên fold giữ lại.

**M1 — Histogram bỏ thông tin vị trí.** *Dự đoán:* xáo trộn ảnh chấm thành lưới 4×4 thì histogram giữ nguyên, HOG giảm.

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

![M1 — Xáo trộn lưới 4×4](../results/figures/M1_shuffle.png)

*Kết luận:* khớp. Histogram và moment màu không đổi đúng đến chữ số cuối (độ sụt 0,000); HOG và vector cạnh rơi về mức đoán mò (0,20–0,29); CCV sụt 0,05–0,08 vì có thông tin liên kết không gian.

**M2 — HSV tách độ sáng khỏi sắc màu.** *Dự đoán:* khi nhân kênh V với 0,6 hoặc 1,4, histogram RGB sụt nhiều nhất, HSV ít hơn, H-S bền nhất.

| Đặc trưng | Bộ phân lớp | Ảnh gốc | V×0,6 | Sụt (V×0,6) | V×1,4 | Sụt (V×1,4) |
|---|---|---|---|---|---|---|
| Hist RGB (3×32) | KNN (k=5) | 0,964 ± 0,012 | 0,092 ± 0,002 | +0,873 ± 0,013 | 0,205 ± 0,019 | +0,760 ± 0,024 |
| Hist RGB (3×32) | LogReg | 0,976 ± 0,013 | 0,212 ± 0,054 | +0,763 ± 0,064 | 0,279 ± 0,038 | +0,696 ± 0,036 |
| Hist HSV (8,8,8) | KNN (k=5) | 0,965 ± 0,014 | 0,302 ± 0,042 | +0,663 ± 0,043 | 0,589 ± 0,020 | +0,376 ± 0,025 |
| Hist HSV (8,8,8) | LogReg | 0,991 ± 0,005 | 0,318 ± 0,035 | +0,673 ± 0,035 | 0,662 ± 0,004 | +0,329 ± 0,009 |
| Hist H-S (8,8) | KNN (k=5) | 0,976 ± 0,006 | 0,975 ± 0,009 | +0,000 ± 0,008 | 0,975 ± 0,006 | +0,001 ± 0,002 |
| Hist H-S (8,8) | LogReg | 0,985 ± 0,008 | 0,959 ± 0,014 | +0,027 ± 0,011 | 0,974 ± 0,007 | +0,011 ± 0,004 |

*Kết luận:* khớp một phần. Chỉ histogram H-S (bỏ hẳn kênh V) là bền (sụt ≤ 0,027). Histogram HSV đủ 3 kênh vẫn sụt 0,33–0,67 vì kênh V vẫn nằm trong đặc trưng: tách V ra thành một trục riêng thôi chưa đủ, phải bỏ nó khỏi vector.

**M3 — Số bin.** *Dự đoán:* ít bin thì mất khả năng phân biệt, nhiều bin thì vector thưa và nhạy nhiễu.

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

![M3 — Quét số bin](../results/figures/M3_bins.png)

*Kết luận:* vế "ít bin" khớp (4 bin mỗi kênh: 0,914–0,936). Vế "nhiều bin" chỉ thấy ở KNN trên histogram 3 chiều: 16³ = 4.096 chiều, 87% phần tử bằng 0, KNN giảm còn 0,917, trong khi LogReg vẫn 0,992. Histogram RGB từng kênh từ 8 đến 256 bin cho kết quả ngang nhau (chênh nhỏ hơn độ lệch chuẩn).

**M4 — Moment màu gọn nhưng thô.** *Dự đoán:* 9 chiều nhanh nhưng kém histogram.

| Đặc trưng | Số chiều | Rút đặc trưng (ms/ảnh) | KNN (k=5) | Huấn luyện KNN (k=5) (s) | Dự đoán KNN (k=5) (ms/ảnh) | LogReg | Huấn luyện LogReg (s) | Dự đoán LogReg (ms/ảnh) | Linear SVM |
|---|---|---|---|---|---|---|---|---|---|
| Hist RGB (3×32) | 96 | 0,15 | 0,964 ± 0,012 | 0,003 | 0,010 | 0,976 ± 0,013 | 0,038 | 0,002 | 0,960 ± 0,012 |
| Hist HSV (8,8,8) | 512 | 0,23 | 0,965 ± 0,014 | 0,008 | 0,024 | 0,991 ± 0,005 | 0,774 | 0,007 | 0,989 ± 0,008 |
| Hist H-S (8,8) | 64 | 0,20 | 0,976 ± 0,006 | 0,002 | 0,009 | 0,985 ± 0,008 | 0,017 | 0,001 | 0,958 ± 0,007 |
| Moment màu | 9 | 4,04 | 0,972 ± 0,008 | 0,002 | 0,007 | 0,908 ± 0,009 | 0,017 | 0,001 | 0,838 ± 0,011 |
| Màu chủ đạo (k=3) | 12 | 1,58 | 0,945 ± 0,005 | 0,002 | 0,007 | 0,899 ± 0,009 | 0,018 | 0,001 | 0,826 ± 0,013 |
| CCV (64 màu) | 128 | 1,90 | 0,946 ± 0,011 | 0,003 | 0,012 | 0,975 ± 0,006 | 0,016 | 0,001 | 0,965 ± 0,014 |
| Hist 64 màu | 64 | 0,50 | 0,966 ± 0,010 | 0,002 | 0,008 | 0,968 ± 0,008 | 0,014 | 0,001 | 0,945 ± 0,020 |

![M4 — Các đặc trưng màu](../results/figures/M4_color_features.png)

*Kết luận:* khớp với bộ phân lớp tuyến tính (LogReg 0,908, Linear SVM 0,838). **Trái dự đoán với KNN** (0,972, ngang histogram) và về thời gian: rút moment màu (khoảng 4 ms/ảnh) chậm hơn rút histogram (0,2 ms/ảnh) vì phải tính lũy thừa trên từng điểm ảnh. Giải thích ở P6: 9 con số đủ tách 4 lớp nếu biên quyết định được phép cong.

**M5 — CCV bổ sung thông tin liên kết không gian.** *Dự đoán:* CCV tốt hơn histogram cùng mức lượng tử hóa và nhạy với bố cục.

| Đặc trưng | Số chiều | Bộ phân lớp | Ảnh gốc | Xáo trộn 4×4 | Độ sụt |
|---|---|---|---|---|---|
| Hist 64 màu | 64 | KNN (k=5) | 0,966 ± 0,010 | 0,962 ± 0,008 | +0,003 ± 0,004 |
| Hist 64 màu | 64 | LogReg | 0,968 ± 0,008 | 0,966 ± 0,009 | +0,002 ± 0,004 |
| CCV (64 màu) | 128 | KNN (k=5) | 0,946 ± 0,011 | 0,864 ± 0,018 | +0,082 ± 0,015 |
| CCV (64 màu) | 128 | LogReg | 0,975 ± 0,006 | 0,922 ± 0,011 | +0,053 ± 0,009 |

*Kết luận:* vế "nhạy với bố cục" khớp (xáo trộn làm CCV sụt 0,05–0,08, histogram ≤ 0,003). Vế "tốt hơn histogram" không được xác nhận: với LogReg 0,975 so với 0,968 (chênh nhỏ hơn độ lệch chuẩn), với KNN còn kém hơn. Trên ảnh một quả chuối nền trơn, hầu hết điểm ảnh cùng màu nằm trong vùng liên thông lớn, nên phần "không liên kết" gần như rỗng và mang ít thông tin.

**M6 — Giống chuối khác nhau ở màu hay ở hình dạng?** *Dự đoán:* chuyển ảnh sang xám thì đặc trưng màu mất tác dụng, đặc trưng cạnh và HOG gần như không đổi.

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

![M6 — Ảnh xám](../results/figures/M6_grayscale.png)

*Kết luận:* khớp. Khi học lại trên ảnh xám, đặc trưng màu sụt 0,04–0,23; cạnh và HOG đổi ≤ 0,005, và HOG trên ảnh xám vẫn đạt 0,986. Bốn giống phân biệt được bằng cả màu (hist HSV 0,991) lẫn hình dạng (HOG 0,986). Mục 5.10 (S2) cho thấy một phần của cả hai con số này đến từ nền ảnh.

### 5.6 Thí nghiệm đặc trưng cạnh và HOG: C1–C5

**C1 — Toán tử cạnh và nhiễu.** *Tính chất:* Roberts dùng mặt nạ 2×2 nên nhạy nhiễu; Prewitt và Sobel có làm trơn; Laplacian là đạo hàm bậc hai nên nhạy nhiễu nhất. *Dự đoán:* thêm nhiễu Gauss thì độ sụt xếp theo thứ tự Prewitt ≈ Sobel < Roberts < Laplacian. Nhóm thêm mức σ = 2 vì từ σ = 10 mọi toán tử đều rơi về mức đoán mò.

| Toán tử | Bộ phân lớp | Ảnh gốc | Sụt σ=2 | Sụt σ=5 | Sụt σ=10 | Sụt σ=20 |
|---|---|---|---|---|---|---|
| Roberts (2×2) | KNN (k=5) | 0,901 ± 0,019 | +0,461 ± 0,017 | +0,789 ± 0,019 | +0,813 ± 0,019 | +0,813 ± 0,019 |
| Roberts (2×2) | LogReg | 0,932 ± 0,011 | +0,669 ± 0,015 | +0,839 ± 0,007 | +0,844 ± 0,010 | +0,844 ± 0,010 |
| Prewitt (3×3) | KNN (k=5) | 0,890 ± 0,019 | +0,026 ± 0,020 | +0,523 ± 0,025 | +0,799 ± 0,014 | +0,803 ± 0,019 |
| Prewitt (3×3) | LogReg | 0,892 ± 0,007 | +0,198 ± 0,019 | +0,768 ± 0,013 | +0,805 ± 0,007 | +0,805 ± 0,007 |
| Sobel (3×3) | KNN (k=5) | 0,890 ± 0,019 | +0,023 ± 0,022 | +0,535 ± 0,021 | +0,800 ± 0,017 | +0,802 ± 0,019 |
| Sobel (3×3) | LogReg | 0,894 ± 0,008 | +0,226 ± 0,020 | +0,776 ± 0,013 | +0,806 ± 0,007 | +0,806 ± 0,007 |
| Laplacian (bậc hai) | KNN (k=5) | 0,886 ± 0,014 | +0,601 ± 0,008 | +0,797 ± 0,014 | +0,798 ± 0,014 | +0,798 ± 0,014 |
| Laplacian (bậc hai) | LogReg | 0,914 ± 0,015 | +0,826 ± 0,015 | +0,827 ± 0,015 | +0,827 ± 0,015 | +0,827 ± 0,015 |

| Toán tử | σ=2 | σ=5 | σ=10 | σ=20 |
|---|---|---|---|---|
| Roberts (2×2) | 0,35 | 1,00 | 2,14 | 4,47 |
| Prewitt (3×3) | 0,14 | 0,43 | 0,98 | 2,11 |
| Sobel (3×3) | 0,15 | 0,46 | 1,03 | 2,22 |
| Laplacian (bậc hai) | 0,78 | 2,16 | 4,55 | 9,37 |

![C1 — Toán tử cạnh và nhiễu](../results/figures/C1_edge_noise.png)

*Kết luận:* khớp. Ở σ = 5, vector đặc trưng thay đổi 43–46% với Prewitt/Sobel, 100% với Roberts và 216% với Laplacian. Ngoài dự đoán: vector "tổng theo hàng/cột" kém bền với mọi toán tử, sập hoàn toàn từ σ = 10, vì nhiễu cộng thêm gradient ở mọi điểm ảnh nền và tổng theo hàng/cột cộng dồn hết phần đó.

**C2 — HOG chuẩn hóa block bền với độ sáng.** *Dự đoán:* lặp lại M2 với HOG; bản có chuẩn hóa bền hơn bản không chuẩn hóa.

| Đặc trưng | Số chiều | Bộ phân lớp | Ảnh gốc | Sụt (V×0,6) | Sụt (V×1,4) |
|---|---|---|---|---|---|
| HOG, chuẩn hóa block L2 | 3780 | KNN (k=5) | 0,946 ± 0,007 | +0,009 ± 0,007 | +0,163 ± 0,008 |
| HOG, chuẩn hóa block L2 | 3780 | LogReg | 0,985 ± 0,004 | +0,004 ± 0,004 | +0,215 ± 0,033 |
| HOG, không chuẩn hóa block | 1152 | KNN (k=5) | 0,845 ± 0,015 | +0,394 ± 0,023 | -0,032 ± 0,011 |
| HOG, không chuẩn hóa block | 1152 | LogReg | 0,960 ± 0,005 | +0,080 ± 0,009 | +0,057 ± 0,006 |
| Hist RGB (3×32) | 96 | KNN (k=5) | 0,964 ± 0,012 | +0,873 ± 0,013 | +0,760 ± 0,024 |
| Hist RGB (3×32) | 96 | LogReg | 0,976 ± 0,013 | +0,763 ± 0,064 | +0,696 ± 0,036 |
| Hist HSV (8,8,8) | 512 | KNN (k=5) | 0,965 ± 0,014 | +0,663 ± 0,043 | +0,376 ± 0,025 |
| Hist HSV (8,8,8) | 512 | LogReg | 0,991 ± 0,005 | +0,673 ± 0,035 | +0,329 ± 0,009 |
| Hist H-S (8,8) | 64 | KNN (k=5) | 0,976 ± 0,006 | +0,000 ± 0,008 | +0,001 ± 0,002 |
| Hist H-S (8,8) | 64 | LogReg | 0,985 ± 0,008 | +0,027 ± 0,011 | +0,011 ± 0,004 |

| Ảnh | Tỉ lệ cell gần như không có gradient | Tỉ lệ điểm ảnh bão hòa (kênh lớn nhất = 255) |
|---|---|---|
| Ảnh gốc | 0,002 | 0,005 |
| V×0,6 | 0,003 | 0,000 |
| V×1,4 | 0,141 | 0,687 |

*Kết luận:* khớp khi làm tối (V×0,6: sụt 0,004–0,009 so với 0,08–0,39). **Trái dự đoán khi làm sáng** (V×1,4: bản chuẩn hóa sụt 0,16–0,22, bản không chuẩn hóa chỉ −0,03 đến +0,06). Nguyên nhân: V×1,4 làm 68,7% điểm ảnh bão hòa ở 255 và 14,1% số cell mất hẳn gradient. Đây không còn là phép đổi độ sáng tuyến tính mà chuẩn hóa khử được. Chuẩn hóa L2 còn khuếch đại các block nền có gradient yếu lên độ dài đơn vị, nên khi nền bão hòa thì vector đã chuẩn hóa đổi nhiều hơn.

**C3 — Tham số HOG.** *Dự đoán:* kích thước ảnh, cell và số bin quyết định mức chi tiết.

| Kích thước ảnh (rộng×cao) | Cell | Số bin | Số chiều | Rút đặc trưng (ms/ảnh) | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|---|---|---|---|
| 64×128 | 8×8 | 6 | 2520 | 0,76 | 0,954 ± 0,007 | 0,982 ± 0,007 | 0,977 ± 0,003 |
| 64×128 | 8×8 | 9 | 3780 | 0,94 | 0,946 ± 0,007 | 0,985 ± 0,004 | 0,982 ± 0,004 |
| 64×128 | 8×8 | 12 | 5040 | 1,31 | 0,939 ± 0,011 | 0,985 ± 0,006 | 0,979 ± 0,006 |
| 64×128 | 16×16 | 6 | 504 | 0,96 | 0,920 ± 0,010 | 0,959 ± 0,006 | 0,959 ± 0,008 |
| 64×128 | 16×16 | 9 | 756 | 0,83 | 0,911 ± 0,010 | 0,959 ± 0,005 | 0,958 ± 0,009 |
| 64×128 | 16×16 | 12 | 1008 | 1,00 | 0,910 ± 0,008 | 0,960 ± 0,006 | 0,959 ± 0,009 |
| 128×128 | 8×8 | 6 | 5400 | 1,86 | 0,952 ± 0,013 | 0,986 ± 0,005 | 0,978 ± 0,006 |
| 128×128 | 8×8 | 9 | 8100 | 1,88 | 0,942 ± 0,017 | 0,986 ± 0,004 | 0,982 ± 0,007 |
| 128×128 | 8×8 | 12 | 10800 | 2,32 | 0,941 ± 0,013 | 0,985 ± 0,005 | 0,978 ± 0,010 |
| 128×128 | 16×16 | 6 | 1176 | 1,35 | 0,961 ± 0,007 | 0,983 ± 0,007 | 0,981 ± 0,005 |
| 128×128 | 16×16 | 9 | 1764 | 1,96 | 0,951 ± 0,014 | 0,982 ± 0,007 | 0,981 ± 0,011 |
| 128×128 | 16×16 | 12 | 2352 | 1,89 | 0,954 ± 0,011 | 0,984 ± 0,006 | 0,981 ± 0,010 |

*Kết luận:* khớp một phần. Chỉ tỉ lệ kích thước cell so với kích thước ảnh tạo khác biệt đo được: cell 16×16 trên ảnh 64×128 còn 0,959. Với LogReg, 9/12 cấu hình nằm trong khoảng 0,982–0,986 và số bin (6, 9, 12) không tạo khác biệt vượt độ lệch chuẩn. Đồ án giữ cấu hình của bài giảng (64×128, cell 8, 9 bin, 3.780 chiều).

**C4 — Đặc trưng toàn cục và cục bộ khi bị che.** *Dự đoán:* che một hình vuông 25% diện tích thì histogram (toàn cục) sụt nhiều hơn HOG (ghép từ đặc trưng cục bộ).

| Đặc trưng | Loại | Bộ phân lớp | Ảnh gốc | Che 25% | Độ sụt |
|---|---|---|---|---|---|
| Hist RGB (3×32) | toàn cục | KNN (k=5) | 0,964 ± 0,012 | 0,085 ± 0,000 | +0,879 ± 0,012 |
| Hist RGB (3×32) | toàn cục | LogReg | 0,976 ± 0,013 | 0,085 ± 0,000 | +0,890 ± 0,014 |
| Hist HSV (8,8,8) | toàn cục | KNN (k=5) | 0,965 ± 0,014 | 0,077 ± 0,003 | +0,887 ± 0,013 |
| Hist HSV (8,8,8) | toàn cục | LogReg | 0,991 ± 0,005 | 0,222 ± 0,113 | +0,769 ± 0,117 |
| Hist H-S (8,8) | toàn cục | KNN (k=5) | 0,976 ± 0,006 | 0,843 ± 0,030 | +0,133 ± 0,029 |
| Hist H-S (8,8) | toàn cục | LogReg | 0,985 ± 0,008 | 0,762 ± 0,028 | +0,224 ± 0,021 |
| Moment màu | toàn cục | KNN (k=5) | 0,972 ± 0,008 | 0,312 ± 0,042 | +0,659 ± 0,043 |
| Moment màu | toàn cục | LogReg | 0,908 ± 0,009 | 0,227 ± 0,006 | +0,681 ± 0,010 |
| HOG 64×128 | cục bộ ghép lại | KNN (k=5) | 0,946 ± 0,007 | 0,784 ± 0,030 | +0,162 ± 0,028 |
| HOG 64×128 | cục bộ ghép lại | LogReg | 0,985 ± 0,004 | 0,794 ± 0,027 | +0,192 ± 0,026 |
| Cạnh Sobel | cục bộ ghép lại | KNN (k=5) | 0,890 ± 0,019 | 0,322 ± 0,006 | +0,568 ± 0,023 |
| Cạnh Sobel | cục bộ ghép lại | LogReg | 0,894 ± 0,008 | 0,272 ± 0,014 | +0,621 ± 0,017 |

| Đặc trưng | KNN (k=5), chuẩn hóa | Ảnh gốc | Che 25% | Độ sụt |
|---|---|---|---|---|
| Hist RGB (3×32) | z-score | 0,964 ± 0,012 | 0,085 ± 0,000 | +0,879 ± 0,012 |
| Hist RGB (3×32) | không chuẩn hóa | 0,952 ± 0,013 | 0,859 ± 0,024 | +0,093 ± 0,027 |
| Hist RGB (3×32) | Hellinger | 0,974 ± 0,011 | 0,870 ± 0,028 | +0,104 ± 0,026 |
| Hist HSV (8,8,8) | z-score | 0,965 ± 0,014 | 0,077 ± 0,003 | +0,887 ± 0,013 |
| Hist HSV (8,8,8) | không chuẩn hóa | 0,953 ± 0,006 | 0,920 ± 0,021 | +0,032 ± 0,019 |
| Hist HSV (8,8,8) | Hellinger | 0,976 ± 0,015 | 0,964 ± 0,013 | +0,012 ± 0,002 |
| Hist H-S (8,8) | z-score | 0,976 ± 0,006 | 0,843 ± 0,030 | +0,133 ± 0,029 |
| Hist H-S (8,8) | không chuẩn hóa | 0,947 ± 0,011 | 0,556 ± 0,103 | +0,391 ± 0,107 |
| Hist H-S (8,8) | Hellinger | 0,979 ± 0,005 | 0,605 ± 0,136 | +0,374 ± 0,138 |

![C4 — Che 25%](../results/figures/C4_occlusion.png)

*Kết luận:* với cấu hình mặc định (z-score) thì khớp: Hist RGB/HSV rơi về 0,08–0,22, HOG còn 0,78–0,79. Nhưng phép đo phụ cho thấy **nguyên nhân chính là bước chuẩn hóa z-score**, không phải tính "toàn cục". Hình vuông đen dồn 25% khối lượng histogram vào một bin hầu như luôn bằng 0 trên ảnh huấn luyện; z-score chia cho độ lệch chuẩn rất nhỏ của bin đó nên một chiều chi phối khoảng cách. Dùng Hellinger thì Hist HSV chỉ sụt 0,012.

**C5 — Màu và hình dạng bổ sung cho nhau.** *Dự đoán:* ghép màu + HOG (chuẩn hóa từng phần) tốt hơn từng đặc trưng riêng lẻ.

| Đặc trưng | Cách ghép | Số chiều | KNN (k=5) | LogReg | Linear SVM |
|---|---|---|---|---|---|
| Hist HSV (8,8,8) | — | 512 | 0,965 ± 0,014 | 0,991 ± 0,005 | 0,989 ± 0,008 |
| Hist H-S (8,8) | — | 64 | 0,976 ± 0,006 | 0,985 ± 0,008 | 0,958 ± 0,007 |
| HOG 64×128 | — | 3780 | 0,946 ± 0,007 | 0,985 ± 0,004 | 0,982 ± 0,004 |
| Hist HSV + HOG | ghép thẳng, z-score chung | 4292 | 0,943 ± 0,008 | 0,993 ± 0,003 | 0,991 ± 0,003 |
| Hist HSV + HOG | chuẩn hóa từng phần rồi ghép | 4292 | 0,956 ± 0,012 | 0,993 ± 0,004 | 0,961 ± 0,007 |
| Hist H-S + HOG | ghép thẳng, z-score chung | 3844 | 0,948 ± 0,009 | 0,989 ± 0,003 | 0,989 ± 0,003 |
| Hist H-S + HOG | chuẩn hóa từng phần rồi ghép | 3844 | 0,977 ± 0,005 | 0,992 ± 0,003 | 0,966 ± 0,014 |
| Hist HSV + moment + HOG | ghép thẳng, z-score chung | 4301 | 0,945 ± 0,008 | 0,993 ± 0,003 | 0,992 ± 0,003 |
| Hist HSV + moment + HOG | chuẩn hóa từng phần rồi ghép | 4301 | 0,985 ± 0,009 | 0,991 ± 0,004 | 0,959 ± 0,007 |

![C5 — Ghép đặc trưng](../results/figures/C5_fusion.png)

*Kết luận:* khớp với KNN: chuẩn hóa từng phần nâng Hist H-S + HOG từ 0,948 (ghép thẳng) lên 0,977. Sau khi chọn siêu tham số (mục 5.7), cấu hình ghép đạt 0,996 ± 0,003, hơn HOG riêng 0,010 (lớn hơn độ lệch chuẩn) và hơn màu riêng 0,005 (bằng độ lệch chuẩn, chưa đủ cơ sở phân biệt).

### 5.7 Thí nghiệm bộ phân lớp: P1–P6 và cấu hình truyền thống đã chốt

**P1 — Chọn k.** *Dự đoán:* k nhỏ nhạy nhiễu, k lớn kém nhạy với đặc điểm cục bộ; so quy tắc k = số lớp + 1 với k chọn bằng cross-validation.

| Đặc trưng | k = số lớp + 1 = 5 | k tốt nhất theo CV | Macro-F1 tại k tốt nhất | Chênh so với k=5 | k=1 | k=31 |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0,972 ± 0,008 | 1 | 0,977 ± 0,003 | +0,006 | 0,977 ± 0,003 | 0,937 ± 0,012 |
| Hist HSV (512 chiều) | 0,965 ± 0,014 | 1 | 0,975 ± 0,005 | +0,011 | 0,975 ± 0,005 | 0,867 ± 0,026 |
| HOG 64×128 (3780 chiều) | 0,946 ± 0,007 | 1 | 0,967 ± 0,011 | +0,020 | 0,967 ± 0,011 | 0,871 ± 0,012 |

![P1 — KNN theo k](../results/figures/P1_knn_k.png)

*Kết luận:* vế "k lớn" khớp (k = 31 thấp nhất). Vế "k nhỏ nhạy nhiễu" **không xuất hiện**: k = 1 tốt nhất với cả ba đặc trưng, hơn quy tắc k = 5 từ 0,006 đến 0,020. Các lớp tách nhau rõ và mỗi quả thường có nhiều ảnh gần giống nhau, nên láng giềng gần nhất hầu như luôn cùng lớp; tăng k chỉ kéo thêm điểm ở xa, phần nhiều thuộc lớp đông nhất.

**P2 — Độ đo và thang đo.** *Dự đoán:* KNN phụ thuộc độ đo và chuẩn hóa; sau khi chuẩn hóa về vector đơn vị, Euclidean và cosine cho cùng thứ hạng láng giềng.

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

*Kết luận:* khớp. Sau chuẩn hóa đơn vị, 100% ảnh truy vấn có cùng tập 10 láng giềng theo Euclidean và theo cosine (trước chuẩn hóa: 11% với moment, 18% với hist HSV); macro-F1 của hai cách trùng nhau. Manhattan tốt nhất trên HOG (0,978 so với 0,954 của Euclidean).

**P3 — Ưu tiên điểm gần.** *Dự đoán:* `weights='distance'` giúp khi các lớp chồng lấn, nhất là với lớp ít ảnh.

| Đặc trưng | k | Macro-F1 uniform | Macro-F1 distance | Recall Bangla Kola (lớp ít ảnh nhất) uniform | Recall Bangla Kola distance | Recall thấp nhất trong 4 lớp uniform → distance |
|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 5 | 0,972 ± 0,008 | 0,972 ± 0,008 | 0,984 ± 0,013 | 0,984 ± 0,013 | 0,922 → 0,924 |
| Moment màu (9 chiều) | 15 | 0,955 ± 0,008 | 0,961 ± 0,006 | 0,942 ± 0,026 | 0,968 ± 0,018 | 0,888 → 0,888 |
| Hist HSV (512 chiều) | 5 | 0,965 ± 0,014 | 0,966 ± 0,013 | 0,981 ± 0,007 | 0,981 ± 0,007 | 0,879 → 0,888 |
| Hist HSV (512 chiều) | 15 | 0,931 ± 0,013 | 0,933 ± 0,014 | 0,952 ± 0,018 | 0,955 ± 0,022 | 0,756 → 0,765 |
| HOG 64×128 (3780 chiều) | 5 | 0,946 ± 0,007 | 0,949 ± 0,008 | 0,955 ± 0,021 | 0,952 ± 0,026 | 0,816 → 0,830 |
| HOG 64×128 (3780 chiều) | 15 | 0,912 ± 0,011 | 0,914 ± 0,010 | 0,920 ± 0,033 | 0,923 ± 0,036 | 0,697 → 0,706 |

*Kết luận:* chưa đủ cơ sở phân biệt. Macro-F1 đổi ≤ 0,006; recall của Bangla Kola (lớp ít ảnh nhất) đổi từ −0,003 đến +0,026, nhỏ hơn hoặc xấp xỉ độ lệch chuẩn giữa các fold.

**P4 — K-D Tree và Ball Tree.** *Dự đoán:* chỉ tăng tốc, không đổi kết quả, và kém hiệu quả khi nhiều chiều.

| Đặc trưng | brute: dựng (ms) | brute: dự đoán 420 ảnh (ms) | kd_tree: dựng (ms) | kd_tree: dự đoán 420 ảnh (ms) | kd_tree: kết quả trùng brute | ball_tree: dựng (ms) | ball_tree: dự đoán 420 ảnh (ms) | ball_tree: kết quả trùng brute |
|---|---|---|---|---|---|---|---|---|
| Moment màu (9 chiều) | 0,7 | 1,8 | 1,5 | 4,0 | có | 1,6 | 4,9 | có |
| Hist HSV (512 chiều) | 1,5 | 8,1 | 38,1 | 363,7 | có | 32,7 | 271,6 | có |
| HOG 64×128 (3780 chiều) | 7,3 | 63,3 | 276,2 | 3795,9 | có | 303,9 | 3196,5 | có |

| Số điểm tham chiếu | brute: dự đoán 420 ảnh (ms) | kd_tree: dự đoán 420 ảnh (ms) | ball_tree: dự đoán 420 ảnh (ms) |
|---|---|---|---|
| 1680 | 1,9 | 3,7 | 4,5 |
| 8400 | 21,3 | 5,8 | 8,4 |
| 33600 | 111,3 | 10,6 | 24,0 |
| 100800 | 487,0 | 20,6 | 68,3 |

![P4 — Thuật toán tìm láng giềng](../results/figures/P4_knn_algorithms.png)

*Kết luận:* khớp, kèm điều kiện. Kết quả trùng Brute Force trong mọi trường hợp. Với 3.780 chiều, K-D Tree chậm hơn Brute Force khoảng 60 lần. Ngay cả với 9 chiều, khi chỉ có 1.680 ảnh tham chiếu thì cây vẫn chậm hơn; từ 8.400 điểm trở lên K-D Tree mới nhanh hơn (6,1 ms so với 22,3 ms). Cây chỉ có lợi khi **ít chiều và tập tham chiếu đủ lớn**.

**P5 — Thời gian huấn luyện và dự đoán.** *Dự đoán:* KNN không tốn thời gian huấn luyện nhưng dự đoán chậm; mô hình tuyến tính thì ngược lại.

| Đặc trưng | Bộ phân lớp | Huấn luyện 1680 ảnh (ms) | Dự đoán 420 ảnh (ms) | Dự đoán (ms/ảnh) | Macro-F1 |
|---|---|---|---|---|---|
| Moment màu (9 chiều) | KNN (k=5) | 2,2 | 3,9 | 0,0094 | 0,972 ± 0,008 |
| Moment màu (9 chiều) | Logistic Regression | 14,7 | 0,4 | 0,0010 | 0,908 ± 0,009 |
| Moment màu (9 chiều) | Linear SVM | 5,0 | 0,4 | 0,0011 | 0,838 ± 0,011 |
| Hist HSV (512 chiều) | KNN (k=5) | 8,0 | 21,7 | 0,0518 | 0,965 ± 0,014 |
| Hist HSV (512 chiều) | Logistic Regression | 479,1 | 4,0 | 0,0095 | 0,991 ± 0,005 |
| Hist HSV (512 chiều) | Linear SVM | 464,2 | 2,7 | 0,0065 | 0,989 ± 0,008 |
| HOG 64×128 (3780 chiều) | KNN (k=5) | 70,7 | 84,6 | 0,2014 | 0,946 ± 0,007 |
| HOG 64×128 (3780 chiều) | Logistic Regression | 1019,2 | 13,8 | 0,0328 | 0,985 ± 0,004 |
| HOG 64×128 (3780 chiều) | Linear SVM | 2211,7 | 8,3 | 0,0197 | 0,982 ± 0,004 |

*Kết luận:* khớp. Trên HOG, KNN "huấn luyện" 0,07 s (chỉ lưu dữ liệu) nhưng dự đoán 0,22 ms/ảnh; LogReg huấn luyện 1,2 s nhưng dự đoán chỉ 0,02 ms/ảnh.

**P6 — Biên quyết định tuyến tính.** *Dự đoán:* LogReg kém KNN khi đặc trưng ít chiều, tốt hơn khi nhiều chiều.

| Đặc trưng | KNN (k=5) | Logistic Regression | Linear SVM | LogReg − KNN |
|---|---|---|---|---|
| Moment màu (9 chiều) | 0,972 ± 0,008 | 0,908 ± 0,009 | 0,838 ± 0,011 | -0,064 |
| Hist HSV (512 chiều) | 0,965 ± 0,014 | 0,991 ± 0,005 | 0,989 ± 0,008 | +0,027 |
| HOG 64×128 (3780 chiều) | 0,946 ± 0,007 | 0,985 ± 0,004 | 0,982 ± 0,004 | +0,039 |

![P6 — KNN so với bộ phân lớp tuyến tính](../results/figures/P6_linear_vs_knn.png)

*Kết luận:* khớp. Với 9 chiều, LogReg kém KNN 0,064; với 512 và 3.780 chiều thì hơn 0,027 và 0,039. Trong không gian nhiều chiều, các lớp dễ tách tuyến tính hơn.

**Cấu hình truyền thống đã chốt.** Quét k ∈ {1, 3, 5}, độ đo (Euclidean, Manhattan, Hellinger) cho KNN và tham số C cho LogReg, Linear SVM (`results/select_traditional/table.md`):

| Vai trò | Cấu hình | Macro-F1 (5-fold) | Accuracy |
|---|---|---|---|
| trad_color | Hist HSV (8,8,8) + LogReg C=0,1 | 0,991 ± 0,005 | 0,992 |
| trad_edge | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0,955 ± 0,015 | 0,959 |
| trad_hog | HOG 64×128 + LogReg C=0,1 | 0,986 ± 0,007 | 0,989 |
| trad_fusion | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 0,996 ± 0,003 | 0,997 |

### 5.8 CNN hướng A và hướng B: N1–N8

Hai hướng được trình bày riêng (5.8.1, 5.8.2) trước khi so sánh (5.8.3). Số liệu CNN trong mục này chấm trên val (371 ảnh); kết quả trên test ở mục 5.11.

#### 5.8.1 Hướng A — CNN theo bài giảng

**N1 — Số tham số tính được từ kiến trúc.** Lớp tích chập có (F×F×C + 1)×K tham số, lớp kết nối đầy đủ có (N_in + 1)×N_out tham số.

**Ví dụ tính tay 3.** Lớp `conv2` của `a_lenet`: kernel 5×5, đầu vào 6 kênh, 16 bộ lọc → (5×5×6 + 1) × 16 = 151 × 16 = 2.416 tham số. Lớp `fc1`: ảnh 64×64 qua Conv 5×5 còn 60×60, qua MaxPool còn 30×30, qua Conv 5×5 còn 26×26, qua MaxPool còn 13×13; Flatten cho 13 × 13 × 16 = 2.704 giá trị → (2.704 + 1) × 120 = 324.600 tham số.

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

*Kết luận:* 10/10 lớp khớp đúng `model.summary()`. Lớp Dense đầu tiên chiếm 96% số tham số của `a_lenet` và 99% của `a_dropout`.

Kết quả huấn luyện của hướng A trên val: `a_lenet` 0,973 (ảnh vào 64×64; 32×32 cho 0,954; 128×128 cho 0,956), `a_dropout` 0,940 (0,948 khi có tăng cường ảnh), `a_vgg16` 0,984 với learning rate 1e-4 và 0,990 với 1e-3 (`results/cnn/table_runs.md`).

**N2 — Lượng dữ liệu.** *Tính chất:* CNN học đặc trưng từ dữ liệu nên cần nhiều dữ liệu; transfer learning tận dụng đặc trưng đã học. *Dự đoán:* khi giảm dữ liệu, `a_lenet` sụt nhiều hơn `a_vgg16` và `b_mobilenetv3`.

| Mô hình | 25% train (432 ảnh) | 50% train (864 ảnh) | 100% train (1729 ảnh) | Tăng từ 25% lên 100% |
|---|---|---|---|---|
| a_lenet (LeNet, từ đầu) | 0,924 | 0,950 | 0,973 | +0,050 |
| a_vgg16 (đóng băng) | 0,967 | 0,963 | 0,984 | +0,016 |
| b_mobilenetv3 (fine-tune) | 0,975 | 0,974 | 0,993 | +0,018 |

![N2 — Đường cong học](../results/figures/N2_learning_curve.png)

*Kết luận:* khớp. Từ 25% lên 100% train, `a_lenet` tăng 0,050, hai mô hình pretrained chỉ tăng 0,016–0,018. Với 25% train (432 ảnh), `a_vgg16` (0,967) và `b_mobilenetv3` (0,975) đã ngang hoặc hơn `a_lenet` dùng đủ 1.729 ảnh (0,973).

**N3 — Dropout giảm quá khớp.** *Dự đoán:* `a_dropout` có Dropout có khoảng cách train–val nhỏ hơn và kết quả trên val tốt hơn bản không Dropout.

| Cấu hình | Số lần chạy | Macro-F1 val từng seed | Macro-F1 val (TB ± ĐLC) | Khoảng cách loss val − train ở epoch 40 (TB ± ĐLC) | Loss val thấp nhất (TB ± ĐLC) |
|---|---|---|---|---|---|
| Có Dropout(0,5) | 3 | 0,940, 0,947, 0,935 | 0,941 ± 0,005 | 0,456 ± 0,061 | 0,248 ± 0,021 |
| Không Dropout | 3 | 0,951, 0,960, 0,947 | 0,953 ± 0,006 | 0,376 ± 0,081 | 0,202 ± 0,055 |

![N3 — Dropout](../results/figures/N3_dropout.png)

*Kết luận:* **trái dự đoán**. Qua 3 seed khởi tạo, bản có Dropout(0,5) kém hơn 0,012 (0,941 ± 0,005 so với 0,953 ± 0,006), và khoảng cách loss val − train không nhỏ hơn. Giải thích của nhóm: 99% tham số nằm ở lớp Dense ngay sau Flatten, trước lớp Dropout; lớp này vẫn khớp tập train đến accuracy 0,999 ở cả hai cấu hình. Dropout chỉ làm nhiễu 128 đầu ra của lớp đó nên không chặn được quá khớp ở nơi nó xảy ra, lại làm quá trình tối ưu ồn hơn.

**N5 — Tăng cường ảnh giảm quá khớp.** (Hướng A và B chung một bảng.)

| Cấu hình | Epoch chọn / tổng | Macro-F1 val | Accuracy train (epoch chọn) | Accuracy val (epoch chọn) | Loss train (epoch cuối) | Loss val (epoch cuối) | Khoảng cách loss val − train (epoch cuối) | Loss val thấp nhất |
|---|---|---|---|---|---|---|---|---|
| a_dropout, không tăng cường | 25 / 40 | 0,940 | 0,999 | 0,946 | 0,0022 | 0,5444 | 0,5422 | 0,2713 |
| a_dropout, có tăng cường | 25 / 40 | 0,948 | 0,967 | 0,954 | 0,2613 | 0,3150 | 0,0536 | 0,1286 |
| b_mobilenetv3 giai đoạn 2, không tăng cường | 4 / 9 | 0,997 | 0,999 | 0,997 | 0,0045 | 0,0107 | 0,0062 | 0,0107 |
| b_mobilenetv3 giai đoạn 2, có tăng cường | 12 / 17 | 0,993 | 0,997 | 0,995 | 0,0125 | 0,0183 | 0,0058 | 0,0183 |

![N5 — a_dropout có và không có tăng cường](../results/figures/N5_augment_a_dropout.png)

*Kết luận:* khớp với mô hình học từ đầu. Với `a_dropout`, khoảng cách loss val − train giảm từ 0,542 xuống 0,054 và macro-F1 tăng 0,940 → 0,948; tăng cường tác động vào đầu vào nên hiệu quả hơn Dropout rõ rệt. Với `b_mobilenetv3` pretrained, quá khớp vốn đã nhỏ nên tăng cường không tăng điểm trên val (0,993 so với 0,997), nhưng làm mô hình bền hơn với phép xoay ±30°: 0,987–0,993 so với 0,958–0,978 (bảng N8).

#### 5.8.2 Hướng B — CNN lớn hơn, fine-tune

**N4 — Đóng băng và fine-tune.** *Tính chất:* đóng băng giữ nguyên đặc trưng đã học; fine-tune thích nghi đặc trưng với dữ liệu mới, nhưng learning rate lớn sẽ phá trọng số đã học. *Dự đoán:* giai đoạn 2 tốt hơn giai đoạn 1; lr 1e-3 kém và kém ổn định hơn lr 1e-5.

| Cấu hình | Tham số được huấn luyện | Macro-F1 val | Accuracy val | Loss val (epoch chọn) | Macro-F1 val sau epoch 1 | Macro-F1 val thấp nhất | Epoch chọn / tổng | Thời gian mỗi epoch (s) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small, giai đoạn 1 (đóng băng thân) | 2308 | 0,987 | 0,989 | 0,0505 | 0,872 | 0,872 | 21 / 31 | 0,1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-5 | 917204 | 0,993 | 0,995 | 0,0203 | 0,981 | 0,978 | 12 / 17 | 20,1 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-4 | 917204 | 1,000 | 1,000 | 0,0045 | 0,974 | 0,974 | 7 / 12 | 26,9 |
| MobileNetV3-Small, giai đoạn 2, lr = 1e-3 | 917204 | 0,993 | 0,995 | 0,0184 | 0,964 | 0,917 | 4 / 9 | 26,3 |
| ResNet50, giai đoạn 1 (đóng băng thân) | 8196 | 0,996 | 0,997 | 0,0269 | 0,942 | 0,942 | 10 / 20 | 0,2 |
| ResNet50, giai đoạn 2 (mở băng conv5), lr = 1e-5 | 14961668 | 0,998 | 0,997 | 0,0140 | 0,984 | 0,984 | 3 / 8 | 91,2 |

![N4 — Giai đoạn 1 và giai đoạn 2 theo learning rate](../results/figures/N4_finetune_lr.png)

*Kết luận:* vế đầu khớp, nhưng mức tăng nhỏ vì giai đoạn 1 đã rất cao: MobileNetV3 0,987 → 0,993–1,000, ResNet50 0,996 → 0,998. Vế learning rate: lr 1e-3 làm quá trình huấn luyện mất ổn định (macro-F1 tụt xuống 0,917 ở epoch 2 và dao động tới cuối, trong khi lr 1e-5 không xuống dưới 0,978). Tuy vậy, nhờ dừng sớm theo val, kết quả cuối không kém. ResNet50 chạy được giai đoạn 2 trên CPU (mở băng khối conv5, khoảng 61 s mỗi epoch, 8 epoch).

**Một chi tiết cài đặt quan trọng.** Lần fine-tune đầu để các lớp BatchNormalization mở băng: accuracy trên train ở epoch 1 chỉ 0,356. Phép đo lặp lại được cho thấy nguyên nhân:

| Thân mạng | Cách mở băng | Thay đổi tương đối của embedding (training=True so với suy luận) | Accuracy đầu phân lớp GĐ1, embedding chế độ suy luận | Accuracy đầu phân lớp GĐ1, embedding chế độ huấn luyện |
|---|---|---|---|---|
| b_mobilenetv3 | BN mở (trainable=True) | 0,820 | 1,000 | 0,047 |
| b_mobilenetv3 | BN đóng băng (cách đồ án dùng) | 0,000 | 1,000 | 1,000 |
| b_resnet50 | BN mở (trainable=True) | 0,838 | 1,000 | 0,125 |
| b_resnet50 | BN đóng băng (cách đồ án dùng) | 0,000 | 1,000 | 1,000 |

Khi BN ở chế độ huấn luyện, nó dùng thống kê của từng lô thay cho thống kê đã học trên ImageNet. Embedding đổi 82–84%, và đầu phân lớp học ở giai đoạn 1 chỉ còn đoán đúng 4,7–12,5% ảnh. Giữ BN đóng băng khi fine-tune thì vấn đề biến mất.

#### 5.8.3 So sánh hai hướng

**N6 — Mạng lớn hơn, pretrained, tốt hơn nhưng tốn tài nguyên hơn.**

| Hướng | Mô hình | Biến thể được chốt | Macro-F1 val | Accuracy val | Số tham số | File mô hình (MB) | Thời gian huấn luyện (s) | Dự đoán 1 ảnh (ms, CPU) | Dự đoán theo lô 32 (ms/ảnh) | Cách tính thời gian huấn luyện |
|---|---|---|---|---|---|---|---|---|---|---|
| A | a_lenet | a_lenet | 0,973 | 0,978 | 337976 | 4,1 | 30 | 1,4 | 0,2 | 50 epoch × 0,6 s |
| A | a_dropout | a_dropout_aug | 0,948 | 0,954 | 1746260 | 21,0 | 192 | 2,5 | 0,7 | 40 epoch × 4,8 s |
| A | a_vgg16 | a_vgg16_lr1e-3 | 0,990 | 0,992 | 40669252 | 370,4 | 226 | 130,4 | 67,5 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 129 s; đầu phân lớp 24 epoch × 4,0 s |
| B | b_mobilenetv3 | b_mobilenetv3_lr1e-4 | 1,000 | 1,000 | 941428 | 11,7 | 216 | 12,5 | 2,6 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 25 s; giai đoạn 1: 31 epoch, 4 s; giai đoạn 2: 12 epoch × 15,6 s |
| B | b_resnet50 | b_resnet50 | 0,998 | 0,997 | 23595908 | 214,8 | 591 | 61,1 | 28,0 | tính đặc trưng thân đóng băng cho 1729 ảnh train: 98 s; giai đoạn 1: 20 epoch, 4 s; giai đoạn 2: 8 epoch × 61,2 s |

![N6 — Hướng A và hướng B](../results/figures/N6_a_vs_b.png)

*Kết luận:* vế "tốt hơn" khớp: hướng B đạt 0,998–1,000 trên val, hướng A 0,948–0,990. Vế "tốn tài nguyên hơn" chỉ đúng với ResNet50. MobileNetV3-Small có 0,94 triệu tham số, file 11,7 MB, dự đoán 12,5 ms mỗi ảnh; trong khi `a_vgg16` của hướng A có 40,7 triệu tham số, file 370 MB, dự đoán 130 ms mỗi ảnh. Một kiến trúc được thiết kế cho thiết bị di động có thể vừa chính xác hơn vừa nhẹ hơn mô hình theo bài giảng.

**N7 — Feature vector thủ công và embedding.** *Dự đoán:* embedding do mạng học tốt hơn đặc trưng thủ công với cùng bộ phân lớp.

| Đặc trưng | Loại | Số chiều | KNN (k=5) | LogReg |
|---|---|---|---|---|
| Hist HSV (8,8,8) | thủ công | 512 | 0,982 | 0,997 |
| HOG 64×128 | thủ công | 3780 | 0,945 | 0,987 |
| Hist H-S + HOG (ghép, chuẩn hóa từng phần) | thủ công | 3844 | 0,990 | 0,998 |
| VGG16 ImageNet, trung bình 7×7 (chưa học trên chuối) | embedding | 512 | 0,932 | 0,986 |
| a_vgg16, lớp Dense(256) áp chót | embedding | 256 | 0,986 | 0,986 |
| MobileNetV3 ImageNet, sau GAP (chưa fine-tune) | embedding | 576 | 0,981 | 0,993 |
| b_mobilenetv3_lr1e-4, sau GAP (đã fine-tune) | embedding | 576 | 1,000 | 0,996 |

![N7 — Đặc trưng thủ công và embedding](../results/figures/N7_embedding.png)

*Kết luận:* khớp với KNN (embedding MobileNetV3 đã fine-tune: 1,000; đặc trưng thủ công tốt nhất: 0,990). Với LogReg thì ngang nhau (0,996 so với 0,998). Embedding ImageNet **chưa học trên chuối** (VGG16) cho KNN chỉ 0,932, kém histogram HSV (0,982). Embedding chỉ vượt trội sau khi được học trên chính dữ liệu này.

**N8 — CNN có bền với thay đổi hơn đặc trưng thủ công không?**

| Mô hình | Ảnh gốc | Xáo trộn 4×4 | V×0,6 | V×1,4 | Ảnh xám | Nhiễu σ=5 | Nhiễu σ=10 | Nhiễu σ=20 | Che 25% | Xoay −30° | Xoay −15° | Xoay +15° | Xoay +30° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Truyền thống: Hist HSV (8,8,8) | 0,997 | 0,997 | 0,280 | 0,705 | 0,145 | 0,451 | 0,085 | 0,085 | 0,076 | 0,987 | 0,993 | 0,992 | 0,994 |
| Truyền thống: HOG 64×128 | 0,983 | 0,302 | 0,984 | 0,800 | 0,981 | 0,537 | 0,461 | 0,318 | 0,801 | 0,679 | 0,929 | 0,886 | 0,550 |
| Truyền thống: Hist H-S + HOG | 0,998 | 0,898 | 0,995 | 0,888 | 0,770 | 0,706 | 0,180 | 0,101 | 0,918 | 0,978 | 0,987 | 0,988 | 0,974 |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | 0,990 | 0,097 | 0,984 | 0,969 | 0,953 | 0,981 | 0,976 | 0,932 | 0,850 | 0,888 | 0,973 | 0,980 | 0,906 |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | 1,000 | 0,850 | 0,913 | 0,907 | 0,963 | 0,997 | 0,987 | 0,953 | 0,659 | 0,993 | 1,000 | 0,997 | 0,996 |
| Hướng B, lr 1e-5, có tăng cường (N5): b_mobilenetv3 | 0,993 | 0,775 | 0,899 | 0,936 | 0,914 | 0,980 | 0,972 | 0,949 | 0,506 | 0,987 | 0,990 | 0,993 | 0,993 |
| Hướng B, lr 1e-5, không tăng cường (N5): b_mobilenetv3_noaug | 0,997 | 0,717 | 0,952 | 0,911 | 0,959 | 0,977 | 0,955 | 0,908 | 0,555 | 0,958 | 0,977 | 0,993 | 0,978 |

![N8 — Độ bền theo phép biến đổi](../results/figures/N8_robustness.png)

*Kết luận:* CNN bền hơn hẳn với **nhiễu** (MobileNetV3 0,953 ở σ = 20, histogram và H-S + HOG ≤ 0,101), **xoay** và **ảnh xám**. **Trái dự đoán** với phép che 25%: MobileNetV3 còn 0,659, kém HOG (0,801) và H-S + HOG (0,918). Với xáo trộn 4×4, `a_vgg16` sập về 0,097 vì dùng Flatten, mỗi trọng số gắn với một vị trí cố định; MobileNetV3 dùng GlobalAveragePooling nên giữ được 0,850. Không có phương pháp nào bền với mọi phép biến đổi; mỗi phương pháp bền với đúng loại thay đổi mà cấu trúc của nó bất biến.

### 5.9 Mất cân bằng dữ liệu: I1–I2

**I1 — Accuracy che khuất lớp bị nhận kém.**

| Mô hình (chưa xử lý mất cân bằng) | Chấm trên | Accuracy | Macro-F1 | Accuracy − Macro-F1 | Recall Bangla | Recall Champa | Recall Sabri | Recall Sagor | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Luôn đoán lớp đông nhất (Champa Kola) | train+val | 0,402 | 0,143 | +0,259 | 0,000 | 1,000 | 0,000 | 0,000 | 0,000 (Bangla) |
| HOG 64×128 + KNN k=31 | out-of-fold 5-fold | 0,884 | 0,871 | +0,014 | 0,899 | 1,000 | 0,988 | 0,552 | 0,552 (Sagor) |
| Hist HSV + KNN k=31 | out-of-fold 5-fold | 0,883 | 0,868 | +0,015 | 0,870 | 0,998 | 0,986 | 0,576 | 0,576 (Sagor) |
| Moment màu + LogReg | out-of-fold 5-fold | 0,925 | 0,908 | +0,017 | 0,870 | 0,992 | 0,986 | 0,785 | 0,785 (Sagor) |
| Cạnh Sobel + LogReg | out-of-fold 5-fold | 0,900 | 0,894 | +0,007 | 0,822 | 0,920 | 0,977 | 0,857 | 0,822 (Bangla) |
| Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | out-of-fold 5-fold | 0,959 | 0,955 | +0,004 | 0,950 | 0,995 | 0,988 | 0,870 | 0,870 (Sagor) |
| HOG 64×128 + LogReg C=0,1 | out-of-fold 5-fold | 0,989 | 0,986 | +0,002 | 0,968 | 0,998 | 0,993 | 0,984 | 0,968 (Bangla) |
| Hist HSV (8,8,8) + LogReg C=0,1 | out-of-fold 5-fold | 0,992 | 0,991 | +0,001 | 0,992 | 0,998 | 0,998 | 0,978 | 0,978 (Sagor) |
| Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | out-of-fold 5-fold | 0,997 | 0,996 | +0,000 | 0,995 | 0,996 | 1,000 | 0,996 | 0,995 (Bangla) |
| CNN a_lenet_frac25 | val | 0,930 | 0,924 | +0,006 | 0,879 | 0,953 | 0,961 | 0,899 | 0,879 (Bangla) |
| CNN a_lenet | val | 0,978 | 0,973 | +0,005 | 1,000 | 0,993 | 0,987 | 0,924 | 0,924 (Sagor) |
| CNN a_dropout | val | 0,946 | 0,940 | +0,006 | 0,924 | 0,987 | 0,987 | 0,848 | 0,848 (Sagor) |
| CNN a_vgg16_lr1e-3 | val | 0,992 | 0,990 | +0,002 | 0,985 | 1,000 | 0,987 | 0,987 | 0,985 (Bangla) |
| CNN b_mobilenetv3_s1 | val | 0,989 | 0,987 | +0,002 | 0,970 | 1,000 | 1,000 | 0,975 | 0,970 (Bangla) |
| CNN b_mobilenetv3_lr1e-4 | val | 1,000 | 1,000 | +0,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 (Bangla) |
| CNN b_resnet50_s1 | val | 0,997 | 0,996 | +0,001 | 0,985 | 1,000 | 1,000 | 1,000 | 0,985 (Bangla) |
| CNN b_resnet50 | val | 0,997 | 0,998 | -0,000 | 1,000 | 1,000 | 1,000 | 0,987 | 0,987 (Sagor) |

*Kết luận:* khớp. Bộ phân lớp "luôn đoán Champa Kola" đạt accuracy 0,402 nhưng macro-F1 chỉ 0,143. KNN k = 31 trên HOG đạt accuracy 0,884 nhưng recall Sagor Kola chỉ 0,552. Ở các mô hình tốt, accuracy và macro-F1 chênh ≤ 0,003. Lớp bị nhận kém nhất thường là Sagor Kola (524 ảnh) chứ không phải lớp ít ảnh nhất (Bangla Kola, 444 ảnh).

**I2 — Các cách xử lý mất cân bằng.** Mọi phép lấy mẫu lại chỉ áp dụng trên phần huấn luyện.

| Mô hình | Cách xử lý | Chấm | Macro-F1 | Accuracy | Precision Bangla (ít nhất) | Recall Bangla (ít nhất) | Precision Champa (đông nhất) | Recall Champa (đông nhất) | Recall thấp nhất |
|---|---|---|---|---|---|---|---|---|---|
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Không xử lý | 5-fold CV | 0,996 ± 0,003 | 0,997 | 0,995 | 0,995 | 0,999 | 0,996 | 0,995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Class weight | 5-fold CV | 0,996 ± 0,003 | 0,997 | 0,995 | 0,995 | 0,999 | 0,996 | 0,995 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Oversampling | 5-fold CV | 0,995 ± 0,005 | 0,996 | 0,992 | 0,992 | 0,999 | 0,996 | 0,992 (Bangla) |
| Truyền thống tốt nhất: Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | Tăng cường lớp ít | 5-fold CV | 0,996 ± 0,002 | 0,996 | 0,992 | 0,997 | 0,999 | 0,995 | 0,993 (Sagor) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Không xử lý | 5-fold CV | 0,894 ± 0,008 | 0,900 | 0,849 | 0,822 | 0,921 | 0,920 | 0,822 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Class weight | 5-fold CV | 0,888 ± 0,007 | 0,895 | 0,823 | 0,828 | 0,925 | 0,902 | 0,828 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Oversampling | 5-fold CV | 0,888 ± 0,005 | 0,895 | 0,821 | 0,825 | 0,926 | 0,905 | 0,825 (Bangla) |
| Truyền thống yếu hơn (minh họa): Cạnh Sobel + LogReg | Tăng cường lớp ít | 5-fold CV | 0,884 ± 0,011 | 0,892 | 0,776 | 0,857 | 0,957 | 0,892 | 0,852 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Không xử lý | val | 0,990 | 0,992 | 0,970 | 0,985 | 1,000 | 1,000 | 0,985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Class weight | val | 0,987 | 0,989 | 0,970 | 0,985 | 0,993 | 1,000 | 0,975 (Sagor) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Oversampling | val | 0,987 | 0,989 | 0,956 | 0,985 | 1,000 | 0,993 | 0,985 (Bangla) |
| Hướng A tốt nhất: a_vgg16_lr1e-3 | Tăng cường lớp ít | val | 0,994 | 0,995 | 1,000 | 1,000 | 0,993 | 1,000 | 0,987 (Sabri) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Không xử lý | val | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Class weight | val | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Oversampling | val | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 (Bangla) |
| Hướng B tốt nhất: b_mobilenetv3_lr1e-4 | Tăng cường lớp ít | val | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 (Bangla) |

*Kết luận:* ở mô hình yếu (cạnh Sobel + LogReg), các cách xử lý **đổi precision lấy recall** của lớp ít: tăng cường lớp ít nâng recall Bangla 0,822 → 0,857 nhưng hạ precision Bangla 0,849 → 0,776. Ở các mô hình mạnh không đo được tác động (truyền thống tốt nhất: cả bốn cách 0,995–0,996; `b_mobilenetv3`: cả bốn cách 1,000 trên val). Với tỉ lệ mất cân bằng 2,2 lần, mất cân bằng không phải vấn đề chính của bộ dữ liệu này, nên cấu hình cuối không thêm bước xử lý.

### 5.10 Phân đoạn tách nền: S1–S2

**S1 — Các phương pháp phân đoạn.** *Tính chất:* ngưỡng toàn cục chỉ tốt khi nền và vật tách biệt về mức xám; Otsu tự chọn ngưỡng; k-means gom theo màu. Dữ liệu không có nhãn mask, nên nhóm đánh giá định tính bằng hình, kèm hai chỉ số không cần nhãn: số mảnh rời và tỉ lệ lỗ thủng.

| Phương pháp | Số mảnh rời (trung bình) | Số mảnh rời (trung vị) | Tỉ lệ lỗ thủng trong vật | Tỉ lệ diện tích vật | IoU với phương pháp được chọn |
|---|---|---|---|---|---|
| Ngưỡng toàn cục (T=127) | 11,6 | 9 | 0,085 | 0,105 | 0,689 |
| Otsu (ảnh xám) | 2,9 | 2 | 0,052 | 0,149 | 0,799 |
| Ngưỡng thích nghi | 16,9 | 14 | 0,282 | 0,081 | 0,561 |
| K-means màu (k=2) | 2,5 | 1 | 0,027 | 0,140 | 0,859 |
| Otsu trên khoảng cách màu tới nền | 2,5 | 1 | 0,010 | 0,145 | 0,971 |
| Otsu khoảng cách màu + hậu xử lý (dùng cho S2) | 1,0 | 1 | 0,000 | 0,144 | 1,000 |

![S1 — Các phương pháp phân đoạn](../results/figures/S1_segmentation.png)

*Kết luận:* khớp. Ngưỡng toàn cục và ngưỡng thích nghi tách cả vết thâm trên vỏ thành nền (lỗ thủng 0,085 và 0,282). Ngưỡng thích nghi chỉ bắt được viền và vân. Các cách dựa trên màu (k-means, Otsu trên khoảng cách màu tới nền) cho mask liền nhất, vì nền trơn sáng khác quả chuối về màu rõ hơn về mức xám.

**S2 — Mô hình dựa vào quả chuối hay vào nền?** *Dự đoán:* nếu chỉ dùng vùng nền mà vẫn phân loại tốt thì dữ liệu có thiên lệch theo bối cảnh chụp.

| Đặc trưng + bộ phân lớp | Phiên bản ảnh | Macro-F1 | Accuracy |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0,1 [màu tốt nhất] | Toàn ảnh | 0,991 ± 0,005 | 0,992 ± 0,005 |
| HOG 64×128 + LogReg C=0,1 | Toàn ảnh | 0,986 ± 0,007 | 0,989 ± 0,006 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Toàn ảnh | 0,996 ± 0,003 | 0,997 ± 0,003 |
| Hist HSV (8,8,8) + LogReg C=0,1 [màu tốt nhất] | Chỉ vùng chuối | 0,964 ± 0,010 | 0,967 ± 0,008 |
| HOG 64×128 + LogReg C=0,1 | Chỉ vùng chuối | 0,884 ± 0,010 | 0,888 ± 0,009 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng chuối | 0,958 ± 0,005 | 0,961 ± 0,005 |
| Hist HSV (8,8,8) + LogReg C=0,1 [màu tốt nhất] | Chỉ vùng nền | 0,933 ± 0,008 | 0,943 ± 0,008 |
| HOG 64×128 + LogReg C=0,1 | Chỉ vùng nền | 0,980 ± 0,009 | 0,983 ± 0,008 |
| Hist H-S + HOG + LogReg C=100 [truyền thống tốt nhất] | Chỉ vùng nền | 0,986 ± 0,005 | 0,988 ± 0,005 |

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| HOG của 4 góc ảnh (128×128) + LogReg C=0,1 | 8100 | 0,968 ± 0,008 | 0,973 ± 0,007 |
| Hist HSV (8,8,8) của 4 góc ảnh + LogReg C=0,1 | 512 | 0,939 ± 0,009 | 0,948 ± 0,008 |

| Đặc trưng + bộ phân lớp | Số chiều | Macro-F1 | Accuracy |
|---|---|---|---|
| Màu Lab trung bình của nền (3 số) + KNN k=5 | 3 | 0,937 ± 0,011 | 0,948 ± 0,011 |
| Màu Lab trung bình của nền (3 số) + LogReg | 3 | 0,618 ± 0,027 | 0,693 ± 0,018 |
| Tỉ lệ diện tích quả chuối trong ảnh (1 số) + KNN k=5 | 1 | 0,619 ± 0,022 | 0,642 ± 0,016 |
| HOG của mask nhị phân (chỉ hình bóng) + LogReg C=0,1 | 3780 | 0,857 ± 0,017 | 0,861 ± 0,014 |

| Lớp | Nền: L trung bình ± std | Nền: a | Nền: b | Tỉ lệ diện tích quả chuối |
|---|---|---|---|---|
| Bangla Kola | 201,1 ± 13,7 | 128,1 ± 1,1 | 128,3 ± 2,1 | 0,136 |
| Champa Kola | 190,2 ± 11,7 | 127,2 ± 0,4 | 123,6 ± 0,9 | 0,164 |
| Sabri Kola | 207,4 ± 12,4 | 127,3 ± 1,0 | 119,9 ± 10,4 | 0,054 |
| Sagor Kola | 196,1 ± 20,1 | 128,2 ± 0,7 | 128,2 ± 2,3 | 0,220 |

| Mô hình (học trên ảnh đầy đủ) | Toàn ảnh | Chỉ vùng chuối | Chỉ vùng nền |
|---|---|---|---|
| a_vgg16_lr1e-3 | 0,990 | 0,952 | 0,792 |
| b_mobilenetv3_lr1e-4 | 1,000 | 0,927 | 0,683 |
| b_resnet50 | 0,998 | 0,926 | 0,725 |

![S2 — Toàn ảnh, chỉ chuối, chỉ nền](../results/figures/S2_background_bias.png)

![S2 — Ba phiên bản ảnh](../results/figures/S2_versions.png)

*Kết luận:* **dữ liệu có thiên lệch theo bối cảnh chụp.** Chỉ dùng 4 góc ảnh 64×64, nơi trung bình chỉ 2,5% điểm ảnh thuộc vùng chuối, HOG vẫn đạt 0,968 và histogram màu 0,939. Chỉ 3 con số là màu trung bình của nền đã cho KNN đạt 0,937. CNN học trên ảnh đầy đủ, khi đoán trên ảnh đã xóa quả chuối, vẫn đạt 0,683–0,792 (đoán mò ≈ 0,25). Mỗi lớp được chụp trong điều kiện riêng: nền Sabri Kola xanh hơn (b = 119,9 so với 123,6–128,3 ở các lớp khác), quả Sabri chiếm 5,4% diện tích ảnh so với 22,0% ở Sagor. Khi chỉ nhìn quả chuối, cấu hình truyền thống tốt nhất còn 0,958 và CNN còn 0,926–0,952; các con số này gần với khả năng phân biệt giống bằng chính quả chuối hơn.

### 5.11 Đánh giá cuối trên tập test

Tập test (371 ảnh) chỉ dùng ở bước này, cho 9 cấu hình đã chốt. Nhóm truyền thống học lại trên train+val; CNN học trên train. Khoảng tin cậy 95% tính bằng bootstrap (2.000 lần lấy mẫu lại).

| Nhóm | Mô hình | Macro-F1 test | KTC 95% (bootstrap) | Accuracy test | Số ảnh sai / 371 | Recall thấp nhất | Macro-F1 lúc chọn (5-fold CV hoặc val) |
|---|---|---|---|---|---|---|---|
| Truyền thống (màu) | Hist HSV (8,8,8) + LogReg C=0,1 | 0,994 | [0,984; 1,000] | 0,995 | 2 | 0,985 (Bangla) | 0,991 ± 0,005 |
| Truyền thống (cạnh) | Cạnh Roberts + KNN k=1, Manhattan, ưu tiên điểm gần | 0,964 | [0,944; 0,982] | 0,968 | 12 | 0,885 (Sagor) | 0,955 ± 0,015 |
| Truyền thống (HOG) | HOG 64×128 + LogReg C=0,1 | 0,990 | [0,977; 1,000] | 0,992 | 3 | 0,985 (Bangla) | 0,986 ± 0,007 |
| Truyền thống (ghép) | Hist H-S + HOG + LogReg C=100, chuẩn hóa từng phần | 1,000 | [1,000; 1,000] | 1,000 | 0 | 1,000 (Bangla) | 0,996 ± 0,003 |
| CNN hướng A | a_lenet (biến thể: a_lenet) | 0,978 | [0,960; 0,993] | 0,981 | 7 | 0,955 (Bangla) | 0,973 |
| CNN hướng A | a_dropout (biến thể: a_dropout_aug) | 0,942 | [0,917; 0,965] | 0,943 | 21 | 0,872 (Sagor) | 0,948 |
| CNN hướng A | a_vgg16 (biến thể: a_vgg16_lr1e-3) | 0,986 | [0,971; 0,997] | 0,989 | 4 | 0,970 (Bangla) | 0,990 |
| CNN hướng B | b_mobilenetv3 (biến thể: b_mobilenetv3_lr1e-4) | 1,000 | [1,000; 1,000] | 1,000 | 0 | 1,000 (Bangla) | 1,000 |
| CNN hướng B | b_resnet50 (biến thể: b_resnet50) | 0,997 | [0,988; 1,000] | 0,997 | 1 | 0,987 (Sabri) | 0,998 |

![Macro-F1 trên test kèm khoảng tin cậy](../results/figures/final_test_macro_f1.png)

**Mô hình tốt nhất của từng nhóm (chọn theo CV/val, kết quả trên test):**

- Truyền thống: **Hist H-S + HOG + Logistic Regression** (C = 100, chuẩn hóa từng phần): macro-F1 1,000, đúng 371/371 ảnh.
- CNN hướng A: **`a_vgg16`** (VGG16 đóng băng, lr 1e-3): macro-F1 0,986, sai 4 ảnh.
- CNN hướng B: **`b_mobilenetv3`** (MobileNetV3-Small fine-tune, lr 1e-4): macro-F1 1,000, đúng 371/371 ảnh.

Hai cấu hình đúng toàn bộ tập test không phân biệt được với nhau. Khoảng tin cậy của các cấu hình từ 0,986 trở lên chồng lên nhau; chỉ kết luận được rằng `a_dropout` (0,942), cạnh Roberts + KNN (0,964) và `a_lenet` (0,978) kém hơn nhóm dẫn đầu. Với 0/371 ảnh sai, cận trên 95% của tỉ lệ lỗi thật xấp xỉ 3/371 ≈ 0,8% (quy tắc ba). Điểm trên test bằng hoặc cao hơn điểm lúc chọn ở 6/9 cấu hình; 3 cấu hình còn lại giảm trong khoảng 1–2 ảnh. Không thấy dấu hiệu quá khớp vào tập chọn siêu tham số.

**Phân tích lỗi.** Gộp trên 9 mô hình, có 41 ảnh test bị ít nhất một mô hình đoán sai.

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
| Bangla Kola | 67 | 9 | 0,127 | 0,142 |
| Champa Kola | 149 | 7 | 0,166 | 0,153 |
| Sabri Kola | 77 | 2 | 0,054 | 0,099 |
| Sagor Kola | 78 | 23 | 0,244 | 0,181 |

![Ảnh test bị đoán sai](../results/figures/final_errors_all.png)

Lỗi tập trung ở Sagor Kola: 23/78 ảnh bị ít nhất một mô hình đoán sai, chủ yếu bị nhầm thành Champa Kola. Ảnh Sagor bị sai có quả nhỏ hơn trung bình của lớp (diện tích 0,181 so với 0,244). Ảnh bị nhiều mô hình đoán sai nhất (`Sabri_Kola_5.jpg`, 6/9 mô hình) là một quả Sabri to trên nền trắng, không giống ảnh Sabri điển hình (quả nhỏ, nền xanh nhạt). Cả hai quan sát khớp với thiên lệch theo kích thước quả và nền đã đo ở S2.

### 5.12 Demo

`app/app.py` (Gradio): tải ảnh lên, chọn mô hình (`b_mobilenetv3`, `a_lenet` hoặc cấu hình truyền thống tốt nhất), nhận giống dự đoán cùng xác suất 4 lớp. Thông tin về giống trong `app/variety_info.json` chỉ ghi những gì có nguồn dẫn; chỗ chưa có nguồn để "TODO".

![Giao diện demo](../results/figures/demo_screenshot.png)

## 6. Đánh giá ưu điểm và hạn chế

### 6.1 Ưu điểm

- So sánh đầy đủ các phương pháp đã học trên cùng một quy trình đánh giá: 30 thí nghiệm, 718 dòng kết quả, mỗi nhận định dẫn tới một bảng hoặc hình.
- Quy trình đánh giá chặt: chia tập cố định, test dùng một lần, chọn siêu tham số bằng cross-validation, báo cáo độ lệch chuẩn và khoảng tin cậy, chỉ số chính là macro-F1.
- Ghi nhận cả các kết quả trái dự đoán và thử giải thích (D1 với HOG, M5, P1, C2, C4, N3, N8).
- Phát hiện và đo được thiên lệch theo bối cảnh chụp của bộ dữ liệu (S2), một điều mà chỉ nhìn con số accuracy sẽ không thấy.
- Có hai mô hình đúng 371/371 ảnh test, trong đó một mô hình chỉ 11,7 MB và dự đoán 12,5 ms mỗi ảnh trên CPU.
- Chạy lại được toàn bộ bằng một lệnh; seed cố định; checkpoint và `--resume` cho huấn luyện dài.

### 6.2 Hạn chế

- **Thiên lệch theo bối cảnh chụp (hạn chế lớn nhất).** Chỉ nhìn nền vẫn phân loại được với macro-F1 0,93–0,99 (S2). Con số 1,000 trên test là độ chính xác trên bộ dữ liệu này, không phải độ chính xác nhận diện giống chuối trong điều kiện bất kỳ. Với ảnh chụp ở bối cảnh khác, kết quả có thể thấp hơn nhiều; đồ án chưa có dữ liệu như vậy để đo.
- **Ảnh gần trùng:** 42 cặp ảnh gần như y hệt (Hamming ≤ 2), 7 ảnh test có bản gần trùng trong train. Đo được ảnh hưởng không đáng kể, nhưng pHash không bắt được cùng một quả chụp ở tư thế khác, nên con số này là cận dưới.
- Tập test và val nhỏ (371 ảnh); các mô hình tốt nhất đều trên 0,99 nên không phân biệt được bằng thống kê.
- CNN không dùng cross-validation (do giới hạn CPU); nhiễu giữa các lần huấn luyện mới ước lượng bằng 3 seed cho một mô hình (N3).
- Phân đoạn chưa có nhãn mask nên chưa tính được IoU và Dice.
- Chưa làm phần mở rộng tùy chọn (sliding window + NMS) vì dữ liệu không có nhãn vị trí.
- `app/variety_info.json` còn nhiều mục "TODO" do chưa tìm được nguồn tin cậy về đặc điểm từng giống.

### 6.3 Hướng phát triển

- Thu thêm ảnh của cả 4 giống trong cùng một bối cảnh chụp, hoặc một tập test chụp ở bối cảnh khác, để đo đúng khả năng nhận diện giống.
- Huấn luyện trên ảnh đã tách nền, hoặc tăng cường bằng cách thay nền ngẫu nhiên, để mô hình buộc phải dựa vào quả chuối.
- Gán nhãn mask cho khoảng 20 ảnh để tính IoU/Dice cho phần phân đoạn.

## 7. Tài liệu tham khảo

1. Ferdaus, M. H., Prito, R. H., Rasel, A. A. S., Ahmed, M., Saykot, M. J. H., Shanta, S. S., Akter, S., Das, A. C., Islam, M. M., Hasan, M., Ali, S. (2024). *BananaImageBD: An Extensive Image Dataset of Common Bangladeshi Banana Varieties with Different Ripeness Levels*. Mendeley Data, Version 2. DOI: 10.17632/ptfscwtnyz.2. Giấy phép CC BY 4.0.
2. Bài giảng môn CS231 — Nhập môn Thị giác máy tính: ảnh số, đặc trưng màu, phân loại ảnh, KNN và bộ phân lớp tuyến tính, đặc trưng cạnh, HOG, CNN, phân đoạn ảnh, xác định vị trí và phát hiện đối tượng.
3. Swain, M. J., Ballard, D. H. (1991). Color Indexing. *International Journal of Computer Vision*, 7(1), 11–32.
4. Stricker, M., Orengo, M. (1995). Similarity of Color Images. *Proc. SPIE Storage and Retrieval for Image and Video Databases III*.
5. Pass, G., Zabih, R., Miller, J. (1996). Comparing Images Using Color Coherence Vectors. *Proc. ACM Multimedia*.
6. Dalal, N., Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
7. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Trans. Systems, Man, and Cybernetics*, 9(1), 62–66.
8. Cover, T., Hart, P. (1967). Nearest Neighbor Pattern Classification. *IEEE Trans. Information Theory*, 13(1), 21–27.
9. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11).
10. Simonyan, K., Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
11. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
12. Howard, A. et al. (2019). Searching for MobileNetV3. *ICCV*.
13. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15.
14. Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*.
15. Telea, A. (2004). An Image Inpainting Technique Based on the Fast Marching Method. *Journal of Graphics Tools*, 9(1).
16. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12. Tài liệu TensorFlow/Keras, OpenCV, scikit-image.
