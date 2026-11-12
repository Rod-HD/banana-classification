<!-- File mẫu của báo cáo. Sửa nội dung ở đây rồi chạy: python -m src.build_report (sinh report/report.md và report/report.docx). Các chỗ {{T:...}} được thay bằng bảng số liệu trong results/. -->

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

{{T:results/hand_calc/table_cm_a_vgg16.md}}

Với lớp Bangla Kola: TP = 65 (đường chéo); FP = tổng cột Bangla trừ TP = 65 + 0 + 2 + 0 − 65 = 2 (hai ảnh Sabri bị đoán thành Bangla); FN = tổng hàng Bangla trừ TP = 65 + 0 + 0 + 2 − 65 = 2 (hai ảnh Bangla bị đoán thành Sagor). Precision = 65 / (65 + 2) = 0,970; Recall = 65 / (65 + 2) = 0,970; F1 = 0,970. Kết quả khớp với `classification_report` trong `results/final/a_vgg16/`.

**Ví dụ tính tay 2 — khoảng cách giữa hai vector đặc trưng.** Vector moment màu 9 chiều (HSV đã chia về [0, 1]) của ảnh x = `Bangla_Kola_1.jpg` và y = `Champa_Kola_1.jpg` (`results/hand_calc/`):

{{T:results/hand_calc/table_moments.md}}

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

{{T:results/D1/table.md}}

![D1 — Accuracy theo ba cách chia](../results/figures/D1_leakage.png)

**Kết luận.** Khớp dự đoán với đặc trưng màu: histogram HSV gần như không đổi khi lật hay xoay, nên bản sao trong train là "đáp án có sẵn". Accuracy bị đẩy lên 0,999, thổi phồng +0,021 (KNN k = 1) và +0,009 (LogReg). **Trái dự đoán với HOG:** mức thổi phồng ≈ 0 (+0,005 ± 0,004 và +0,000 ± 0,004), vì HOG mã hóa hướng cạnh theo vị trí nên bản lật hay xoay có vector khác hẳn ảnh gốc. Rò rỉ chỉ gây hại khi đặc trưng bất biến với phép tăng cường.

### 5.5 Thí nghiệm đặc trưng màu: M1–M6

Mọi số liệu dưới đây là macro-F1, 5-fold CV trên train+val; phép biến đổi chỉ áp lên fold giữ lại.

**M1 — Histogram bỏ thông tin vị trí.** *Dự đoán:* xáo trộn ảnh chấm thành lưới 4×4 thì histogram giữ nguyên, HOG giảm.

{{T:results/M1/table.md}}

![M1 — Xáo trộn lưới 4×4](../results/figures/M1_shuffle.png)

*Kết luận:* khớp. Histogram và moment màu không đổi đúng đến chữ số cuối (độ sụt 0,000); HOG và vector cạnh rơi về mức đoán mò (0,20–0,29); CCV sụt 0,05–0,08 vì có thông tin liên kết không gian.

**M2 — HSV tách độ sáng khỏi sắc màu.** *Dự đoán:* khi nhân kênh V với 0,6 hoặc 1,4, histogram RGB sụt nhiều nhất, HSV ít hơn, H-S bền nhất.

{{T:results/M2/table.md}}

*Kết luận:* khớp một phần. Chỉ histogram H-S (bỏ hẳn kênh V) là bền (sụt ≤ 0,027). Histogram HSV đủ 3 kênh vẫn sụt 0,33–0,67 vì kênh V vẫn nằm trong đặc trưng: tách V ra thành một trục riêng thôi chưa đủ, phải bỏ nó khỏi vector.

**M3 — Số bin.** *Dự đoán:* ít bin thì mất khả năng phân biệt, nhiều bin thì vector thưa và nhạy nhiễu.

{{T:results/M3/table.md}}

![M3 — Quét số bin](../results/figures/M3_bins.png)

*Kết luận:* vế "ít bin" khớp (4 bin mỗi kênh: 0,914–0,936). Vế "nhiều bin" chỉ thấy ở KNN trên histogram 3 chiều: 16³ = 4.096 chiều, 87% phần tử bằng 0, KNN giảm còn 0,917, trong khi LogReg vẫn 0,992. Histogram RGB từng kênh từ 8 đến 256 bin cho kết quả ngang nhau (chênh nhỏ hơn độ lệch chuẩn).

**M4 — Moment màu gọn nhưng thô.** *Dự đoán:* 9 chiều nhanh nhưng kém histogram.

{{T:results/M4/table.md}}

![M4 — Các đặc trưng màu](../results/figures/M4_color_features.png)

*Kết luận:* khớp với bộ phân lớp tuyến tính (LogReg 0,908, Linear SVM 0,838). **Trái dự đoán với KNN** (0,972, ngang histogram) và về thời gian: rút moment màu (khoảng 4 ms/ảnh) chậm hơn rút histogram (0,2 ms/ảnh) vì phải tính lũy thừa trên từng điểm ảnh. Giải thích ở P6: 9 con số đủ tách 4 lớp nếu biên quyết định được phép cong.

**M5 — CCV bổ sung thông tin liên kết không gian.** *Dự đoán:* CCV tốt hơn histogram cùng mức lượng tử hóa và nhạy với bố cục.

{{T:results/M5/table.md}}

*Kết luận:* vế "nhạy với bố cục" khớp (xáo trộn làm CCV sụt 0,05–0,08, histogram ≤ 0,003). Vế "tốt hơn histogram" không được xác nhận: với LogReg 0,975 so với 0,968 (chênh nhỏ hơn độ lệch chuẩn), với KNN còn kém hơn. Trên ảnh một quả chuối nền trơn, hầu hết điểm ảnh cùng màu nằm trong vùng liên thông lớn, nên phần "không liên kết" gần như rỗng và mang ít thông tin.

**M6 — Giống chuối khác nhau ở màu hay ở hình dạng?** *Dự đoán:* chuyển ảnh sang xám thì đặc trưng màu mất tác dụng, đặc trưng cạnh và HOG gần như không đổi.

{{T:results/M6/table.md}}

![M6 — Ảnh xám](../results/figures/M6_grayscale.png)

*Kết luận:* khớp. Khi học lại trên ảnh xám, đặc trưng màu sụt 0,04–0,23; cạnh và HOG đổi ≤ 0,005, và HOG trên ảnh xám vẫn đạt 0,986. Bốn giống phân biệt được bằng cả màu (hist HSV 0,991) lẫn hình dạng (HOG 0,986). Mục 5.10 (S2) cho thấy một phần của cả hai con số này đến từ nền ảnh.

### 5.6 Thí nghiệm đặc trưng cạnh và HOG: C1–C5

**C1 — Toán tử cạnh và nhiễu.** *Tính chất:* Roberts dùng mặt nạ 2×2 nên nhạy nhiễu; Prewitt và Sobel có làm trơn; Laplacian là đạo hàm bậc hai nên nhạy nhiễu nhất. *Dự đoán:* thêm nhiễu Gauss thì độ sụt xếp theo thứ tự Prewitt ≈ Sobel < Roberts < Laplacian. Nhóm thêm mức σ = 2 vì từ σ = 10 mọi toán tử đều rơi về mức đoán mò.

{{T:results/C1/table.md}}

{{T:results/C1/table_feature_change.md}}

![C1 — Toán tử cạnh và nhiễu](../results/figures/C1_edge_noise.png)

*Kết luận:* khớp. Ở σ = 5, vector đặc trưng thay đổi 43–46% với Prewitt/Sobel, 100% với Roberts và 216% với Laplacian. Ngoài dự đoán: vector "tổng theo hàng/cột" kém bền với mọi toán tử, sập hoàn toàn từ σ = 10, vì nhiễu cộng thêm gradient ở mọi điểm ảnh nền và tổng theo hàng/cột cộng dồn hết phần đó.

**C2 — HOG chuẩn hóa block bền với độ sáng.** *Dự đoán:* lặp lại M2 với HOG; bản có chuẩn hóa bền hơn bản không chuẩn hóa.

{{T:results/C2/table.md}}

{{T:results/C2/table_saturation.md}}

*Kết luận:* khớp khi làm tối (V×0,6: sụt 0,004–0,009 so với 0,08–0,39). **Trái dự đoán khi làm sáng** (V×1,4: bản chuẩn hóa sụt 0,16–0,22, bản không chuẩn hóa chỉ −0,03 đến +0,06). Nguyên nhân: V×1,4 làm 68,7% điểm ảnh bão hòa ở 255 và 14,1% số cell mất hẳn gradient. Đây không còn là phép đổi độ sáng tuyến tính mà chuẩn hóa khử được. Chuẩn hóa L2 còn khuếch đại các block nền có gradient yếu lên độ dài đơn vị, nên khi nền bão hòa thì vector đã chuẩn hóa đổi nhiều hơn.

**C3 — Tham số HOG.** *Dự đoán:* kích thước ảnh, cell và số bin quyết định mức chi tiết.

{{T:results/C3/table.md}}

*Kết luận:* khớp một phần. Chỉ tỉ lệ kích thước cell so với kích thước ảnh tạo khác biệt đo được: cell 16×16 trên ảnh 64×128 còn 0,959. Với LogReg, 9/12 cấu hình nằm trong khoảng 0,982–0,986 và số bin (6, 9, 12) không tạo khác biệt vượt độ lệch chuẩn. Đồ án giữ cấu hình của bài giảng (64×128, cell 8, 9 bin, 3.780 chiều).

**C4 — Đặc trưng toàn cục và cục bộ khi bị che.** *Dự đoán:* che một hình vuông 25% diện tích thì histogram (toàn cục) sụt nhiều hơn HOG (ghép từ đặc trưng cục bộ).

{{T:results/C4/table.md}}

{{T:results/C4/table_hist_scaling.md}}

![C4 — Che 25%](../results/figures/C4_occlusion.png)

*Kết luận:* với cấu hình mặc định (z-score) thì khớp: Hist RGB/HSV rơi về 0,08–0,22, HOG còn 0,78–0,79. Nhưng phép đo phụ cho thấy **nguyên nhân chính là bước chuẩn hóa z-score**, không phải tính "toàn cục". Hình vuông đen dồn 25% khối lượng histogram vào một bin hầu như luôn bằng 0 trên ảnh huấn luyện; z-score chia cho độ lệch chuẩn rất nhỏ của bin đó nên một chiều chi phối khoảng cách. Dùng Hellinger thì Hist HSV chỉ sụt 0,012.

**C5 — Màu và hình dạng bổ sung cho nhau.** *Dự đoán:* ghép màu + HOG (chuẩn hóa từng phần) tốt hơn từng đặc trưng riêng lẻ.

{{T:results/C5/table.md}}

![C5 — Ghép đặc trưng](../results/figures/C5_fusion.png)

*Kết luận:* khớp với KNN: chuẩn hóa từng phần nâng Hist H-S + HOG từ 0,948 (ghép thẳng) lên 0,977. Sau khi chọn siêu tham số (mục 5.7), cấu hình ghép đạt 0,996 ± 0,003, hơn HOG riêng 0,010 (lớn hơn độ lệch chuẩn) và hơn màu riêng 0,005 (bằng độ lệch chuẩn, chưa đủ cơ sở phân biệt).

### 5.7 Thí nghiệm bộ phân lớp: P1–P6 và cấu hình truyền thống đã chốt

**P1 — Chọn k.** *Dự đoán:* k nhỏ nhạy nhiễu, k lớn kém nhạy với đặc điểm cục bộ; so quy tắc k = số lớp + 1 với k chọn bằng cross-validation.

{{T:results/P1/table.md}}

![P1 — KNN theo k](../results/figures/P1_knn_k.png)

*Kết luận:* vế "k lớn" khớp (k = 31 thấp nhất). Vế "k nhỏ nhạy nhiễu" **không xuất hiện**: k = 1 tốt nhất với cả ba đặc trưng, hơn quy tắc k = 5 từ 0,006 đến 0,020. Các lớp tách nhau rõ và mỗi quả thường có nhiều ảnh gần giống nhau, nên láng giềng gần nhất hầu như luôn cùng lớp; tăng k chỉ kéo thêm điểm ở xa, phần nhiều thuộc lớp đông nhất.

**P2 — Độ đo và thang đo.** *Dự đoán:* KNN phụ thuộc độ đo và chuẩn hóa; sau khi chuẩn hóa về vector đơn vị, Euclidean và cosine cho cùng thứ hạng láng giềng.

{{T:results/P2/table.md}}

{{T:results/P2/table_rank_agreement.md}}

*Kết luận:* khớp. Sau chuẩn hóa đơn vị, 100% ảnh truy vấn có cùng tập 10 láng giềng theo Euclidean và theo cosine (trước chuẩn hóa: 11% với moment, 18% với hist HSV); macro-F1 của hai cách trùng nhau. Manhattan tốt nhất trên HOG (0,978 so với 0,954 của Euclidean).

**P3 — Ưu tiên điểm gần.** *Dự đoán:* `weights='distance'` giúp khi các lớp chồng lấn, nhất là với lớp ít ảnh.

{{T:results/P3/table.md}}

*Kết luận:* chưa đủ cơ sở phân biệt. Macro-F1 đổi ≤ 0,006; recall của Bangla Kola (lớp ít ảnh nhất) đổi từ −0,003 đến +0,026, nhỏ hơn hoặc xấp xỉ độ lệch chuẩn giữa các fold.

**P4 — K-D Tree và Ball Tree.** *Dự đoán:* chỉ tăng tốc, không đổi kết quả, và kém hiệu quả khi nhiều chiều.

{{T:results/P4/table.md}}

{{T:results/P4/table_scaling.md}}

![P4 — Thuật toán tìm láng giềng](../results/figures/P4_knn_algorithms.png)

*Kết luận:* khớp, kèm điều kiện. Kết quả trùng Brute Force trong mọi trường hợp. Với 3.780 chiều, K-D Tree chậm hơn Brute Force khoảng 60 lần. Ngay cả với 9 chiều, khi chỉ có 1.680 ảnh tham chiếu thì cây vẫn chậm hơn; từ 8.400 điểm trở lên K-D Tree mới nhanh hơn (6,1 ms so với 22,3 ms). Cây chỉ có lợi khi **ít chiều và tập tham chiếu đủ lớn**.

**P5 — Thời gian huấn luyện và dự đoán.** *Dự đoán:* KNN không tốn thời gian huấn luyện nhưng dự đoán chậm; mô hình tuyến tính thì ngược lại.

{{T:results/P5/table.md}}

*Kết luận:* khớp. Trên HOG, KNN "huấn luyện" 0,07 s (chỉ lưu dữ liệu) nhưng dự đoán 0,22 ms/ảnh; LogReg huấn luyện 1,2 s nhưng dự đoán chỉ 0,02 ms/ảnh.

**P6 — Biên quyết định tuyến tính.** *Dự đoán:* LogReg kém KNN khi đặc trưng ít chiều, tốt hơn khi nhiều chiều.

{{T:results/P6/table.md}}

![P6 — KNN so với bộ phân lớp tuyến tính](../results/figures/P6_linear_vs_knn.png)

*Kết luận:* khớp. Với 9 chiều, LogReg kém KNN 0,064; với 512 và 3.780 chiều thì hơn 0,027 và 0,039. Trong không gian nhiều chiều, các lớp dễ tách tuyến tính hơn.

**Cấu hình truyền thống đã chốt.** Quét k ∈ {1, 3, 5}, độ đo (Euclidean, Manhattan, Hellinger) cho KNN và tham số C cho LogReg, Linear SVM (`results/select_traditional/table.md`):

{{T:results/select_traditional/table_selected.md}}

### 5.8 CNN hướng A và hướng B: N1–N8

Hai hướng được trình bày riêng (5.8.1, 5.8.2) trước khi so sánh (5.8.3). Số liệu CNN trong mục này chấm trên val (371 ảnh); kết quả trên test ở mục 5.11.

#### 5.8.1 Hướng A — CNN theo bài giảng

**N1 — Số tham số tính được từ kiến trúc.** Lớp tích chập có (F×F×C + 1)×K tham số, lớp kết nối đầy đủ có (N_in + 1)×N_out tham số.

**Ví dụ tính tay 3.** Lớp `conv2` của `a_lenet`: kernel 5×5, đầu vào 6 kênh, 16 bộ lọc → (5×5×6 + 1) × 16 = 151 × 16 = 2.416 tham số. Lớp `fc1`: ảnh 64×64 qua Conv 5×5 còn 60×60, qua MaxPool còn 30×30, qua Conv 5×5 còn 26×26, qua MaxPool còn 13×13; Flatten cho 13 × 13 × 16 = 2.704 giá trị → (2.704 + 1) × 120 = 324.600 tham số.

{{T:results/N1/table.md}}

{{T:results/N1/table_totals.md}}

*Kết luận:* 10/10 lớp khớp đúng `model.summary()`. Lớp Dense đầu tiên chiếm 96% số tham số của `a_lenet` và 99% của `a_dropout`.

Kết quả huấn luyện của hướng A trên val: `a_lenet` 0,973 (ảnh vào 64×64; 32×32 cho 0,954; 128×128 cho 0,956), `a_dropout` 0,940 (0,948 khi có tăng cường ảnh), `a_vgg16` 0,984 với learning rate 1e-4 và 0,990 với 1e-3 (`results/cnn/table_runs.md`).

**N2 — Lượng dữ liệu.** *Tính chất:* CNN học đặc trưng từ dữ liệu nên cần nhiều dữ liệu; transfer learning tận dụng đặc trưng đã học. *Dự đoán:* khi giảm dữ liệu, `a_lenet` sụt nhiều hơn `a_vgg16` và `b_mobilenetv3`.

{{T:results/N2/table.md}}

![N2 — Đường cong học](../results/figures/N2_learning_curve.png)

*Kết luận:* khớp. Từ 25% lên 100% train, `a_lenet` tăng 0,050, hai mô hình pretrained chỉ tăng 0,016–0,018. Với 25% train (432 ảnh), `a_vgg16` (0,967) và `b_mobilenetv3` (0,975) đã ngang hoặc hơn `a_lenet` dùng đủ 1.729 ảnh (0,973).

**N3 — Dropout giảm quá khớp.** *Dự đoán:* `a_dropout` có Dropout có khoảng cách train–val nhỏ hơn và kết quả trên val tốt hơn bản không Dropout.

{{T:results/N3/table_seeds.md}}

![N3 — Dropout](../results/figures/N3_dropout.png)

*Kết luận:* **trái dự đoán**. Qua 3 seed khởi tạo, bản có Dropout(0,5) kém hơn 0,012 (0,941 ± 0,005 so với 0,953 ± 0,006), và khoảng cách loss val − train không nhỏ hơn. Giải thích của nhóm: 99% tham số nằm ở lớp Dense ngay sau Flatten, trước lớp Dropout; lớp này vẫn khớp tập train đến accuracy 0,999 ở cả hai cấu hình. Dropout chỉ làm nhiễu 128 đầu ra của lớp đó nên không chặn được quá khớp ở nơi nó xảy ra, lại làm quá trình tối ưu ồn hơn.

**N5 — Tăng cường ảnh giảm quá khớp.** (Hướng A và B chung một bảng.)

{{T:results/N5/table.md}}

![N5 — a_dropout có và không có tăng cường](../results/figures/N5_augment_a_dropout.png)

*Kết luận:* khớp với mô hình học từ đầu. Với `a_dropout`, khoảng cách loss val − train giảm từ 0,542 xuống 0,054 và macro-F1 tăng 0,940 → 0,948; tăng cường tác động vào đầu vào nên hiệu quả hơn Dropout rõ rệt. Với `b_mobilenetv3` pretrained, quá khớp vốn đã nhỏ nên tăng cường không tăng điểm trên val (0,993 so với 0,997), nhưng làm mô hình bền hơn với phép xoay ±30°: 0,987–0,993 so với 0,958–0,978 (bảng N8).

#### 5.8.2 Hướng B — CNN lớn hơn, fine-tune

**N4 — Đóng băng và fine-tune.** *Tính chất:* đóng băng giữ nguyên đặc trưng đã học; fine-tune thích nghi đặc trưng với dữ liệu mới, nhưng learning rate lớn sẽ phá trọng số đã học. *Dự đoán:* giai đoạn 2 tốt hơn giai đoạn 1; lr 1e-3 kém và kém ổn định hơn lr 1e-5.

{{T:results/N4/table.md}}

![N4 — Giai đoạn 1 và giai đoạn 2 theo learning rate](../results/figures/N4_finetune_lr.png)

*Kết luận:* vế đầu khớp, nhưng mức tăng nhỏ vì giai đoạn 1 đã rất cao: MobileNetV3 0,987 → 0,993–1,000, ResNet50 0,996 → 0,998. Vế learning rate: lr 1e-3 làm quá trình huấn luyện mất ổn định (macro-F1 tụt xuống 0,917 ở epoch 2 và dao động tới cuối, trong khi lr 1e-5 không xuống dưới 0,978). Tuy vậy, nhờ dừng sớm theo val, kết quả cuối không kém. ResNet50 chạy được giai đoạn 2 trên CPU (mở băng khối conv5, khoảng 61 s mỗi epoch, 8 epoch).

**Một chi tiết cài đặt quan trọng.** Lần fine-tune đầu để các lớp BatchNormalization mở băng: accuracy trên train ở epoch 1 chỉ 0,356. Phép đo lặp lại được cho thấy nguyên nhân:

{{T:results/N4/table_batchnorm.md}}

Khi BN ở chế độ huấn luyện, nó dùng thống kê của từng lô thay cho thống kê đã học trên ImageNet. Embedding đổi 82–84%, và đầu phân lớp học ở giai đoạn 1 chỉ còn đoán đúng 4,7–12,5% ảnh. Giữ BN đóng băng khi fine-tune thì vấn đề biến mất.

#### 5.8.3 So sánh hai hướng

**N6 — Mạng lớn hơn, pretrained, tốt hơn nhưng tốn tài nguyên hơn.**

{{T:results/N6/table.md}}

![N6 — Hướng A và hướng B](../results/figures/N6_a_vs_b.png)

*Kết luận:* vế "tốt hơn" khớp: hướng B đạt 0,998–1,000 trên val, hướng A 0,948–0,990. Vế "tốn tài nguyên hơn" chỉ đúng với ResNet50. MobileNetV3-Small có 0,94 triệu tham số, file 11,7 MB, dự đoán 12,5 ms mỗi ảnh; trong khi `a_vgg16` của hướng A có 40,7 triệu tham số, file 370 MB, dự đoán 130 ms mỗi ảnh. Một kiến trúc được thiết kế cho thiết bị di động có thể vừa chính xác hơn vừa nhẹ hơn mô hình theo bài giảng.

**N7 — Feature vector thủ công và embedding.** *Dự đoán:* embedding do mạng học tốt hơn đặc trưng thủ công với cùng bộ phân lớp.

{{T:results/N7/table.md}}

![N7 — Đặc trưng thủ công và embedding](../results/figures/N7_embedding.png)

*Kết luận:* khớp với KNN (embedding MobileNetV3 đã fine-tune: 1,000; đặc trưng thủ công tốt nhất: 0,990). Với LogReg thì ngang nhau (0,996 so với 0,998). Embedding ImageNet **chưa học trên chuối** (VGG16) cho KNN chỉ 0,932, kém histogram HSV (0,982). Embedding chỉ vượt trội sau khi được học trên chính dữ liệu này.

**N8 — CNN có bền với thay đổi hơn đặc trưng thủ công không?**

{{T:results/N8/table.md}}

![N8 — Độ bền theo phép biến đổi](../results/figures/N8_robustness.png)

*Kết luận:* CNN bền hơn hẳn với **nhiễu** (MobileNetV3 0,953 ở σ = 20, histogram và H-S + HOG ≤ 0,101), **xoay** và **ảnh xám**. **Trái dự đoán** với phép che 25%: MobileNetV3 còn 0,659, kém HOG (0,801) và H-S + HOG (0,918). Với xáo trộn 4×4, `a_vgg16` sập về 0,097 vì dùng Flatten, mỗi trọng số gắn với một vị trí cố định; MobileNetV3 dùng GlobalAveragePooling nên giữ được 0,850. Không có phương pháp nào bền với mọi phép biến đổi; mỗi phương pháp bền với đúng loại thay đổi mà cấu trúc của nó bất biến.

### 5.9 Mất cân bằng dữ liệu: I1–I2

**I1 — Accuracy che khuất lớp bị nhận kém.**

{{T:results/I1/table.md}}

*Kết luận:* khớp. Bộ phân lớp "luôn đoán Champa Kola" đạt accuracy 0,402 nhưng macro-F1 chỉ 0,143. KNN k = 31 trên HOG đạt accuracy 0,884 nhưng recall Sagor Kola chỉ 0,552. Ở các mô hình tốt, accuracy và macro-F1 chênh ≤ 0,003. Lớp bị nhận kém nhất thường là Sagor Kola (524 ảnh) chứ không phải lớp ít ảnh nhất (Bangla Kola, 444 ảnh).

**I2 — Các cách xử lý mất cân bằng.** Mọi phép lấy mẫu lại chỉ áp dụng trên phần huấn luyện.

{{T:results/I2/table.md}}

*Kết luận:* ở mô hình yếu (cạnh Sobel + LogReg), các cách xử lý **đổi precision lấy recall** của lớp ít: tăng cường lớp ít nâng recall Bangla 0,822 → 0,857 nhưng hạ precision Bangla 0,849 → 0,776. Ở các mô hình mạnh không đo được tác động (truyền thống tốt nhất: cả bốn cách 0,995–0,996; `b_mobilenetv3`: cả bốn cách 1,000 trên val). Với tỉ lệ mất cân bằng 2,2 lần, mất cân bằng không phải vấn đề chính của bộ dữ liệu này, nên cấu hình cuối không thêm bước xử lý.

### 5.10 Phân đoạn tách nền: S1–S2

**S1 — Các phương pháp phân đoạn.** *Tính chất:* ngưỡng toàn cục chỉ tốt khi nền và vật tách biệt về mức xám; Otsu tự chọn ngưỡng; k-means gom theo màu. Dữ liệu không có nhãn mask, nên nhóm đánh giá định tính bằng hình, kèm hai chỉ số không cần nhãn: số mảnh rời và tỉ lệ lỗ thủng.

{{T:results/S1/table.md}}

![S1 — Các phương pháp phân đoạn](../results/figures/S1_segmentation.png)

*Kết luận:* khớp. Ngưỡng toàn cục và ngưỡng thích nghi tách cả vết thâm trên vỏ thành nền (lỗ thủng 0,085 và 0,282). Ngưỡng thích nghi chỉ bắt được viền và vân. Các cách dựa trên màu (k-means, Otsu trên khoảng cách màu tới nền) cho mask liền nhất, vì nền trơn sáng khác quả chuối về màu rõ hơn về mức xám.

**S2 — Mô hình dựa vào quả chuối hay vào nền?** *Dự đoán:* nếu chỉ dùng vùng nền mà vẫn phân loại tốt thì dữ liệu có thiên lệch theo bối cảnh chụp.

{{T:results/S2/table.md}}

{{T:results/S2/table_corners.md}}

{{T:results/S2/table_extreme.md}}

{{T:results/S2/table_background_stats.md}}

{{T:results/S2/table_cnn.md}}

![S2 — Toàn ảnh, chỉ chuối, chỉ nền](../results/figures/S2_background_bias.png)

![S2 — Ba phiên bản ảnh](../results/figures/S2_versions.png)

*Kết luận:* **dữ liệu có thiên lệch theo bối cảnh chụp.** Chỉ dùng 4 góc ảnh 64×64, nơi trung bình chỉ 2,5% điểm ảnh thuộc vùng chuối, HOG vẫn đạt 0,968 và histogram màu 0,939. Chỉ 3 con số là màu trung bình của nền đã cho KNN đạt 0,937. CNN học trên ảnh đầy đủ, khi đoán trên ảnh đã xóa quả chuối, vẫn đạt 0,683–0,792 (đoán mò ≈ 0,25). Mỗi lớp được chụp trong điều kiện riêng: nền Sabri Kola xanh hơn (b = 119,9 so với 123,6–128,3 ở các lớp khác), quả Sabri chiếm 5,4% diện tích ảnh so với 22,0% ở Sagor. Khi chỉ nhìn quả chuối, cấu hình truyền thống tốt nhất còn 0,958 và CNN còn 0,926–0,952; các con số này gần với khả năng phân biệt giống bằng chính quả chuối hơn.

### 5.11 Đánh giá cuối trên tập test

Tập test (371 ảnh) chỉ dùng ở bước này, cho 9 cấu hình đã chốt. Nhóm truyền thống học lại trên train+val; CNN học trên train. Khoảng tin cậy 95% tính bằng bootstrap (2.000 lần lấy mẫu lại).

{{T:results/final/table.md}}

![Macro-F1 trên test kèm khoảng tin cậy](../results/figures/final_test_macro_f1.png)

**Mô hình tốt nhất của từng nhóm (chọn theo CV/val, kết quả trên test):**

- Truyền thống: **Hist H-S + HOG + Logistic Regression** (C = 100, chuẩn hóa từng phần): macro-F1 1,000, đúng 371/371 ảnh.
- CNN hướng A: **`a_vgg16`** (VGG16 đóng băng, lr 1e-3): macro-F1 0,986, sai 4 ảnh.
- CNN hướng B: **`b_mobilenetv3`** (MobileNetV3-Small fine-tune, lr 1e-4): macro-F1 1,000, đúng 371/371 ảnh.

Hai cấu hình đúng toàn bộ tập test không phân biệt được với nhau. Khoảng tin cậy của các cấu hình từ 0,986 trở lên chồng lên nhau; chỉ kết luận được rằng `a_dropout` (0,942), cạnh Roberts + KNN (0,964) và `a_lenet` (0,978) kém hơn nhóm dẫn đầu. Với 0/371 ảnh sai, cận trên 95% của tỉ lệ lỗi thật xấp xỉ 3/371 ≈ 0,8% (quy tắc ba). Điểm trên test bằng hoặc cao hơn điểm lúc chọn ở 6/9 cấu hình; 3 cấu hình còn lại giảm trong khoảng 1–2 ảnh. Không thấy dấu hiệu quá khớp vào tập chọn siêu tham số.

**Phân tích lỗi.** Gộp trên 9 mô hình, có 41 ảnh test bị ít nhất một mô hình đoán sai.

{{T:results/final/table_error_pairs.md}}

{{T:results/final/table_error_context.md}}

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

1. Ferdaus, M. H., Prito, R. H., Rasel, A. A. S., Ahmed, M., Saykot, M. J. H., Shanta, S. S., Akter, S., Das, A. C., Islam, M. M., Hasan, M., Ali, M. S. (2024). *BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh* (Version 2) [Data set]. Mendeley Data. DOI: 10.17632/ptfscwtnyz.2. Giấy phép CC BY 4.0.
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
