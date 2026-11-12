# Kế hoạch thực nghiệm — Nhận diện giống chuối từ ảnh (CS231)

Tài liệu này là đặc tả của đồ án: phạm vi, phương pháp, các thí nghiệm phải chạy và sản phẩm phải nộp. Khi thực tế thay đổi so với kế hoạch thì sửa tài liệu này trước rồi mới sửa code.

## 1. Bài toán

- **Input:** một ảnh màu chụp quả hoặc nải chuối.
- **Output:** tên giống chuối (1 trong 4 lớp) kèm xác suất từng lớp.
- **Ngoài phạm vi:** không xác định hàm lượng dinh dưỡng của quả chuối cụ thể, không nhận diện giống ngoài 4 lớp, không phân loại độ chín.

## 2. Dữ liệu

`data/raw/<lớp>/` chứa 2.471 ảnh gốc của bộ BananaImageBD (Mendeley Data, DOI 10.17632/ptfscwtnyz.2, CC BY 4.0). Hai điểm code đọc ảnh phải xử lý: có 93 file đuôi `.JPG` viết hoa; 2.469 ảnh có kích thước 256×256 nhưng 2 ảnh thì không, nên luôn resize trước khi rút đặc trưng.

| Thư mục | Lớp | Số ảnh |
|---|---|---|
| `bangla_kola` | Bangla Kola | 444 |
| `champa_kola` | Champa Kola | 994 |
| `sabri_kola` | Sabri Kola | 509 |
| `sagor_kola` | Sagor Kola | 524 |

Lớp lớn nhất gấp khoảng 2,2 lần lớp nhỏ nhất.

## 3. Sản phẩm phải nộp (theo buổi giới thiệu đồ án)

1. **Báo cáo tổng kết** (pdf hoặc docx) gồm: thông tin thành viên và đánh giá mức độ hoàn thành; lý do và tầm quan trọng của bài toán; phát biểu bài toán (Input, Output); phương pháp giải quyết; kết quả thực nghiệm (dataset, độ đo, lập trình, kết quả); đánh giá ưu điểm và hạn chế; tài liệu tham khảo.
2. **Source code** (link GitHub).
3. **Slide trình bày** trên lớp, cùng các mục như báo cáo.

## 4. Phạm vi phương pháp

**Quyết định đã chốt:** thử *tất cả* phương pháp đã học mà áp dụng được cho bài toán phân loại ảnh, so sánh chúng bằng thực nghiệm và giải thích kết quả bằng tính chất của từng phương pháp.

| Bài học | Kỹ thuật dùng trong đồ án |
|---|---|
| Ảnh số | Không gian màu RGB, HSV, Lab, ảnh xám. Độ đo khoảng cách (Euclidean, Manhattan) và độ đo tương đồng (cosine). Chuẩn hóa vector về vector đơn vị trước khi đo. |
| Đặc trưng màu | Histogram RGB từng kênh; histogram HSV 3 chiều (8,8,8); moment màu (mean, std, skewness trên HSV, 9 chiều); màu chủ đạo bằng k-means (k=3, sắp theo tỉ lệ giảm dần); Color Coherence Vector (CCV). |
| Phân loại ảnh | Hai giai đoạn huấn luyện và phân lớp; hướng truyền thống (rút đặc trưng + bộ phân lớp) và hướng học sâu. Độ đo: Accuracy, Precision, Recall, F-score, Confusion Matrix. |
| KNN và bộ phân lớp tuyến tính | KNN: chọn k, độ đo khoảng cách, bình chọn đa số so với ưu tiên điểm gần, Brute Force so với K-D Tree và Ball Tree. Bộ phân lớp tuyến tính: Logistic Regression. |
| Đặc trưng cạnh | Gradient; toán tử Roberts, Prewitt, Sobel, Laplacian. Vector đặc trưng cạnh: tổng độ lớn gradient theo hàng và theo cột của ảnh 256×256 (512 chiều). |
| HOG | Resize, tính gradient, histogram 9 bin trên cell 8×8, block 16×16 chuẩn hóa L2, ghép thành vector. Khoảng cách Hellinger cho đặc trưng dạng histogram. HOG + SVM. |
| CNN | Hai hướng tách riêng, xem mục 6.5. |
| Phân đoạn ảnh | Ngưỡng toàn cục, Otsu, ngưỡng thích nghi; gom cụm k-means. Dùng để tách quả chuối khỏi nền trước khi rút đặc trưng. |
| Xác định vị trí, phát hiện đối tượng | Không phải bài toán phân loại: dữ liệu không có nhãn bounding box nên không tính được IoU hay mAP. Phần mở rộng tùy chọn: sliding window + bộ phân lớp tốt nhất + NMS, chỉ minh họa định tính. |

Linear SVM chỉ được nhắc trong môn như bộ phân lớp đi kèm đặc trưng (HOG + SVM), nên dùng như một bộ phân lớp bổ sung và ghi rõ điều này trong báo cáo.

Thư viện: OpenCV, scikit-image, scikit-learn cho phần truyền thống; TensorFlow/Keras cho CNN (cùng thư viện với ví dụ trong bài giảng).

## 5. Quy trình đánh giá

- Chia một lần, cố định: stratified 70/15/15 (train/val/test), `seed=42`, lưu `data/splits/{train,val,test}.csv`.
- **Tập test chỉ dùng một lần ở cuối** cho các cấu hình đã chốt.
- Ở mục 6, cụm "ảnh test" trong các thí nghiệm kiểm chứng (D1, M1, M2, M6, C1, C2, C4, N8) nghĩa là phần ảnh giữ lại để chấm trong giai đoạn phát triển: fold giữ lại của cross-validation (nhóm truyền thống) hoặc tập val (CNN). D1 chạy trong tập train+val. Tập test chính thức không dùng cho các thí nghiệm này (cập nhật ở bước 1 của nhật ký).
- Nhóm truyền thống: gộp train+val, chọn siêu tham số bằng 5-fold stratified cross-validation, báo cáo trung bình ± độ lệch chuẩn. Chênh lệch nhỏ hơn độ lệch chuẩn thì kết luận là "chưa đủ cơ sở phân biệt", không tuyên bố hơn kém.
- Nhóm CNN: huấn luyện trên train, chọn epoch và siêu tham số theo val.
- Chỉ số chính: **macro-F1** (vì dữ liệu mất cân bằng). Luôn báo cáo kèm Accuracy, Precision và Recall từng lớp, Confusion Matrix dạng số lượng và dạng chuẩn hóa theo hàng.
- Kiểm tra ảnh gần trùng giữa các tập (perceptual hash) và báo cáo số lượng; nếu có thì nêu trong phần hạn chế.
- Ghi thời gian rút đặc trưng, huấn luyện và dự đoán của mỗi cấu hình.

## 6. Giả thuyết và thí nghiệm kiểm chứng

Mỗi nhận định trong báo cáo đi theo ba bước: tính chất của phương pháp (lý thuyết) → dự đoán → thí nghiệm kiểm chứng. Kết quả trái dự đoán vẫn phải báo cáo đúng như đo được và thử giải thích.

### 6.1 Dữ liệu

| # | Tính chất | Thí nghiệm |
|---|---|---|
| D1 | Tăng cường ảnh *trước khi* chia tập làm bản sao của cùng một ảnh rơi vào cả train lẫn test, kết quả bị thổi phồng. Đây là lý do không dùng bản "Augmented" có sẵn của bộ dữ liệu. | Tự tạo bản tăng cường (lật, xoay) của ảnh gốc. So hai cách với cùng một mô hình: (a) tăng cường rồi mới chia ngẫu nhiên, (b) chia trước rồi chỉ tăng cường tập train. Chênh lệch độ chính xác trên test là mức thổi phồng do rò rỉ. |

### 6.2 Đặc trưng màu

| # | Tính chất | Thí nghiệm |
|---|---|---|
| M1 | Histogram chỉ đếm tần suất màu, bỏ thông tin vị trí. | Cắt ảnh test thành lưới 4×4 rồi xáo trộn các ô. Đặc trưng histogram không đổi nên độ chính xác phải giữ nguyên; HOG và CNN phải giảm. |
| M2 | HSV tách độ sáng (V) khỏi sắc màu (H, S). | Nhân kênh V của ảnh test với 0,6 và 1,4. So độ sụt của histogram RGB, histogram HSV và histogram chỉ dùng H-S. |
| M3 | Ít bin thì mất khả năng phân biệt, nhiều bin thì vector thưa và nhạy nhiễu. | Quét số bin: 4, 8, 16, 32, 64, 256 mỗi kênh. |
| M4 | Moment màu rất gọn (9 chiều) nhưng mô tả thô. | So moment với histogram về độ chính xác và thời gian. |
| M5 | CCV bổ sung thông tin liên kết không gian cho histogram. | So CCV với histogram cùng mức lượng tử hóa; lặp lại phép xáo trộn M1 để xem CCV có nhạy với bố cục không. |
| M6 | Giống chuối khác nhau ở màu hay ở hình dạng? | Chuyển ảnh sang xám: đặc trưng màu mất tác dụng, đặc trưng cạnh và HOG gần như không đổi. So mức sụt. |

### 6.3 Đặc trưng cạnh và HOG

| # | Tính chất | Thí nghiệm |
|---|---|---|
| C1 | Roberts dùng mặt nạ 2×2 nên nhạy nhiễu; Prewitt và Sobel có làm trơn; Laplacian là đạo hàm bậc hai nên nhạy nhiễu nhất. | Rút vector cạnh với từng toán tử; thêm nhiễu Gauss (σ = 5, 10, 20) vào ảnh test và so độ sụt. |
| C2 | HOG mô tả hình dạng cục bộ và chuẩn hóa theo block nên bền với thay đổi độ sáng. | Lặp lại M2 với HOG; so HOG có và không chuẩn hóa block. |
| C3 | Tham số HOG quyết định mức chi tiết. | Quét kích thước ảnh (64×128 như bài giảng, 128×128), cell (8, 16), số bin (6, 9, 12). Ghi số chiều của vector. |
| C4 | Histogram màu là đặc trưng toàn cục, HOG ghép từ đặc trưng cục bộ. | Che một vùng vuông 25% diện tích ảnh test (occlusion); so độ sụt của hai loại. |
| C5 | Màu và hình dạng bổ sung cho nhau. | Ghép màu + HOG (chuẩn hóa từng phần trước khi ghép); so với từng đặc trưng riêng lẻ. |

### 6.4 Bộ phân lớp

| # | Tính chất | Thí nghiệm |
|---|---|---|
| P1 | k nhỏ thì nhạy nhiễu, k lớn thì kém nhạy với đặc điểm cục bộ. | Vẽ đường cong theo k = 1, 3, 5, …, 31. So quy tắc k = số lớp + 1 với k chọn bằng cross-validation. |
| P2 | KNN phụ thuộc độ đo và thang đo của đặc trưng. | Euclidean, Manhattan, cosine, Hellinger; có và không chuẩn hóa về vector đơn vị. Kiểm chứng: sau khi chuẩn hóa đơn vị, Euclidean và cosine cho cùng thứ hạng láng giềng. |
| P3 | Ưu tiên điểm gần giúp khi các lớp chồng lấn. | `weights='uniform'` so với `'distance'`; xem riêng recall của lớp ít ảnh nhất. |
| P4 | K-D Tree và Ball Tree chỉ tăng tốc, không đổi kết quả, và kém hiệu quả khi số chiều lớn. | Đo thời gian dự đoán của brute, kd_tree, ball_tree trên đặc trưng 9 chiều và trên đặc trưng hàng nghìn chiều; kiểm tra kết quả trùng nhau. |
| P5 | KNN không tốn thời gian huấn luyện nhưng dự đoán chậm; mô hình tuyến tính thì ngược lại. | Bảng thời gian huấn luyện và dự đoán của KNN, Logistic Regression, Linear SVM. |
| P6 | Bộ phân lớp tuyến tính chỉ tạo biên quyết định tuyến tính. | So Logistic Regression với KNN trên cùng đặc trưng, ở đặc trưng ít chiều (moment) và nhiều chiều (HOG). |

### 6.5 CNN: hai hướng tách riêng

**Hướng A — CNN theo bài giảng**

| Mô hình | Mô tả |
|---|---|
| `a_lenet` | Kiểu LeNet-5, huấn luyện từ đầu: Conv(6, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(120) → Dense(84) → Softmax(4). |
| `a_dropout` | Conv(32, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Flatten → Dense(128) → Dropout(0,5) → Dense(64) → Softmax(4), đầu vào 128×128×3, huấn luyện từ đầu. |
| `a_vgg16` | Transfer learning đúng ví dụ bài giảng: VGG16 ImageNet, đóng băng phần tích chập, thêm Flatten → Dense(1024) → Dense(256) → Softmax(4). |

**Hướng B — CNN lớn hơn, fine-tune**

| Mô hình | Mô tả |
|---|---|
| `b_mobilenetv3` | MobileNetV3-Small ImageNet, 224×224. Giai đoạn 1: đóng băng thân, huấn luyện đầu phân lớp. Giai đoạn 2: mở băng, fine-tune toàn mạng với learning rate nhỏ, có tăng cường ảnh. |
| `b_resnet50` | ResNet50 ImageNet (kiến trúc đã gặp ở bài xác định vị trí). Giai đoạn 1 như trên là bắt buộc. Giai đoạn 2 mở băng khối tích chập cuối nếu thời gian chạy trên CPU cho phép; nếu không chạy thì ghi rõ lý do kèm thời gian đo được của 1 epoch. |

Hướng B dùng kiến trúc ngoài ví dụ của bài giảng nhưng vẫn là khái niệm đã học (transfer learning: tái sử dụng kiến trúc và trọng số). Báo cáo phải ghi rõ điều này và trình bày hai hướng trong hai mục riêng trước khi so sánh.

Cấu hình chung: Adam, cross-entropy, lưu đồ thị loss và accuracy theo epoch, dừng sớm theo macro-F1 trên val.

| # | Tính chất | Thí nghiệm |
|---|---|---|
| N1 | Số tham số tính được từ kiến trúc: lớp tích chập (F×F×C+1)×K, lớp kết nối đầy đủ (N_in+1)×N_out. | Tính tay số tham số từng lớp của `a_lenet` và `a_dropout`, đối chiếu với `model.summary()`. Ghi tổng số tham số của cả năm mô hình. |
| N2 | CNN học đặc trưng từ dữ liệu nên cần nhiều dữ liệu; transfer learning tận dụng đặc trưng đã học. | Huấn luyện `a_lenet`, `a_vgg16`, `b_mobilenetv3` với 25%, 50%, 100% tập train; vẽ đường cong học. |
| N3 | Dropout giảm quá khớp. | `a_dropout` có và không có Dropout; so khoảng cách giữa đường train và đường val. |
| N4 | Đóng băng giữ nguyên đặc trưng đã học; fine-tune thích nghi đặc trưng với dữ liệu mới nhưng learning rate lớn sẽ phá trọng số đã học. | Hướng B: so giai đoạn 1 với giai đoạn 2; ở giai đoạn 2 so learning rate 1e-3 với 1e-5. |
| N5 | Tăng cường ảnh giảm quá khớp khi dữ liệu ít. | `a_dropout` và `b_mobilenetv3` có và không có tăng cường ảnh. |
| N6 | Mạng lớn hơn và pretrained cho kết quả tốt hơn nhưng tốn tài nguyên hơn. | Bảng so sánh hướng A với hướng B: macro-F1, Accuracy, số tham số, dung lượng file mô hình, thời gian huấn luyện, thời gian dự đoán mỗi ảnh. |
| N7 | Feature vector do người thiết kế, embedding vector do mạng học được. | Lấy embedding ở lớp áp chót của `a_vgg16` và `b_mobilenetv3`, đưa vào KNN và Logistic Regression; so với cùng bộ phân lớp trên đặc trưng thủ công. |
| N8 | CNN có bền với thay đổi hơn đặc trưng thủ công không? | Chạy lại các phép biến đổi M1, M2, M6, C1, C4 và xoay ±15°, ±30° cho mô hình tốt nhất của hướng A và của hướng B. |

### 6.6 Mất cân bằng dữ liệu

| # | Tính chất | Thí nghiệm |
|---|---|---|
| I1 | Accuracy có thể cao trong khi lớp ít ảnh bị nhận kém. | So Accuracy với macro-F1 và recall từng lớp ở mô hình chưa xử lý. |
| I2 | Mỗi cách xử lý tác động khác nhau lên precision và recall của lớp thiểu số. | Trên mô hình tốt nhất của nhóm truyền thống, của hướng A và của hướng B: không xử lý, class weight, oversampling, tăng cường ảnh cho lớp thiểu số. Mọi phép lấy mẫu lại chỉ áp dụng trên phần huấn luyện. |

### 6.7 Phân đoạn tách nền

| # | Tính chất | Thí nghiệm |
|---|---|---|
| S1 | Ngưỡng toàn cục chỉ tốt khi nền và vật tách biệt về mức xám; Otsu tự chọn ngưỡng; k-means gom theo màu. | Áp dụng ngưỡng toàn cục, Otsu, ngưỡng thích nghi, k-means trên một mẫu ảnh mỗi lớp. Đánh giá định tính bằng hình. Nếu nhóm tự gán nhãn mask cho khoảng 20 ảnh thì tính thêm IoU và Dice. |
| S2 | Mô hình có thể dựa vào nền thay vì quả chuối. | Rút đặc trưng tốt nhất trên ba phiên bản: toàn ảnh, chỉ vùng chuối, chỉ vùng nền. Nếu "chỉ nền" vẫn cho độ chính xác cao thì dữ liệu có thiên lệch theo bối cảnh chụp và phải nêu trong phần hạn chế. |

## 7. Nhật ký thực nghiệm (bắt buộc)

File `docs/nhat-ky-thuc-nghiem.md` ghi lại toàn bộ quá trình. Sau mỗi bước hoàn chỉnh, thêm một mục theo đúng mẫu ở đầu file, gồm: mục tiêu; đã làm gì (file code, lệnh chạy); lý do chọn cách này; phương án khác đã cân nhắc và vì sao không chọn; bằng chứng; kết quả và kết luận; ảnh hưởng tới bước sau.

Quy tắc ghi:

- Mỗi lý do chọn hoặc không chọn phải gắn một trong hai nhãn. **[đo]**: có số liệu so sánh, kèm đường dẫn tới file trong `results/`. **[lập luận]**: dựa trên tính chất phương pháp hoặc giới hạn tài nguyên, chưa đo.
- Ưu tiên **[đo]**. Phương án nào kiểm chứng được trong vài phút thì phải chạy để có số, không dừng ở lập luận.
- So sánh hiệu quả giữa các cách phải có bảng số cụ thể (chỉ số, độ lệch chuẩn nếu có, thời gian), không viết chung chung "tốt hơn".
- Ghi cả những gì thất bại hoặc trái dự đoán.
- Cuối file có **Bảng quyết định** tổng hợp: quyết định, phương án thay thế, bằng chứng, nhãn.

Phần "Phương pháp" và "Kết quả thực nghiệm" của báo cáo được viết lại từ nhật ký này.

## 8. Cấu trúc sản phẩm trong repo

- `src/`: code Python chạy bằng dòng lệnh (`python -m src.<module> --help`): chia dữ liệu, rút đặc trưng, huấn luyện, đánh giá, các phép biến đổi kiểm chứng.
- `results/<thí_nghiệm>/`: `metrics.json`, `classification_report.txt`, hình Confusion Matrix; `results/summary.csv` và `results/summary.md` tổng hợp mọi cấu hình.
- `results/figures/`: hình dùng trong báo cáo (đường cong theo k, đường cong học, biểu đồ độ sụt theo từng phép biến đổi, ảnh phân đoạn, ảnh dự đoán sai).
- `models/`: mô hình dùng cho demo (mỗi file dưới 50 MB).
- `app/app.py`: demo tải ảnh lên và nhận giống chuối dự đoán cùng xác suất 4 lớp. `app/variety_info.json` chỉ ghi thông tin có nguồn dẫn; chỗ chưa có nguồn để `"TODO"`.
- `report/`: báo cáo `report.md` và bản `.docx` theo đúng các mục ở phần 3. Bảng thành viên và mức độ hoàn thành để trống cho nhóm tự điền.
- `slides/`: slide trình bày `.pptx` theo các mục ở phần 3, có ghi chú thuyết trình cho từng slide.
- `docs/nhat-ky-thuc-nghiem.md`: nhật ký theo mục 7.
- `docs/hoi-dap-bao-ve.md`: các câu hỏi có thể gặp khi trình bày, mỗi câu trả lời dẫn tới một bảng hoặc hình cụ thể của đồ án.
- `README.md`: cách cài đặt, cách chạy lại toàn bộ thí nghiệm và demo, bảng kết quả chính.

## 9. Nguyên tắc

- **Không ghi số liệu không có thật.** Mọi con số trong báo cáo, slide, README, nhật ký lấy từ file trong `results/` do code sinh ra. Thí nghiệm chưa chạy thì ghi là chưa có.
- Mỗi nhận định trong báo cáo dẫn tới một bảng hoặc hình cụ thể.
- Cố định seed (`random`, `numpy`, `tensorflow`) và lưu phiên bản thư viện vào `results/environment.txt`.
- Trong báo cáo có ít nhất một ví dụ tính tay: Precision và Recall của một lớp từ Confusion Matrix, số tham số của một lớp CNN, khoảng cách giữa hai vector đặc trưng.

## 10. Lưu ý khi chạy

- Máy chỉ có CPU. Lưu đặc trưng đã rút vào `data/cache/` để không tính lại. Với mô hình đóng băng, tính trước đầu ra phần tích chập một lần rồi chỉ huấn luyện phần đầu phân lớp.
- Lệnh chạy lâu thì chạy nền, ghi log vào `logs/`, lưu checkpoint mỗi epoch vào `checkpoints/` và hỗ trợ `--resume`.
- Commit sau mỗi bước hoàn chỉnh, làm trên branch `dev`. Mỗi commit của một bước thí nghiệm đi kèm mục nhật ký tương ứng.
- Khi tiếp tục công việc sau khi bị gián đoạn, đọc nhật ký để biết đã làm tới đâu.

## 11. Thứ tự thực hiện

1. Chia dữ liệu, kiểm tra ảnh gần trùng, thống kê dữ liệu, thí nghiệm D1.
2. Đặc trưng màu + KNN và Logistic Regression (mục 6.2, 6.4).
3. Đặc trưng cạnh, HOG, ghép đặc trưng (mục 6.3).
4. CNN hướng A.
5. CNN hướng B, rồi so sánh hai hướng (mục 6.5).
6. Mất cân bằng dữ liệu (mục 6.6).
7. Phân đoạn tách nền (mục 6.7).
8. Đánh giá cuối trên test, phân tích lỗi, tổng hợp bảng và hình.
9. Demo, báo cáo, slide, tài liệu hỏi đáp, README.
