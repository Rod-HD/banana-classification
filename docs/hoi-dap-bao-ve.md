# Hỏi đáp khi bảo vệ đồ án

Các câu hỏi có thể gặp khi trình bày (51 câu), kèm câu trả lời ngắn và bảng hoặc hình làm bằng chứng. Mọi số liệu lấy từ `results/`.

## A. Dữ liệu và cách đánh giá

**1. Vì sao không dùng bản "Augmented" (7.413 ảnh) có sẵn của bộ dữ liệu?**
Vì bản tăng cường sinh ra từ cùng ảnh gốc; khi chia ngẫu nhiên, bản sao của ảnh test sẽ nằm trong train. Thí nghiệm D1 đo được mức thổi phồng: +0,021 accuracy với histogram HSV + KNN (k = 1) và +0,009 với histogram HSV + LogReg. → `results/D1/table.md`, `results/figures/D1_leakage.png`.

**2. Vậy sao với HOG thì rò rỉ gần như không ảnh hưởng?**
HOG mã hóa hướng cạnh theo vị trí, nên bản lật hay xoay của một ảnh có vector khác hẳn ảnh gốc và không còn là láng giềng gần của nó. Mức thổi phồng đo được: +0,005 ± 0,004 (KNN) và +0,000 ± 0,004 (LogReg). Rò rỉ chỉ gây hại khi đặc trưng bất biến với phép tăng cường. → `results/D1/table.md`.

**3. Trong dữ liệu có ảnh trùng hoặc gần trùng không?**
Không có file trùng tuyệt đối (md5). Theo pHash có 42 cặp gần như y hệt (Hamming ≤ 2); 7 ảnh test có bản gần trùng trong train. Chia lại sao cho các ảnh gần trùng nằm cùng một fold thì macro-F1 đổi không quá ±0,007, nhỏ hơn độ lệch chuẩn. Trên test, hai mô hình tốt nhất vẫn đúng 100% ở 321 ảnh không có bản gần trùng. → `results/data/near_duplicates.json`, `results/D1/table_near_duplicates.md`, `results/final/table_near_duplicates.md`.

**4. Vì sao dùng macro-F1 mà không dùng accuracy?**
Vì dữ liệu mất cân bằng (Champa Kola 994 ảnh, Bangla Kola 444 ảnh). Bộ phân lớp luôn đoán Champa Kola có accuracy 0,402 nhưng macro-F1 chỉ 0,143; KNN k = 31 trên HOG có accuracy 0,884 nhưng recall Sagor Kola chỉ 0,552. → `results/I1/table.md`, `results/figures/I1_accuracy_vs_macro_f1.png`.

**5. Tập test có bị dùng để chọn mô hình không?**
Không. Siêu tham số của nhóm truyền thống chọn bằng 5-fold CV trên train+val; CNN chọn theo val. Test chỉ dùng một lần ở bước 8, và mô hình đại diện mỗi nhóm cũng được chọn theo điểm CV/val. Các thí nghiệm biến đổi ảnh (M1, M2, C4, N8…) chấm trên fold giữ lại hoặc val. → `src/final_eval.py`, nhật ký bước 1 và bước 8.

**6. Độ lệch chuẩn "± 0,005" tính thế nào, và khi nào nói A hơn B?**
Là độ lệch chuẩn của macro-F1 qua 5 fold. Nếu chênh lệch giữa hai cấu hình nhỏ hơn độ lệch chuẩn, nhóm kết luận "chưa đủ cơ sở phân biệt". Trên test thì dùng khoảng tin cậy bootstrap 95%. → `results/select_traditional/table.md`, `results/final/table.md`.

## B. Đặc trưng thủ công

**7. Giống chuối khác nhau ở màu hay ở hình dạng?**
Cả hai đều đủ: histogram HSV + LogReg đạt 0,991, HOG trên ảnh xám đạt 0,986. Khi bỏ màu, các đặc trưng màu sụt 0,04–0,23, còn HOG và đặc trưng cạnh gần như không đổi. → `results/M6/table.md`, `results/figures/M6_grayscale.png`. Lưu ý: một phần của cả hai con số đến từ nền ảnh (câu 23).

**8. HSV có thật sự bền với độ sáng không?**
Chỉ khi bỏ hẳn kênh V. Nhân V với 0,6 thì histogram HSV đủ 3 kênh sụt 0,66–0,67, gần bằng histogram RGB; histogram H-S chỉ sụt 0,000–0,027. → `results/M2/table.md`, `results/figures/M2_brightness.png`.

**9. Nhiều bin hơn có tốt hơn không?**
Đến một mức. 4 bin mỗi kênh thì kém rõ (0,914–0,936). Histogram HSV 16³ = 4.096 chiều có 87% phần tử bằng 0, KNN giảm còn 0,917, còn LogReg vẫn 0,992. → `results/M3/table.md`, `results/figures/M3_bins.png`.

**10. CCV có tốt hơn histogram không?**
Không đo được: 0,975 so với 0,968 với LogReg (chênh nhỏ hơn độ lệch chuẩn), với KNN còn kém hơn. CCV chỉ khác histogram ở chỗ nhạy với bố cục: xáo trộn 4×4 làm CCV sụt 0,05–0,08, còn histogram không đổi. → `results/M5/table.md`.

**11. Toán tử cạnh nào bền với nhiễu nhất? Vì sao?**
Prewitt và Sobel, vì có làm trơn. Ở σ = 5, vector đặc trưng thay đổi 43–46% với Prewitt/Sobel, 100% với Roberts (mặt nạ 2×2), 216% với Laplacian (đạo hàm bậc hai). → `results/C1/table_feature_change.md`, `results/figures/C1_edge_noise.png`.

**12. Chuẩn hóa block của HOG có tác dụng gì?**
Khử thay đổi độ sáng dạng nhân: khi nhân V với 0,6, bản có chuẩn hóa chỉ sụt 0,004–0,009, bản không chuẩn hóa sụt 0,08–0,39. Khi làm sáng V×1,4 thì ngược lại, vì 68,7% điểm ảnh bị bão hòa ở 255, không còn là phép nhân tuyến tính. → `results/C2/table.md`, `results/C2/table_saturation.md`.

**13. Tính tay HOG ra 3.780 chiều thế nào?**
Ảnh 64×128, cell 8×8 → lưới 8×16 cell. Block 2×2 cell trượt 1 cell → 7×15 block. Mỗi block 4 cell × 9 bin = 36 số → 7 × 15 × 36 = 3.780. HOG tự cài lệch tối đa 2,2·10⁻⁷ so với skimage. → `python -m src.features.hog`, `results/C3/table.md`.

**14. Vì sao histogram sập khi bị che 25% còn HOG thì không?**
Phần lớn là do bước chuẩn hóa z-score. Ô đen dồn 25% khối lượng histogram vào một bin hầu như luôn bằng 0 trên ảnh huấn luyện; z-score chia cho độ lệch chuẩn rất nhỏ của bin đó nên một chiều chi phối khoảng cách. Dùng Hellinger thì histogram HSV chỉ sụt 0,012. → `results/C4/table.md`, `results/C4/table_hist_scaling.md`.

## C. Bộ phân lớp

**15. Quy tắc k = số lớp + 1 có tốt không?**
Không bằng k chọn bằng CV: k = 1 tốt nhất với cả ba đặc trưng, hơn k = 5 từ 0,006 đến 0,020. → `results/P1/table.md`, `results/figures/P1_knn_k.png`.

**16. Euclidean và cosine khác nhau thế nào?**
Trên vector đơn vị, ‖x − y‖² = 2 − 2·cos(x, y), nên hai độ đo cho cùng thứ hạng láng giềng: đo được 100% ảnh truy vấn có cùng 10 láng giềng; trước khi chuẩn hóa chỉ 11–18%. Ví dụ tính tay ở mục 5.2 của báo cáo (√(2 − 2 × 0,9330) = 0,3662). → `results/P2/table_rank_agreement.md`, `results/hand_calc/metrics.json`.

**17. K-D Tree có làm KNN chính xác hơn không?**
Không, kết quả luôn trùng Brute Force; cây chỉ thay đổi tốc độ. Với 3.780 chiều, K-D Tree chậm hơn Brute Force khoảng 60 lần. Với 9 chiều, cây chỉ nhanh hơn khi có từ 8.400 điểm tham chiếu trở lên. → `results/P4/table.md`, `results/P4/table_scaling.md`.

**18. Khi nào bộ phân lớp tuyến tính thua KNN?**
Khi đặc trưng ít chiều. Với moment màu 9 chiều, LogReg kém KNN 0,064; với 512 và 3.780 chiều, LogReg hơn 0,027 và 0,039. → `results/P6/table.md`, `results/figures/P6_linear_vs_knn.png`.

**19. Vì sao lại dùng Linear SVM?**
Môn học nhắc tới SVM như bộ phân lớp đi kèm HOG ("HOG + SVM"), nên đồ án dùng nó như bộ phân lớp bổ sung để đối chiếu, không phải phương pháp chính. Kết quả gần LogReg: 0,983 so với 0,986 trên HOG. → `results/select_traditional/table.md`.

## D. CNN

**20. Tính số tham số của một lớp tích chập thế nào?**
(F×F×C + 1) × K. Ví dụ `conv2` của `a_lenet`: (5×5×6 + 1) × 16 = 2.416. Lớp kết nối đầy đủ: (N_in + 1) × N_out, ví dụ `fc1`: (13×13×16 + 1) × 120 = 324.600. 10/10 lớp khớp `model.summary()`. → `results/N1/table.md`.

**21. Vì sao Dropout lại làm kết quả kém đi?**
Đo qua 3 seed: có Dropout 0,941 ± 0,005, không Dropout 0,953 ± 0,006. 99% tham số nằm ở lớp Dense ngay sau Flatten, đứng trước lớp Dropout, và lớp này vẫn khớp train đến accuracy 0,999. Dropout không chặn được quá khớp ở đó. Tăng cường ảnh thì giảm khoảng cách loss val − train từ 0,542 xuống 0,054. → `results/N3/table_seeds.md`, `results/N5/table.md`.

**22. Transfer learning giúp được bao nhiêu khi ít dữ liệu?**
Với 25% train (432 ảnh), `a_vgg16` đạt 0,967 và `b_mobilenetv3` đạt 0,975, ngang hoặc hơn `a_lenet` dùng đủ 1.729 ảnh (0,973). Từ 25% lên 100%, `a_lenet` tăng 0,050, các mô hình pretrained chỉ tăng 0,016–0,018. → `results/N2/table.md`, `results/figures/N2_learning_curve.png`.

**23. Fine-tune với learning rate lớn thì sao?**
lr 1e-3 làm quá trình huấn luyện dao động: macro-F1 trên val tụt xuống 0,917, trong khi lr 1e-5 không xuống dưới 0,978. Nhờ dừng sớm theo val nên kết quả cuối không kém. → `results/N4/table.md`, `results/figures/N4_finetune_lr.png`.

**24. Vì sao phải đóng băng BatchNorm khi fine-tune?**
Ở chế độ huấn luyện, BN dùng thống kê của lô thay cho thống kê ImageNet: embedding đổi 82–84%, đầu phân lớp học ở giai đoạn 1 chỉ còn đúng 4,7–12,5%. Lần fine-tune đầu (BN mở) có accuracy train ở epoch 1 chỉ 0,356. → `results/N4/table_batchnorm.md`.

**25. Mạng lớn hơn có luôn tốn tài nguyên hơn không?**
Không. MobileNetV3-Small (hướng B): 0,94 triệu tham số, 11,7 MB, 12,5 ms mỗi ảnh, đạt 1,000 trên val. `a_vgg16` (hướng A): 40,7 triệu tham số, 370 MB, 130 ms mỗi ảnh. → `results/N6/table.md`, `results/figures/N6_a_vs_b.png`.

**26. Embedding của CNN có tốt hơn đặc trưng thủ công không?**
Chỉ sau khi được học trên dữ liệu chuối. Embedding MobileNetV3 đã fine-tune + KNN đạt 1,000; embedding VGG16 ImageNet chưa học trên chuối + KNN chỉ 0,932, kém histogram HSV (0,982). Với LogReg, đặc trưng thủ công tốt nhất (0,998) ngang embedding (0,996). → `results/N7/table.md`.

**27. CNN có bền hơn đặc trưng thủ công không?**
Bền hơn với nhiễu, xoay và ảnh xám, nhưng kém hơn khi bị che 25% (MobileNetV3 0,659 so với HOG 0,801 và H-S + HOG 0,918). Với xáo trộn 4×4, `a_vgg16` (Flatten) còn 0,097, MobileNetV3 (GlobalAveragePooling) còn 0,850. → `results/N8/table.md`, `results/figures/N8_robustness.png`.

## E. Mất cân bằng, phân đoạn và thiên lệch dữ liệu

**28. Xử lý mất cân bằng có giúp không?**
Ở mô hình mạnh thì không đo được tác động (truyền thống tốt nhất 0,995–0,996 với cả 4 cách; `b_mobilenetv3` 1,000 với cả 4 cách). Ở mô hình yếu (cạnh Sobel), các cách xử lý đổi precision lấy recall của lớp ít: recall Bangla 0,822 → 0,857, precision Bangla 0,849 → 0,776. → `results/I2/table.md`, `results/figures/I2_imbalance.png`.

**29. Phương pháp phân đoạn nào tốt nhất?**
Các cách dựa trên màu (k-means, Otsu trên khoảng cách màu tới nền), vì nền trơn sáng khác quả chuối về màu rõ hơn về mức xám. Ngưỡng toàn cục và ngưỡng thích nghi tách cả vết thâm thành nền. Chưa có nhãn mask nên chưa tính được IoU/Dice. → `results/S1/table.md`, `results/figures/S1_segmentation.png`.

**30. Kết quả 100% trên test có đáng tin không?**
Đáng tin **trên bộ dữ liệu này**: hai mô hình độc lập (truyền thống và CNN) đều đúng 371/371 ảnh, và test không được dùng để chọn mô hình. Nhưng S2 cho thấy dữ liệu thiên lệch theo bối cảnh chụp: chỉ dùng 4 góc ảnh 64×64 (2,5% điểm ảnh thuộc vùng chuối), HOG vẫn đạt 0,968; CNN đoán trên ảnh đã xóa quả vẫn đạt 0,683–0,792. Vì vậy 100% không có nghĩa mô hình nhận diện được giống chuối trong điều kiện bất kỳ. → `results/S2/table_corners.md`, `results/S2/table_cnn.md`, `results/figures/S2_background_bias.png`.

**31. Thiên lệch nền cụ thể là gì?**
Mỗi lớp được chụp trong điều kiện riêng: nền Sabri Kola xanh hơn (kênh b của Lab 119,9 so với 123,6–128,3 ở các lớp khác); quả Sabri chiếm 5,4% diện tích ảnh so với 22,0% ở Sagor. Chỉ 3 con số màu nền đã cho KNN đạt 0,937; chỉ diện tích quả đạt 0,619. → `results/S2/table_background_stats.md`, `results/S2/table_extreme.md`.

**32. Nếu chỉ nhìn quả chuối thì mô hình đạt bao nhiêu?**
Cấu hình truyền thống tốt nhất 0,958 (5-fold CV); CNN học trên ảnh đầy đủ rồi đoán trên ảnh đã tô nền 0,926–0,952 (val). → `results/S2/table.md`, `results/S2/table_cnn.md`.

**33. Mô hình hay sai ở đâu?**
41 ảnh test bị ít nhất 1 trong 9 mô hình đoán sai; nhầm nhiều nhất là Sagor → Champa. Ảnh Sagor bị sai có quả nhỏ hơn trung bình lớp (diện tích 0,181 so với 0,244). Ảnh `Sabri_Kola_5.jpg` (quả to, nền trắng, không giống Sabri điển hình) bị 6/9 mô hình đoán sai. → `results/final/table_error_pairs.md`, `results/final/table_error_context.md`, `results/figures/final_errors_all.png`.

**34. Muốn cải thiện thì làm gì tiếp?**
Thu ảnh của cả 4 giống trong cùng một bối cảnh chụp, hoặc một tập test ở bối cảnh khác, để đo đúng khả năng nhận diện giống; huấn luyện trên ảnh đã tách nền hoặc thay nền ngẫu nhiên; gán nhãn mask để đánh giá phân đoạn bằng IoU/Dice. → Báo cáo mục 6.3.

## F. Phần mở rộng: ba nhánh so sánh với pipeline gốc

**35. Phần mở rộng gồm những gì và so sánh thế nào?**
Ba nhánh độc lập, mỗi nhánh đổi một thứ so với pipeline gốc (nhánh G): X đổi cách xử lý dữ liệu; T đổi tập huấn luyện sang bản Augmented của tác giả; B đổi phương pháp sang phương pháp của bài báo. Cả bốn nhánh chấm trên cùng 371 ảnh test gốc và cùng bộ kiểm tra khó. → `docs/ke-hoach-thuc-nghiem.md` mục 12, `results/so_sanh_nhanh/table_arms.md`.

**36. Vì sao phải thêm "bộ kiểm tra khó"? Điểm bền là gì?**
Trên ảnh gốc, hai cấu hình của pipeline gốc đã đúng 371/371 ảnh nên không còn chỗ để một nhánh khác thể hiện hơn kém. Bộ kiểm tra khó gồm 6 phiên bản biến đổi của tập test: nền hoán đổi, V×0,6, V×1,4, nhiễu σ = 10, xoay +30°, che 25%. Điểm bền là trung bình macro-F1 trên 6 phiên bản đó. Với pipeline gốc, macro-F1 trên ảnh gốc là 0,94–1,00 nhưng điểm bền chỉ 0,49–0,90. → `results/so_sanh_nhanh/G/table.md`, `results/figures/E_stress_examples.png`.

**37. "Nền hoán đổi" đo điều gì?**
Quả chuối của mỗi ảnh test được ghép lên nền lấy từ ảnh của một lớp khác. Nếu mô hình nhìn quả chuối thì kết quả không đổi; nếu nhìn nền thì kết quả tụt. Với pipeline gốc: HOG còn 0,145, cấu hình truyền thống tốt nhất còn 0,410, `b_mobilenetv3` còn 0,794. Đây là xác nhận thứ hai cho kết luận của S2. → `results/so_sanh_nhanh/G/table.md`.

**38. Dùng bản Augmented của tác giả thì kết quả có tốt hơn không?**
Trên ảnh gốc thì không: trung bình 9 cấu hình 0,979 so với 0,983 của pipeline gốc; HOG giảm từ 0,990 xuống 0,972 vì ảnh xoay 90° và lật không giống ảnh test. Về độ bền thì tăng nhẹ: điểm bền trung bình 0,769 so với 0,750, và phần tăng nằm đúng ở các phép có trong công thức tăng cường (HOG với xoay +30°: 0,505 → 0,688). → `results/E2_aug_tac_gia/khong_ro_ri/table.md`, `results/so_sanh_nhanh/table_models.md`.

**39. Lấy bộ 7.413 ảnh Augmented rồi chia ngẫu nhiên thì có vấn đề gì?**
Bản Augmented là bản xuất Roboflow, mỗi ảnh gốc có 3 file (1 gần nguyên gốc, 2 tăng cường). Chia ngẫu nhiên thì 90,6% ảnh test có file cùng ảnh gốc nằm trong train. Trên nhóm đó HOG đúng 0,859, trên nhóm còn lại 0,808. Lưu ý điểm của cách chia này không cao hơn cách chia theo ảnh gốc, vì tập test của nó gồm cả ảnh đã xoay, lật, thêm nhiễu nên khó hơn; hai con số không cùng thước đo. → `results/E2_aug_tac_gia/table_leak_stats.md`, `results/E2_aug_tac_gia/table_leak_accuracy.md`.

**40. Bài báo trên bộ dữ liệu này làm gì và đạt bao nhiêu?**
Ferdaus và cộng sự (2025, *Data in Brief*) mô tả dữ liệu, không huấn luyện mô hình. Ergün (2025, *Scientific Reports*) rút đặc trưng bằng ViT pretrained rồi phân lớp bằng SVM, chấm bằng 5-fold trên bản Augmented, công bố accuracy 99,70%. Không bài nào dùng YOLO. → `docs/nhat-ky-thuc-nghiem.md` bước 12.

**41. Nhóm có tái hiện được kết quả của bài báo không?**
Có, ở cấu hình chính: ViT-B/16 + SVM đạt 99,49 ± 0,15% theo đúng giao thức của bài báo (bài báo: 99,70%). Nhưng khoảng cách giữa ViT và CNN thì không tái hiện được: bài báo công bố ViT hơn ResNet + SVM 3,01 điểm, nhóm đo được chỉ 0,22–0,90 điểm (ResNet50 + SVM 99,16%, MobileNetV3 + SVM 99,27%). → `results/E3_bai_bao/table_protocols.md`, `results/figures/E3_papers_protocols.png`.

**42. Con số 99,70% của bài báo có đáng tin không?**
Về cơ bản là có, nhưng hơi lạc quan. Giao thức của bài báo (5-fold trên bản Augmented, không giữ nhóm) để các file cùng ảnh gốc rơi vào cả hai phía. Khi giữ nhóm theo ảnh gốc, ViT + SVM còn 99,20% (thấp hơn 0,28 điểm). Mức thổi phồng dương ở cả 19/19 tổ hợp đã thử, từ +0,11 đến +0,90 điểm. → `results/E3_bai_bao/table_protocols.md`.

**43. ViT có tốt hơn CNN của đồ án không? Sao không dùng ViT làm mô hình chính?**
Trên ảnh gốc thì ngang nhau: ViT + SVM sai 1/371 ảnh, `b_mobilenetv3` và cấu hình truyền thống tốt nhất sai 0/371. Trên bộ kiểm tra khó thì đặc trưng ViT bền hơn: điểm bền 0,919–0,937 so với cao nhất 0,895 của pipeline gốc. Nhóm không dùng ViT làm mô hình chính vì ViT nằm ngoài nội dung môn học, có khoảng 86 triệu tham số và rút đặc trưng chậm hơn MobileNetV3 khoảng 23 lần trên CPU. → `results/E3_bai_bao/table_test.md`, `results/E3_bai_bao/summary.json`.

**44. Hướng "chỉ dùng tới CNN" của phương pháp bài báo cho kết quả thế nào?**
CNN pretrained làm bộ rút đặc trưng + bộ phân lớp đã học (SVM, Logistic Regression, MLP) đạt macro-F1 0,990–0,997 trên test gốc, ngang ViT; điểm bền 0,63–0,89, thấp hơn ViT và thấp hơn fine-tune cùng mạng có tăng cường (MobileNetV3 đóng băng + SVM tuyến tính 0,836; fine-tune 0,875). → `results/E3_bai_bao/table_test.md`.

**45. Pipeline gốc đã xử lý và tăng cường dữ liệu chưa? Nhánh X thêm gì?**
Pipeline gốc chỉ resize ảnh về 256×256 và chuẩn hóa giá trị điểm ảnh; tăng cường chỉ bật lúc huấn luyện cho `a_dropout` và hai mô hình hướng B. Nhánh X thêm ba phần: làm sạch (loại 35 ảnh train gần trùng), tiền xử lý (5 mức: không; cân bằng trắng; nền phẳng; cả hai; thêm cắt theo quả) và tăng cường chỉ trên train (3 mức: không; hình học + quang học; thêm thay nền ngẫu nhiên). Tổ hợp được chọn trên val bằng 7 phép thử. → `results/E1_xu_ly/table_cleaning.md`, `results/E1_xu_ly/table_selection.md`, `results/figures/E1_preprocess_examples.png`.

**46. Cách xử lý dữ liệu nào tốt nhất?**
Trên val, "nền phẳng + tăng cường hình học/quang học" (X-a) đạt điểm chọn cao nhất, 0,889 ± 0,006, so với 0,804 khi không xử lý, và phân biệt được với mọi tổ hợp khác theo bootstrap. Khi chạy lại pipeline: X-a đưa phép nền hoán đổi của `b_mobilenetv3` từ 0,794 lên 0,995 nhưng làm HOG, cạnh, `a_lenet`, `a_vgg16` mất điểm trên ảnh gốc (trung bình 9 cấu hình 0,983 → 0,942). X-b (chỉ tăng cường có thay nền, không đổi ảnh lúc dự đoán) giữ `b_mobilenetv3` ở 371/371 ảnh gốc và đạt 0,975 ở nền hoán đổi. Trên ảnh gốc thì không cách nào hơn pipeline gốc, vì nó đã đúng 371/371. → `results/E1_xu_ly/pipeline/table.md`, `results/E1_xu_ly/pipeline_b/table.md`, `results/so_sanh_nhanh/table_models.md`.

**47. Vì sao không dùng cân bằng trắng theo nền giấy trắng?**
Vì nó chuyển thông tin của nền vào màu quả chuối. Hệ số cân bằng tính từ màu nền rồi nhân vào cả ảnh; khi nền đổi, màu quả lệch trung bình 12,9 đơn vị Lab (không cân bằng trắng: 0,0), và histogram HSV + LogReg sụt 0,351 ở phép nền hoán đổi, trong khi chỉ làm phẳng nền thì sụt 0,023. Lần chọn đầu của nhóm có bước này; nhóm phát hiện lỗi qua phép nền hoán đổi rồi sửa. → `results/E1_xu_ly/table_wb_leak.md`, nhật ký bước 13.

**48. "Mỗi quả chụp bốn mặt" ảnh hưởng gì tới kết quả?**
Bài báo mô tả dữ liệu cho biết mỗi quả được chụp từ bốn mặt. Đo được: 20,0% số ảnh có láng giềng gần nhất cùng lớp mang số thứ tự cách không quá 3 (ngẫu nhiên: 1,0%), và cross-validation chia theo khối ảnh liên tiếp thấp hơn chia ngẫu nhiên 0,024–0,069. Tức ảnh của cùng một quả nằm ở cả train lẫn test khi chia ngẫu nhiên, nên mọi con số chấm bằng cách chia ngẫu nhiên, kể cả 1,000 trên test, lạc quan hơn so với khi gặp quả chưa từng thấy. Dữ liệu không có nhãn "quả nào" nên chưa chia lại được. → `results/E1_xu_ly/table_adjacent.md`, `results/E1_xu_ly/table_blocked_cv.md`.

**49. Tóm lại, trong bốn nhánh thì nhánh nào tốt nhất?**
Trên ảnh gốc không nhánh nào hơn pipeline gốc: cấu hình đại diện của các nhánh sai từ 0 đến 2 ảnh trên 371. Khác biệt nằm ở bộ kiểm tra khó, và mỗi nhánh mạnh ở một chỗ: nhánh X gần như xóa được sự phụ thuộc vào nền (nền hoán đổi 0,975–0,995 so với 0,794); nhánh T bền đều hơn ở các phép có trong công thức tăng cường (điểm bền 0,909); ViT + SVM của bài báo có điểm bền cao nhất (0,919) nhưng nặng và nằm ngoài môn học; CNN đóng băng + SVM kém bền nhất (0,836). Điểm bền trung bình che mất sự đánh đổi: nhánh X được ở nền hoán đổi nhưng mất ở phép che 25% (0,44–0,45 so với 0,66). → `results/so_sanh_nhanh/table_arms.md`, `results/so_sanh_nhanh/table_stress.md`, `results/figures/E_compare_stress.png`.

## G. Cài đặt và chạy lại

**50. Làm sao chạy lại toàn bộ kết quả?**
`bash scripts/setup_env.sh` rồi `bash scripts/run_all.sh` (hoặc từng bước, ví dụ `bash scripts/run_all.sh 2 3`; phần mở rộng là các bước 10–14). Seed cố định, phiên bản thư viện ghi ở `results/environment.txt`. Huấn luyện CNN có checkpoint mỗi epoch và `--resume`. → `README.md`.

**51. Demo chạy thế nào?**
`python app/app.py` mở giao diện web; `python app/app.py --image <ảnh>` dự đoán trên dòng lệnh. Mô hình mặc định là `b_mobilenetv3` (11,7 MB); có thêm bản huấn luyện với nền thay ngẫu nhiên (nhánh X-b), bền hơn khi nền ảnh khác bộ dữ liệu. `a_vgg16` không đưa vào demo vì file 370 MB vượt giới hạn 50 MB. → `results/figures/demo_screenshot.png`, `models/models.json`.
