## 3. Kịch bản demo (khoảng 2 phút)

Chuẩn bị trước buổi báo cáo:

- Mở sẵn chương trình bằng lệnh `python app/app.py` và mở trình duyệt ở địa chỉ http://127.0.0.1:7860. Lần dự đoán đầu tiên của mỗi mô hình mất vài giây để nạp mô hình, nên bấm thử một lần trước khi lên trình bày.
- Ở tab 6, bấm sẵn **Tính metric** một lần cho tập test để lần bấm trên lớp ra kết quả ngay.
- Chỉ dùng các ảnh ví dụ có sẵn trong demo (4 ảnh lấy từ tập test). Đây là những ảnh nhóm đã chạy thử và biết chắc kết quả.

Các bước demo và lời nói:

| Bước | Thao tác | Lời nói |
|---|---|---|
| 1 | Tab **1. Dự đoán**: bấm ảnh ví dụ Sabri Kola, bấm **Dự đoán** | "Đây là một ảnh trong tập test. Chương trình trả về giống Sabri Kola, kèm xác suất của bốn lớp và thời gian dự đoán. Mô hình đang dùng là MobileNetV3-Small đã fine-tune." |
| 2 | Tab **2. So sánh mô hình**: bấm ảnh ví dụ Champa Kola, bấm **Chạy tất cả mô hình** | "Cùng một ảnh chạy qua cả bốn mô hình: hai CNN, mô hình có thay nền lúc huấn luyện và mô hình truyền thống. Cả bốn cùng trả về Champa Kola." |
| 3 | Tab **3. Giải thích (Grad-CAM)**: bấm ảnh ví dụ Bangla Kola, bấm **Tạo bản đồ nhiệt** | "Grad-CAM cho biết mô hình dựa vào vùng nào của ảnh. Vùng màu đỏ là vùng ảnh hưởng mạnh nhất tới kết quả." |
| 4 | Tab **6. Đánh giá theo metric**: chọn mô hình LeNet, bấm **Tính metric** | "Tab này tính lại các độ đo trên toàn bộ tập test. Với LeNet: accuracy 0.9811, macro-F1 0.9777, sai 7 trên 371 ảnh, đúng bằng con số trong báo cáo. Bên dưới là precision, recall, F1 của từng lớp, ma trận nhầm lẫn và các ảnh bị đoán sai." |

Nếu còn thời gian thì mở thêm tab **4. Thử độ bền**, chọn mẫu "Nhiễu σ=10" để cho thấy dự đoán không đổi khi ảnh bị nhiễu.

**Khi thầy hoặc các bạn yêu cầu chạy một ảnh khác:**

1. Trước khi chạy, nói trước dự đoán của mình. Nếu là ảnh chuối thuộc 4 giống, chụp một quả trên nền trơn: "Em dự đoán chương trình sẽ nhận đúng." Nếu là ảnh ngoài phạm vi (giống khác, quả khác, con vật, nải chuối, nền phức tạp): "Ảnh này nằm ngoài phạm vi bài toán. Chương trình vẫn chạy được và sẽ trả về một trong bốn giống, vì đầu ra của bài toán chỉ có bốn lớp."
2. Chạy ảnh. Chương trình không báo lỗi với bất kỳ ảnh nào.
3. Khi được hỏi kết quả đúng hay sai với ảnh ngoài phạm vi: "Chương trình chạy đúng theo phát biểu bài toán: output là một trong bốn giống. Còn ảnh này không thuộc tập nhãn của bài toán nên em không đánh giá đúng sai cho nó được. Muốn chương trình trả lời là không phải bốn giống này thì phải phát biểu lại bài toán với lớp thứ năm, phần đó nhóm em chưa làm và đã ghi trong hạn chế."
4. Không nói rằng ảnh ngoài phạm vi sẽ có xác suất thấp. Nhóm đã thử: ảnh logo của trường cho ra Sagor Kola với xác suất 0.988. Mô hình chỉ biết chia xác suất cho 4 lớp, nên xác suất cao không có nghĩa là ảnh thuộc phạm vi bài toán. Dòng nhắc "mô hình không chắc chắn" trong demo chỉ hiện khi xác suất cao nhất dưới 0.6.
5. Không tự chọn ảnh khó để chạy. Chỉ chạy ảnh ngoài kịch bản khi được yêu cầu.

## 4. Nếu thiếu thời gian

- Còn khoảng 10 phút: slide 4 (hình minh họa) chỉ nói một câu; slide 14 (điểm khác biệt) chỉ đọc dòng đầu và dòng cuối của bảng; gộp slide 18 và 19 mỗi slide một câu; demo chỉ làm bước 1 và bước 4.
- Còn khoảng 7 phút: mỗi cặp slide "là gì" và "dùng như thế nào" chỉ nói slide "dùng như thế nào"; bỏ slide 18, 19, 21; demo chỉ làm bước 1.
- Không bỏ các slide: lý do (2), phát biểu bài toán (3), hình minh họa (4), dataset (15), độ đo (16), kết quả (17), demo (22), kết luận (24). Mỗi slide này ứng với một mục chấm điểm trong phiếu đánh giá.

## 5. Câu hỏi dự trù

Các câu dưới đây là những câu dễ bị hỏi nhất. Danh sách đầy đủ hơn (kèm đường dẫn tới bảng số liệu) nằm ở `docs/hoi-dap-bao-ve.md`. Khi trả lời có thể mở slide phụ lục tương ứng.

### 5.1 Về bài toán và chương trình demo

**1. Đưa vào ảnh con sư tử (hoặc quả táo, hoặc giống chuối khác) thì chương trình ra gì? Như vậy là đúng hay sai?**
Chương trình vẫn chạy và trả về một trong bốn giống, vì theo phát biểu bài toán, output chỉ gồm bốn lớp. Xét theo ràng buộc đó thì chương trình không sai. Còn ảnh đó không thuộc tập nhãn nên không có nhãn đúng để so. Nếu muốn chương trình trả lời "không phải bốn giống này" thì bài toán phải có thêm lớp thứ năm và dữ liệu cho lớp đó. Nhóm em chưa làm phần này và đã ghi trong hạn chế. Xác suất trả về cũng không dùng để phát hiện ảnh ngoài phạm vi được: nhóm em thử ảnh logo của trường thì mô hình trả về Sagor Kola với xác suất 0.988.

**2. Ảnh chuối chụp ngoài thực tế, ví dụ cả nải chuối hoặc chuối để trên bàn, thì có nhận đúng không?**
Nhóm em chưa có dữ liệu như vậy để đo nên không khẳng định được. Mô hình chỉ học ảnh một quả chuối trên nền trơn. Thí nghiệm của nhóm em cho thấy mô hình dựa một phần vào nền: khi ghép quả chuối lên nền của ảnh lớp khác, macro-F1 của MobileNetV3 giảm từ 1.000 xuống 0.794. Nên với ảnh ngoài thực tế, em dự đoán kết quả sẽ kém hơn trên bộ dữ liệu.

**3. Kết quả 100% như vậy có đáng tin không? Có bị quá khớp không?**
Tập test chỉ dùng một lần ở cuối, mọi lựa chọn đều làm trên train và val, nên đây không phải quá khớp vào tập test. Nhưng con số này chỉ đúng trên bộ dữ liệu này, vì hai lý do nhóm em đo được: dữ liệu thiên lệch theo nền (chỉ nhìn nền vẫn đạt 0.93 đến 0.99), và mỗi quả được chụp bốn mặt nên ảnh của cùng một quả có thể nằm ở cả train lẫn test. Với 0 ảnh sai trên 371 ảnh, tỉ lệ lỗi thật vẫn có thể tới khoảng 0.8%.

**4. Dataset này nhóm tự thu thập hay lấy có sẵn?**
Dataset có sẵn: BananaImageBD của Ferdaus và cộng sự, công bố trên Mendeley Data, giấy phép CC BY 4.0. Nhóm em chỉ chia lại thành train, val, test.

**5. Đề tài của nhóm khác gì so với bài báo đã có trên bộ dữ liệu này?**
Bài báo dùng Vision Transformer kết hợp SVM và báo cáo accuracy 99.70% theo 5-fold trên bản Augmented. Nhóm em khác ở ba điểm: dùng các phương pháp trong môn học (đặc trưng thủ công, KNN, CNN); chia dữ liệu trên ảnh gốc trước rồi mới tăng cường để tránh rò rỉ; và kiểm tra thêm mô hình nhìn quả hay nhìn nền, độ bền khi ảnh thay đổi. Nhóm em cũng chạy lại phương pháp của bài báo và đạt 99.49%.

### 5.2 Về khái niệm cơ bản

**6. Feature vector khác embedding như thế nào?**
Feature vector là vector mà mình hiểu ý nghĩa và cách tính của từng phần tử. Ví dụ phần tử thứ i của histogram là tỉ lệ điểm ảnh có màu rơi vào khoảng thứ i. Embedding là vector lấy ra từ mạng học sâu, mình không giải thích được ý nghĩa của từng phần tử. Trong đồ án, histogram màu, moment màu, HOG là feature vector, còn vector 576 chiều của MobileNetV3 là embedding.

**7. Đặc trưng toàn cục và đặc trưng cục bộ khác nhau thế nào? HOG và CNN thuộc loại nào?**
Khác nhau ở cách tính mỗi phần tử. Nếu mỗi phần tử được tính từ toàn bộ ảnh thì là toàn cục, ví dụ histogram màu. Nếu mỗi phần tử chỉ tính từ một vùng nhỏ của ảnh thì là cục bộ. HOG là cục bộ vì mỗi giá trị tính trên một cell. Đặc trưng của CNN cũng là cục bộ vì mỗi giá trị tính trên một cửa sổ tích chập. Vision Transformer tính tương quan giữa mọi mảnh của ảnh nên là toàn cục.

**8. Tại sao phải chuẩn hóa vector trước khi tính khoảng cách?**
Vì hai vector cùng hướng nhưng khác độ lớn thì độ tương đồng là 100%, nhưng nếu không chuẩn hóa thì khoảng cách Euclidean giữa chúng lại khác 0. Chuẩn hóa về vector đơn vị thì hai vector đó trùng nhau và khoảng cách bằng 0. Nhóm em kiểm chứng ở thí nghiệm P2: sau khi chuẩn hóa, Euclidean và cosine cho cùng 10 láng giềng ở 100% ảnh truy vấn.

**9. Ý nghĩa của một con số trong ma trận nhầm lẫn?**
Hàng là lớp thật, cột là lớp dự đoán. Ví dụ với a_vgg16 trên test, hàng Bangla Kola cột Sagor Kola bằng 2, nghĩa là có 2 ảnh Bangla Kola bị đoán nhầm thành Sagor Kola. Nếu chia theo tổng hàng (67 ảnh) thì được 0.030, tức 3.0% số ảnh Bangla Kola bị đoán nhầm thành Sagor Kola. Ô trên đường chéo là 65 trên 67, bằng 0.970, chính là recall của lớp Bangla Kola. Khi chia theo hàng thì tổng mỗi hàng bằng 1.

**10. Precision và recall khác nhau thế nào? Vì sao chọn macro-F1 mà không chọn accuracy?**
Precision của lớp c là trong các ảnh mô hình gọi là c, bao nhiêu phần đúng. Recall của lớp c là trong các ảnh thật sự là c, mô hình tìm ra bao nhiêu phần. Dữ liệu của nhóm em lệch lớp, Champa Kola gấp 2.2 lần Bangla Kola. Một bộ phân lớp luôn đoán Champa Kola vẫn có accuracy 0.402 nhưng macro-F1 chỉ 0.143. Macro-F1 tính mỗi lớp trọng số như nhau nên lớp ít ảnh bị nhận kém sẽ lộ ra.

**11. Ảnh 64 × 128 của HOG thì số nào là chiều rộng, số nào là chiều cao?**
Khi nói kích thước ảnh thì chiều rộng đứng trước: rộng 64, cao 128. Khi lưu thành ma trận thì số hàng đứng trước, nên ma trận có 128 hàng và 64 cột.

**12. Vector HOG 3780 chiều tính ra như thế nào?**
Ảnh 64 × 128 chia thành cell 8 × 8 được 8 × 16 cell. Block gồm 2 × 2 cell, trượt mỗi lần 1 cell, nên có 7 × 15 = 105 block. Mỗi block có 4 cell, mỗi cell 9 bin, tức 36 số. Tổng là 105 × 36 = 3780.

### 5.3 Về phương pháp và kết quả

**13. MobileNetV3 (hoặc ResNet, VGG) bên trong gồm những thành phần nào?**
Em chưa tìm hiểu kỹ chi tiết bên trong, vì nhóm em chỉ sử dụng nó làm phần thân rút đặc trưng. Chi tiết kiến trúc có trong bài báo gốc mà nhóm em để link ở slide. Phần em nắm là cách nhóm em dùng: phiên bản Small, trọng số ImageNet của Keras, ảnh vào 224 × 224, đầu ra sau GlobalAveragePooling là vector 576 chiều, nhóm em thay lớp phân loại 1000 lớp bằng lớp Softmax 4 lớp rồi fine-tune hai giai đoạn.

**14. Tại sao dùng MobileNetV3-Small mà không dùng phiên bản Large hay mạng lớn hơn?**
Nhóm em chỉ có CPU và dữ liệu chỉ có 2471 ảnh. Phiên bản Small đã đạt macro-F1 1.000 trên val nên nhóm em không thử phiên bản Large. Nhóm em có thử một mạng lớn hơn là ResNet50, kết quả 0.997 trên test, không hơn MobileNetV3-Small mà file nặng hơn khoảng 22 lần (95 MB so với 4.3 MB).

**15. Vì sao không dùng bản Augmented có sẵn của tác giả để có nhiều dữ liệu hơn?**
Bản Augmented chứa ảnh gốc và các bản tăng cường của nó. Nếu chia ngẫu nhiên bản này thì bản sao của ảnh test nằm trong train. Thí nghiệm D1 của nhóm em đo được accuracy bị thổi phồng khoảng 0.02 với histogram màu và KNN. Nhóm em chia trên ảnh gốc trước, rồi chỉ tăng cường phần huấn luyện. Ở phần mở rộng, nhóm em có thử huấn luyện bằng bản Augmented theo cách không rò rỉ, kết quả trên ảnh gốc không tốt hơn.

**16. Vì sao KNN tốt nhất với k = 1?**
Các lớp tách nhau khá rõ, và mỗi quả có nhiều ảnh gần giống nhau, nên láng giềng gần nhất hầu như luôn cùng lớp. Tăng k chỉ kéo thêm các điểm ở xa, phần nhiều thuộc lớp đông nhất. Kết quả này chọn bằng 5-fold cross-validation chứ không phải chọn theo test.

**17. Vì sao Dropout không giúp ở mô hình a_dropout?**
Đây là kết quả trái với dự đoán của nhóm em. Qua 3 seed, bản có Dropout đạt 0.941, bản không có đạt 0.953. Nhóm em giải thích: 99% tham số của mô hình nằm ở lớp Dense ngay sau Flatten, đứng trước lớp Dropout. Lớp này vẫn khớp tập train gần 100%, nên Dropout đặt phía sau không chặn được quá khớp ở đó. Tăng cường ảnh thì giúp rõ hơn, vì nó tác động vào đầu vào.

**18. Vì sao phải đóng băng BatchNorm khi fine-tune?**
Lần đầu nhóm em để BatchNorm mở, accuracy trên train ở epoch đầu chỉ còn khoảng 0.36. Khi BatchNorm ở chế độ huấn luyện, nó dùng thống kê của từng lô ảnh thay cho thống kê đã học trên ImageNet, nên embedding thay đổi trên 80% và lớp Softmax học ở giai đoạn 1 không còn dùng được. Giữ BatchNorm đóng băng thì hết hiện tượng này.

**19. Vì sao HOG dùng ảnh 64 × 128 trong khi ảnh gốc là hình vuông?**
Nhóm em dùng cấu hình của bài giảng và của bài báo gốc. Nhóm em có thử thêm ảnh 128 × 128 và các cỡ cell, số bin khác ở thí nghiệm C3: phần lớn cấu hình cho kết quả ngang nhau, chênh lệch nhỏ hơn độ lệch chuẩn giữa các fold, nên nhóm em giữ cấu hình của bài giảng.

**20. Grad-CAM là gì?**
Là cách xem CNN dựa vào vùng nào của ảnh để ra quyết định. Lấy bản đồ đặc trưng của lớp tích chập cuối, tính gradient của điểm lớp theo bản đồ đó, lấy trung bình gradient làm trọng số cho từng kênh, rồi cộng các kênh lại và giữ phần dương. Nhóm em tự cài đặt theo bài báo của Selvaraju và cộng sự năm 2017.

**21. Vision Transformer là gì, hoạt động ra sao?**
Vision Transformer nằm ngoài nội dung môn học, nhóm em chưa tìm hiểu kỹ. Nhóm em chỉ dùng mô hình ViT-B/16 có sẵn của torchvision làm bộ rút đặc trưng 768 chiều, đúng như bài báo mô tả, để đối chiếu kết quả. Chi tiết có trong bài báo của Dosovitskiy và cộng sự năm 2021.

**22. Thời gian chạy của các mô hình?**
Trên CPU 4 nhân: MobileNetV3-Small dự đoán một ảnh mất 12.5 ms, a_vgg16 mất 130 ms. Chạy lại toàn bộ thí nghiệm với trọng số có sẵn mất khoảng 2.5 giờ, huấn luyện lại từ đầu mất khoảng 8 giờ.

### 5.4 Về cách làm việc của nhóm

Các câu này nhóm tự chuẩn bị câu trả lời theo đúng thực tế, vì có thể bị yêu cầu mở lên cho xem:

- Nhóm trao đổi qua kênh nào, họp ở đâu, bao lâu một lần?
- Ai làm phần nào? Ai là người ghép code?
- Phần nào nhóm tự viết, phần nào tham khảo, tham khảo từ đâu? (xem mục 7.2 và 7.3 của báo cáo)
- Mã nguồn nằm ở đâu, tổ chức thư mục như thế nào? (mở trang GitHub, file README mô tả cấu trúc thư mục)

## 6. Những điểm cần nhớ khi báo cáo

- Slide đầu chỉ có tên đề tài và thành viên, không ghi tên giảng viên hướng dẫn. Không cần slide mục lục.
- Thứ tự: lý do, phát biểu bài toán, phương pháp, rồi mới tới dataset, độ đo, kết quả. Không nói dataset ngay sau phát biểu bài toán.
- Phương pháp có sẵn: nói ngắn nó là gì, chỉ vào link tài liệu gốc, rồi nói kỹ nhóm dùng như thế nào (phiên bản nào, bao nhiêu chiều, sửa gì cho bài toán 4 lớp).
- Slide độ đo phải có, nhưng chỉ cần nói một câu vì đây là các độ đo cơ bản đã học.
- Ai làm phần nào thì tự trình bày phần đó ở mục phương pháp. Phần chung một bạn đại diện trình bày.
- Demo chỉ chạy ảnh đã chuẩn bị. Khi được yêu cầu chạy ảnh khác thì nói dự đoán trước rồi mới chạy.
- Cuối phần kết luận có câu: nhóm đã cố gắng và dành nhiều thời gian, tuy nhiên vẫn còn những phần chưa đạt.
- Khi nhóm khác báo cáo thì nên đặt câu hỏi.
- Nộp bài: file báo cáo (`report/report.pdf`) và link GitHub của mã nguồn, không nén mã nguồn để nộp.
- Trong báo cáo, mục 7.3 (công cụ và sự hỗ trợ khác) và hai cột "Công việc phụ trách", "Mức độ hoàn thành" ở mục 1 do nhóm tự điền trước khi nộp.
