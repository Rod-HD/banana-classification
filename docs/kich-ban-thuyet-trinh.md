# Kịch bản thuyết trình đồ án CS231.R12: Nhận diện giống chuối từ ảnh

File này do `python -m src.make_slides` sinh ra từ cùng danh sách slide với `slides/bao-cao-do-an.pdf`. Muốn sửa lời thuyết trình thì sửa trường `notes` trong `src/make_slides.py`; phần demo và hỏi đáp sửa ở `docs/kich-ban-phan-co-dinh.md`.

Tổng thời gian dự kiến của 26 slide chính: khoảng 18 phút (gồm khoảng 2 phút demo). Các slide phụ lục không trình bày, chỉ mở khi được hỏi.

## 1. Phân công trình bày

Phần phương pháp (mục 3 của phiếu đánh giá, 30 điểm) được chấm riêng từng thành viên, nên ai làm phần nào thì trình bày phần đó. Các phần còn lại là điểm chung, một bạn đại diện trình bày.

| Phần | Slide | Người trình bày |
|---|---|---|
| Phần chung (mở đầu, bài toán, dataset, độ đo, kết quả, demo, kết luận) | 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 |  |
| Phần 1: đặc trưng màu và KNN | 6, 7 |  |
| Phần 2: cạnh, HOG, ghép đặc trưng, bộ phân lớp tuyến tính | 8, 9 |  |
| Phần 3: CNN hướng A và hướng B | 10, 11, 12, 13, 14 |  |

Cột "Người trình bày" để nhóm tự điền theo đúng phần mỗi bạn đã làm.

## 2. Lời thuyết trình theo từng slide

### Slide 1. Nhận diện giống chuối từ ảnh

*Chung, khoảng 20 giây.*

Kính chào thầy và các bạn. Nhóm em gồm ba thành viên: Hoàng Quốc Duy, Nguyễn Thái Duy, Nguyễn Anh Khoa. Hôm nay nhóm em xin trình bày đồ án nhận diện giống chuối từ ảnh. Bài trình bày đi theo thứ tự: lý do chọn đề tài, phát biểu bài toán, phương pháp, thực nghiệm, demo, rồi nhận xét và kết luận.

### Slide 2. Lý do chọn đề tài

*Chung, khoảng 45 giây.*

Lý do thứ nhất là ý nghĩa thực tế. Chuối có nhiều giống, giá bán và cách dùng khác nhau, mà ở chợ người ta vẫn phân biệt bằng mắt và kinh nghiệm. Bộ dữ liệu BananaImageBD được thu ngay tại chợ đầu mối và cửa hàng ở Bangladesh, tác giả hướng tới ứng dụng trong nông nghiệp và chuỗi cung ứng. Lý do thứ hai là bài toán này hợp với môn học. Bốn giống khác nhau cả về màu lẫn hình dạng, nên nhóm em so sánh được đặc trưng màu với đặc trưng hình dạng. Dữ liệu nhỏ và lệch lớp, nên thấy rõ hiện tượng quá khớp và lợi ích của transfer learning. Ảnh chụp một quả trên nền trơn, nên kiểm tra được mô hình đang nhìn quả chuối hay nhìn nền. Về cơ sở thực hiện: dữ liệu công khai, có giấy phép, và đã có một bài báo công bố kết quả 99.70% trên chính bộ dữ liệu này để nhóm em đối chiếu.

### Slide 3. Phát biểu bài toán

*Chung, khoảng 50 giây.*

Đây là bài toán phân loại ảnh nhiều lớp. Cho trước tập nhãn C gồm bốn giống chuối, và tập huấn luyện D gồm các cặp ảnh và nhãn, trong đó mỗi ảnh là ảnh màu 256 nhân 256 chụp một quả chuối. Cần tìm hàm phân loại f. Với một ảnh x chưa gặp, f trả về vector xác suất p gồm bốn thành phần không âm, tổng bằng một. Nhãn dự đoán là lớp có xác suất lớn nhất, và ta muốn nhãn này trùng với nhãn thật. Khi sử dụng, input là một ảnh màu chụp một quả chuối, output là tên giống và xác suất của bốn lớp. Nhóm em lưu ý một ràng buộc: đầu ra chỉ gồm bốn lớp, không có lớp khác. Nên nếu đưa vào ảnh ngoài phạm vi, chương trình vẫn chạy và vẫn trả về một trong bốn giống.

### Slide 4. Hình minh họa bài toán

*Chung, khoảng 25 giây.*

Đây là hình minh họa. Bên trái là bốn ảnh lấy từ tập test, mỗi ảnh một giống. Ảnh đi qua mô hình f. Bên phải là đầu ra: xác suất của bốn lớp, và nhãn dự đoán là lớp có cột cao nhất. Các xác suất trong hình là kết quả thật của mô hình MobileNetV3 trong đồ án.

### Slide 5. Tổng quan phương pháp

*Chung, khoảng 35 giây.*

Nhóm em giải bài toán theo hai hướng. Hướng truyền thống: rút feature vector do con người thiết kế, gồm màu, cạnh và HOG, rồi đưa vào bộ phân lớp. Hướng học sâu: dùng CNN, mạng tự học đặc trưng từ dữ liệu. Phần phương pháp chia làm ba phần, mỗi bạn trình bày phần mình làm: phần một là đặc trưng màu và KNN, phần hai là cạnh, HOG, ghép đặc trưng và bộ phân lớp tuyến tính, phần ba là CNN. Các phương pháp này đều đã có sẵn, nên với mỗi phương pháp nhóm em chỉ nói ngắn gọn nó là gì, được đề xuất ở đâu, rồi nói rõ nhóm em dùng nó như thế nào.

### Slide 6. Phần 1. Đặc trưng màu và KNN: là gì

*Phần 1, khoảng 45 giây.*

Em xin trình bày phần một: đặc trưng màu và KNN. Histogram màu đếm tỉ lệ điểm ảnh rơi vào từng khoảng màu, nó không quan tâm điểm ảnh nằm ở đâu. Moment màu tóm tắt phân bố màu của mỗi kênh bằng ba con số. CCV bổ sung cho histogram thông tin điểm ảnh có thuộc một vùng liên thông lớn hay không. Cả ba đều là feature vector toàn cục: mình hiểu ý nghĩa của từng phần tử, và mỗi phần tử được tính từ toàn bộ ảnh. Bộ phân lớp đi kèm là KNN: ảnh mới được gán nhãn theo k ảnh huấn luyện gần nó nhất. Tài liệu gốc của từng phương pháp nằm ở cuối slide.

### Slide 7. Phần 1. Đặc trưng màu và KNN: nhóm dùng như thế nào

*Phần 1, khoảng 50 giây.*

Đây là cách nhóm em dùng. Histogram RGB lấy 32 bin mỗi kênh, được 96 chiều. Histogram HSV lấy 8 bin mỗi kênh, được 512 chiều. Histogram H-S chỉ lấy hai kênh H và S, bỏ kênh độ sáng, được 64 chiều. Moment màu 9 chiều, màu chủ đạo 12 chiều, CCV 128 chiều. Trước khi đo khoảng cách, vector được chuẩn hóa. Lý do là hai vector cùng hướng nhưng khác độ lớn thì độ tương đồng là một trăm phần trăm, nhưng nếu không chuẩn hóa, khoảng cách Euclidean giữa chúng lại khác không. Với KNN, nhóm em khảo sát số láng giềng k, độ đo và cách bình chọn bằng 5-fold cross-validation, và k bằng 1 cho kết quả tốt nhất.

### Slide 8. Phần 2. Cạnh, HOG và bộ phân lớp tuyến tính: là gì

*Phần 2, khoảng 45 giây.*

Em xin trình bày phần hai. Đặc trưng màu không biết gì về hình dạng, nên nhóm em dùng thêm đặc trưng cạnh và HOG. Các toán tử Roberts, Prewitt, Sobel xấp xỉ đạo hàm bậc nhất của ảnh, Laplacian xấp xỉ đạo hàm bậc hai. HOG do Dalal và Triggs đề xuất năm 2005: ảnh được chia thành các cell, mỗi cell có một histogram của hướng gradient, rồi chuẩn hóa theo block. HOG mô tả hình dạng của vật. Khác với histogram màu, HOG là đặc trưng cục bộ: mỗi phần tử tính trên một cell, nên nó giữ thông tin vị trí. Bộ phân lớp ở phần này là Logistic Regression và Linear SVM, đều là bộ phân lớp tuyến tính.

### Slide 9. Phần 2. Cạnh, HOG, ghép đặc trưng: nhóm dùng như thế nào

*Phần 2, khoảng 55 giây.*

Về cách dùng. HOG nhóm em tự cài đặt theo đúng các bước trong bài giảng: ảnh xám 64 nhân 128, cell 8 nhân 8, 9 bin, block 16 nhân 16 có chuẩn hóa L2, ra vector 3780 chiều. Nhóm em tự cài đặt để có thể bật tắt bước chuẩn hóa block trong thí nghiệm, và đã đối chiếu với thư viện scikit-image, hai kết quả gần như trùng nhau. Vector cạnh là tổng độ lớn gradient theo từng hàng và từng cột, 512 chiều. Điểm khác của nhóm em là cách ghép: phần màu chỉ có 64 chiều, phần HOG có 3780 chiều. Nếu ghép thẳng thì HOG lấn át. Nên mỗi phần được chuẩn hóa riêng rồi chia cho căn số chiều, để hai phần đóng góp ngang nhau. Hình bên phải cho thấy ghép theo cách này tốt hơn từng đặc trưng riêng lẻ. Cấu hình cuối cùng của hướng truyền thống là histogram H-S ghép HOG, phân lớp bằng Logistic Regression.

### Slide 10. Phần 3. CNN hướng A (theo bài giảng): là gì

*Phần 3, khoảng 40 giây.*

Em xin trình bày phần ba, CNN. Hướng A là các mô hình đúng như ví dụ trong bài giảng. LeNet gồm hai cặp lớp tích chập và lấy mẫu, sau đó là các lớp kết nối đầy đủ. Dropout là kỹ thuật tắt ngẫu nhiên một số nơ-ron lúc huấn luyện để giảm quá khớp. VGG16 là mạng 16 lớp, được huấn luyện sẵn trên ImageNet với 1000 lớp. Transfer learning nghĩa là dùng lại phần tích chập đã học của VGG16 để rút đặc trưng cho bài toán của mình. Khác với feature vector ở hai phần trước, đặc trưng do CNN sinh ra gọi là embedding, mình không giải thích được ý nghĩa của từng phần tử.

### Slide 11. Phần 3. CNN hướng A: nhóm dùng như thế nào

*Phần 3, khoảng 50 giây.*

Nhóm em dùng như sau. Mô hình a_lenet giữ cấu trúc LeNet, đổi ảnh vào thành ảnh màu 64 nhân 64 và lớp ra thành 4 lớp, huấn luyện từ đầu. Mô hình a_dropout là CNN hai lớp tích chập, có Dropout 0.5, ảnh vào 128 nhân 128. Với VGG16: mạng gốc huấn luyện trên 1000 lớp, còn bài của nhóm em chỉ có 4 lớp. Nên nhóm em bỏ ba lớp kết nối đầy đủ ở cuối, giữ nguyên và đóng băng phần tích chập, rồi thêm phần đầu mới gồm Flatten, hai lớp Dense và lớp Softmax 4 lớp. Chỉ phần đầu mới được huấn luyện.

### Slide 12. Phần 3. CNN hướng B (fine-tune): là gì

*Phần 3, khoảng 35 giây.*

Hướng B dùng mạng lớn hơn, huấn luyện sẵn trên ImageNet rồi fine-tune. MobileNetV3 là họ mạng thiết kế cho thiết bị di động, có hai phiên bản Large và Small. ResNet dùng kết nối tắt để huấn luyện được mạng rất sâu. Fine-tune khác với đóng băng ở chỗ mình huấn luyện tiếp cả phần thân trên dữ liệu của mình, với learning rate nhỏ. Chi tiết kiến trúc bên trong của hai mạng này có trong bài báo gốc ở cuối slide, nhóm em chỉ sử dụng chúng làm phần thân rút đặc trưng.

### Slide 13. Phần 3. CNN hướng B: nhóm dùng như thế nào

*Phần 3, khoảng 55 giây.*

Nhóm em dùng phiên bản MobileNetV3-Small, embedding sau lớp GlobalAveragePooling có 576 chiều, và ResNet50 với embedding 2048 chiều. Ảnh vào 224 nhân 224. Mạng gốc phân loại 1000 lớp, nên nhóm em thay phần đầu bằng lớp Softmax 4 lớp. Huấn luyện gồm hai giai đoạn. Giai đoạn một đóng băng phần thân, chỉ huấn luyện lớp Softmax mới. Giai đoạn hai mở băng phần thân và huấn luyện tiếp với learning rate nhỏ, có tăng cường ảnh như lật, xoay, phóng to thu nhỏ. Một chi tiết nhóm em rút ra khi làm: khi fine-tune phải giữ các lớp BatchNorm đóng băng. Lần đầu nhóm em để mở, accuracy trên train tụt xuống còn khoảng 0.36. Hình bên phải là macro-F1 trên val theo epoch của giai đoạn một và giai đoạn hai với các learning rate khác nhau.

### Slide 14. Điểm khác biệt giữa các phương pháp

*Phần 3, khoảng 40 giây.*

Bảng này tóm tắt điểm khác biệt giữa các phương pháp. Histogram và moment màu là feature vector toàn cục, ít chiều, không giữ vị trí. HOG và vector cạnh là feature vector cục bộ, giữ được vị trí. Ba dòng cuối là embedding của CNN: mỗi giá trị tính trên một cửa sổ tích chập nên cũng là cục bộ. Khác nhau ở chỗ có học đặc trưng từ dữ liệu hay không: LeNet học từ đầu nên cần nhiều dữ liệu, VGG16 đóng băng thì không học lại, MobileNetV3 fine-tune thì chỉnh lại từ trọng số ImageNet. Từ bảng này nhóm em đặt ra các dự đoán và kiểm chứng ở phần thực nghiệm.

### Slide 15. Dataset

*Chung, khoảng 40 giây.*

Về dataset. Nhóm em dùng bộ BananaImageBD, là bộ dữ liệu có sẵn, công bố trên Mendeley Data. Có 2471 ảnh gốc kích thước 256 nhân 256 của bốn giống, trong đó Champa Kola nhiều nhất với 994 ảnh và Bangla Kola ít nhất với 444 ảnh. Nhóm em chia cố định thành train, val và test theo tỉ lệ 70, 15, 15, giữ nguyên tỉ lệ các lớp. Tập test chỉ dùng một lần ở cuối.

### Slide 16. Độ đo đánh giá

*Chung, khoảng 20 giây.*

Nhóm em sử dụng các độ đo cơ bản của bài toán phân loại đã học trong môn học: accuracy, precision, recall, F1 và ma trận nhầm lẫn. Chỉ số chính là macro-F1, vì dữ liệu lệch lớp.

### Slide 17. Kết quả trên tập test (371 ảnh)

*Chung, khoảng 50 giây.*

Đây là kết quả cuối trên tập test 371 ảnh. Hướng truyền thống tốt nhất là histogram H-S ghép HOG với Logistic Regression, macro-F1 1.000, đúng cả 371 ảnh. CNN hướng B với MobileNetV3 cũng đạt 1.000, đúng cả 371 ảnh. Trong hướng A, VGG16 đóng băng tốt nhất với 0.986, còn LeNet huấn luyện từ đầu đạt 0.978. Khoảng tin cậy của các mô hình dẫn đầu chồng lên nhau, nên nhóm em không kết luận mô hình nào hơn mô hình nào trong nhóm này.

### Slide 18. Kết quả: màu hay hình dạng?

*Chung, khoảng 40 giây.*

Một số kết quả kiểm chứng tính chất của đặc trưng. Khi xáo trộn ảnh thành lưới 4 nhân 4, histogram màu không đổi một chút nào, vì nó không giữ vị trí, còn HOG thì rơi về mức đoán mò. Khi chuyển ảnh sang xám, đặc trưng màu sụt, còn HOG vẫn đạt 0.986. Vậy bốn giống chuối phân biệt được bằng cả màu lẫn hình dạng, và ghép hai loại lại cho kết quả tốt nhất.

### Slide 19. Kết quả: CNN và lượng dữ liệu

*Chung, khoảng 45 giây.*

Với CNN, hình bên phải là macro-F1 theo lượng dữ liệu huấn luyện. Chỉ với 25 phần trăm dữ liệu, hai mô hình dùng trọng số ImageNet đã ngang LeNet dùng toàn bộ dữ liệu. Đây là lợi ích của transfer learning khi dữ liệu ít. MobileNetV3-Small vừa chính xác nhất vừa nhẹ: file chỉ 4.3 megabyte và dự đoán 12.5 mili giây mỗi ảnh trên CPU, trong khi VGG16 nặng 163 megabyte. Cũng có kết quả trái dự đoán: Dropout không giúp ở mô hình a_dropout. Nhóm em giải thích là 99 phần trăm tham số nằm ở lớp Dense đứng trước lớp Dropout, nên Dropout không chặn được quá khớp ở đó.

### Slide 20. Khác tài liệu tham khảo (1): mô hình nhìn quả hay nhìn nền?

*Chung, khoảng 50 giây.*

Phần này là thực nghiệm nhóm em tự thiết kế, không có trong tài liệu tham khảo. Kết quả gần 100 phần trăm làm nhóm em nghi ngờ: mô hình đang nhìn quả chuối hay nhìn nền? Nhóm em dùng các phương pháp phân đoạn đã học để tách ảnh thành phần chỉ có quả và phần chỉ có nền. Kết quả là chỉ dùng bốn góc ảnh, nơi gần như không có chuối, HOG vẫn đạt 0.968. CNN dự đoán trên ảnh đã xóa quả chuối vẫn đạt khoảng 0.7 đến 0.8, trong khi đoán mò chỉ là 0.25. Nghĩa là bộ dữ liệu có thiên lệch theo bối cảnh chụp: mỗi giống được chụp với nền và cỡ quả riêng, và mô hình học được cả điều đó. Đây là hạn chế quan trọng nhất mà nhóm em tìm ra.

### Slide 21. Khác tài liệu tham khảo (2): bộ kiểm tra khó, đối chiếu bài báo

*Chung, khoảng 50 giây.*

Vì trên ảnh gốc hai mô hình đã đúng hết, nhóm em tạo thêm bộ kiểm tra khó gồm sáu phép biến đổi trên tập test, trong đó có phép thay nền: ghép quả chuối lên nền của ảnh thuộc lớp khác. Với phép thay nền, MobileNetV3 giảm từ 1.000 xuống 0.794. Khi nhóm em thêm vào tập train các ảnh có nền thay ngẫu nhiên, con số này lên lại 0.975. Nhóm em cũng chạy lại phương pháp của bài báo đã công bố trên bộ dữ liệu này, là Vision Transformer kết hợp SVM, và đạt 99.49 phần trăm, gần với 99.70 phần trăm bài báo công bố. Tuy nhiên cách chia fold của bài báo để ảnh gốc và bản tăng cường của nó nằm ở cả hai phía, nên kết quả cao hơn cách chia đúng ở cả 19 trên 19 tổ hợp nhóm em thử. Vision Transformer nằm ngoài nội dung môn học, nhóm em chỉ dùng để đối chiếu.

### Slide 22. Chương trình demo

*Chung, khoảng 120 giây.*

Nhóm em xin chạy chương trình demo. Chương trình viết bằng Gradio. (Chuyển sang cửa sổ demo, làm theo kịch bản demo trong file kịch bản.)

### Slide 23. Nhận xét

*Chung, khoảng 45 giây.*

Về nhận xét. Ưu điểm: có hai mô hình đúng toàn bộ 371 ảnh test, trong đó MobileNetV3 rất nhẹ. Nhóm em so sánh mọi phương pháp trên cùng một quy trình đánh giá, và kiểm chứng tính chất của từng phương pháp bằng 30 thí nghiệm, ghi lại cả những kết quả trái dự đoán. Hạn chế: thứ nhất, dữ liệu thiên lệch theo nền, nên con số 100 phần trăm chỉ đúng trên bộ dữ liệu này, chưa chắc đúng với ảnh chụp ở nơi khác. Thứ hai, mỗi quả được chụp bốn mặt mà khi chia ngẫu nhiên thì ảnh của cùng một quả có thể nằm ở cả train và test, nên kết quả lạc quan hơn thực tế. Thứ ba, nhóm em chưa có ảnh chụp thật ngoài bộ dữ liệu để đánh giá, và chương trình chưa từ chối được ảnh ngoài bốn giống.

### Slide 24. Kết luận

*Chung, khoảng 35 giây.*

Kết luận. Nhóm em đã xây dựng được chương trình nhận diện bốn giống chuối, cả hướng truyền thống lẫn hướng học sâu đều đạt macro-F1 bằng 1 trên tập test. Nhóm em rút ra ba điều: phương pháp tốt khi tính chất của nó khớp với dữ liệu, cách đánh giá quan trọng không kém mô hình, và độ chính xác cao chưa chắc mô hình đã học đúng thứ cần học. Nhóm em đã rất cố gắng và dành nhiều thời gian cho đồ án này. Tuy nhiên vẫn còn những phần chưa đạt như mong muốn, như chưa đánh giá được trên ảnh chụp thật ngoài bộ dữ liệu và chưa xử lý được ảnh ngoài phạm vi bốn giống. Đó cũng là hướng phát triển tiếp theo.

### Slide 25. Tài liệu tham khảo

*Chung, khoảng 10 giây.*

Đây là các tài liệu tham khảo chính. Danh sách đầy đủ nằm trong báo cáo.

### Slide 26. Cảm ơn thầy và các bạn đã lắng nghe

*Chung, khoảng 10 giây.*

Phần trình bày của nhóm em đến đây là hết. Em cảm ơn thầy và các bạn đã lắng nghe. Nhóm em xin nhận câu hỏi và góp ý ạ.

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
