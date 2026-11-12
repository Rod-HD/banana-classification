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

---

## Bảng quyết định

| Quyết định | Phương án thay thế | Bằng chứng | Nhãn |
|---|---|---|---|
| Dùng ảnh gốc Version 2 | Bản Augmented; Version 1 | Bước 0.1; chờ D1 | [lập luận] |
| TensorFlow/Keras cho CNN | PyTorch | Bước 0.2 | [lập luận] |
| Chia 70/15/15, test dùng một lần | Chỉ train/test; cross-validation cho CNN | Bước 0.3 | [lập luận] |
| macro-F1 là chỉ số chính | Accuracy | Bước 0.3; chờ I1 | [lập luận] |
