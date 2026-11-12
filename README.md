# Nhận diện giống chuối từ ảnh — Đồ án CS231

Mô hình thị giác máy tính tự động nhận diện 4 giống chuối Bangladesh (Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola) từ hình ảnh. Đồ án so sánh các phương pháp đã học trong môn: đặc trưng màu, đặc trưng cạnh, HOG, KNN, bộ phân lớp tuyến tính và CNN, có thử nghiệm xử lý mất cân bằng dữ liệu.

- **Input:** ảnh chụp chuối.
- **Output:** giống chuối dự đoán và xác suất từng lớp.

- Kế hoạch thực nghiệm: [docs/ke-hoach-thuc-nghiem.md](docs/ke-hoach-thuc-nghiem.md)
- Nhật ký thực nghiệm (đã làm gì, vì sao, bằng chứng): [docs/nhat-ky-thuc-nghiem.md](docs/nhat-ky-thuc-nghiem.md)

> Trạng thái: đang xây dựng. Bảng kết quả và hướng dẫn chạy sẽ được cập nhật khi các thí nghiệm hoàn tất.

## Dữ liệu

| Lớp | Số ảnh gốc |
|---|---|
| Bangla Kola | 444 |
| Champa Kola | 994 |
| Sabri Kola | 509 |
| Sagor Kola | 524 |
| **Tổng** | **2.471** |

Ảnh nằm trong `data/raw/<lớp>/`. Tên thư mục lớp đã được đổi sang dạng `snake_case` so với bản gốc; nội dung ảnh giữ nguyên.

**Nguồn và giấy phép.** Dữ liệu thuộc bộ *BananaImageBD* của Ankan Chandra Das, Mohammad Manzurul Islam, Mahamudul Hasan và Md Sawkat Ali, phát hành trên Mendeley Data (Version 2, DOI: [10.17632/ptfscwtnyz.2](https://doi.org/10.17632/ptfscwtnyz.2)) theo giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Repo này chỉ dùng phần ảnh gốc phân loại giống chuối.

Tải lại dữ liệu từ nguồn gốc:

```bash
bash scripts/download_data.sh
```

## Cấu trúc thư mục

```
data/raw/        ảnh gốc theo lớp
data/splits/     danh sách train/val/test cố định
docs/            kế hoạch và nhật ký thực nghiệm
src/             code rút đặc trưng, huấn luyện, đánh giá
results/         chỉ số, bảng tổng hợp và hình của từng thí nghiệm
models/          mô hình dùng cho demo
app/             demo tải ảnh và dự đoán
report/          báo cáo
slides/          slide trình bày
scripts/         script tải dữ liệu, cài môi trường
```

## Cài đặt

```bash
pip install -r requirements.txt
```
