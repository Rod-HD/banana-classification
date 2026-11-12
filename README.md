# Nhận diện giống chuối từ ảnh — Đồ án CS231

Nhận diện 4 giống chuối Bangladesh (Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola) từ ảnh. Đồ án thử tất cả các phương pháp đã học trong môn: đặc trưng màu, đặc trưng cạnh, HOG, KNN, bộ phân lớp tuyến tính, CNN (hướng theo bài giảng và hướng fine-tune mạng lớn hơn), xử lý mất cân bằng dữ liệu và phân đoạn tách nền. Các phương pháp được so sánh qua 30 thí nghiệm kiểm chứng.

- **Input:** ảnh chụp chuối. **Output:** giống chuối dự đoán và xác suất từng lớp.
- Kế hoạch thực nghiệm: [docs/ke-hoach-thuc-nghiem.md](docs/ke-hoach-thuc-nghiem.md)
- Nhật ký thực nghiệm (đã làm gì, vì sao, bằng chứng): [docs/nhat-ky-thuc-nghiem.md](docs/nhat-ky-thuc-nghiem.md)
- Báo cáo: [report/report.md](report/report.md) ([.docx](report/report.docx)) · Slide: [slides/bao-cao-do-an.pptx](slides/bao-cao-do-an.pptx) · Hỏi đáp khi bảo vệ: [docs/hoi-dap-bao-ve.md](docs/hoi-dap-bao-ve.md)
- Tổng hợp mọi kết quả: [results/summary.md](results/summary.md), [results/summary.csv](results/summary.csv)

## Kết quả chính

Đánh giá cuối trên tập test (371 ảnh, chỉ dùng một lần). Nhóm truyền thống học trên train+val và chọn siêu tham số bằng 5-fold CV; CNN học trên train và chọn theo val. Nguồn: [results/final/table.md](results/final/table.md).

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

- Mô hình tốt nhất của nhóm truyền thống: **Hist H-S + HOG + Logistic Regression**. Hướng A: **`a_vgg16`**. Hướng B: **`b_mobilenetv3`** (11,7 MB, khoảng 12,5 ms/ảnh trên CPU).
- **Lưu ý quan trọng (thí nghiệm S2):** bộ dữ liệu có thiên lệch theo bối cảnh chụp. Chỉ dùng 4 góc ảnh 64×64, hầu như không chứa quả chuối, HOG + LogReg vẫn đạt macro-F1 0,968 (5-fold CV). Con số gần 100% là độ chính xác trên bộ dữ liệu này, không phải trong điều kiện chụp bất kỳ. Xem [results/S2/](results/S2/) và mục 5.10 của báo cáo.

## Cài đặt

Cần Python 3.13 và kết nối mạng (lần đầu chạy CNN, Keras tải trọng số ImageNet của VGG16, ResNet50, MobileNetV3). Máy chỉ cần CPU.

```bash
bash scripts/setup_env.sh
```

Lệnh trên cài đúng các phiên bản trong `requirements.txt`, tức các phiên bản đã dùng để sinh `results/` (`results/environment.txt`). Ảnh gốc đã có sẵn trong `data/raw/`. Để tải lại từ nguồn và kiểm tra mã băm:

```bash
bash scripts/download_data.sh
```

## Chạy lại toàn bộ thí nghiệm

```bash
bash scripts/run_all.sh
```

Lệnh này chạy lần lượt 9 bước theo mục 11 của kế hoạch và ghi lại mọi bảng, hình, mô hình, báo cáo và slide; tổng cộng khoảng 3 giờ trên CPU 4 nhân. Có thể chạy từng bước, ví dụ:

```bash
bash scripts/run_all.sh 2 3
```

| Bước | Nội dung | Module chính | Thời gian ước lượng (CPU 4 nhân) |
|---|---|---|---|
| 1 | Chia dữ liệu, ảnh gần trùng, D1 | `src.split_data`, `src.exp_d1` | khoảng 2 phút |
| 2 | Đặc trưng màu M1–M6, bộ phân lớp P1–P6 | `src.exp_color`, `src.exp_classifier` | khoảng 6 phút |
| 3 | Cạnh, HOG C1–C5, chốt cấu hình truyền thống | `src.exp_shape`, `src.exp_select` | khoảng 25 phút |
| 4 | CNN hướng A | `src.cnn.train` | khoảng 20 phút |
| 5 | CNN hướng B, N1–N8 | `src.cnn.train`, `src.exp_cnn` | khoảng 70 phút |
| 6 | Mất cân bằng I1–I2 | `src.cnn.train`, `src.exp_imbalance` | khoảng 40 phút |
| 7 | Phân đoạn S1–S2 | `src.exp_segment` | khoảng 10 phút |
| 8 | Đánh giá cuối trên test, phân tích lỗi, tổng hợp | `src.final_eval`, `src.error_analysis`, `src.summarize` | khoảng 2 phút |
| 9 | Mô hình demo, báo cáo (md + docx), slide | `src.export_demo`, `src.build_report`, `src.make_slides` | khoảng 1 phút |

Ghi chú:

- Seed cố định (42) cho `random`, `numpy`, TensorFlow; TensorFlow bật chế độ tính toán tất định. Bước 1–3 (truyền thống) cho lại đúng các con số đã commit. CNN huấn luyện trên máy hoặc phiên bản thư viện khác có thể lệch nhẹ ở chữ số cuối.
- Đặc trưng đã rút và đầu ra của các thân CNN đóng băng được lưu ở `data/cache/` (không commit). `FRESH=1 bash scripts/run_all.sh` xóa cache trước khi chạy.
- Huấn luyện CNN lưu checkpoint mỗi epoch vào `checkpoints/<run>/`. Nếu bị ngắt, chạy lại đúng lệnh cũ sẽ tiếp tục từ epoch dở dang (`--resume`). Log nằm trong `logs/`.
- Mỗi module có hướng dẫn riêng: `python -m src.<module> --help`. Ví dụ huấn luyện một CNN:

```bash
python -m src.cnn.train --model b_mobilenetv3 --stage 2 --lr 1e-4 --augment --tag lr1e-4
```

## Demo

```bash
python app/app.py
```

Lệnh trên mở giao diện web tại http://127.0.0.1:7860: tải ảnh lên, chọn mô hình, nhận giống dự đoán và xác suất 4 lớp. Dự đoán một ảnh trên dòng lệnh:

```bash
python app/app.py --image app/examples/sagor_kola_sagor_kola_110.jpg --model b_mobilenetv3
```

Mô hình nằm trong `models/` (`b_mobilenetv3`, `a_lenet`, cấu hình truyền thống tốt nhất; mỗi file dưới 50 MB). `app/variety_info.json` chỉ ghi thông tin có nguồn dẫn, chỗ chưa có nguồn để `"TODO"`. Mô hình chỉ học ảnh một quả chuối trên nền trơn của 4 giống; với ảnh khác bối cảnh, kết quả không đáng tin.

## Dữ liệu

| Lớp | Số ảnh gốc | train | val | test |
|---|---|---|---|---|
| Bangla Kola | 444 | 311 | 66 | 67 |
| Champa Kola | 994 | 695 | 150 | 149 |
| Sabri Kola | 509 | 356 | 76 | 77 |
| Sagor Kola | 524 | 367 | 79 | 78 |
| **Tổng** | **2.471** | **1.729** | **371** | **371** |

Ảnh nằm trong `data/raw/<lớp>/`; danh sách chia tập cố định (stratified 70/15/15, seed 42) nằm trong `data/splits/`. Tên thư mục lớp đã được đổi sang dạng `snake_case` so với bản gốc; nội dung ảnh giữ nguyên.

**Nguồn và giấy phép.** Dữ liệu thuộc bộ *BananaImageBD* (Mendeley Data, Version 2, DOI: [10.17632/ptfscwtnyz.2](https://doi.org/10.17632/ptfscwtnyz.2)) của nhóm tác giả East West University, phát hành theo giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Repo này chỉ dùng phần ảnh gốc phân loại giống chuối. Bốn ảnh trong `app/examples/` được sao từ tập test của bộ dữ liệu này.

## Cấu trúc thư mục

```
data/raw/         ảnh gốc theo lớp
data/splits/      danh sách train/val/test cố định
src/              code: rút đặc trưng, huấn luyện, đánh giá, các thí nghiệm (python -m src.<module>)
src/features/     đặc trưng màu, cạnh, HOG
src/cnn/          kiến trúc, huấn luyện, dự đoán CNN
results/          kết quả từng thí nghiệm (results/<mã>/), hình (results/figures/), tổng hợp (summary.*)
models/           mô hình dùng cho demo
app/              demo (app.py), thông tin giống chuối, ảnh ví dụ
report/           báo cáo (report_template.md → report.md, report.docx)
slides/           slide trình bày
docs/             kế hoạch, nhật ký thực nghiệm, hỏi đáp khi bảo vệ
scripts/          cài môi trường, tải dữ liệu, chạy toàn bộ thí nghiệm
```
