# Nhận diện giống chuối từ ảnh — Đồ án CS231

Nhận diện 4 giống chuối Bangladesh (Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola) từ ảnh. Đồ án thử tất cả các phương pháp đã học trong môn: đặc trưng màu, đặc trưng cạnh, HOG, KNN, bộ phân lớp tuyến tính, CNN (hướng theo bài giảng và hướng fine-tune mạng lớn hơn), xử lý mất cân bằng dữ liệu và phân đoạn tách nền. Các phương pháp được so sánh qua 30 thí nghiệm kiểm chứng.

- **Input:** ảnh chụp chuối. **Output:** giống chuối dự đoán và xác suất từng lớp.
- Kế hoạch thực nghiệm: [docs/ke-hoach-thuc-nghiem.md](docs/ke-hoach-thuc-nghiem.md)
- Nhật ký thực nghiệm (đã làm gì, vì sao, bằng chứng): [docs/nhat-ky-thuc-nghiem.md](docs/nhat-ky-thuc-nghiem.md)
- Báo cáo: [report/report.md](report/report.md) ([.docx](report/report.docx)) · Slide Beamer style UIT: [slides/bao-cao-do-an.pdf](slides/bao-cao-do-an.pdf) ([.tex](slides/bao-cao-do-an.tex), [bản kèm ghi chú](slides/bao-cao-do-an-ghi-chu.pdf), [.pptx](slides/bao-cao-do-an.pptx)) · Hỏi đáp khi bảo vệ: [docs/hoi-dap-bao-ve.md](docs/hoi-dap-bao-ve.md)
- Tổng hợp mọi kết quả: [results/summary.md](results/summary.md), [results/summary.csv](results/summary.csv)

## Kết quả chính

Đánh giá cuối trên tập test (371 ảnh, chỉ dùng một lần). Nhóm truyền thống học trên train+val và chọn siêu tham số bằng 5-fold CV; CNN học trên train và chọn theo val. Nguồn: [results/final/table.md](results/final/table.md).

<!-- T:results/final/table.md -->
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
<!-- /T -->

- Mô hình tốt nhất của nhóm truyền thống: **Hist H-S + HOG + Logistic Regression**. Hướng A: **`a_vgg16`**. Hướng B: **`b_mobilenetv3`** (4,3 MB, khoảng 12,5 ms/ảnh trên CPU).
- **Lưu ý quan trọng (thí nghiệm S2):** bộ dữ liệu có thiên lệch theo bối cảnh chụp. Chỉ dùng 4 góc ảnh 64×64, hầu như không chứa quả chuối, HOG + LogReg vẫn đạt macro-F1 0,968 (5-fold CV). Con số gần 100% là độ chính xác trên bộ dữ liệu này, không phải trong điều kiện chụp bất kỳ. Xem [results/S2/](results/S2/) và mục 5.10 của báo cáo.

## Phần mở rộng: ba nhánh so sánh độc lập với pipeline gốc

Ba nhánh, mỗi nhánh đổi một thứ so với pipeline gốc (G), cùng chấm trên 371 ảnh test gốc và trên **bộ kiểm tra khó** (6 phiên bản biến đổi của tập test: nền hoán đổi, V×0,6, V×1,4, nhiễu, xoay, che 25%; **điểm bền** là trung bình macro-F1 trên 6 phiên bản). Chi tiết: mục 12 của kế hoạch, mục 5.13 của báo cáo, nhật ký bước 10–14.

- **X — xử lý dữ liệu do nhóm thiết kế** (làm sạch, tiền xử lý, tăng cường): [results/E1_xu_ly/](results/E1_xu_ly/). X-a: nền phẳng + tăng cường hình học/quang học; X-b: tăng cường có thay nền ngẫu nhiên, không đổi ảnh lúc dự đoán.
- **T — bản Augmented của tác giả bộ dữ liệu** (7.413 ảnh, chia theo ảnh gốc để không rò rỉ): [results/E2_aug_tac_gia/](results/E2_aug_tac_gia/).
- **B — phương pháp của bài báo** (Ergün 2025: đặc trưng pretrained + SVM): [results/E3_bai_bao/](results/E3_bai_bao/). B1: CNN làm bộ rút đặc trưng (trong phạm vi môn học); B2: ViT như bài báo.

Cấu hình đại diện của mỗi nhánh (nguồn: [results/so_sanh_nhanh/table_stress.md](results/so_sanh_nhanh/table_stress.md)):

<!-- T:results/so_sanh_nhanh/table_stress.md -->
| Nhánh | Cấu hình đại diện | Ảnh gốc | Nền hoán đổi | V×0,6 | V×1,4 | Nhiễu σ=10 | Xoay +30° | Che 25% | Điểm bền |
|---|---|---|---|---|---|---|---|---|---|
| G — pipeline gốc | b_mobilenetv3 | 1,000 | 0,794 | 0,919 | 0,890 | 0,989 | 1,000 | 0,659 | 0,875 |
| X-a — xử lý dữ liệu: tiền xử lý lúc suy luận | b_mobilenetv3 | 0,995 | 0,995 | 0,943 | 0,915 | 0,988 | 0,998 | 0,450 | 0,881 |
| X-b — xử lý dữ liệu: tăng cường có thay nền | b_mobilenetv3 | 1,000 | 0,975 | 0,954 | 0,935 | 0,995 | 1,000 | 0,441 | 0,883 |
| T — Augmented của tác giả (không rò rỉ) | b_mobilenetv3 | 0,998 | 0,867 | 0,954 | 0,904 | 0,993 | 1,000 | 0,736 | 0,909 |
| B1 — bài báo, trong phạm vi môn học (CNN + bộ phân lớp) | MobileNetV3-Small + SVM tuyến tính | 0,993 | 0,742 | 0,965 | 0,856 | 0,897 | 0,981 | 0,575 | 0,836 |
| B2 — bài báo, đúng phương pháp (ViT + SVM) | ViT-B/16 + SVM | 0,997 | 0,857 | 0,990 | 0,978 | 0,986 | 0,978 | 0,728 | 0,919 |
<!-- /T -->

- Trên ảnh gốc, không nhánh nào hơn pipeline gốc (0–2 ảnh sai trên 371). Khác biệt nằm ở bộ kiểm tra khó: nhánh X gần như xóa được sự phụ thuộc vào nền; nhánh T bền đều hơn một chút; ViT + SVM của bài báo bền nhất nhưng nặng và nằm ngoài môn học.
- Tái hiện kết quả bài báo: ViT + SVM đạt 99,49% theo giao thức của bài báo (công bố: 99,70%); khi giữ các file cùng ảnh gốc trong một fold còn 99,20% ([results/E3_bai_bao/table_protocols.md](results/E3_bai_bao/table_protocols.md)).
- Hạn chế đo được thêm: mỗi quả được chụp nhiều mặt và cách chia ngẫu nhiên không tách theo quả, nên các con số chia ngẫu nhiên lạc quan khoảng 0,02–0,07 so với chia theo khối ảnh liên tiếp ([results/E1_xu_ly/table_blocked_cv.md](results/E1_xu_ly/table_blocked_cv.md)).

## Tải repo

```bash
git clone https://github.com/Rod-HD/banana-classification.git
```

Bản clone đầy đủ tải khoảng 1,5 GB (3,2 GB trên đĩa), phần lớn là trọng số CNN trong `checkpoints/`. Nếu chỉ cần code, kết quả, báo cáo, slide và mô hình demo (khoảng 120 MB), clone không kèm `checkpoints/` và `data/aug_author/`:

```bash
git clone --filter=blob:none --sparse https://github.com/Rod-HD/banana-classification.git
```

```bash
cd banana-classification
```

```bash
git sparse-checkout set --no-cone '/*' '!/checkpoints/' '!/data/aug_author/'
```

Với bản clone gọn, demo và mọi kết quả vẫn dùng được; muốn chạy lại các bước có CNN thì cần `checkpoints/` (hoặc huấn luyện lại bằng `RETRAIN=1`).

## Cài đặt

Cần Python 3.13 và kết nối mạng (lần đầu chạy, Keras và torchvision tải trọng số ImageNet của VGG16, ResNet50, MobileNetV3, SqueezeNet, ViT-B/16). Máy chỉ cần CPU. PyTorch chỉ dùng cho bước 12. Slide Beamer cần `pdflatex` (texlive có gói `vntex`); script cài đặt thử cài bằng `apt-get`, nếu không có thì bước dựng slide chỉ sinh file `.tex` và `.pptx`.

```bash
bash scripts/setup_env.sh
```

Lệnh trên cài đúng các phiên bản trong `requirements.txt`, tức các phiên bản đã dùng để sinh `results/` (`results/environment.txt`). Ảnh gốc đã có sẵn trong `data/raw/`. Để tải lại từ nguồn và kiểm tra mã băm:

```bash
bash scripts/download_data.sh
```

## Những gì có sẵn trong repo (không cần chạy lại)

| Thứ có sẵn | Vị trí | Ghi chú |
|---|---|---|
| Kết quả mọi thí nghiệm | `results/` | bảng, hình, `metrics.json` của từng thí nghiệm |
| Trọng số của 54 lần huấn luyện CNN | `checkpoints/<run>/best.keras` | kiến trúc + trọng số tốt nhất theo macro-F1 trên val; `state.json` là lịch sử huấn luyện |
| Mô hình cho demo | `models/` | chạy `python app/app.py` được ngay |
| Bản Augmented của tác giả | `data/aug_author/` | 7.413 ảnh, dùng ở bước 11–12 |
| Log của lần chạy đã sinh ra `results/` | `logs/` | |
| Báo cáo, slide, nhật ký, hỏi đáp | `report/`, `slides/`, `docs/` | |

Kiểm tra trọng số trong repo cho lại đúng kết quả đã ghi (nạp từng `best.keras`, dự đoán lại, so với `metrics.json`; khoảng 5 phút):

```bash
python -m src.cnn.checkpoints
```

Kết quả của lần kiểm tra gần nhất: [results/cnn/table_checkpoints.md](results/cnn/table_checkpoints.md).

## Chạy lại toàn bộ thí nghiệm

```bash
bash scripts/run_all.sh
```

Lệnh này chạy lần lượt các bước 1–8 theo mục 11 của kế hoạch, phần mở rộng (bước 10–14), rồi bước 9 (mô hình demo, báo cáo, slide); ghi lại mọi bảng, hình, mô hình, báo cáo và slide. Vì trọng số đã có trong `checkpoints/`, các bước huấn luyện CNN không huấn luyện lại: mỗi lệnh nạp trọng số đã lưu rồi chấm lại. Muốn huấn luyện lại từ đầu:

```bash
RETRAIN=1 bash scripts/run_all.sh
```

Khi huấn luyện lại từ đầu, tổng cộng khoảng 8 giờ trên CPU 4 nhân (bước 1–9: khoảng 3 giờ; phần mở rộng: khoảng 5 giờ); cột thời gian trong bảng dưới là của trường hợp này. Có thể chạy từng bước, ví dụ:

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
| 10 | Bộ kiểm tra khó; nhánh G chấm lại | `src.stress`, `src.pipeline` | khoảng 10 phút |
| 11 | Nhánh T: bản Augmented của tác giả (có sẵn trong `data/aug_author/`) | `src.exp_arm_t` | khoảng 100 phút |
| 12 | Nhánh B: phương pháp của bài báo | `src.exp_papers` | khoảng 45 phút |
| 13 | Nhánh X: làm sạch, tiền xử lý, tăng cường; S2 chạy lại | `src.exp_arm_x`, `src.exp_segment --only S2V2` | khoảng 130 phút |
| 14 | So sánh bốn nhánh, tổng hợp | `src.compare_arms`, `src.summarize` | khoảng 1 phút |
| 9 | Mô hình demo, báo cáo (md + docx), slide (Beamer + pptx); chạy sau cùng | `src.export_demo`, `src.build_report`, `src.make_slides` | khoảng 2 phút |

Ghi chú:

- Seed cố định (42) cho `random`, `numpy`, TensorFlow; TensorFlow bật chế độ tính toán tất định. Bước 1–3 (truyền thống) cho lại đúng các con số đã commit. CNN huấn luyện trên máy hoặc phiên bản thư viện khác có thể lệch nhẹ ở chữ số cuối.
- Bảng trong báo cáo, slide và README được sinh lại từ `results/` ở bước 9 (`python -m src.build_report`, `python -m src.make_slides`), nên sau khi chạy lại thí nghiệm thì tài liệu tự khớp với số liệu mới.
- Đặc trưng đã rút và đầu ra của các thân CNN đóng băng được lưu ở `data/cache/`. Thư mục này không commit (khoảng 8 GB, có file trên 1 GB) và tự sinh lại khi chạy, không cần huấn luyện. `FRESH=1 bash scripts/run_all.sh` xóa cache trước khi chạy.
- Huấn luyện CNN lưu checkpoint mỗi epoch vào `checkpoints/<run>/`. Nếu bị ngắt, chạy lại đúng lệnh cũ sẽ tiếp tục từ epoch dở dang (`--resume`); run đã xong thì dùng lại `best.keras`. Log nằm trong `logs/`.
- Bước 11 và 13 mặc định tính lại mọi cấu hình của nhánh. `SKIP_DONE=1 bash scripts/run_all.sh 11 13` bỏ qua cấu hình đã có kết quả, dùng khi chạy tiếp một lần chạy bị ngắt.
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

Mô hình nằm trong `models/` (`b_mobilenetv3`; `b_mobilenetv3_thay_nen` là bản huấn luyện có thay nền ngẫu nhiên của nhánh X-b, bền hơn khi nền ảnh khác bộ dữ liệu; `a_lenet`; cấu hình truyền thống tốt nhất; mỗi file dưới 50 MB). `app/variety_info.json` chỉ ghi thông tin có nguồn dẫn, chỗ chưa có nguồn để `"TODO"`. Mô hình chỉ học ảnh một quả chuối trên nền trơn của 4 giống; với ảnh khác bối cảnh, kết quả không đáng tin.

## Dữ liệu

| Lớp | Số ảnh gốc | train | val | test |
|---|---|---|---|---|
| Bangla Kola | 444 | 311 | 66 | 67 |
| Champa Kola | 994 | 695 | 150 | 149 |
| Sabri Kola | 509 | 356 | 76 | 77 |
| Sagor Kola | 524 | 367 | 79 | 78 |
| **Tổng** | **2.471** | **1.729** | **371** | **371** |

Ảnh nằm trong `data/raw/<lớp>/`; danh sách chia tập cố định (stratified 70/15/15, seed 42) nằm trong `data/splits/`. Tên thư mục lớp đã được đổi sang dạng `snake_case` so với bản gốc; nội dung ảnh giữ nguyên.

**Nguồn và giấy phép.** Dữ liệu thuộc bộ *BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh* (Ferdaus và cộng sự, Mendeley Data, Version 2, DOI: [10.17632/ptfscwtnyz.2](https://doi.org/10.17632/ptfscwtnyz.2); bài báo mô tả: *Data in Brief* 58, 111239), phát hành theo giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Repo chứa phần ảnh gốc phân loại giống chuối (`data/raw/`) và bản Augmented của tác giả dùng ở bước 11–12 (`data/aug_author/`, 7.413 ảnh, cùng nguồn và giấy phép); tải lại bản Augmented từ nguồn và kiểm tra mã băm bằng `bash scripts/download_data.sh augmented`. Logo trong `slides/figures/uit-logo.png` là logo của Trường ĐH Công nghệ Thông tin, lấy từ trang uit.edu.vn, chỉ dùng cho slide của đồ án. Bốn ảnh trong `app/examples/` được sao từ tập test của bộ dữ liệu này.

## Cấu trúc thư mục

```
data/raw/         ảnh gốc theo lớp
data/splits/      danh sách train/val/test cố định
data/aug_author/  bản Augmented của tác giả (7.413 ảnh)
data/cache/       đặc trưng đã rút (không commit; tự sinh lại khi chạy)
src/              code: rút đặc trưng, huấn luyện, đánh giá, các thí nghiệm (python -m src.<module>)
src/features/     đặc trưng màu, cạnh, HOG
src/cnn/          kiến trúc, huấn luyện, dự đoán CNN
results/          kết quả từng thí nghiệm (results/<mã>/), hình (results/figures/), tổng hợp (summary.*)
models/           mô hình dùng cho demo
checkpoints/      trọng số của từng lần huấn luyện CNN: <run>/best.keras, <run>/state.json
logs/             log của các lần chạy
app/              demo (app.py), thông tin giống chuối, ảnh ví dụ
report/           báo cáo (report_template.md → report.md, report.docx)
slides/           slide trình bày: Beamer LaTeX style UIT (.tex, .pdf, theme beamerthemeUIT.sty) và bản .pptx
docs/             kế hoạch, nhật ký thực nghiệm, hỏi đáp khi bảo vệ
scripts/          cài môi trường, tải dữ liệu, chạy toàn bộ thí nghiệm
```
