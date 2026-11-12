"""Dựng slide trình bày (có ghi chú thuyết trình cho từng slide) từ một danh sách slide duy nhất:
  slides/bao-cao-do-an.tex + .pdf   bản chính: Beamer LaTeX, style UIT (kèm bản -ghi-chu.pdf có trang ghi chú)
  slides/bao-cao-do-an.pptx         bản PowerPoint cùng nội dung

    python -m src.make_slides
Bảng trên slide được đọc trực tiếp từ results/; hình lấy từ results/figures/.
"""
from __future__ import annotations

import argparse
import json

from .config import RESULTS_DIR, ROOT
from .pptx_builder import build

F = RESULTS_DIR / "figures"
R = RESULTS_DIR


def j(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def slides():
    fin = {k: j(f"final/{k}/metrics.json") for k in ["trad_fusion", "a_vgg16", "b_mobilenetv3"]}
    s = []
    s.append({"layout": "title", "title": "Nhận diện giống chuối từ ảnh",
              "bullets": ["Đồ án môn CS231 — Nhập môn Thị giác máy tính", "So sánh đặc trưng màu, cạnh, HOG, KNN, bộ phân lớp tuyến tính và CNN",
                          "Mã nguồn: github.com/Rod-HD/banana-classification"],
              "notes": "Chào thầy/cô và các bạn. Nhóm trình bày đồ án nhận diện 4 giống chuối Bangladesh từ ảnh. "
                       "Mục tiêu không chỉ là có mô hình chính xác mà là thử tất cả phương pháp đã học, so sánh bằng thực nghiệm "
                       "và giải thích kết quả bằng tính chất của từng phương pháp."})
    s.append({"title": "Thành viên và mức độ hoàn thành",
              "table": [["STT", "Họ và tên", "MSSV", "Công việc", "Mức độ hoàn thành"]] + [[str(i), "", "", "", ""] for i in range(1, 5)],
              "notes": "Phần này nhóm tự điền: thành viên, công việc phụ trách và mức độ hoàn thành."})
    s.append({"title": "Lý do chọn đề tài",
              "bullets": ["Phân loại giống chuối ở chợ đầu mối, cửa hàng hiện làm bằng mắt",
                          "BananaImageBD: ảnh thu thập tại chợ đầu mối và cửa hàng ở Bangladesh, hướng tới nông nghiệp chính xác và chuỗi cung ứng",
                          "Phù hợp để so sánh các phương pháp đã học:",
                          ("4 giống khác nhau cả về màu lẫn hình dạng", 1),
                          ("Dữ liệu nhỏ (2.471 ảnh) và mất cân bằng (2,2 lần)", 1),
                          ("Ảnh một quả trên nền trơn: dùng được phân đoạn tách nền", 1)],
              "notes": "Bài toán có ý nghĩa thực tế và đồng thời là bài toán tốt để kiểm chứng các tính chất đã học: "
                       "màu so với hình dạng, quá khớp khi dữ liệu ít, transfer learning, mất cân bằng dữ liệu."})
    s.append({"title": "Phát biểu bài toán",
              "bullets": ["**Input:** một ảnh màu chụp quả hoặc nải chuối",
                          "**Output:** 1 trong 4 giống (Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola) và xác suất từng lớp",
                          "**Ngoài phạm vi:** dinh dưỡng, giống ngoài 4 lớp, độ chín"],
              "image": F / "data_samples.png",
              "notes": "Input là ảnh màu, output là tên giống cùng xác suất 4 lớp. Hình bên phải là ảnh mẫu của 4 giống. "
                       "Có thể thấy Sabri Kola thường là quả nhỏ trên nền xanh nhạt; chi tiết này quan trọng ở phần S2."})
    s.append({"title": "Dữ liệu và quy trình đánh giá",
              "bullets": ["BananaImageBD v2 (Mendeley Data, CC BY 4.0): 2.471 ảnh gốc",
                          "Chia stratified 70/15/15 cố định, seed 42: 1.729 / 371 / 371",
                          "Truyền thống: 5-fold CV trên train+val; CNN: chọn theo val",
                          "**Test chỉ dùng một lần ở cuối**",
                          "Chỉ số chính: macro-F1 (dữ liệu mất cân bằng)",
                          "Ảnh gần trùng (pHash, Hamming ≤ 2): 42 cặp; ảnh hưởng không đo được"],
              "image": F / "data_split_counts.png",
              "notes": "Quy trình đánh giá chốt từ đầu để mọi phương pháp so sánh được với nhau. Test chỉ dùng một lần. "
                       "Chênh lệch nhỏ hơn độ lệch chuẩn giữa các fold thì nhóm kết luận là chưa đủ cơ sở phân biệt. "
                       "Bảng ảnh gần trùng ở results/data/near_duplicates.json."})
    s.append({"title": "Phương pháp — hướng truyền thống",
              "bullets": ["**Đặc trưng màu:** histogram RGB, HSV (8,8,8), H-S; moment màu (9 chiều); màu chủ đạo k-means; CCV",
                          "**Đặc trưng cạnh:** Roberts, Prewitt, Sobel, Laplacian → tổng theo hàng/cột (512 chiều)",
                          "**HOG** tự cài đặt: 64×128, cell 8×8, 9 bin, block 16×16 L2 → 3.780 chiều (lệch skimage 2·10⁻⁷)",
                          "**Ghép:** chuẩn hóa từng phần rồi ghép",
                          "**Bộ phân lớp:** KNN (k, độ đo, uniform/distance, brute/kd/ball tree), Logistic Regression; Linear SVM làm bộ phân lớp bổ sung"],
              "notes": "Đây là tất cả đặc trưng và bộ phân lớp đã học. HOG được nhóm tự cài đặt theo các bước của bài giảng để có thể bỏ bước "
                       "chuẩn hóa block cho thí nghiệm C2, và đã đối chiếu với skimage."})
    s.append({"title": "Phương pháp — CNN hai hướng",
              "table": [["Hướng", "Mô hình", "Mô tả"],
                        ["A (bài giảng)", "a_lenet", "Kiểu LeNet-5, ảnh 64×64, học từ đầu"],
                        ["A (bài giảng)", "a_dropout", "2 Conv + Dense(128) + Dropout(0,5), ảnh 128×128"],
                        ["A (bài giảng)", "a_vgg16", "VGG16 ImageNet đóng băng + Dense(1024) + Dense(256)"],
                        ["B (lớn hơn)", "b_mobilenetv3", "MobileNetV3-Small: GĐ1 đóng băng, GĐ2 fine-tune + tăng cường"],
                        ["B (lớn hơn)", "b_resnet50", "ResNet50: GĐ1 đóng băng, GĐ2 mở khối conv5"]],
              "notes": "Hướng A làm đúng các ví dụ trong bài giảng. Hướng B dùng kiến trúc ngoài bài giảng nhưng vẫn là khái niệm đã học: "
                       "transfer learning. Chung: Adam, cross-entropy, dừng sớm theo macro-F1 trên val. "
                       "Khi fine-tune, BatchNorm phải giữ đóng băng (slide N4)."})
    s.append({"title": "D1 — Rò rỉ khi tăng cường ảnh trước khi chia", "image": F / "D1_leakage.png",
              "bullets": ["Histogram HSV + KNN: thổi phồng +0,021", "HOG: ≈ 0 (trái dự đoán)", "Rò rỉ chỉ hại khi đặc trưng bất biến với phép tăng cường"],
              "notes": "Dự đoán: tăng cường rồi mới chia làm bản sao của ảnh test nằm trong train nên kết quả bị thổi phồng. "
                       "Đúng với histogram vì histogram gần như không đổi khi lật/xoay. Với HOG mức thổi phồng gần 0 vì bản lật có vector khác hẳn. "
                       "Đây là lý do không dùng bản Augmented có sẵn. Bảng: results/D1/table.md."})
    s.append({"title": "M1, M2 — Histogram bỏ vị trí; HSV tách độ sáng", "image": F / "M1_shuffle.png",
              "bullets": ["M1: xáo trộn 4×4 → histogram sụt đúng 0,000; HOG, cạnh về mức đoán mò",
                          "M2: nhân V×0,6 → Hist RGB sụt 0,76–0,87, Hist HSV sụt 0,66–0,67, **Hist H-S sụt ≤ 0,027**",
                          "Tách V thành một trục chưa đủ, phải bỏ V khỏi vector"],
              "notes": "M1 khớp hoàn toàn: histogram không chứa thông tin vị trí. M2 khớp một phần: HSV đủ 3 kênh vẫn chứa V nên vẫn sụt mạnh; "
                       "chỉ H-S bền. Bảng: results/M1/table.md, results/M2/table.md."})
    s.append({"title": "M3–M6 — Số bin, moment, CCV, ảnh xám", "image": F / "M6_grayscale.png",
              "bullets": ["M3: 4 bin kém rõ; 16³ bin làm KNN giảm (87% phần tử = 0)",
                          "M4: moment 9 chiều: KNN 0,972 nhưng LogReg 0,908",
                          "M5: CCV nhạy bố cục nhưng không tốt hơn histogram",
                          "M6: học lại trên ảnh xám → màu sụt 0,04–0,23; HOG vẫn 0,986"],
              "notes": "Hình là M6. Giống chuối phân biệt được cả bằng màu lẫn hình dạng. Moment màu ít chiều nên biên quyết định tuyến tính không đủ "
                       "(liên hệ P6). CCV không giúp vì ảnh một quả nền trơn, hầu như mọi điểm ảnh đều thuộc vùng liên thông lớn."})
    s.append({"title": "C1, C2 — Toán tử cạnh và HOG", "image": F / "C1_edge_noise.png",
              "bullets": ["C1: σ = 5 → vector đổi 43–46% (Prewitt/Sobel), 100% (Roberts), 216% (Laplacian): đúng thứ tự dự đoán",
                          "C2: chuẩn hóa block bền khi làm tối (sụt ≤ 0,009)",
                          "Làm sáng V×1,4: **trái dự đoán** — 68,7% điểm ảnh bão hòa, 14,1% cell mất gradient"],
              "notes": "C1 khớp lý thuyết: toán tử có làm trơn bền hơn, đạo hàm bậc hai nhạy nhất. C2: phép làm sáng gây bão hòa, không còn là "
                       "phép nhân tuyến tính nên chuẩn hóa không khử được. Bảng: results/C1, results/C2."})
    s.append({"title": "C3–C5 — Tham số HOG, che 25%, ghép đặc trưng", "image": F / "C5_fusion.png",
              "bullets": ["C3: chỉ cell/kích thước ảnh tạo khác biệt; số bin không",
                          "C4: histogram sập khi che — do **z-score**; Hellinger chỉ sụt 0,012",
                          "C5: H-S + HOG: 0,996 ± 0,003 (màu riêng 0,991, HOG riêng 0,986)"],
              "notes": "C4 là ví dụ cho thấy kết luận phụ thuộc cả bước chuẩn hóa chứ không chỉ đặc trưng. C5: ghép màu và hình dạng tốt nhất, "
                       "nhưng hơn màu riêng chỉ 0,005, bằng độ lệch chuẩn."})
    s.append({"title": "P1–P3 — KNN: k, độ đo, cách bình chọn", "image": F / "P1_knn_k.png",
              "bullets": ["P1: k = 1 tốt nhất, hơn quy tắc k = 5 từ 0,006–0,020",
                          "P2: chuẩn hóa đơn vị → Euclidean và cosine cùng 10 láng giềng ở 100% ảnh",
                          "P3: uniform/distance chênh ≤ 0,006 → chưa đủ cơ sở phân biệt"],
              "notes": "P1 trái với vế 'k nhỏ nhạy nhiễu' vì các lớp tách nhau rõ và có nhiều ảnh gần giống nhau. "
                       "P2 kiểm chứng đúng điều lý thuyết dự đoán: trên vector đơn vị, khoảng cách Euclidean là hàm đơn điệu của cosine."})
    s.append({"title": "P4–P6 — Tốc độ và biên quyết định", "image": F / "P6_linear_vs_knn.png",
              "bullets": ["P4: K-D Tree chậm hơn brute ~60 lần ở 3.780 chiều; chỉ nhanh hơn khi ít chiều và ≥ 8.400 điểm",
                          "P5: KNN huấn luyện 0,07 s nhưng dự đoán 0,22 ms/ảnh; LogReg ngược lại",
                          "P6: 9 chiều: LogReg kém KNN 0,064; 3.780 chiều: hơn 0,039"],
              "notes": "Kết quả của cây tìm kiếm luôn trùng brute force, chỉ khác tốc độ. Bộ phân lớp tuyến tính cần nhiều chiều để các lớp tách được."})
    s.append({"title": "Cấu hình truyền thống đã chốt (5-fold CV)", "table": R / "select_traditional" / "table_selected.md",
              "notes": "Sau khi quét siêu tham số: cấu hình ghép H-S + HOG + Logistic Regression tốt nhất, macro-F1 0,996 ± 0,003."})
    s.append({"title": "Hướng A — N1, N2: tham số và lượng dữ liệu", "image": F / "N2_learning_curve.png",
              "bullets": ["N1: tính tay khớp 10/10 lớp; ví dụ conv2 của a_lenet: (5×5×6+1)×16 = 2.416",
                          "Lớp Dense sau Flatten chiếm 96–99% số tham số",
                          "N2: 25% → 100% dữ liệu: a_lenet +0,050; pretrained +0,016–0,018"],
              "notes": "Ví dụ tính tay số tham số. Đường cong học cho thấy mô hình học từ đầu cần nhiều dữ liệu hơn; với 25% dữ liệu, "
                       "mô hình pretrained đã ngang a_lenet dùng đủ dữ liệu."})
    s.append({"title": "Hướng A — N3, N5: Dropout và tăng cường ảnh", "image": F / "N5_augment_a_dropout.png",
              "bullets": ["N3 (3 seed): có Dropout 0,941 ± 0,005; không Dropout 0,953 ± 0,006 → **trái dự đoán**",
                          "N5: tăng cường ảnh: khoảng cách loss val − train 0,542 → 0,054",
                          "Quá khớp nằm ở lớp Dense trước Dropout"],
              "notes": "Dropout(0,5) đặt sau lớp Dense chứa 99% tham số nên không chặn được quá khớp ở đó. Tăng cường ảnh tác động vào đầu vào nên hiệu quả hơn nhiều."})
    s.append({"title": "Hướng B — N4: đóng băng và fine-tune", "image": F / "N4_finetune_lr.png",
              "bullets": ["MobileNetV3: GĐ1 0,987 → GĐ2 0,993 (1e-5) / 1,000 (1e-4)",
                          "lr 1e-3: dao động, tụt tới 0,917",
                          "BatchNorm để mở khi fine-tune: embedding đổi 82%, đầu phân lớp chỉ đúng 4,7% → phải đóng băng BN"],
              "notes": "Fine-tune giúp nhưng mức tăng nhỏ vì giai đoạn 1 đã cao. Learning rate lớn làm mất ổn định. Chi tiết BatchNorm là bài học cài đặt quan trọng: "
                       "results/N4/table_batchnorm.md."})
    s.append({"title": "N6 — So sánh hướng A và hướng B", "table": R / "N6" / "table.md",
              "cols": ["Hướng", "Mô hình", "Macro-F1 val", "Số tham số", "File mô hình (MB)", "Thời gian huấn luyện (s)", "Dự đoán 1 ảnh (ms, CPU)"],
              "notes": "Hướng B tốt hơn. Nhưng 'lớn hơn' không có nghĩa là nặng hơn: MobileNetV3 chỉ 0,94 triệu tham số, 11,7 MB, 12,5 ms mỗi ảnh, "
                       "trong khi a_vgg16 của hướng A nặng 370 MB và mất 130 ms."})
    s.append({"title": "N7, N8 — Embedding và độ bền", "image": F / "N8_robustness.png",
              "bullets": ["N7: embedding MobileNetV3 đã fine-tune + KNN = 1,000; embedding ImageNet chưa học + KNN = 0,932",
                          "N8: CNN bền với nhiễu, xoay, ảnh xám",
                          "Che 25%: MobileNetV3 0,659 < HOG 0,801 (trái dự đoán)",
                          "Xáo trộn: a_vgg16 (Flatten) 0,097; MobileNetV3 (GAP) 0,850"],
              "notes": "Không phương pháp nào bền với mọi thay đổi; mỗi phương pháp bền với đúng loại thay đổi mà cấu trúc của nó bất biến."})
    s.append({"title": "I1, I2 — Mất cân bằng dữ liệu", "image": F / "I2_imbalance.png",
              "bullets": ["I1: luôn đoán lớp đông nhất: accuracy 0,402, macro-F1 0,143",
                          "KNN k=31: accuracy 0,884 nhưng recall Sagor 0,552",
                          "I2: ở mô hình yếu, các cách xử lý đổi precision lấy recall; ở mô hình mạnh không tác động"],
              "notes": "Accuracy có thể che khuất lớp bị nhận kém; vì vậy chỉ số chính là macro-F1. Với tỉ lệ 2,2 lần, mất cân bằng không phải vấn đề chính."})
    s.append({"title": "S1 — Phân đoạn tách nền", "image": F / "S1_segmentation.png",
              "notes": "Ngưỡng toàn cục và thích nghi tách cả vết thâm thành nền. Các cách dựa trên màu (k-means, Otsu trên khoảng cách màu tới nền) "
                       "cho mask liền nhất. Không có nhãn mask nên đánh giá định tính, kèm số mảnh rời và tỉ lệ lỗ thủng."})
    s.append({"title": "S2 — Mô hình dựa vào quả chuối hay vào nền?", "image": F / "S2_background_bias.png",
              "bullets": ["Chỉ 4 góc ảnh 64×64: HOG 0,968; histogram 0,939",
                          "Chỉ 3 số màu nền + KNN: 0,937",
                          "CNN đoán trên ảnh đã xóa quả: 0,683–0,792",
                          "**Dữ liệu thiên lệch theo bối cảnh chụp**",
                          "Chỉ nhìn quả: truyền thống 0,958, CNN 0,926–0,952"],
              "notes": "Đây là phát hiện quan trọng nhất. Mỗi giống được chụp trong điều kiện riêng: nền Sabri xanh hơn, quả Sabri chiếm 5,4% ảnh so với 22% ở Sagor. "
                       "Vì vậy con số gần 100% là độ chính xác trên bộ dữ liệu này, không phải trong điều kiện bất kỳ."})
    s.append({"title": "Kết quả cuối trên tập test (371 ảnh)", "table": R / "final" / "table.md",
              "cols": ["Nhóm", "Mô hình", "Macro-F1 test", "KTC 95% (bootstrap)", "Accuracy test", "Số ảnh sai / 371"], "table_size": 11,
              "notes": f"Truyền thống tốt nhất (H-S + HOG + LogReg) và b_mobilenetv3 đều đúng 371/371 ảnh, macro-F1 "
                       f"{fin['trad_fusion']['macro_f1']:.3f} và {fin['b_mobilenetv3']['macro_f1']:.3f}. a_vgg16 tốt nhất hướng A với "
                       f"{fin['a_vgg16']['macro_f1']:.3f}. Khoảng tin cậy chồng lên nhau nên không tuyên bố hơn kém giữa các mô hình dẫn đầu."})
    s.append({"title": "Phân tích lỗi", "image": F / "final_errors_all.png",
              "bullets": ["41 ảnh test bị ít nhất 1/9 mô hình đoán sai",
                          "Nhầm nhiều nhất: Sagor → Champa",
                          "Ảnh Sagor bị sai có quả nhỏ hơn trung bình lớp (0,181 so với 0,244)",
                          "Sabri_Kola_5: quả to, nền trắng → 6/9 mô hình sai"],
              "notes": "Lỗi khớp với thiên lệch đo ở S2: ảnh không giống ảnh điển hình của lớp về kích thước quả và nền thì dễ bị đoán sai."})
    s.append({"title": "Demo", "image": F / "demo_screenshot.png",
              "notes": "Chạy: python app/app.py. Tải ảnh, chọn mô hình, nhận giống và xác suất 4 lớp. Thông tin giống chỉ ghi những gì có nguồn."})
    s.append({"title": "Ưu điểm và hạn chế",
              "bullets": ["**Ưu điểm**", ("30 thí nghiệm, mỗi nhận định có bảng hoặc hình; ghi cả kết quả trái dự đoán", 1),
                          ("Quy trình đánh giá chặt: test dùng một lần, CV, độ lệch chuẩn, khoảng tin cậy", 1),
                          ("Mô hình 11,7 MB, 12,5 ms/ảnh, đúng 371/371 ảnh test", 1), ("Chạy lại toàn bộ bằng một lệnh", 1),
                          "**Hạn chế**", ("Thiên lệch theo bối cảnh chụp (S2): chỉ nhìn nền vẫn đạt 0,93–0,99", 1),
                          ("Test nhỏ, các mô hình dẫn đầu không phân biệt được", 1), ("Chưa có nhãn mask (IoU/Dice), chưa có ảnh ngoài bối cảnh", 1),
                          "**Hướng phát triển:** ảnh cùng bối cảnh cho 4 giống; huấn luyện trên ảnh đã tách nền"],
              "notes": "Nhóm nhấn mạnh hạn chế thiên lệch nền vì nó quyết định cách hiểu con số 100%."})
    s.append({"title": "Tài liệu tham khảo",
              "bullets": ["BananaImageBD, Mendeley Data v2, DOI 10.17632/ptfscwtnyz.2 (CC BY 4.0)", "Bài giảng CS231",
                          "Dalal & Triggs 2005 (HOG); Pass et al. 1996 (CCV); Stricker & Orengo 1995 (moment màu); Otsu 1979",
                          "LeCun et al. 1998; Simonyan & Zisserman 2015; He et al. 2016; Howard et al. 2019",
                          "Srivastava et al. 2014 (Dropout); Ioffe & Szegedy 2015 (BatchNorm); scikit-learn, TensorFlow/Keras, OpenCV"],
              "size": 16, "notes": "Danh sách đầy đủ ở mục 7 của báo cáo. Cảm ơn thầy/cô và các bạn đã lắng nghe."})
    return s


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from .beamer_builder import build_tex, compile_pdf
    sl = slides()
    out = ROOT / "slides" / "bao-cao-do-an.pptx"; out.parent.mkdir(exist_ok=True)
    build(sl, out, "Nhận diện giống chuối từ ảnh — Đồ án CS231")
    print("đã ghi", out.relative_to(ROOT))
    tex = ROOT / "slides" / "bao-cao-do-an.tex"
    build_tex(sl, tex, title="Nhận diện giống chuối từ ảnh", subtitle="Đồ án môn CS231 — Nhập môn Thị giác máy tính",
              author="Nhóm thực hiện: ....................", institute="Giảng viên: ....................", date="")
    print("đã ghi", tex.relative_to(ROOT))
    for f in compile_pdf(tex):
        print("đã ghi", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
