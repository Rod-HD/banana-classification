"""Dựng slide trình bày và kịch bản thuyết trình từ một danh sách slide duy nhất:

  slides/bao-cao-do-an.tex + .pdf   bản chính: Beamer LaTeX, style UIT (kèm bản -ghi-chu.pdf có lời thuyết trình sau mỗi slide)
  slides/bao-cao-do-an.pptx         bản PowerPoint cùng nội dung
  docs/kich-ban-thuyet-trinh.md     lời thuyết trình cho từng slide, kịch bản demo, câu hỏi dự trù (phần cuối lấy từ docs/hoi-dap-bao-ve.md)

    python -m src.make_slides
Thứ tự slide theo phiếu đánh giá đồ án: thành viên và tên đề tài, lý do, phát biểu bài toán và hình minh họa, phương pháp
(mỗi phương pháp có sẵn tối đa hai slide: nó là gì và tài liệu gốc, nhóm dùng nó như thế nào), dataset, độ đo, kết quả, demo,
nhận xét và kết luận, tài liệu tham khảo. Phụ lục giữ các slide chi tiết của từng thí nghiệm để dùng khi hỏi đáp.
Bảng trên slide đọc trực tiếp từ results/; hình lấy từ results/figures/.
"""
from __future__ import annotations

import argparse
import json

from .config import RESULTS_DIR, ROOT
from .pptx_builder import build
from .slides_detail import detail_slides

F = RESULTS_DIR / "figures"
R = RESULTS_DIR

TITLE = "Nhận diện giống chuối từ ảnh"
COURSE = "CS231.R12: Nhập môn Thị giác máy tính"
MEMBERS = [("Hoàng Quốc Duy", "24520373"), ("Nguyễn Thái Duy", "24520390"), ("Nguyễn Anh Khoa", "24520819")]
REPO = "github.com/Rod-HD/banana-classification"
# Tên người trình bày từng phần phương pháp (ai làm phần nào thì trình bày phần đó). Để trống thì slide chỉ ghi tên phần.
PRESENTERS = {"Phần 1": "", "Phần 2": "", "Phần 3": ""}


def j(path):
    return json.loads((R / path).read_text(encoding="utf-8"))


def part(name: str) -> str:
    return f"{name} ({PRESENTERS[name]})" if PRESENTERS.get(name) else name


PROBLEM_TEX = r"""
\textbf{Cho:}
\begin{itemize}
\item Tập nhãn $C=\{\text{Bangla Kola},\ \text{Champa Kola},\ \text{Sabri Kola},\ \text{Sagor Kola}\}$, số lớp $K=4$
\item Tập huấn luyện $D=\{(x_i,y_i)\}_{i=1}^{N}$: $x_i\in\{0,\dots,255\}^{256\times256\times3}$ là ảnh màu chụp một quả chuối, $y_i\in C$ là giống của quả chuối
\end{itemize}
\textbf{Tìm:} hàm phân loại $f$ sao cho với ảnh $x$ chưa gặp,
\[ f(x)=p=(p_1,p_2,p_3,p_4),\quad p_k\ge 0,\quad \sum_{k=1}^{4}p_k=1,\qquad \hat{y}=\arg\max_k p_k \ \text{trùng với nhãn thật } y \]
\begin{columns}[T]
\begin{column}{0.47\textwidth}
\begin{block}{Input}
Một ảnh màu $x$ chụp \textbf{một quả chuối} (chương trình tự đưa về $256\times256$)
\end{block}
\end{column}
\begin{column}{0.47\textwidth}
\begin{block}{Output}
Nhãn $\hat{y}\in C$ (tên giống) và vector xác suất $p$ của 4 lớp
\end{block}
\end{column}
\end{columns}
\vspace{1mm}
{\footnotesize \textbf{Ràng buộc:} đầu ra chỉ gồm 4 lớp, không có lớp ``khác''. Ảnh ngoài phạm vi (giống khác, không phải chuối) vẫn được gán vào một trong 4 giống.}
"""

METRIC_TEX = r"""
Với mỗi lớp $c$ (coi $c$ là dương, 3 lớp còn lại là âm):
\[ \text{Precision}_c=\frac{TP}{TP+FP},\qquad \text{Recall}_c=\frac{TP}{TP+FN},\qquad F1_c=\frac{2\cdot \text{Precision}_c\cdot \text{Recall}_c}{\text{Precision}_c+\text{Recall}_c} \]
\[ \text{Accuracy}=\frac{\text{số ảnh đoán đúng}}{\text{tổng số ảnh}},\qquad \textbf{Macro-F1}=\frac{1}{4}\sum_{c=1}^{4}F1_c \]
\begin{itemize}
\item \textbf{Macro-F1 là chỉ số chính}: dữ liệu lệch lớp (994 so với 444 ảnh), mỗi lớp được tính trọng số như nhau
\item \textbf{Ma trận nhầm lẫn}: hàng là lớp thật, cột là lớp dự đoán. Chia theo tổng hàng thì đường chéo là recall
\item Ví dụ (\texttt{a\_vgg16}, test): hàng Bangla Kola có 67 ảnh, $65/67=0.970$ đoán đúng, $2/67=0.030$ bị nhầm thành Sagor Kola
\item Kết quả cuối kèm khoảng tin cậy 95\% (bootstrap); truyền thống chọn tham số bằng 5-fold cross-validation
\end{itemize}
"""


def main_slides():
    fin = {k: j(f"final/{k}/metrics.json") for k in ["trad_fusion", "a_vgg16", "b_mobilenetv3", "a_lenet"]}
    names = ", ".join(n for n, _ in MEMBERS)
    s = []
    s.append({"layout": "title", "title": TITLE, "who": "Chung", "sec": 20,
              "bullets": [f"Đồ án môn {COURSE}"] + [f"{n} ({m})" for n, m in MEMBERS] + [f"Mã nguồn: {REPO}"],
              "notes": f"Kính chào thầy và các bạn. Nhóm em gồm ba thành viên: {names}. Hôm nay nhóm em xin trình bày đồ án nhận diện giống chuối từ ảnh. "
                       "Bài trình bày đi theo thứ tự: lý do chọn đề tài, phát biểu bài toán, phương pháp, thực nghiệm, demo, rồi nhận xét và kết luận."})
    s.append({"section": "Giới thiệu", "title": "Lý do chọn đề tài", "who": "Chung", "sec": 45, "image": F / "data_samples.png", "ratio": 0.56,
              "bullets": ["Chuối có nhiều giống, giá bán và cách dùng khác nhau; ở chợ, việc nhận ra giống vẫn làm bằng mắt",
                          "**BananaImageBD**: ảnh 4 giống chuối phổ biến ở Bangladesh, thu tại chợ đầu mối và cửa hàng, hướng tới ứng dụng nông nghiệp và chuỗi cung ứng",
                          "Phù hợp để áp dụng kiến thức môn học:",
                          ("4 giống khác nhau cả màu lẫn hình dạng: so sánh được đặc trưng màu với hình dạng", 1),
                          ("Dữ liệu nhỏ (2471 ảnh), lệch lớp (2.2 lần): thấy rõ quá khớp, lợi ích của transfer learning", 1),
                          ("Ảnh một quả trên nền trơn: kiểm tra được mô hình nhìn quả hay nhìn nền", 1),
                          "**Cơ sở thực hiện:** dữ liệu công khai (CC BY 4.0); có bài báo đạt 99.70% trên cùng dữ liệu để đối chiếu"],
              "notes": "Lý do thứ nhất là ý nghĩa thực tế. Chuối có nhiều giống, giá bán và cách dùng khác nhau, mà ở chợ người ta vẫn phân biệt bằng mắt và kinh nghiệm. "
                       "Bộ dữ liệu BananaImageBD được thu ngay tại chợ đầu mối và cửa hàng ở Bangladesh, tác giả hướng tới ứng dụng trong nông nghiệp và chuỗi cung ứng. "
                       "Lý do thứ hai là bài toán này hợp với môn học. Bốn giống khác nhau cả về màu lẫn hình dạng, nên nhóm em so sánh được đặc trưng màu với đặc trưng hình dạng. "
                       "Dữ liệu nhỏ và lệch lớp, nên thấy rõ hiện tượng quá khớp và lợi ích của transfer learning. Ảnh chụp một quả trên nền trơn, nên kiểm tra được mô hình đang nhìn quả chuối hay nhìn nền. "
                       "Về cơ sở thực hiện: dữ liệu công khai, có giấy phép, và đã có một bài báo công bố kết quả 99.70% trên chính bộ dữ liệu này để nhóm em đối chiếu."})
    s.append({"title": "Phát biểu bài toán", "who": "Chung", "sec": 50, "tex": PROBLEM_TEX,
              "bullets": ["**Cho:** tập nhãn C gồm 4 giống (Bangla Kola, Champa Kola, Sabri Kola, Sagor Kola); tập huấn luyện D gồm các cặp (ảnh màu 256 × 256 × 3 chụp một quả chuối, nhãn giống)",
                          "**Tìm:** hàm f sao cho với ảnh x chưa gặp, f(x) = p = (p1, p2, p3, p4), p_k ≥ 0, tổng bằng 1, và nhãn ŷ = argmax p_k trùng nhãn thật",
                          "**Input:** một ảnh màu chụp một quả chuối", "**Output:** nhãn ŷ thuộc C và vector xác suất p của 4 lớp",
                          "**Ràng buộc:** đầu ra chỉ gồm 4 lớp; ảnh ngoài phạm vi vẫn được gán vào một trong 4 giống"],
              "notes": "Đây là bài toán phân loại ảnh nhiều lớp. Cho trước tập nhãn C gồm bốn giống chuối, và tập huấn luyện D gồm các cặp ảnh và nhãn, trong đó mỗi ảnh là ảnh màu 256 nhân 256 chụp một quả chuối. "
                       "Cần tìm hàm phân loại f. Với một ảnh x chưa gặp, f trả về vector xác suất p gồm bốn thành phần không âm, tổng bằng một. Nhãn dự đoán là lớp có xác suất lớn nhất, và ta muốn nhãn này trùng với nhãn thật. "
                       "Khi sử dụng, input là một ảnh màu chụp một quả chuối, output là tên giống và xác suất của bốn lớp. "
                       "Nhóm em lưu ý một ràng buộc: đầu ra chỉ gồm bốn lớp, không có lớp khác. Nên nếu đưa vào ảnh ngoài phạm vi, chương trình vẫn chạy và vẫn trả về một trong bốn giống."})
    s.append({"title": "Hình minh họa bài toán", "who": "Chung", "sec": 25, "image": F / "problem_io.png",
              "notes": "Đây là hình minh họa. Bên trái là bốn ảnh lấy từ tập test, mỗi ảnh một giống. Ảnh đi qua mô hình f. Bên phải là đầu ra: xác suất của bốn lớp, và nhãn dự đoán là lớp có cột cao nhất. "
                       "Các xác suất trong hình là kết quả thật của mô hình MobileNetV3 trong đồ án."})
    s.append({"section": "Phương pháp", "title": "Tổng quan phương pháp", "who": "Chung", "sec": 35, "image": F / "method_overview.png", "stack": True, "img_h": "0.56\\textheight",
              "bullets": [f"**{part('Phần 1')}:** đặc trưng màu và KNN", f"**{part('Phần 2')}:** cạnh, HOG, ghép đặc trưng, bộ phân lớp tuyến tính",
                          f"**{part('Phần 3')}:** CNN (hướng A theo bài giảng, hướng B fine-tune)",
                          "Cùng một cách chia dữ liệu và một bộ độ đo cho mọi phương pháp"],
              "notes": "Nhóm em giải bài toán theo hai hướng. Hướng truyền thống: rút feature vector do con người thiết kế, gồm màu, cạnh và HOG, rồi đưa vào bộ phân lớp. "
                       "Hướng học sâu: dùng CNN, mạng tự học đặc trưng từ dữ liệu. Phần phương pháp chia làm ba phần, mỗi bạn trình bày phần mình làm: phần một là đặc trưng màu và KNN, "
                       "phần hai là cạnh, HOG, ghép đặc trưng và bộ phân lớp tuyến tính, phần ba là CNN. Các phương pháp này đều đã có sẵn, nên với mỗi phương pháp nhóm em chỉ nói ngắn gọn nó là gì, "
                       "được đề xuất ở đâu, rồi nói rõ nhóm em dùng nó như thế nào."})
    # ---- Phần 1
    s.append({"title": "Phần 1. Đặc trưng màu và KNN: là gì", "who": "Phần 1", "sec": 45,
              "bullets": ["**Histogram màu** (Swain và Ballard, 1991): tỉ lệ điểm ảnh rơi vào từng khoảng màu; không giữ vị trí điểm ảnh",
                          "**Moment màu** (Stricker và Orengo, 1995): trung bình, độ lệch chuẩn, độ lệch của từng kênh màu",
                          "**Color Coherence Vector, CCV** (Pass và cộng sự, 1996): với mỗi màu, tách điểm ảnh thuộc vùng liên thông lớn và điểm ảnh rời rạc",
                          "Đều là **feature vector toàn cục**: biết rõ ý nghĩa từng phần tử, mỗi phần tử tính từ toàn bộ ảnh",
                          "**KNN** (Cover và Hart, 1967): gán nhãn theo k ảnh huấn luyện gần nhất; không có bước huấn luyện"],
              "links": [("Color Indexing", "https://doi.org/10.1007/BF00130487"), ("Moment màu", "https://doi.org/10.1117/12.205308"),
                        ("CCV", "https://doi.org/10.1145/244130.244148"), ("KNN", "https://doi.org/10.1109/TIT.1967.1053964")],
              "notes": "Em xin trình bày phần một: đặc trưng màu và KNN. Histogram màu đếm tỉ lệ điểm ảnh rơi vào từng khoảng màu, nó không quan tâm điểm ảnh nằm ở đâu. "
                       "Moment màu tóm tắt phân bố màu của mỗi kênh bằng ba con số. CCV bổ sung cho histogram thông tin điểm ảnh có thuộc một vùng liên thông lớn hay không. "
                       "Cả ba đều là feature vector toàn cục: mình hiểu ý nghĩa của từng phần tử, và mỗi phần tử được tính từ toàn bộ ảnh. "
                       "Bộ phân lớp đi kèm là KNN: ảnh mới được gán nhãn theo k ảnh huấn luyện gần nó nhất. Tài liệu gốc của từng phương pháp nằm ở cuối slide."})
    s.append({"title": "Phần 1. Đặc trưng màu và KNN: nhóm dùng như thế nào", "who": "Phần 1", "sec": 50,
              "table": [["Đặc trưng", "Cách dùng trong đồ án", "Số chiều"],
                        ["Histogram RGB", "32 bin mỗi kênh, ghép 3 kênh", "96"], ["Histogram HSV", "(8, 8, 8) bin cho (H, S, V)", "512"],
                        ["Histogram H-S", "(8, 8) bin, bỏ kênh độ sáng V", "64"], ["Moment màu", "3 moment × 3 kênh HSV", "9"],
                        ["Màu chủ đạo", "k-means, k = 3: (tỉ lệ, R, G, B) mỗi cụm", "12"], ["CCV", "64 màu, vùng liên thông từ 1% ảnh", "128"]],
              "bullets": ["Histogram chia cho tổng số điểm ảnh (tổng bằng 1)",
                          "**Chuẩn hóa trước khi đo khoảng cách**: z-score từng chiều, hoặc đưa về vector đơn vị",
                          "KNN: khảo sát k, độ đo (Euclidean, Manhattan, cosine, Hellinger), cách bình chọn",
                          "Chọn bằng 5-fold cross-validation: **k = 1** tốt nhất"],
              "notes": "Đây là cách nhóm em dùng. Histogram RGB lấy 32 bin mỗi kênh, được 96 chiều. Histogram HSV lấy 8 bin mỗi kênh, được 512 chiều. "
                       "Histogram H-S chỉ lấy hai kênh H và S, bỏ kênh độ sáng, được 64 chiều. Moment màu 9 chiều, màu chủ đạo 12 chiều, CCV 128 chiều. "
                       "Trước khi đo khoảng cách, vector được chuẩn hóa. Lý do là hai vector cùng hướng nhưng khác độ lớn thì độ tương đồng là một trăm phần trăm, "
                       "nhưng nếu không chuẩn hóa, khoảng cách Euclidean giữa chúng lại khác không. Với KNN, nhóm em khảo sát số láng giềng k, độ đo và cách bình chọn bằng 5-fold cross-validation, và k bằng 1 cho kết quả tốt nhất."})
    # ---- Phần 2
    s.append({"title": "Phần 2. Cạnh, HOG và bộ phân lớp tuyến tính: là gì", "who": "Phần 2", "sec": 45,
              "bullets": ["**Toán tử cạnh**: Roberts, Prewitt, Sobel xấp xỉ đạo hàm bậc nhất; Laplacian xấp xỉ đạo hàm bậc hai",
                          "**HOG** (Dalal và Triggs, CVPR 2005): chia ảnh thành cell, mỗi cell một histogram hướng gradient, chuẩn hóa theo block; mô tả **hình dạng**",
                          "HOG là **feature vector cục bộ**: mỗi phần tử tính trên một cell, nên giữ được thông tin vị trí",
                          "**Logistic Regression, Linear SVM**: bộ phân lớp tuyến tính, học trọng số cho từng chiều của đặc trưng"],
              "links": [("HOG", "https://doi.org/10.1109/CVPR.2005.177"), ("SVM", "https://doi.org/10.1007/BF00994018"),
                        ("scikit-learn", "https://scikit-learn.org")],
              "notes": "Em xin trình bày phần hai. Đặc trưng màu không biết gì về hình dạng, nên nhóm em dùng thêm đặc trưng cạnh và HOG. "
                       "Các toán tử Roberts, Prewitt, Sobel xấp xỉ đạo hàm bậc nhất của ảnh, Laplacian xấp xỉ đạo hàm bậc hai. "
                       "HOG do Dalal và Triggs đề xuất năm 2005: ảnh được chia thành các cell, mỗi cell có một histogram của hướng gradient, rồi chuẩn hóa theo block. HOG mô tả hình dạng của vật. "
                       "Khác với histogram màu, HOG là đặc trưng cục bộ: mỗi phần tử tính trên một cell, nên nó giữ thông tin vị trí. "
                       "Bộ phân lớp ở phần này là Logistic Regression và Linear SVM, đều là bộ phân lớp tuyến tính."})
    s.append({"title": "Phần 2. Cạnh, HOG, ghép đặc trưng: nhóm dùng như thế nào", "who": "Phần 2", "sec": 55, "image": F / "C5_fusion.png", "ratio": 0.5,
              "bullets": ["**HOG tự cài đặt theo bài giảng**: ảnh xám 64 × 128, cell 8 × 8, 9 bin, block 16 × 16 chuẩn hóa L2, được **3780 chiều**",
                          "Đối chiếu với scikit-image: lệch tối đa 2.2·10⁻⁷",
                          "**Vector cạnh**: tổng độ lớn gradient theo 256 hàng và 256 cột, 512 chiều",
                          "**Ghép màu với HOG**: chuẩn hóa z-score từng phần, chia cho căn số chiều để hai phần đóng góp ngang nhau",
                          "Cấu hình chốt bằng 5-fold CV: **Hist H-S + HOG + Logistic Regression (C = 100)**, 3844 chiều"],
              "notes": "Về cách dùng. HOG nhóm em tự cài đặt theo đúng các bước trong bài giảng: ảnh xám 64 nhân 128, cell 8 nhân 8, 9 bin, block 16 nhân 16 có chuẩn hóa L2, ra vector 3780 chiều. "
                       "Nhóm em tự cài đặt để có thể bật tắt bước chuẩn hóa block trong thí nghiệm, và đã đối chiếu với thư viện scikit-image, hai kết quả gần như trùng nhau. "
                       "Vector cạnh là tổng độ lớn gradient theo từng hàng và từng cột, 512 chiều. "
                       "Điểm khác của nhóm em là cách ghép: phần màu chỉ có 64 chiều, phần HOG có 3780 chiều. Nếu ghép thẳng thì HOG lấn át. "
                       "Nên mỗi phần được chuẩn hóa riêng rồi chia cho căn số chiều, để hai phần đóng góp ngang nhau. Hình bên phải cho thấy ghép theo cách này tốt hơn từng đặc trưng riêng lẻ. "
                       "Cấu hình cuối cùng của hướng truyền thống là histogram H-S ghép HOG, phân lớp bằng Logistic Regression."})
    # ---- Phần 3
    s.append({"title": "Phần 3. CNN hướng A (theo bài giảng): là gì", "who": "Phần 3", "sec": 40,
              "bullets": ["**LeNet-5** (LeCun và cộng sự, 1998): hai cặp lớp tích chập và lấy mẫu, sau đó là các lớp kết nối đầy đủ",
                          "**Dropout** (Srivastava và cộng sự, 2014): tắt ngẫu nhiên nơ-ron lúc huấn luyện để giảm quá khớp",
                          "**VGG16** (Simonyan và Zisserman, 2015): 13 lớp tích chập 3 × 3 và 3 lớp kết nối đầy đủ, huấn luyện trên ImageNet **1000 lớp**",
                          "**Transfer learning**: dùng lại phần tích chập đã học làm bộ rút đặc trưng",
                          "Đặc trưng do CNN sinh ra là **embedding**: không giải thích được ý nghĩa từng phần tử"],
              "links": [("LeNet", "https://doi.org/10.1109/5.726791"), ("Dropout", "https://jmlr.org/papers/v15/srivastava14a.html"),
                        ("VGG16", "https://arxiv.org/abs/1409.1556")],
              "notes": "Em xin trình bày phần ba, CNN. Hướng A là các mô hình đúng như ví dụ trong bài giảng. LeNet gồm hai cặp lớp tích chập và lấy mẫu, sau đó là các lớp kết nối đầy đủ. "
                       "Dropout là kỹ thuật tắt ngẫu nhiên một số nơ-ron lúc huấn luyện để giảm quá khớp. VGG16 là mạng 16 lớp, được huấn luyện sẵn trên ImageNet với 1000 lớp. "
                       "Transfer learning nghĩa là dùng lại phần tích chập đã học của VGG16 để rút đặc trưng cho bài toán của mình. "
                       "Khác với feature vector ở hai phần trước, đặc trưng do CNN sinh ra gọi là embedding, mình không giải thích được ý nghĩa của từng phần tử."})
    s.append({"title": "Phần 3. CNN hướng A: nhóm dùng như thế nào", "who": "Phần 3", "sec": 50, "colspec": "l p{7.6cm} l p{2.6cm}",
              "table": [["Mô hình", "Cách dùng trong đồ án", "Ảnh vào", "Số tham số"],
                        ["a_lenet", "Cấu trúc LeNet, đổi ảnh vào thành ảnh màu, lớp ra 4 lớp; huấn luyện từ đầu", "64 × 64 × 3", "337976"],
                        ["a_dropout", "Conv(32) → Conv(16) → Dense(128) → Dropout(0.5) → Dense(64) → Softmax(4); huấn luyện từ đầu", "128 × 128 × 3", "1746260"],
                        ["a_vgg16", "Bỏ 3 lớp FC của VGG16, đóng băng phần tích chập (ra 7 × 7 × 512), thêm Flatten → Dense(1024) → Dense(256) → Softmax(4)", "224 × 224 × 3",
                         "40669252 (huấn luyện 25954564)"]],
              "notes": "Nhóm em dùng như sau. Mô hình a_lenet giữ cấu trúc LeNet, đổi ảnh vào thành ảnh màu 64 nhân 64 và lớp ra thành 4 lớp, huấn luyện từ đầu. "
                       "Mô hình a_dropout là CNN hai lớp tích chập, có Dropout 0.5, ảnh vào 128 nhân 128. "
                       "Với VGG16: mạng gốc huấn luyện trên 1000 lớp, còn bài của nhóm em chỉ có 4 lớp. Nên nhóm em bỏ ba lớp kết nối đầy đủ ở cuối, giữ nguyên và đóng băng phần tích chập, "
                       "rồi thêm phần đầu mới gồm Flatten, hai lớp Dense và lớp Softmax 4 lớp. Chỉ phần đầu mới được huấn luyện."})
    s.append({"title": "Phần 3. CNN hướng B (fine-tune): là gì", "who": "Phần 3", "sec": 35,
              "bullets": ["**MobileNetV3** (Howard và cộng sự, ICCV 2019): họ mạng thiết kế cho thiết bị di động, có hai phiên bản Large và Small",
                          "**ResNet** (He và cộng sự, CVPR 2016): dùng kết nối tắt để huấn luyện được mạng rất sâu",
                          "Cả hai có sẵn trọng số huấn luyện trên ImageNet (Keras Applications)",
                          "**Fine-tune**: không đóng băng hẳn, mà huấn luyện tiếp trọng số ImageNet trên dữ liệu mới với learning rate nhỏ"],
              "links": [("MobileNetV3", "https://arxiv.org/abs/1905.02244"), ("ResNet", "https://arxiv.org/abs/1512.03385"),
                        ("Keras Applications", "https://keras.io/api/applications/")],
              "notes": "Hướng B dùng mạng lớn hơn, huấn luyện sẵn trên ImageNet rồi fine-tune. MobileNetV3 là họ mạng thiết kế cho thiết bị di động, có hai phiên bản Large và Small. "
                       "ResNet dùng kết nối tắt để huấn luyện được mạng rất sâu. Fine-tune khác với đóng băng ở chỗ mình huấn luyện tiếp cả phần thân trên dữ liệu của mình, với learning rate nhỏ. "
                       "Chi tiết kiến trúc bên trong của hai mạng này có trong bài báo gốc ở cuối slide, nhóm em chỉ sử dụng chúng làm phần thân rút đặc trưng."})
    s.append({"title": "Phần 3. CNN hướng B: nhóm dùng như thế nào", "who": "Phần 3", "sec": 55, "image": F / "N4_finetune_lr.png", "ratio": 0.5,
              "bullets": ["Phiên bản: **MobileNetV3-Small** (embedding **576 chiều**) và **ResNet50** (2048 chiều), ảnh 224 × 224",
                          "Mạng gốc phân loại 1000 lớp, bài toán có 4 lớp: thay phần đầu bằng GlobalAveragePooling → Dropout(0.2) → Softmax(4)",
                          "**Giai đoạn 1:** đóng băng phần thân, chỉ huấn luyện lớp Softmax",
                          "**Giai đoạn 2:** mở băng (toàn mạng với MobileNetV3, khối conv5 với ResNet50), learning rate 1e-4 và 1e-5, có tăng cường ảnh",
                          "Giữ BatchNorm đóng băng khi fine-tune; Adam, cross-entropy, dừng sớm theo macro-F1 trên val"],
              "notes": "Nhóm em dùng phiên bản MobileNetV3-Small, embedding sau lớp GlobalAveragePooling có 576 chiều, và ResNet50 với embedding 2048 chiều. Ảnh vào 224 nhân 224. "
                       "Mạng gốc phân loại 1000 lớp, nên nhóm em thay phần đầu bằng lớp Softmax 4 lớp. Huấn luyện gồm hai giai đoạn. "
                       "Giai đoạn một đóng băng phần thân, chỉ huấn luyện lớp Softmax mới. Giai đoạn hai mở băng phần thân và huấn luyện tiếp với learning rate nhỏ, có tăng cường ảnh như lật, xoay, phóng to thu nhỏ. "
                       "Một chi tiết nhóm em rút ra khi làm: khi fine-tune phải giữ các lớp BatchNorm đóng băng. Lần đầu nhóm em để mở, accuracy trên train tụt xuống còn khoảng 0.36. "
                       "Hình bên phải là macro-F1 trên val theo epoch của giai đoạn một và giai đoạn hai với các learning rate khác nhau."})
    s.append({"title": "Điểm khác biệt giữa các phương pháp", "who": "Phần 3", "sec": 40, "table_size": 10, "colspec": "p{3.6cm} p{1.7cm} p{2.4cm} p{1.4cm} p{1.3cm} p{2.4cm}",
              "table": [["Phương pháp", "Loại biểu diễn", "Toàn cục hay cục bộ", "Số chiều", "Giữ vị trí", "Học đặc trưng từ dữ liệu"],
                        ["Histogram màu, moment màu", "feature vector", "toàn cục", "9 đến 512", "không", "không"],
                        ["Vector cạnh, HOG", "feature vector", "cục bộ (hàng, cột, cell)", "512, 3780", "có", "không"],
                        ["CNN học từ đầu (LeNet)", "embedding", "cục bộ (cửa sổ tích chập)", "84", "có", "có, cần nhiều dữ liệu"],
                        ["CNN ImageNet đóng băng (VGG16)", "embedding", "cục bộ (cửa sổ tích chập)", "7 × 7 × 512", "có", "chỉ phần đầu mới"],
                        ["CNN ImageNet fine-tune (MobileNetV3)", "embedding", "cục bộ (cửa sổ tích chập)", "576", "mất sau GAP", "có, chỉnh từ ImageNet"]],
              "notes": "Bảng này tóm tắt điểm khác biệt giữa các phương pháp. Histogram và moment màu là feature vector toàn cục, ít chiều, không giữ vị trí. "
                       "HOG và vector cạnh là feature vector cục bộ, giữ được vị trí. Ba dòng cuối là embedding của CNN: mỗi giá trị tính trên một cửa sổ tích chập nên cũng là cục bộ. "
                       "Khác nhau ở chỗ có học đặc trưng từ dữ liệu hay không: LeNet học từ đầu nên cần nhiều dữ liệu, VGG16 đóng băng thì không học lại, MobileNetV3 fine-tune thì chỉnh lại từ trọng số ImageNet. "
                       "Từ bảng này nhóm em đặt ra các dự đoán và kiểm chứng ở phần thực nghiệm."})
    # ---- Thực nghiệm
    s.append({"section": "Thực nghiệm", "title": "Dataset", "who": "Chung", "sec": 40, "image": F / "data_split_counts.png", "ratio": 0.46,
              "bullets": ["**BananaImageBD**, Version 2 (Mendeley Data, CC BY 4.0), dataset có sẵn của Ferdaus và cộng sự",
                          "**2471 ảnh** gốc, 256 × 256, 4 giống: Champa 994, Sagor 524, Sabri 509, Bangla 444",
                          "Mỗi quả đặt trên giấy trắng, chụp bằng điện thoại từ bốn mặt",
                          "Chia cố định, giữ tỉ lệ lớp: **train 1729, val 371, test 371**",
                          "Tập test chỉ dùng một lần ở cuối"],
              "links": [("Dataset", "https://data.mendeley.com/datasets/ptfscwtnyz/2"), ("Bài báo mô tả", "https://doi.org/10.1016/j.dib.2024.111239")],
              "notes": "Về dataset. Nhóm em dùng bộ BananaImageBD, là bộ dữ liệu có sẵn, công bố trên Mendeley Data. Có 2471 ảnh gốc kích thước 256 nhân 256 của bốn giống, "
                       "trong đó Champa Kola nhiều nhất với 994 ảnh và Bangla Kola ít nhất với 444 ảnh. Nhóm em chia cố định thành train, val và test theo tỉ lệ 70, 15, 15, giữ nguyên tỉ lệ các lớp. Tập test chỉ dùng một lần ở cuối."})
    s.append({"title": "Độ đo đánh giá", "who": "Chung", "sec": 20, "tex": METRIC_TEX,
              "bullets": ["Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2·P·R / (P + R)", "Accuracy = số ảnh đoán đúng / tổng số ảnh",
                          "**Macro-F1** = trung bình F1 của 4 lớp, là chỉ số chính vì dữ liệu lệch lớp", "Ma trận nhầm lẫn: hàng là lớp thật, cột là lớp dự đoán",
                          "Khoảng tin cậy 95% (bootstrap) cho kết quả cuối; 5-fold cross-validation để chọn tham số"],
              "notes": "Nhóm em sử dụng các độ đo cơ bản của bài toán phân loại đã học trong môn học: accuracy, precision, recall, F1 và ma trận nhầm lẫn. "
                       "Chỉ số chính là macro-F1, vì dữ liệu lệch lớp."})
    s.append({"title": "Kết quả trên tập test (371 ảnh)", "who": "Chung", "sec": 50, "table": R / "final" / "table.md",
              "cols": ["Nhóm", "Mô hình", "Macro-F1 test", "KTC 95% (bootstrap)", "Số ảnh sai / 371"], "table_size": 11, "colspec": "p{2.7cm} p{5.6cm} l l l",
              "notes": f"Đây là kết quả cuối trên tập test 371 ảnh. Hướng truyền thống tốt nhất là histogram H-S ghép HOG với Logistic Regression, macro-F1 {fin['trad_fusion']['macro_f1']:.3f}, đúng cả 371 ảnh. "
                       f"CNN hướng B với MobileNetV3 cũng đạt {fin['b_mobilenetv3']['macro_f1']:.3f}, đúng cả 371 ảnh. Trong hướng A, VGG16 đóng băng tốt nhất với {fin['a_vgg16']['macro_f1']:.3f}, "
                       f"còn LeNet huấn luyện từ đầu đạt {fin['a_lenet']['macro_f1']:.3f}. Khoảng tin cậy của các mô hình dẫn đầu chồng lên nhau, nên nhóm em không kết luận mô hình nào hơn mô hình nào trong nhóm này."})
    s.append({"title": "Kết quả: màu hay hình dạng?", "who": "Chung", "sec": 40, "image": F / "M6_grayscale.png", "ratio": 0.42,
              "bullets": ["Xáo trộn ảnh thành lưới 4 × 4: histogram không đổi (sụt 0.000), HOG rơi về mức đoán mò",
                          "Chuyển sang ảnh xám: đặc trưng màu sụt 0.04 đến 0.23, **HOG vẫn đạt 0.986**",
                          "Bốn giống phân biệt được bằng **cả màu** (Hist HSV 0.991) **lẫn hình dạng** (HOG 0.986)",
                          "Ghép màu với HOG tốt nhất: 0.996 ± 0.003 (5-fold)"],
              "notes": "Một số kết quả kiểm chứng tính chất của đặc trưng. Khi xáo trộn ảnh thành lưới 4 nhân 4, histogram màu không đổi một chút nào, vì nó không giữ vị trí, còn HOG thì rơi về mức đoán mò. "
                       "Khi chuyển ảnh sang xám, đặc trưng màu sụt, còn HOG vẫn đạt 0.986. Vậy bốn giống chuối phân biệt được bằng cả màu lẫn hình dạng, và ghép hai loại lại cho kết quả tốt nhất."})
    s.append({"title": "Kết quả: CNN và lượng dữ liệu", "who": "Chung", "sec": 45, "image": F / "N2_learning_curve.png", "ratio": 0.46,
              "bullets": ["Chỉ 25% dữ liệu train: mô hình có trọng số ImageNet (0.967, 0.975) đã ngang LeNet dùng đủ dữ liệu (0.973)",
                          "**MobileNetV3-Small**: 0.94 triệu tham số, file **4.3 MB**, 12.5 ms mỗi ảnh trên CPU",
                          "a_vgg16: 40.7 triệu tham số, file 163 MB, 130 ms mỗi ảnh",
                          "Trái dự đoán: Dropout không giúp ở a_dropout (0.941 so với 0.953 khi bỏ Dropout, 3 seed)"],
              "notes": "Với CNN, hình bên phải là macro-F1 theo lượng dữ liệu huấn luyện. Chỉ với 25 phần trăm dữ liệu, hai mô hình dùng trọng số ImageNet đã ngang LeNet dùng toàn bộ dữ liệu. "
                       "Đây là lợi ích của transfer learning khi dữ liệu ít. MobileNetV3-Small vừa chính xác nhất vừa nhẹ: file chỉ 4.3 megabyte và dự đoán 12.5 mili giây mỗi ảnh trên CPU, trong khi VGG16 nặng 163 megabyte. "
                       "Cũng có kết quả trái dự đoán: Dropout không giúp ở mô hình a_dropout. Nhóm em giải thích là 99 phần trăm tham số nằm ở lớp Dense đứng trước lớp Dropout, nên Dropout không chặn được quá khớp ở đó."})
    s.append({"title": "Khác tài liệu tham khảo (1): mô hình nhìn quả hay nhìn nền?", "who": "Chung", "sec": 50, "image": F / "S2_background_bias.png", "ratio": 0.44,
              "bullets": ["Dùng phân đoạn (Otsu, k-means) tách ảnh thành \"chỉ quả\" và \"chỉ nền\"",
                          "Chỉ dùng 4 góc ảnh (gần như không có chuối): HOG vẫn đạt 0.968, histogram màu 0.939",
                          "CNN dự đoán trên ảnh đã xóa quả chuối: 0.683 đến 0.792 (đoán mò khoảng 0.25)",
                          "**Dữ liệu thiên lệch theo bối cảnh chụp**: mỗi giống có nền và cỡ quả riêng",
                          "Chỉ nhìn quả: truyền thống 0.958, CNN 0.926 đến 0.952"],
              "notes": "Phần này là thực nghiệm nhóm em tự thiết kế, không có trong tài liệu tham khảo. Kết quả gần 100 phần trăm làm nhóm em nghi ngờ: mô hình đang nhìn quả chuối hay nhìn nền? "
                       "Nhóm em dùng các phương pháp phân đoạn đã học để tách ảnh thành phần chỉ có quả và phần chỉ có nền. Kết quả là chỉ dùng bốn góc ảnh, nơi gần như không có chuối, HOG vẫn đạt 0.968. "
                       "CNN dự đoán trên ảnh đã xóa quả chuối vẫn đạt khoảng 0.7 đến 0.8, trong khi đoán mò chỉ là 0.25. "
                       "Nghĩa là bộ dữ liệu có thiên lệch theo bối cảnh chụp: mỗi giống được chụp với nền và cỡ quả riêng, và mô hình học được cả điều đó. Đây là hạn chế quan trọng nhất mà nhóm em tìm ra."})
    s.append({"title": "Khác tài liệu tham khảo (2): bộ kiểm tra khó, đối chiếu bài báo", "who": "Chung", "sec": 50,
              "table": R / "so_sanh_nhanh" / "table_stress.md", "table_size": 9, "colspec": "p{5.4cm} p{3.4cm} l l l l",
              "cols": ["Nhánh", "Cấu hình đại diện", "Ảnh gốc", "Nền hoán đổi", "Che 25%", "Điểm bền"],
              "bullets": ["**Bộ kiểm tra khó**: 6 phép biến đổi trên tập test (thay nền, tối, sáng, nhiễu, xoay, che)",
                          "Thay nền: MobileNetV3 từ 1.000 xuống **0.794**; thêm ảnh train có nền thay ngẫu nhiên (X-b): **0.975**",
                          "Chạy lại phương pháp của bài báo (ViT + SVM): **99.49%**, bài báo công bố 99.70%",
                          "Cách chia fold của bài báo có rò rỉ: cao hơn chia đúng ở 19/19 tổ hợp (+0.11 đến +0.90 điểm)"],
              "notes": "Vì trên ảnh gốc hai mô hình đã đúng hết, nhóm em tạo thêm bộ kiểm tra khó gồm sáu phép biến đổi trên tập test, trong đó có phép thay nền: ghép quả chuối lên nền của ảnh thuộc lớp khác. "
                       "Với phép thay nền, MobileNetV3 giảm từ 1.000 xuống 0.794. Khi nhóm em thêm vào tập train các ảnh có nền thay ngẫu nhiên, con số này lên lại 0.975. "
                       "Nhóm em cũng chạy lại phương pháp của bài báo đã công bố trên bộ dữ liệu này, là Vision Transformer kết hợp SVM, và đạt 99.49 phần trăm, gần với 99.70 phần trăm bài báo công bố. "
                       "Tuy nhiên cách chia fold của bài báo để ảnh gốc và bản tăng cường của nó nằm ở cả hai phía, nên kết quả cao hơn cách chia đúng ở cả 19 trên 19 tổ hợp nhóm em thử. Vision Transformer nằm ngoài nội dung môn học, nhóm em chỉ dùng để đối chiếu."})
    s.append({"title": "Chương trình demo", "who": "Chung", "sec": 120, "image": F / "demo_screenshot.png",
              "bullets": ["Viết bằng Gradio, chạy: python app/app.py", "Dự đoán một ảnh: giống, xác suất 4 lớp, thời gian",
                          "So sánh 4 mô hình trên cùng một ảnh", "Grad-CAM: vùng ảnh mà CNN dựa vào",
                          "Thử độ bền: đổi sáng, nhiễu, xoay, che, thay nền", "Tính lại accuracy, precision, recall, F1, ma trận nhầm lẫn trên val và test"],
              "notes": "Nhóm em xin chạy chương trình demo. Chương trình viết bằng Gradio. (Chuyển sang cửa sổ demo, làm theo kịch bản demo trong file kịch bản.)"})
    s.append({"section": "Kết luận", "title": "Nhận xét", "who": "Chung", "sec": 45, "size": 17,
              "bullets": ["**Ưu điểm**",
                          ("Hai mô hình đúng 371/371 ảnh test; MobileNetV3-Small chỉ 4.3 MB, 12.5 ms mỗi ảnh", 1),
                          ("So sánh mọi phương pháp trên cùng quy trình: test dùng một lần, cross-validation, khoảng tin cậy", 1),
                          ("30 thí nghiệm kiểm chứng tính chất của phương pháp, ghi cả kết quả trái dự đoán", 1),
                          ("Chạy lại được toàn bộ bằng một lệnh; có demo", 1),
                          "**Hạn chế**",
                          ("Dữ liệu thiên lệch theo nền: chỉ nhìn nền vẫn đạt 0.93 đến 0.99, nên kết quả 100% chỉ đúng trên bộ dữ liệu này", 1),
                          ("Mỗi quả chụp bốn mặt, chia ngẫu nhiên không tách theo quả: kết quả lạc quan hơn 0.02 đến 0.07", 1),
                          ("Chưa có ảnh chụp thật ngoài bộ dữ liệu; chương trình không từ chối được ảnh ngoài 4 giống", 1)],
              "notes": "Về nhận xét. Ưu điểm: có hai mô hình đúng toàn bộ 371 ảnh test, trong đó MobileNetV3 rất nhẹ. Nhóm em so sánh mọi phương pháp trên cùng một quy trình đánh giá, "
                       "và kiểm chứng tính chất của từng phương pháp bằng 30 thí nghiệm, ghi lại cả những kết quả trái dự đoán. "
                       "Hạn chế: thứ nhất, dữ liệu thiên lệch theo nền, nên con số 100 phần trăm chỉ đúng trên bộ dữ liệu này, chưa chắc đúng với ảnh chụp ở nơi khác. "
                       "Thứ hai, mỗi quả được chụp bốn mặt mà khi chia ngẫu nhiên thì ảnh của cùng một quả có thể nằm ở cả train và test, nên kết quả lạc quan hơn thực tế. "
                       "Thứ ba, nhóm em chưa có ảnh chụp thật ngoài bộ dữ liệu để đánh giá, và chương trình chưa từ chối được ảnh ngoài bốn giống."})
    s.append({"title": "Kết luận", "who": "Chung", "sec": 35,
              "bullets": ["Đã xây dựng chương trình nhận diện 4 giống chuối; hai hướng đều đạt macro-F1 1.000 trên tập test",
                          "Phương pháp tốt khi tính chất của nó khớp với dữ liệu: histogram không biết vị trí, HOG không biết màu, CNN học từ đầu cần nhiều dữ liệu",
                          "Cách đánh giá quan trọng không kém mô hình: tăng cường trước khi chia tập làm kết quả đẹp hơn thực tế",
                          "Độ chính xác cao chưa chắc mô hình đã học đúng thứ cần học: ở đây mô hình dựa một phần vào nền",
                          "**Hướng phát triển:** ảnh ở bối cảnh khác, chia dữ liệu theo quả, thêm lớp \"không thuộc 4 giống\""],
              "notes": "Kết luận. Nhóm em đã xây dựng được chương trình nhận diện bốn giống chuối, cả hướng truyền thống lẫn hướng học sâu đều đạt macro-F1 bằng 1 trên tập test. "
                       "Nhóm em rút ra ba điều: phương pháp tốt khi tính chất của nó khớp với dữ liệu, cách đánh giá quan trọng không kém mô hình, và độ chính xác cao chưa chắc mô hình đã học đúng thứ cần học. "
                       "Nhóm em đã rất cố gắng và dành nhiều thời gian cho đồ án này. Tuy nhiên vẫn còn những phần chưa đạt như mong muốn, như chưa đánh giá được trên ảnh chụp thật ngoài bộ dữ liệu "
                       "và chưa xử lý được ảnh ngoài phạm vi bốn giống. Đó cũng là hướng phát triển tiếp theo."})
    s.append({"title": "Tài liệu tham khảo", "who": "Chung", "sec": 10, "size": 14,
              "bullets": ["[1] Ferdaus và cộng sự (2024). BananaImageBD, Version 2. Mendeley Data. DOI 10.17632/ptfscwtnyz.2 (CC BY 4.0); bài báo mô tả: Data in Brief 58, 111239 (2025)",
                          "[2] Bài giảng môn CS231, Nhập môn Thị giác máy tính, Trường ĐH Công nghệ Thông tin",
                          "[3] Ergün (2025). High precision banana variety identification using vision transformer based feature extraction and SVM. Scientific Reports 15, 10366",
                          "[4] Swain, Ballard (1991), Color Indexing; Stricker, Orengo (1995); Pass, Zabih, Miller (1996), CCV; Dalal, Triggs (2005), HOG; Otsu (1979); Cover, Hart (1967), KNN",
                          "[5] LeCun và cộng sự (1998), LeNet; Simonyan, Zisserman (2015), VGG; He và cộng sự (2016), ResNet; Howard và cộng sự (2019), MobileNetV3",
                          "[6] Srivastava và cộng sự (2014), Dropout; Ioffe, Szegedy (2015), BatchNorm; Kingma, Ba (2015), Adam; Selvaraju và cộng sự (2017), Grad-CAM; Dosovitskiy và cộng sự (2021), ViT",
                          "[7] Thư viện: scikit-learn, scikit-image, OpenCV, TensorFlow và Keras, Gradio",
                          "Danh sách đầy đủ (40 tài liệu) và nguồn của từng hình ở mục 7 của báo cáo"],
              "notes": "Đây là các tài liệu tham khảo chính. Danh sách đầy đủ nằm trong báo cáo."})
    s.append({"title": "Cảm ơn thầy và các bạn đã lắng nghe", "who": "Chung", "sec": 10,
              "bullets": ["Nhóm xin nhận câu hỏi và góp ý", f"Mã nguồn, báo cáo, slide và demo: {REPO}"],
              "notes": "Phần trình bày của nhóm em đến đây là hết. Em cảm ơn thầy và các bạn đã lắng nghe. Nhóm em xin nhận câu hỏi và góp ý ạ."})
    return s


def all_slides():
    main = main_slides()
    extra = detail_slides()
    if extra:
        extra[0]["appendix"] = True
        for x in extra:
            x["title"] = "Phụ lục. " + x["title"]
            x.setdefault("who", "Phụ lục")
    return main, extra


def write_script(main, out):
    """Kịch bản thuyết trình: lời nói cho từng slide chính, lấy từ trường notes; phần demo và hỏi đáp lấy từ docs/kich-ban-phan-co-dinh.md."""
    total = sum(x.get("sec", 0) for x in main)
    L = ["# Kịch bản thuyết trình đồ án CS231.R12: Nhận diện giống chuối từ ảnh", "",
         "File này do `python -m src.make_slides` sinh ra từ cùng danh sách slide với `slides/bao-cao-do-an.pdf`. "
         "Muốn sửa lời thuyết trình thì sửa trường `notes` trong `src/make_slides.py`; phần demo và hỏi đáp sửa ở `docs/kich-ban-phan-co-dinh.md`.", "",
         f"Tổng thời gian dự kiến của {len(main)} slide chính: khoảng {round(total / 60)} phút (gồm khoảng 2 phút demo). "
         "Các slide phụ lục không trình bày, chỉ mở khi được hỏi.", "",
         "## 1. Phân công trình bày", "",
         "Phần phương pháp (mục 3 của phiếu đánh giá, 30 điểm) được chấm riêng từng thành viên, nên ai làm phần nào thì trình bày phần đó. Các phần còn lại là điểm chung, một bạn đại diện trình bày.", "",
         "| Phần | Slide | Người trình bày |", "|---|---|---|"]
    groups = {}
    for k, x in enumerate(main, 1):
        groups.setdefault(x["who"], []).append(k)
    label = {"Chung": "Phần chung (mở đầu, bài toán, dataset, độ đo, kết quả, demo, kết luận)", "Phần 1": "Phần 1: đặc trưng màu và KNN",
             "Phần 2": "Phần 2: cạnh, HOG, ghép đặc trưng, bộ phân lớp tuyến tính", "Phần 3": "Phần 3: CNN hướng A và hướng B"}
    for w in ["Chung", "Phần 1", "Phần 2", "Phần 3"]:
        L.append(f"| {label[w]} | {', '.join(map(str, groups.get(w, [])))} | {PRESENTERS.get(w, '')} |")
    L += ["", "Cột \"Người trình bày\" để nhóm tự điền theo đúng phần mỗi bạn đã làm.", "", "## 2. Lời thuyết trình theo từng slide", ""]
    for k, x in enumerate(main, 1):
        L += [f"### Slide {k}. {x['title']}", "", f"*{x['who']}, khoảng {x.get('sec', 30)} giây.*", "", x.get("notes", ""), ""]
    fixed = ROOT / "docs" / "kich-ban-phan-co-dinh.md"
    if fixed.exists():
        L += [fixed.read_text(encoding="utf-8").strip(), ""]
    out.write_text("\n".join(L), encoding="utf-8")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from .beamer_builder import build_tex, compile_pdf
    main_s, extra = all_slides()
    sl = main_s + extra
    out = ROOT / "slides" / "bao-cao-do-an.pptx"; out.parent.mkdir(exist_ok=True)
    build(sl, out, f"{TITLE}: đồ án {COURSE.split(':')[0]}")
    print("đã ghi", out.relative_to(ROOT))
    tex = ROOT / "slides" / "bao-cao-do-an.tex"
    author = r" \\ ".join(f"{n} \\quad {m}" for n, m in MEMBERS)
    build_tex(sl, tex, title=TITLE, subtitle=f"Đồ án môn {COURSE}", author_tex=author, institute="", date="")
    print("đã ghi", tex.relative_to(ROOT))
    for f in compile_pdf(tex):
        print("đã ghi", f.relative_to(ROOT))
    script = ROOT / "docs" / "kich-ban-thuyet-trinh.md"
    write_script(main_s, script)
    print("đã ghi", script.relative_to(ROOT))


if __name__ == "__main__":
    main()
