### Chốt cấu hình truyền thống: cấu hình tốt nhất của mỗi bộ phân lớp trên từng đặc trưng (5-fold CV)

| Họ | Đặc trưng | Cấu hình tốt nhất của bộ phân lớp | Số chiều | Macro-F1 | Accuracy | Huấn luyện (s) | Dự đoán (ms/ảnh) |
|---|---|---|---|---|---|---|---|
| màu | Hist HSV (8,8,8) | KNN k=1, Hellinger, ưu tiên điểm gần | 512 | 0.987 ± 0.006 | 0.989 ± 0.006 | 0.00 | 0.022 |
| màu | Hist HSV (8,8,8) | LogReg C=0.1 | 512 | 0.991 ± 0.005 | 0.992 ± 0.005 | 0.87 | 0.003 |
| màu | Hist HSV (8,8,8) | Linear SVM C=0.1 | 512 | 0.990 ± 0.004 | 0.991 ± 0.004 | 2.69 | 0.006 |
| màu | Hist H-S (8,8) | KNN k=1, Manhattan, ưu tiên điểm gần | 64 | 0.986 ± 0.004 | 0.986 ± 0.004 | 0.00 | 0.035 |
| màu | Hist H-S (8,8) | LogReg C=1 | 64 | 0.985 ± 0.008 | 0.986 ± 0.007 | 0.02 | 0.001 |
| màu | Hist H-S (8,8) | Linear SVM C=0.1 | 64 | 0.973 ± 0.004 | 0.977 ± 0.003 | 0.03 | 0.001 |
| màu | Moment màu | KNN k=1, Manhattan, ưu tiên điểm gần | 9 | 0.978 ± 0.004 | 0.981 ± 0.003 | 0.00 | 0.009 |
| màu | Moment màu | LogReg C=10 | 9 | 0.910 ± 0.011 | 0.926 ± 0.008 | 0.03 | 0.001 |
| màu | Moment màu | Linear SVM C=1 | 9 | 0.874 ± 0.007 | 0.894 ± 0.004 | 0.01 | 0.001 |
| cạnh | Cạnh Roberts | KNN k=1, Manhattan, ưu tiên điểm gần | 512 | 0.955 ± 0.015 | 0.959 ± 0.013 | 0.01 | 0.182 |
| cạnh | Cạnh Roberts | LogReg C=1 | 512 | 0.932 ± 0.011 | 0.940 ± 0.009 | 4.53 | 0.003 |
| cạnh | Cạnh Roberts | Linear SVM C=0.01 | 512 | 0.925 ± 0.014 | 0.937 ± 0.012 | 0.52 | 0.005 |
| cạnh | Cạnh Sobel | KNN k=1, Manhattan, ưu tiên điểm gần | 512 | 0.952 ± 0.014 | 0.956 ± 0.012 | 0.01 | 0.185 |
| cạnh | Cạnh Sobel | LogReg C=1 | 512 | 0.894 ± 0.008 | 0.900 ± 0.006 | 5.81 | 0.004 |
| cạnh | Cạnh Sobel | Linear SVM C=0.1 | 512 | 0.883 ± 0.014 | 0.895 ± 0.013 | 0.98 | 0.005 |
| HOG | HOG 64×128 | KNN k=1, Manhattan, ưu tiên điểm gần | 3780 | 0.983 ± 0.004 | 0.984 ± 0.003 | 0.06 | 1.820 |
| HOG | HOG 64×128 | LogReg C=0.1 | 3780 | 0.986 ± 0.007 | 0.989 ± 0.006 | 3.04 | 0.026 |
| HOG | HOG 64×128 | Linear SVM C=0.001 | 3780 | 0.983 ± 0.003 | 0.985 ± 0.002 | 1.63 | 0.021 |
| ghép | Hist HSV + HOG | KNN k=1, Manhattan, ưu tiên điểm gần, chuẩn hóa từng phần | 4292 | 0.989 ± 0.003 | 0.989 ± 0.002 | 0.10 | 2.122 |
| ghép | Hist HSV + HOG | LogReg C=10, chuẩn hóa từng phần | 4292 | 0.996 ± 0.003 | 0.997 ± 0.002 | 2.48 | 0.035 |
| ghép | Hist HSV + HOG | Linear SVM C=1, chuẩn hóa từng phần | 4292 | 0.996 ± 0.004 | 0.996 ± 0.003 | 1.22 | 0.027 |
| ghép | Hist H-S + HOG | KNN k=1, Manhattan, ưu tiên điểm gần, chuẩn hóa từng phần | 3844 | 0.990 ± 0.003 | 0.990 ± 0.003 | 0.07 | 1.877 |
| ghép | Hist H-S + HOG | LogReg C=100, chuẩn hóa từng phần | 3844 | 0.996 ± 0.003 | 0.997 ± 0.003 | 1.07 | 0.032 |
| ghép | Hist H-S + HOG | Linear SVM C=1, chuẩn hóa từng phần | 3844 | 0.996 ± 0.003 | 0.997 ± 0.003 | 1.00 | 0.025 |
