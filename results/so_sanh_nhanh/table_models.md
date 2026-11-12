### So sánh từng cấu hình qua các nhánh: macro-F1 trên test gốc (371 ảnh), số ảnh sai, điểm bền (trung bình macro-F1 trên 6 phép thử khó)

| Cấu hình | G: macro-F1 test | G: số ảnh sai | G: điểm bền | T: macro-F1 test | T: số ảnh sai | T: điểm bền |
|---|---|---|---|---|---|---|
| Hist HSV + LogReg | 0.994 | 2 | 0.487 | 0.997 | 1 | 0.555 |
| Cạnh Roberts + KNN | 0.964 | 12 | 0.654 | 0.948 | 18 | 0.689 |
| HOG + LogReg | 0.990 | 3 | 0.617 | 0.972 | 8 | 0.620 |
| Hist H-S + HOG + LogReg | 1.000 | 0 | 0.728 | 0.989 | 4 | 0.745 |
| a_lenet | 0.978 | 7 | 0.837 | 0.966 | 10 | 0.780 |
| a_dropout | 0.942 | 21 | 0.769 | 0.963 | 13 | 0.796 |
| a_vgg16 | 0.986 | 4 | 0.886 | 0.985 | 5 | 0.905 |
| b_mobilenetv3 | 1.000 | 0 | 0.875 | 0.998 | 1 | 0.909 |
| b_resnet50 | 0.997 | 1 | 0.895 | 0.997 | 1 | 0.924 |
| **Trung bình 9 cấu hình** | **0.983** | **5.6** | **0.750** | **0.979** | **6.8** | **0.769** |
