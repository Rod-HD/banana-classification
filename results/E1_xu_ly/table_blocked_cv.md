### X1: Macro-F1 trên train+val: chia ngẫu nhiên so với chia theo khối ảnh có số thứ tự liên tiếp

| Mô hình | 5-fold ngẫu nhiên phân tầng | 5-fold theo khối ảnh liên tiếp | Chênh |
|---|---|---|---|
| Hist HSV (8,8,8) + LogReg C=0.1 | 0.991 ± 0.005 | 0.922 ± 0.042 | +0.069 |
| HOG 64×128 + LogReg C=0.1 | 0.986 ± 0.007 | 0.933 ± 0.062 | +0.053 |
| Hist H-S + HOG + LogReg C=100 | 0.996 ± 0.003 | 0.973 ± 0.017 | +0.024 |
| Embedding MobileNetV3 (ImageNet) + LogReg | 0.995 ± 0.005 | 0.964 ± 0.025 | +0.031 |
