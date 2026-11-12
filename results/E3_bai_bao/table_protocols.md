### E3 — Phương pháp của bài báo trên bản Augmented (7.413 ảnh), 5-fold: như bài báo (i) so với giữ nhóm theo ảnh gốc (ii)

| Hướng | Bộ rút đặc trưng | Bộ phân lớp | Bài báo công bố: accuracy (%) | (i) Như bài báo: accuracy (%) | (i) Macro-F1 | (ii) Giữ nhóm theo ảnh gốc: accuracy (%) | (ii) Macro-F1 | Thổi phồng (i) − (ii), điểm % |
|---|---|---|---|---|---|---|---|---|
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | SVM (RBF, C=1) | 96.69 | 99.16 ± 0.15 | 0.9900 | 98.97 ± 0.38 | 0.9881 | +0.19 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | SVM tuyến tính (C=0,01) | — | 99.15 ± 0.16 | 0.9901 | 98.85 ± 0.33 | 0.9864 | +0.30 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | MLP (1 lớp ẩn 256) | 97.76 | 98.97 ± 0.28 | 0.9881 | 98.87 ± 0.20 | 0.9868 | +0.11 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | Logistic Regression | — | 99.12 ± 0.15 | 0.9894 | 98.96 ± 0.29 | 0.9876 | +0.16 |
| B1 | ResNet50 (ImageNet), sau GAP, 2048 chiều | KNN (k=5) | — | 93.96 ± 0.47 | 0.9356 | 93.05 ± 0.75 | 0.9258 | +0.90 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | SVM (RBF, C=1) | — | 99.27 ± 0.21 | 0.9912 | 99.08 ± 0.31 | 0.9891 | +0.19 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | SVM tuyến tính (C=0,01) | — | 98.80 ± 0.34 | 0.9858 | 98.60 ± 0.39 | 0.9836 | +0.20 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | MLP (1 lớp ẩn 256) | — | 99.22 ± 0.11 | 0.9909 | 98.91 ± 0.32 | 0.9875 | +0.31 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | Logistic Regression | — | 98.68 ± 0.22 | 0.9844 | 98.50 ± 0.36 | 0.9828 | +0.18 |
| B1 | MobileNetV3-Small (ImageNet), sau GAP, 576 chiều | KNN (k=5) | — | 96.07 ± 0.62 | 0.9572 | 95.55 ± 0.81 | 0.9516 | +0.53 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | SVM (RBF, C=1) | 97.63 | 98.58 ± 0.42 | 0.9836 | 98.12 ± 0.29 | 0.9783 | +0.46 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | SVM tuyến tính (C=0,01) | — | 98.03 ± 0.43 | 0.9772 | 97.65 ± 0.14 | 0.9726 | +0.38 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | MLP (1 lớp ẩn 256) | 95.88 | 98.57 ± 0.20 | 0.9837 | 98.35 ± 0.24 | 0.9815 | +0.22 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | Logistic Regression | — | 97.59 ± 0.41 | 0.9727 | 97.33 ± 0.40 | 0.9693 | +0.26 |
| B1 | SqueezeNet 1.1 (ImageNet), sau GAP, 512 chiều | KNN (k=5) | — | 91.73 ± 0.33 | 0.9079 | 91.61 ± 0.70 | 0.9064 | +0.12 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | SVM (RBF, C=1) | 99.70 | 99.49 ± 0.15 | 0.9937 | 99.20 ± 0.33 | 0.9905 | +0.28 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | SVM tuyến tính (C=0,01) | — | 99.41 ± 0.19 | 0.9929 | 99.12 ± 0.37 | 0.9895 | +0.28 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | MLP (1 lớp ẩn 256) | 99.29 | 99.50 ± 0.21 | 0.9942 | 99.03 ± 0.38 | 0.9886 | +0.47 |
| B2 | ViT-B/16 (ImageNet), token phân lớp, 768 chiều | GBM (HistGradientBoosting) | 97.39 | 97.71 ± 0.36 | 0.9734 | 97.14 ± 0.60 | 0.9666 | +0.57 |
