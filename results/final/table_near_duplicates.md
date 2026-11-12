### Accuracy trên test tách theo việc có ảnh gần trùng (pHash) trong tập huấn luyện

| Ngưỡng Hamming | Mô hình | Ảnh test có ảnh gần trùng trong train+val | Accuracy trên nhóm đó | Ảnh test không có | Accuracy trên nhóm không có |
|---|---|---|---|---|---|
| ≤ 2 | trad_fusion | 9 | 1.000 | 362 | 1.000 |
| ≤ 2 | a_vgg16 | 9 | 1.000 | 362 | 0.989 |
| ≤ 2 | b_mobilenetv3 | 9 | 1.000 | 362 | 1.000 |
| ≤ 4 | trad_fusion | 50 | 1.000 | 321 | 1.000 |
| ≤ 4 | a_vgg16 | 50 | 1.000 | 321 | 0.988 |
| ≤ 4 | b_mobilenetv3 | 50 | 1.000 | 321 | 1.000 |
