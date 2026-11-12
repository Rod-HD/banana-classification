### Phép che 25%: accuracy trên test theo tỉ lệ diện tích quả chuối bị ô đen che (nhóm bị che ≥ 90% gồm 5 ảnh Bangla Kola, 35 ảnh Sabri Kola)

| Nhánh | Cấu hình | Quả bị che ≥ 90% (40 ảnh) | Quả bị che 30–90% (286 ảnh) | Quả bị che < 30% (45 ảnh) |
|---|---|---|---|---|
| G | b_mobilenetv3 | 0.175 | 0.699 | 0.911 |
| G | Hist H-S + HOG + LogReg | 0.950 | 0.965 | 0.978 |
| X-a | b_mobilenetv3 | 0.125 | 0.451 | 0.667 |
| X-a | Hist H-S + HOG + LogReg | 0.750 | 0.850 | 0.844 |
| X-b | b_mobilenetv3 | 0.125 | 0.448 | 0.822 |
| X-b | Hist H-S + HOG + LogReg | 0.975 | 0.815 | 0.933 |
| T | b_mobilenetv3 | 0.550 | 0.717 | 1.000 |
| T | Hist H-S + HOG + LogReg | 0.975 | 0.962 | 1.000 |
