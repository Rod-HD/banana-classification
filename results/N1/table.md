### N1: Số tham số từng lớp: tính tay theo công thức so với Keras

| Mô hình | Lớp | Công thức | Tính tay | model.summary() | Khớp | Ghi chú |
|---|---|---|---|---|---|---|
| a_lenet | conv1 | (5×5×3+1)×6 | 456 | 456 | có |  |
| a_lenet | conv2 | (5×5×6+1)×16 | 2416 | 2416 | có |  |
| a_lenet | fc1 | (2704+1)×120 | 324600 | 324600 | có | Flatten: 13×13×16 = 2704 |
| a_lenet | fc2 | (120+1)×84 | 10164 | 10164 | có |  |
| a_lenet | softmax | (84+1)×4 | 340 | 340 | có |  |
| a_lenet | Tổng |  | 337976 | 337976 | có | đầu vào 64×64×3 |
| a_dropout | conv1 | (5×5×3+1)×32 | 2432 | 2432 | có |  |
| a_dropout | conv2 | (5×5×32+1)×16 | 12816 | 12816 | có |  |
| a_dropout | fc1 | (13456+1)×128 | 1722496 | 1722496 | có | Flatten: 29×29×16 = 13456 |
| a_dropout | fc2 | (128+1)×64 | 8256 | 8256 | có |  |
| a_dropout | softmax | (64+1)×4 | 260 | 260 | có |  |
| a_dropout | Tổng |  | 1746260 | 1746260 | có | đầu vào 128×128×3 |
