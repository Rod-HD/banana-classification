### Ảnh test bị từ 2 mô hình trở lên đoán sai

| Ảnh | Lớp đúng | Số mô hình sai / 9 | Các mô hình sai → lớp đoán |
|---|---|---|---|
| data/raw/sabri_kola/Sabri_Kola_5.jpg | Sabri Kola | 6 | trad_edge→Sagor; trad_hog→Sagor; a_lenet→Bangla; a_dropout→Sagor; a_vgg16→Bangla; b_resnet50→Bangla |
| data/raw/bangla_kola/Bangla_Kola_189.jpg | Bangla Kola | 3 | a_lenet→Sagor; a_dropout→Sagor; a_vgg16→Sagor |
| data/raw/bangla_kola/Bangla_Kola_422.jpg | Bangla Kola | 2 | trad_hog→Sagor; a_dropout→Sagor |
| data/raw/bangla_kola/Bangla_Kola_195.jpg | Bangla Kola | 2 | a_lenet→Sagor; a_dropout→Sagor |
