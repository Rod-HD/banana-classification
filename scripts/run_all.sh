#!/usr/bin/env bash
# Chạy lại toàn bộ thí nghiệm của đồ án theo đúng thứ tự ở mục 11 của kế hoạch.
# Kết quả ghi vào results/, hình vào results/figures/, log vào logs/.
#
#   bash scripts/run_all.sh            # chạy tất cả các bước
#   bash scripts/run_all.sh 2 3        # chỉ chạy bước 2 và 3
# Các bước: 1 dữ liệu + D1 | 2 màu, bộ phân lớp | 3 cạnh, HOG, chốt cấu hình | 4 CNN hướng A | 5 CNN hướng B + N1–N8
#           6 mất cân bằng | 7 phân đoạn | 8 đánh giá cuối trên test | 9 demo, báo cáo, slide
# Phần mở rộng (mục 12 của kế hoạch), cần kết quả của bước 1–8:
#           10 bộ kiểm tra khó + nhánh G | 11 nhánh T (bản Augmented của tác giả) | 12 nhánh B (phương pháp của bài báo)
#           13 nhánh X (xử lý dữ liệu) | 14 so sánh bốn nhánh, tổng hợp, rồi dựng lại báo cáo và slide
#   FRESH=1 bash scripts/run_all.sh    # xóa data/cache/ trước (rút lại toàn bộ đặc trưng)
#   RETRAIN=1 bash scripts/run_all.sh  # xóa checkpoints/ trước (huấn luyện lại mọi CNN từ đầu; thời gian huấn luyện đã ghi là 3,3 giờ)
#   SKIP_DONE=1 bash scripts/run_all.sh 11 13   # nhánh T, X: bỏ qua cấu hình đã có kết quả (chạy tiếp sau khi bị ngắt)
#
# Trọng số CNN đã huấn luyện có sẵn trong checkpoints/<run>/best.keras, nên mặc định các bước huấn luyện
# không chạy lại: mỗi lệnh train nạp trọng số đã lưu rồi chấm lại.
#
# Yêu cầu: đã cài thư viện (bash scripts/setup_env.sh) và có ảnh trong data/raw/
# (có sẵn trong repo; tải lại từ nguồn bằng bash scripts/download_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p logs
[ "${FRESH:-0}" = "1" ] && rm -rf data/cache
[ "${RETRAIN:-0}" = "1" ] && rm -rf checkpoints

STEPS=("$@")
want() { [ ${#STEPS[@]} -eq 0 ] && return 0; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
run() { echo ">>> python -m $*"; python -m "$@" 2>&1 | tee "logs/$(echo "$1" | tr '.' '_').log"; }
# Huấn luyện một CNN; log vào logs/cnn_<tên>.log; --resume: run đã xong thì dùng lại trọng số, run bị ngắt thì chạy tiếp.
train() { local name="$1"; shift; echo ">>> train $name: $*"
          python -m src.cnn.train "$@" --resume 2>&1 | grep --line-buffered -vE "^(WARNING|I0000|W0000|E0000)|oneDNN|cpu_feature|To enable" \
            | tee "logs/cnn_$name.log" | grep --line-buffered -E "run=|KẾT QUẢ|tiếp tục|dùng lại"; }

if want 1; then   # Bước 1: chia dữ liệu, ảnh gần trùng, thống kê, D1
  run src.env_info
  run src.split_data
  run src.exp_d1
fi
if want 2; then   # Bước 2: đặc trưng màu (M1–M6), bộ phân lớp (P1–P6)
  run src.exp_color
  run src.exp_classifier
fi
if want 3; then   # Bước 3: cạnh, HOG, ghép đặc trưng (C1–C5), chốt cấu hình truyền thống
  run src.exp_shape
  run src.exp_select
fi
if want 4; then   # Bước 4: CNN hướng A (theo bài giảng)
  train a_lenet           --model a_lenet
  train a_lenet_s32       --model a_lenet --size 32 --tag s32
  train a_lenet_s128      --model a_lenet --size 128 --tag s128
  train a_lenet_frac25    --model a_lenet --frac 0.25
  train a_lenet_frac50    --model a_lenet --frac 0.5
  train a_dropout         --model a_dropout --epochs 40 --patience 40 --train-eval
  train a_dropout_nodrop  --model a_dropout --no-dropout --tag nodrop --epochs 40 --patience 40 --train-eval
  train a_dropout_aug     --model a_dropout --augment --tag aug --epochs 40 --patience 40 --train-eval
  train a_vgg16           --model a_vgg16
  train a_vgg16_lr1e-3    --model a_vgg16 --lr 1e-3 --tag lr1e-3
  train a_vgg16_frac25    --model a_vgg16 --frac 0.25
  train a_vgg16_frac50    --model a_vgg16 --frac 0.5
fi
if want 5; then   # Bước 5: CNN hướng B (lớn hơn, fine-tune), rồi các phép đo N1–N8
  train b_mobilenetv3_s1        --model b_mobilenetv3 --stage 1
  train b_mobilenetv3           --model b_mobilenetv3 --stage 2 --lr 1e-5 --augment --train-eval
  train b_mobilenetv3_lr1e-4    --model b_mobilenetv3 --stage 2 --lr 1e-4 --augment --tag lr1e-4
  train b_mobilenetv3_lr1e-3    --model b_mobilenetv3 --stage 2 --lr 1e-3 --augment --tag lr1e-3
  train b_mobilenetv3_noaug     --model b_mobilenetv3 --stage 2 --lr 1e-5 --tag noaug --train-eval
  train b_mobilenetv3_s1_frac25 --model b_mobilenetv3 --stage 1 --frac 0.25
  train b_mobilenetv3_frac25    --model b_mobilenetv3 --stage 2 --lr 1e-5 --augment --frac 0.25
  train b_mobilenetv3_s1_frac50 --model b_mobilenetv3 --stage 1 --frac 0.5
  train b_mobilenetv3_frac50    --model b_mobilenetv3 --stage 2 --lr 1e-5 --augment --frac 0.5
  train b_resnet50_s1           --model b_resnet50 --stage 1
  train b_resnet50              --model b_resnet50 --stage 2 --lr 1e-5 --augment
  for sd in 43 44; do           # N3: lặp lại với 2 seed khởi tạo khác
    train a_dropout_seed$sd        --model a_dropout --epochs 40 --patience 40 --train-eval --seed $sd --tag seed$sd
    train a_dropout_nodrop_seed$sd --model a_dropout --no-dropout --epochs 40 --patience 40 --train-eval --seed $sd --tag nodrop_seed$sd
  done
  run src.exp_cnn
fi
if want 6; then   # Bước 6: mất cân bằng (I1–I2). Run gốc của I2 là biến thể được chốt ở bước 5:
                  # a_vgg16_lr1e-3 (hướng A) và b_mobilenetv3_lr1e-4 (hướng B).
  for m in cw os minaug; do
    flag=$([ $m = cw ] && echo --class-weight || ([ $m = os ] && echo --oversample || echo --minority-aug))
    train a_vgg16_lr1e-3_$m        --model a_vgg16 --lr 1e-3 $flag --tag lr1e-3_$m
    train b_mobilenetv3_lr1e-4_$m  --model b_mobilenetv3 --stage 2 --lr 1e-4 --augment $flag --tag lr1e-4_$m
  done
  run src.exp_imbalance
  python -m src.exp_cnn --only RUNS N4BN N6 N8 2>&1 | tee logs/src_exp_cnn_N6.log   # đo lại thời gian khi máy không chạy tác vụ khác
fi
if want 7; then   # Bước 7: phân đoạn tách nền (S1–S2); S2CNN cần các mô hình CNN của bước 5
  run src.exp_segment --only S1 S2 S2CNN
fi
if want 8; then   # Bước 8: đánh giá cuối trên test, tổng hợp
  run src.final_eval
  run src.error_analysis
  run src.summarize
fi
if want 10; then  # Bước 10: bộ kiểm tra khó; nhánh G (pipeline gốc) chấm trên test và bộ kiểm tra khó
  run src.stress
  run src.pipeline
fi
if want 11; then  # Bước 11: nhánh T: 9 cấu hình với bản Augmented của tác giả (có sẵn trong data/aug_author/; tự tải nếu thiếu)
  [ -d data/aug_author/bangla_kola ] || bash scripts/download_data.sh augmented
  run src.exp_arm_t
fi
if want 12; then  # Bước 12: nhánh B: phương pháp của bài báo (CNN hoặc ViT làm bộ rút đặc trưng + SVM/MLP/GBM); cần PyTorch
  [ -d data/aug_author/bangla_kola ] || bash scripts/download_data.sh augmented
  run src.exp_papers
fi
if want 13; then  # Bước 13: nhánh X: làm sạch, tiền xử lý, tăng cường, chạy lại 9 cấu hình; kiểm tra lại S2 bằng mask mới
  run src.exp_arm_x
  run src.exp_segment --only S2V2
fi
if want 14; then  # Bước 14: so sánh bốn nhánh, tổng hợp
  run src.compare_arms
  run src.summarize
fi
if want 9 || want 14; then   # Bước 9 (chạy lại sau bước 14): mô hình cho demo, ví dụ tính tay, báo cáo (md + docx), slide
  run src.export_demo
  run src.hand_calc
  run src.build_report
  run src.make_slides
fi
