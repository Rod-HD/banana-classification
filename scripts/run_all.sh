#!/usr/bin/env bash
# Chạy lại toàn bộ thí nghiệm của đồ án theo đúng thứ tự ở mục 11 của kế hoạch.
# Kết quả ghi vào results/, hình vào results/figures/, log vào logs/.
#
#   bash scripts/run_all.sh            # chạy tất cả các bước
#   bash scripts/run_all.sh 2 3        # chỉ chạy bước 2 và 3
#   FRESH=1 bash scripts/run_all.sh    # xóa data/cache/ trước (rút lại toàn bộ đặc trưng)
#
# Yêu cầu: đã cài thư viện (bash scripts/setup_env.sh) và có ảnh trong data/raw/
# (có sẵn trong repo; tải lại từ nguồn bằng bash scripts/download_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p logs
[ "${FRESH:-0}" = "1" ] && rm -rf data/cache

STEPS=("$@")
want() { [ ${#STEPS[@]} -eq 0 ] && return 0; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
run() { echo ">>> python -m $*"; python -m "$@" 2>&1 | tee "logs/$(echo "$1" | tr '.' '_').log"; }

if want 1; then   # Bước 1: chia dữ liệu, ảnh gần trùng, thống kê, D1
  run src.env_info
  run src.split_data
  run src.exp_d1
fi
if want 2; then   # Bước 2: đặc trưng màu (M1–M6), bộ phân lớp (P1–P6)
  run src.exp_color
  run src.exp_classifier
fi
