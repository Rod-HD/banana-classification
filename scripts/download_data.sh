#!/usr/bin/env bash
# Tải ảnh BananaImageBD (phần phân loại giống) từ Mendeley Data, kiểm tra SHA-256 và
# sắp xếp vào thư mục theo lớp với tên snake_case.
#   bash scripts/download_data.sh             # ảnh gốc -> data/raw/<lớp>/
#   bash scripts/download_data.sh augmented   # bản Augmented của tác giả (7.413 ảnh) -> data/aug_author/<lớp>/
set -euo pipefail

if [ "${1:-raw}" = "augmented" ]; then
  URL="https://data.mendeley.com/public-files/datasets/ptfscwtnyz/files/dd3545ea-a7eb-41d4-a05c-8da3098c4aff/file_downloaded"
  SHA256="e56e2f874dd86146bc4e912299c693fbc0485e9e3f83eade850df03a99240ea2"
  FOLDER="Augmented Banana Classification Dataset"; DEST="data/aug_author"
else
  URL="https://data.mendeley.com/public-files/datasets/ptfscwtnyz/files/2073f7f0-0531-466d-939a-21da4c0b1d3b/file_downloaded"
  SHA256="df5c7ceac7be073fff761b7f8fbc9a7c65d34ee9fb7b4ac64905964b2aabbf1f"
  FOLDER="Banana Classification Dataset"; DEST="data/raw"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fsSL -o "$TMP/banana.zip" "$URL"
echo "$SHA256  $TMP/banana.zip" | sha256sum -c -
unzip -q "$TMP/banana.zip" -d "$TMP/x"

for cls in "Bangla Kola" "Champa Kola" "Sabri Kola" "Sagor Kola"; do
  dst="$ROOT/$DEST/$(echo "$cls" | tr 'A-Z ' 'a-z_')"
  mkdir -p "$dst"
  cp "$TMP/x/$FOLDER/$cls"/* "$dst"/
  echo "$dst: $(ls "$dst" | wc -l) ảnh"
done
