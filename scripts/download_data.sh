#!/usr/bin/env bash
# Tải lại ảnh gốc BananaImageBD (phần phân loại giống) từ Mendeley Data và
# sắp xếp vào data/raw/<lớp>/ với tên thư mục snake_case.
set -euo pipefail

URL="https://data.mendeley.com/public-files/datasets/ptfscwtnyz/files/2073f7f0-0531-466d-939a-21da4c0b1d3b/file_downloaded"
SHA256="df5c7ceac7be073fff761b7f8fbc9a7c65d34ee9fb7b4ac64905964b2aabbf1f"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fsSL -o "$TMP/banana.zip" "$URL"
echo "$SHA256  $TMP/banana.zip" | sha256sum -c -
unzip -q "$TMP/banana.zip" -d "$TMP/x"

for cls in "Bangla Kola" "Champa Kola" "Sabri Kola" "Sagor Kola"; do
  dst="$ROOT/data/raw/$(echo "$cls" | tr 'A-Z ' 'a-z_')"
  mkdir -p "$dst"
  cp "$TMP/x/Banana Classification Dataset/$cls"/* "$dst"/
  echo "$dst: $(ls "$dst" | wc -l) ảnh"
done
