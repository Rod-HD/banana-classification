#!/bin/bash
# Cài các thư viện cần cho đồ án (máy chỉ có CPU).
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.." || exit 1
pip install -q -r requirements.txt
# Slide Beamer cần pdflatex. Nếu máy dùng apt và chưa có thì thử cài (không bắt buộc: thiếu pdflatex thì chỉ sinh file .tex).
if ! command -v pdflatex >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-lang-other texlive-pictures lmodern >/dev/null 2>&1 || echo "Không cài được texlive; bỏ qua."
fi
