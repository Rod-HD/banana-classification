#!/bin/bash
# Cài các thư viện cần cho đồ án (máy chỉ có CPU).
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.." || exit 1
pip install -q -r requirements.txt
