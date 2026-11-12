"""Tiện ích ghi bảng kết quả (markdown, json) và thiết lập hình vẽ."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RESULTS_DIR

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    cols = cols or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def exp_dir(name: str) -> Path:
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def save_table(exp: str, rows: list[dict], name: str = "table", cols=None, title: str = "") -> str:
    """Ghi results/<exp>/<name>.md và trả về chuỗi markdown."""
    md = (f"### {title}\n\n" if title else "") + md_table(rows, cols)
    (exp_dir(exp) / f"{name}.md").write_text(md)
    return md


def pm(mean, std, d=3):
    return f"{mean:.{d}f} ± {std:.{d}f}"
