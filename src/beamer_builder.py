"""Dựng slide Beamer (LaTeX) theo style UIT từ cùng danh sách slide với bản pptx (src.make_slides).

Sinh slides/<tên>.tex rồi biên dịch bằng pdflatex thành:
  <tên>.pdf           slide trình chiếu
  <tên>-ghi-chu.pdf   slide kèm trang ghi chú thuyết trình (\\note) sau mỗi slide
Theme: slides/beamerthemeUIT.sty; logo: slides/figures/uit-logo.png.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .pptx_builder import md_rows

SUBS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"), ("×", r"$\times$"), ("±", r"$\pm$"), ("→", r"$\rightarrow$"), ("≤", r"$\le$"), ("≥", r"$\ge$"), ("≈", r"$\approx$"),
        ("−", r"$-$"), ("°", r"$^\circ$"), ("σ", r"$\sigma$"), ("Δ", r"$\Delta$"), ("·", r"$\cdot$"), ("…", r"\ldots{}"), ("⁻⁷", r"$^{-7}$"),
        ("⁻⁴", r"$^{-4}$"), ("⁻⁵", r"$^{-5}$"), ("²", r"$^2$"), ("³", r"$^3$"), ("—", "---"), ("–", "--"), ("“", "``"), ("”", "''"), ("‖", r"$\Vert$"),
        ("<", r"$<$"), (">", r"$>$"), ("|", r"$|$")]


def esc(text: str) -> str:
    """Thoát ký tự đặc biệt của LaTeX; **đậm** -> \\textbf."""
    parts = re.split(r"(\*\*[^*]+\*\*)", str(text))
    out = []
    for p in parts:
        bold = p.startswith("**") and p.endswith("**") and len(p) > 4
        t = p[2:-2] if bold else p
        for a, b in SUBS:
            t = t.replace(a, b)
        out.append(r"\textbf{" + t + "}" if bold else t)
    return "".join(out)


def _bullets(items, size="") -> str:
    out, level = [], -1
    for it in items:
        text, lv = (it, 0) if isinstance(it, str) else it
        while level < lv:
            out.append(r"\begin{itemize}" + (size if level < 0 else "")); level += 1
        while level > lv:
            out.append(r"\end{itemize}"); level -= 1
        out.append(r"\item " + esc(text))
    while level >= 0:
        out.append(r"\end{itemize}"); level -= 1
    return "\n".join(out)


def _rows(s):
    rows = md_rows(s["table"], s.get("max_rows"))
    if s.get("cols"):
        idx = [rows[0].index(c) for c in s["cols"]]; rows = [[r[i] for i in idx] for r in rows]
    return rows


def _table(s, max_h: str = "0.74\\textheight") -> str:
    rows = _rows(s)
    nc = len(rows[0])
    body = [" & ".join(r"\textbf{" + esc(c) + "}" for c in rows[0]) + r" \\ \midrule"]
    body += [" & ".join(esc(c) for c in r) + r" \\" for r in rows[1:]]
    spec = "l" * nc
    return ("\\begin{center}\\begin{adjustbox}{max width=\\linewidth,max totalheight=" + max_h + "}\n\\rowcolors{2}{uitbg}{white}\n"
            f"\\begin{{tabular}}{{{spec}}}\\toprule\n" + "\n".join(body) + "\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\\end{center}")


def _image(path, slides_dir: Path, width="\\linewidth", height="0.74\\textheight") -> str:
    rel = os.path.relpath(Path(path).resolve(), slides_dir.resolve()).replace(os.sep, "/")
    return f"\\begin{{center}}\\includegraphics[width={width},height={height},keepaspectratio]{{{rel}}}\\end{{center}}"


def frame(s, slides_dir: Path) -> str:
    if s.get("layout") == "title":
        return ""
    out = []
    if s.get("section") and not s.get("toc"):
        out.append(f"\\section{{{esc(s['section'])}}}")
    out.append(f"\\begin{{frame}}{{{esc(s['title'])}}}")
    if s.get("toc"):
        out.append("\\tableofcontents")
        if s.get("notes"):
            out.append("\\note{" + esc(s["notes"]) + "}")
        out.append("\\end{frame}")
        return "\n".join(out)
    side = s.get("bullets") and (s.get("image") or s.get("table"))
    size = "\\small" if side or len(s.get("bullets") or []) > 6 else ""
    if side and s.get("table") and len(_rows(s)[0]) > 4:      # bảng nhiều cột: chữ ở trên, bảng rộng hết khung ở dưới
        out.append(_bullets(s["bullets"], "\\footnotesize"))
        out.append(_table(s, "0.40\\textheight"))
    elif side:
        out.append("\\begin{columns}[T]\n\\begin{column}{0.36\\textwidth}")
        out.append(_bullets(s["bullets"], "\\footnotesize"))
        out.append("\\end{column}\n\\begin{column}{0.62\\textwidth}")
        out.append(_image(s["image"], slides_dir, height="0.72\\textheight") if s.get("image") else _table(s))
        out.append("\\end{column}\n\\end{columns}")
    elif s.get("bullets"):
        out.append(_bullets(s["bullets"], size))
    elif s.get("image"):
        out.append(_image(s["image"], slides_dir))
    elif s.get("table"):
        out.append(_table(s))
    if s.get("notes"):
        out.append("\\note{" + esc(s["notes"]) + "}")
    out.append("\\end{frame}")
    return "\n".join(out)


def build_tex(slides: list[dict], out_tex: Path, title: str, subtitle: str = "", author: str = "", institute: str = "", date: str = "") -> str:
    slides_dir = out_tex.parent
    t0 = next((s for s in slides if s.get("layout") == "title"), {})
    pre = f"""% Slide trình bày đồ án — Beamer, style UIT. Biên dịch: pdflatex {out_tex.name} (2 lần), hoặc tải thư mục slides/ lên Overleaf.
% Bật trang ghi chú thuyết trình: đổi \\setbeameroption{{hide notes}} thành \\setbeameroption{{show notes}}.
\\documentclass[aspectratio=169,10pt]{{beamer}}
\\usepackage[utf8]{{vietnam}}
\\usepackage{{lmodern}}
\\usepackage{{booktabs,adjustbox,colortbl}}
\\usepackage[table]{{xcolor}}
\\usetheme{{UIT}}
\\ifdefined\\shownotes\\setbeameroption{{show notes}}\\else\\setbeameroption{{hide notes}}\\fi
\\title[{esc(title)}]{{{esc(title)}}}
\\subtitle{{{esc(subtitle)}}}
\\author{{{esc(author)}}}
\\institute{{{esc(institute)}}}
\\date{{{esc(date)}}}
\\begin{{document}}
\\begin{{frame}}[plain]
\\titlepage
{("\\note{" + esc(t0["notes"]) + "}") if t0.get("notes") else ""}
\\end{{frame}}
"""
    body = "\n\n".join(f for f in (frame(s, slides_dir) for s in slides) if f)
    tex = pre + body + "\n\\end{document}\n"
    out_tex.write_text(tex, encoding="utf-8")
    return tex


def compile_pdf(out_tex: Path) -> list[Path]:
    """Biên dịch bản slide và bản kèm ghi chú. Trả về danh sách PDF đã sinh (rỗng nếu máy không có pdflatex)."""
    if not shutil.which("pdflatex"):
        print("Không tìm thấy pdflatex: chỉ sinh file .tex (biên dịch trên Overleaf hoặc cài texlive).")
        return []
    d, name = out_tex.parent, out_tex.stem
    done = []
    for job, pre in [(name, ""), (name + "-ghi-chu", "\\def\\shownotes{1}")]:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}", pre + f"\\input{{{out_tex.name}}}"]
        for _ in range(2):
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2500:]); raise SystemExit(f"pdflatex lỗi khi biên dịch {job}")
        for ext in (".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb"):
            (d / (job + ext)).unlink(missing_ok=True)
        done.append(d / (job + ".pdf"))
    return done
