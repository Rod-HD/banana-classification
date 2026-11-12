"""Chuyển báo cáo Markdown (report/report.md) sang Word (report/report.docx).

    python -m src.md2docx report/report.md report/report.docx

Hỗ trợ phần Markdown mà báo cáo dùng: tiêu đề #..####, đoạn văn, **đậm**, *nghiêng*, `code`, danh sách
gạch đầu dòng và đánh số, bảng, hình ![chú thích](đường_dẫn), khối code ``` và đường kẻ ---.
Thuộc tính tác giả của file: "Nhóm đồ án CS231".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AUTHOR = "Nhóm đồ án CS231"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(par, text, size=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        elif part.startswith("[") and "](" in part:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part).groups()
            r = par.add_run(label if url.startswith("#") else f"{label} ({url})" if label != url else label)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        if size:
            r.font.size = Pt(size)


def shade(cell, color="D9E2F3"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def _section(doc, landscape: bool):
    """Mở một section mới, trang ngang hoặc trang dọc (dùng cho bảng nhiều cột)."""
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    w, h = sec.page_width, sec.page_height
    if landscape != (w > h):
        sec.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = h, w
    return sec


def add_table(doc, lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|\s*:?-{2,}", l.strip())]
    ncol = max(len(r) for r in rows)
    wide = ncol >= 11
    if wide:
        _section(doc, True)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 9 if ncol <= 5 else 8 if ncol <= 8 else 7
    for i, r in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]
            add_inline(p, r[j] if j < len(r) else "", size)
            if i == 0:
                for run in p.runs:
                    run.bold = True
                shade(cell)
    if wide:
        _section(doc, False)
    else:
        doc.add_paragraph()


def convert(src: Path, dst: Path):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)
    width = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    lines = src.read_text(encoding="utf-8").splitlines()
    i = 0
    title_done = False
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            p = doc.add_paragraph(); r = p.add_run("\n".join(buf)); r.font.name = "Consolas"; r.font.size = Pt(9)
            i = j + 1; continue
        if l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            add_table(doc, lines[i:j]); i = j; continue
        m = re.match(r"^(#{1,4})\s+(.*)", l)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and not title_done:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(text); r.bold = True; r.font.size = Pt(18); title_done = True
            else:
                h = doc.add_heading(level=min(level, 3) if level > 1 else 1); add_inline(h, text)
            i += 1; continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", l.strip())
        if m:
            cap, path = m.groups()
            img = (src.parent / path).resolve()
            if img.exists():
                doc.add_picture(str(img), width=width)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(cap); r.italic = True; r.font.size = Pt(10)
            i += 1; continue
        if l.strip() == "---":
            i += 1; continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", l)
        if m:
            indent, mark, text = m.groups()
            style = "List Number" if mark[0].isdigit() else "List Bullet"
            if len(indent) >= 2:
                style += " 2"
            p = doc.add_paragraph(style=style); add_inline(p, text)
            i += 1; continue
        if not l.strip():
            i += 1; continue
        buf = [l.strip()]; j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||!\[|```|\s*([-*]|\d+\.)\s)", lines[j]):
            buf.append(lines[j].strip()); j += 1
        p = doc.add_paragraph(); add_inline(p, " ".join(buf)); p.paragraph_format.space_after = Pt(6)
        i = j
    cp = doc.core_properties
    cp.author = AUTHOR; cp.last_modified_by = AUTHOR; cp.title = "Nhận diện giống chuối từ ảnh — Báo cáo đồ án CS231"
    cp.comments = ""; cp.keywords = "CS231, phân loại ảnh, giống chuối"
    doc.save(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="report/report.md")
    ap.add_argument("dst", nargs="?", default="report/report.docx")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.dst))
    print("đã ghi", a.dst)


if __name__ == "__main__":
    main()
