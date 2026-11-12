"""Dựng file .pptx từ danh sách slide mô tả bằng dict (dùng bởi src.make_slides).

Mỗi slide: {"title": str, "bullets": [str | (str, cấp)], "image": đường_dẫn, "table": đường_dẫn_md hoặc list hàng,
            "notes": str (ghi chú thuyết trình), "layout": "title" | "content"}.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

AUTHOR = "Nhóm đồ án CS231"
W, H = Inches(13.333), Inches(7.5)
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def md_rows(path_or_rows, max_rows=None):
    if isinstance(path_or_rows, (list, tuple)):
        rows = path_or_rows
    else:
        lines = [l for l in Path(path_or_rows).read_text(encoding="utf-8").splitlines() if l.startswith("|")]
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|\s*:?-{2,}", l)]
        rows = [[re.sub(r"(?<![\d.,])(\d+)\.(\d+)(?![\d.])", r"\1,\2", c) for c in r] for r in rows]
    return rows if max_rows is None else rows[:max_rows + 1]


def _text(tf, items, size):
    tf.clear(); tf.word_wrap = True
    first = True
    for it in items:
        text, level = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.level = level
        parts = re.split(r"(\*\*[^*]+\*\*)", text)
        for part in parts:
            if not part:
                continue
            r = p.add_run(); r.text = part.strip("*") if part.startswith("**") else part
            r.font.size = Pt(size - 2 * level); r.font.bold = part.startswith("**")
        p.space_after = Pt(6)


def build(slides: list[dict], out: Path, title: str):
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    for k, s in enumerate(slides, 1):
        sl = prs.slides.add_slide(blank)
        if s.get("layout") == "title":
            tb = sl.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.6))
            _text(tb.text_frame, [s["title"]], 40); tb.text_frame.paragraphs[0].runs[0].font.bold = True
            tb.text_frame.paragraphs[0].runs[0].font.color.rgb = ACCENT
            if s.get("bullets"):
                sb = sl.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.7), Inches(2.8)); _text(sb.text_frame, s["bullets"], 22)
        else:
            bar = sl.shapes.add_shape(1, 0, 0, W, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
            tb = sl.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.9))
            _text(tb.text_frame, [s["title"]], 28)
            for r in tb.text_frame.paragraphs[0].runs:
                r.font.bold = True; r.font.color.rgb = ACCENT
            top = Inches(1.2); left = Inches(0.5); avail_w = Inches(12.3); avail_h = H - top - Inches(0.5)
            has_side = s.get("bullets") and (s.get("image") or s.get("table"))
            if s.get("bullets"):
                bw = Inches(4.6) if has_side else avail_w
                bb = sl.shapes.add_textbox(left, top, bw, avail_h); _text(bb.text_frame, s["bullets"], s.get("size", 18 if has_side else 20))
                if has_side:
                    left = left + bw + Inches(0.2); avail_w = avail_w - bw - Inches(0.2)
            if s.get("image"):
                im = Image.open(s["image"]); ar = im.width / im.height
                w = avail_w; h = Emu(int(w / ar))
                if h > avail_h:
                    h = avail_h; w = Emu(int(h * ar))
                sl.shapes.add_picture(str(s["image"]), left + (avail_w - w) // 2, top, w, h)
            elif s.get("table"):
                rows = md_rows(s["table"], s.get("max_rows"))
                if s.get("cols"):
                    idx = [rows[0].index(c) for c in s["cols"]]; rows = [[r[i] for i in idx] for r in rows]
                nr, nc = len(rows), len(rows[0])
                rh = min(Inches(0.45), Emu(int(avail_h / nr)))
                t = sl.shapes.add_table(nr, nc, left, top, avail_w, Emu(rh * nr)).table
                fs = s.get("table_size", 14 if nc <= 4 and nr <= 8 else 11 if nc <= 7 else 9)
                for i, r in enumerate(rows):
                    for j, v in enumerate(r):
                        c = t.cell(i, j); c.text = v
                        for p in c.text_frame.paragraphs:
                            for run in p.runs:
                                run.font.size = Pt(fs); run.font.bold = i == 0 or ("**" in v)
                                run.text = run.text.replace("**", "")
        num = sl.shapes.add_textbox(W - Inches(1.0), H - Inches(0.45), Inches(0.8), Inches(0.35))
        _text(num.text_frame, [str(k)], 11)
        if s.get("notes"):
            sl.notes_slide.notes_text_frame.text = s["notes"]
    cp = prs.core_properties
    cp.author = AUTHOR; cp.last_modified_by = AUTHOR; cp.title = title; cp.comments = ""
    prs.save(out)
