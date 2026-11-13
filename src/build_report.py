"""Dựng báo cáo từ file mẫu: thay mỗi chỗ {{T:đường_dẫn_bảng.md}} bằng bảng tương ứng trong results/
(giữ dấu chấm thập phân), ghi report/report.md, chuyển sang report/report.docx rồi xuất report/report.pdf
(cần LibreOffice; không có thì bỏ qua bước PDF).
Đồng thời cập nhật các bảng trong README.md (giữa <!-- T:... --> và <!-- /T -->).

    python -m src.build_report
"""
from __future__ import annotations

import argparse
import re

from .config import ROOT
from .md2docx import convert


def table(path: str) -> str:
    lines = [l for l in (ROOT / path).read_text(encoding="utf-8").strip().splitlines() if l.startswith("|")]
    return "\n".join(lines)


def to_pdf(docx):
    """Xuất PDF từ file docx bằng LibreOffice (nếu có). Trả về đường dẫn PDF hoặc None."""
    import shutil
    import subprocess
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)], capture_output=True, text=True, timeout=600)
    pdf = docx.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def render(template: str) -> str:
    return re.sub(r"\{\{T:([^}]+)\}\}", lambda m: table(m.group(1)), template)


def refresh_readme(path="README.md"):
    """Cập nhật các bảng trong README nằm giữa cặp chú thích <!-- T:đường_dẫn --> ... <!-- /T --> bằng bảng mới nhất trong results/."""
    f = ROOT / path
    txt = f.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- T:([^ ]+) -->\n).*?(<!-- /T -->)",
                 lambda m: m.group(1) + (table(m.group(2)) if (ROOT / m.group(2)).exists() else "(chưa có kết quả)") + "\n" + m.group(3), txt, flags=re.S)
    if new != txt:
        f.write_text(new, encoding="utf-8")
    return new != txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="report/report_template.md")
    ap.add_argument("--out", default="report/report.md")
    a = ap.parse_args()
    tpl = (ROOT / a.template).read_text(encoding="utf-8")
    tpl = re.sub(r"^<!--.*?-->\s*", "", tpl, flags=re.S)
    md = render(tpl)
    (ROOT / a.out).write_text(md, encoding="utf-8")
    convert(ROOT / a.out, (ROOT / a.out).with_suffix(".docx"))
    print("đã ghi", a.out, "và", str((ROOT / a.out).with_suffix(".docx").relative_to(ROOT)))
    pdf = to_pdf((ROOT / a.out).with_suffix(".docx"))
    print("đã ghi " + str(pdf.relative_to(ROOT)) if pdf else "không có LibreOffice: bỏ qua bước xuất PDF")
    if refresh_readme():
        print("đã cập nhật bảng trong README.md")


if __name__ == "__main__":
    main()
