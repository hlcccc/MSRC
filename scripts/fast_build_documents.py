"""Build concise Word and Markdown delivery artifacts from the same source."""
import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def font(style, western='Times New Roman', east='宋体', size=11):
    style.font.name, style.font.size = western, Pt(size)
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east)


def build(source, destination):
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.2), Cm(2.0)
    section.left_margin = section.right_margin = Cm(2.3)
    normal = document.styles['Normal']
    font(normal)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in [('Title', 18), ('Heading 1', 13), ('Subtitle', 11)]:
        style = document.styles[name]
        font(style, east='黑体', size=size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
    title = document.add_paragraph('课题二快速测试流程', 'Title')
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('全量历史证据复核与现场抽测', 'Subtitle')
    code = False
    table = None
    for line in source.read_text(encoding='utf-8').splitlines()[1:]:
        if line.startswith('```'):
            code = not code
            continue
        if code:
            p = document.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(line)
            run.font.name, run.font.size = 'Consolas', Pt(8.5)
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), 'F2F4F6')
            p._p.get_or_add_pPr().append(shade)
            continue
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if all(c == '---' for c in cells):
                continue
            if table is None:
                table = document.add_table(rows=1, cols=len(cells))
                table.style = 'Table Grid'
                row = table.rows[0]
                header = True
                repeat = OxmlElement('w:tblHeader')
                row._tr.get_or_add_trPr().append(repeat)
            else:
                row, header = table.add_row(), False
            for cell, text in zip(row.cells, cells):
                cell.text = text
                for run in cell.paragraphs[0].runs:
                    run.font.size = Pt(9)
                    run.bold = header
            continue
        table = None
        if not line.strip():
            continue
        if line.startswith('## '):
            document.add_paragraph(line[3:], 'Heading 1')
        else:
            p = document.add_paragraph()
            for i, part in enumerate(re.split('(`[^`]+`)', line)):
                run = p.add_run(part.strip('`') if part.startswith('`') else part)
                if part.startswith('`'):
                    run.font.name, run.font.size = 'Consolas', Pt(9)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build(root / 'docs/FAST_REVIEW_FLOW.md', args.out / '课题二快速测试流程.docx')
    import shutil
    shutil.copyfile(root / 'docs/FAST_REVIEW_FLOW.md', args.out / '课题二快速测试流程.md')
    shutil.copyfile(root / 'docs/FAST_REVIEW_DELIVERY.md', args.out / '交付说明.md')
