"""Build concise Word and Markdown delivery artifacts from the same source."""
import argparse
from copy import deepcopy
import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'docs/FLOW_SAFETY_MULTILAYER_FITTED.docx'


def blocks(markdown):
    code = None
    for line in markdown.splitlines()[1:]:
        if line.startswith('```'):
            if code is None:
                code = []
            else:
                yield 'code', '\n'.join(code)
                code = None
        elif code is not None:
            code.append(line)
        elif line.startswith('## '):
            yield 'heading', line[3:]
        elif line.strip():
            yield 'body', line
    if code is not None:
        raise ValueError('unclosed command block')


def build(source, destination, template=TEMPLATE):
    document = Document(template)
    paragraphs = document.paragraphs
    prototypes = {
        'title': paragraphs[0],
        'intro': paragraphs[1],
        'heading': next(p for p in paragraphs if p.style.name == 'Heading 1'),
        'body': next(p for p in paragraphs[2:] if p.style.name == 'Normal' and p.text),
        'code': next(p for p in paragraphs if p.style.name == 'Command'),
    }
    properties = {k: deepcopy(p._p.pPr) for k, p in prototypes.items()}
    body = document._element.body
    for element in list(body):
        if element.tag != qn('w:sectPr'):
            body.remove(element)

    def add(kind, text):
        p = document.add_paragraph()
        if properties[kind] is not None:
            p._p.append(deepcopy(properties[kind]))
        pp = p._p.get_or_add_pPr()
        for child in list(pp):
            if child.tag in (qn('w:rPr'), qn('w:shd'), qn('w:pageBreakBefore')):
                pp.remove(child)
        title = kind in ('title', 'intro')
        command = kind == 'code'
        p.paragraph_format.keep_with_next = kind in ('title', 'intro', 'heading')
        if command:
            p.paragraph_format.keep_together = True
            shade = OxmlElement('w:shd')
            shade.set(qn('w:val'), 'clear')
            shade.set(qn('w:fill'), '000000')
            pp.append(shade)
        for part in re.split('(`[^`]+`)', text):
            inline = not command and part.startswith('`')
            run = p.add_run(part[1:-1] if inline else part)
            name = 'Consolas' if inline else ('Times New Roman' if title else '宋体')
            run.font.name = name
            fonts = run._r.get_or_add_rPr().rFonts
            fonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = Pt(10 if inline else (14 if title else 12))
            run.font.color.rgb = RGBColor.from_string('FFFFFF' if command else '000000')
            if title:
                run.bold = True
        return p

    add('title', '7. 测试流程')
    add('intro', '指标1.11不确定性量化建模、1.12不确定性来源分析和1.13多模态生成内容风险评估快速测试流程：')
    for kind, text in blocks(source.read_text(encoding='utf-8')):
        add(kind, text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    args = parser.parse_args()
    root = ROOT
    build(root / 'docs/FAST_REVIEW_FLOW.md', args.out / '课题二快速测试流程.docx', args.template)
    import shutil
    shutil.copyfile(root / 'docs/FAST_REVIEW_FLOW.md', args.out / '课题二快速测试流程.md')
    shutil.copyfile(root / 'docs/FAST_REVIEW_DELIVERY.md', args.out / '交付说明.md')
