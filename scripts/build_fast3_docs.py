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

    original = source.read_text(encoding='utf-8')
    content_blocks = list(blocks(original))
    assert content_blocks and content_blocks[0][0] == 'body'
    add('title', original.splitlines()[0][2:])
    add('intro', content_blocks.pop(0)[1])
    for kind, text in content_blocks:
        add(kind, text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    reread = Document(destination)
    content = '\n'.join(p.text for p in reread.paragraphs)
    commands = re.findall(r'```(?:bash|cmd)\n(.*?)\n```', original, flags=re.S)
    headings = [int(value) for value in re.findall(r'^## (\d+) ', original, flags=re.M)]
    assert headings and headings == list(range(1, len(headings) + 1))
    assert all(command in content for command in commands)
    for p in reread.paragraphs:
        if p.style.name == 'Command':
            assert p._p.pPr.find(qn('w:shd')).get(qn('w:fill')) == '000000'
            assert all(str(run.font.color.rgb) == 'FFFFFF' for run in p.runs)
    return dict(steps=len(headings), eight_steps=len(headings) == 8,
        command_blocks=len(commands), commands_exact=True,
        template_section_and_styles_retained=True, black_command_background_white_text=True,
        native_pagination_rendered=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    args = parser.parse_args()
    root = ROOT
    import json
    result = build(root / 'docs/FAST3_FLOW.md', args.out / '测试流程_三张5090_无需现场训练.docx', args.template)
    import shutil
    shutil.copyfile(root / 'docs/FAST3_FLOW.md', args.out / '测试流程_三张5090_无需现场训练.md')
    shutil.copyfile(root / 'docs/FAST3_DELIVERY.md', args.out / '交付说明.md')
    shutil.copyfile(root / 'docs/FAST3_AUDIT.md', args.out / '逐步审计.md')
    shutil.copyfile(root / 'resources/FAST3_ASSET_SOURCES.json', args.out / 'FAST3_ASSET_SOURCES.json')
    shutil.copyfile(root / 'configs/local-models.fast3.json', args.out / 'local-models.fast3.json')
    (args.out / 'document_qa.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
