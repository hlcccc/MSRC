"""Build the concise local-three-GPU flow with the supplied Word paragraph styles."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'docs/FAST3_FLOW.docx'
TITLE = '课题二测试流程_无需训练版_一机三卡'


def blocks(text):
    pending, code, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith('```'):
            if pending:
                yield 'body', '\n'.join(pending)
                pending.clear()
            if in_code:
                yield 'code', '\n'.join(code)
                code.clear()
            in_code = not in_code
        elif in_code:
            code.append(line)
        elif line.startswith(('# ', '## ')):
            if pending:
                yield 'body', '\n'.join(pending)
                pending.clear()
            yield ('title', line[2:]) if line.startswith('# ') else ('heading', line[3:])
        elif line.strip():
            pending.append(line)
        elif pending:
            yield 'body', '\n'.join(pending)
            pending.clear()
    if pending:
        yield 'body', '\n'.join(pending)
    if in_code:
        raise ValueError('unclosed command block')


def signature(element):
    if element is None:
        return None
    return (element.tag, tuple(sorted(element.attrib.items())), element.text,
        tuple(signature(child) for child in element))


def validate(source, word):
    text = source.read_text(encoding='utf-8')
    document = Document(word)
    commands = re.findall(r'```bash\n(.*?)\n```', text, flags=re.S)
    assert commands == [p.text for p in document.paragraphs if p.style.name == 'Command']
    headings = [v for k, v in blocks(text) if k == 'heading']
    assert headings == [p.text for p in document.paragraphs if p.style.name == 'Heading 1']
    assert [int(h.split()[0]) for h in headings] == list(range(1, len(headings) + 1))
    rendered = '\n'.join(p.text for p in document.paragraphs)
    assert [p.text for p in document.paragraphs] == [value for _, value in blocks(text)]
    assert document.paragraphs[0].text == '7. 测试流程'
    assert all(name + ext in rendered for name in ('report', 'signal_report', 'calibration_report', 'warning_report')
        for ext in ('.html', '.md'))
    assert 'runs/local3_fitted_001/fast3/joint/report.html' in rendered
    assert all(value not in rendered for value in ('--hosts', '--phase cluster', 'PORT1', 'USER1@', '三台服务器',
        '独占RTX', 'POPE 4500', 'TextVQA 2496', '五个实验臂', 'fast3_fitted_001'))
    return dict(steps=len(headings), command_blocks=len(commands),
        commands_exact_between_markdown_and_word=True, report_paths_and_all_metric_file_pairs_present=True,
        topology='one_host_three_gpus', no_three_machine_commands=True,
        removed_hardware_and_dataset_scope_commentary=True, native_pagination_rendered=False)


def build(source, destination, template=TEMPLATE):
    original_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    document = Document(template)
    paragraphs = document.paragraphs
    models = {kind: deepcopy(paragraph._p) for kind, paragraph in {
        'title': paragraphs[0], 'intro': paragraphs[1],
        'heading': next(p for p in paragraphs if p.style.name == 'Heading 1'),
        'body': next(p for p in paragraphs[2:] if p.style.name == 'Normal' and p.text.startswith('下载')),
        'code': next(p for p in paragraphs if p.style.name == 'Command'),
        'success': next(p for p in paragraphs if p.text.startswith(('成功提示：', '文件校验均显示OK'))),
    }.items()}
    body = document._element.body
    for element in list(body):
        if element.tag != qn('w:sectPr'):
            body.remove(element)
    preserved = True
    for index, (kind, text) in enumerate(blocks(source.read_text(encoding='utf-8'))):
        if index == 1:
            kind = 'intro'
        elif kind == 'body' and text.startswith(('成功提示：', '文件校验均显示OK', '流程完成提示：')):
            kind = 'success'
        model = models[kind]
        element = deepcopy(model)
        original_run = element.find(qn('w:r'))
        properties = deepcopy(original_run.find(qn('w:rPr'))) if original_run is not None else None
        for child in list(element):
            if child.tag != qn('w:pPr'):
                element.remove(child)
        body.insert(len(body) - 1, element)
        run = Paragraph(element, document._body).add_run(text)
        if properties is not None:
            run._r.insert(0, properties)
        preserved &= signature(element.find(qn('w:pPr'))) == signature(model.find(qn('w:pPr')))
    document.core_properties.title = TITLE
    document.core_properties.author = ''
    document.core_properties.last_modified_by = ''
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    assert preserved
    qa = validate(source, destination)
    qa.update(template_name=template.name, template_sha256=original_hash,
        template_paragraph_formatting_preserved=True,
        word_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        markdown_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if template.resolve() != destination.resolve():
        assert original_hash == hashlib.sha256(template.read_bytes()).hexdigest()
        qa['template_unchanged'] = True
    return qa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    args = parser.parse_args()
    qa = build(ROOT / 'docs/FAST3_FLOW.md', args.out / (TITLE + '.docx'), args.template)
    shutil.copyfile(ROOT / 'docs/FAST3_FLOW.md', args.out / (TITLE + '.md'))
    (args.out / 'document_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(qa, ensure_ascii=True))
