"""Build the concise single-5090 Word flow with the user's original paragraph styles."""
import argparse
import json
from pathlib import Path
import shutil

from build_fast3_docs import ROOT, TEMPLATE, build

TITLE = '课题二测试流程_无需训练版_单卡5090'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    args = parser.parse_args()
    source = ROOT / 'docs/FAST_SINGLE_FLOW.md'
    qa = build(source, args.out / (TITLE + '.docx'), args.template,
        title=TITLE, run_id='single_fitted_001', topology='one_host_one_gpu')
    shutil.copyfile(source, args.out / (TITLE + '.md'))
    (args.out / 'document_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(qa, ensure_ascii=True))


if __name__ == '__main__':
    main()
