"""Package a committed isolated source version; large evidence stays out of Git."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

from fast_review.common import ROOT, load, require, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    target = args.out.resolve()
    target.mkdir(parents=True, exist_ok=True)
    release = load(ROOT / 'resources/fast_review/release.json')
    require(sha(args.evidence) == release['archive_sha256'], 'wrong evidence archive')
    version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    zip_path = target / 'MSRC-fast-review-code-20261102.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=MSRC/', '-o', str(zip_path), version],
                   cwd=ROOT, check=True)
    shutil.copyfile(args.evidence, target / args.evidence.name)
    from scripts.fast_build_documents import build
    build(ROOT / 'docs/FAST_REVIEW_FLOW.md', target / '课题二快速测试流程.docx')
    shutil.copyfile(ROOT / 'docs/FAST_REVIEW_FLOW.md', target / '课题二快速测试流程.md')
    shutil.copyfile(ROOT / 'docs/FAST_REVIEW_DELIVERY.md', target / '交付说明.md')
    files = sorted(p for p in target.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
    (target / 'SHA256SUMS.txt').write_text(''.join(sha(p) + '  ' + p.name + '\n' for p in files),
                                       encoding='utf-8', newline='\n')
    print('DELIVERY_PACKAGED ' + version)
    print(str(target))


if __name__ == '__main__':
    main()
