"""Maintainer-only export of hashes and small compatibility files, never weights.

The third-party flow does NOT invoke this script or access the reference server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.build_benign import records_digest


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_catalog(assets, output):
    models = json.loads((assets / 'models.lock.json').read_text())['models']
    paired = json.loads((Path(__file__).resolve().parents[1] / 'assets/plugguard/base_models.json').read_text())
    result, memo = {}, {}
    for role, spec in models.items():
        key = (spec['repo'], spec.get('revision') or paired['generator']['revision'])
        if key not in memo:
            url = 'https://huggingface.co/api/models/%s/revision/%s?blobs=true' % key
            with urllib.request.urlopen(url, timeout=30) as response:
                memo[key] = json.load(response)
        result[role] = memo[key]
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def export(assets, benign, output, catalog):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((assets / 'models.lock.json').read_text())['models']
    paired = json.loads((assets / 'plugguard/base_models.json').read_text())
    models, overrides = {}, {}
    catalog = json.loads(catalog.read_text())
    for role, spec in original.items():
        revision = spec.get('revision') or paired['generator']['revision']
        models[role] = dict(repo=spec['repo'], revision=revision, files=spec['files'])
        public = catalog[role]
        assert public['sha'] == revision and public['id'] == spec['repo']
        siblings = {item['rfilename']: item for item in public['siblings']}
        changes = {}
        for name, expected in spec['files'].items():
            if Path(name).suffix in {'.safetensors', '.bin', '.pt', '.pth'}:
                continue
            source = (Path(spec['path']) / name).read_bytes()
            assert digest(source) == expected
            item = siblings.get(name, {})
            blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
            matches = (item.get('lfs', {}).get('sha256') == expected
                       if item.get('lfs') else item.get('blobId') == blob)
            if not matches:
                if len(source) > 64 * 1024:
                    raise ValueError('unexpected large metadata change: %s/%s' % (role, name))
                changes[name] = source.decode('utf-8')
                print('METADATA_OVERRIDE', role, name, len(source), flush=True)
        overrides[role] = changes
    (output / 'models.json').write_text(json.dumps({'models': models}, indent=2) + '\n')
    (output / 'model_overrides.json').write_text(json.dumps(overrides, indent=2, ensure_ascii=False) + '\n')
    rows = [json.loads(line) for line in benign.read_text().splitlines() if line.strip()]
    assert len(rows) == 1680 and all(r['label'] == 0 and 'answer' not in r for r in rows)
    expected = dict(count=1680, records_sha256=records_digest(rows),
        alpaca_sha256='06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073',
        font_sha256=digest(Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').read_bytes()),
        images={Path(r['image']).name: digest(Path(r['image']).read_bytes()) for r in rows})
    (output / 'benign.expected.json').write_text(json.dumps(expected, indent=2) + '\n')
    lines = [line for line in (assets / 'MANIFEST.sha256').read_text().splitlines()
             if line.split('  ', 1)[1].startswith(('data/', 'raw/'))]
    (output / 'inputs.sha256').write_text('\n'.join(lines) + '\n')
    (output / 'DejaVuSans.ttf').write_bytes(Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').read_bytes())
    (output / 'FONT-LICENSE.txt').write_bytes(Path('/usr/share/doc/fonts-dejavu-core/copyright').read_bytes())
    print('EXPORTED_PUBLIC_REPRODUCTION: hashes, raw-data recipe checks, font and metadata only', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-assets', type=Path, required=True)
    parser.add_argument('--benign', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--public-catalog', type=Path)
    parser.add_argument('--download-catalog', type=Path)
    args = parser.parse_args()
    if args.download_catalog:
        public_catalog(args.reference_assets, args.download_catalog)
    else:
        if not (args.benign and args.output and args.public_catalog):
            parser.error('export requires --benign, --output and --public-catalog')
        export(args.reference_assets, args.benign, args.output, args.public_catalog)
