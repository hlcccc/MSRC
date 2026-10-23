"""Download pinned public raw data and rebuild acceptance inputs without a package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from pipeline import common
from pipeline.common import initialize, inside, save, sha


def recipe_root():
    return common.ROOT / 'resources/reproduction'


def expected_inputs():
    return dict((name, digest) for digest, name in
                (line.split('  ', 1) for line in
                 (recipe_root() / 'inputs.sha256').read_text(encoding='utf-8').splitlines() if line))


def fetch(url, target, expected=None, *, md5=None, attempts=3):
    """Reuse only hash-verified downloads; atomically publish a complete new file."""
    target = inside(target)
    algorithm = hashlib.md5 if md5 else hashlib.sha256
    required = md5 or expected

    def valid(path):
        if not path.is_file() or not required:
            return False
        h = algorithm()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest() == required

    if target.exists():
        if valid(target):
            return target
        raise ValueError('existing public download failed its frozen digest: %s' % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = inside(target.with_name(target.name + '.part'))
    for attempt in range(attempts):
        try:
            print('DOWNLOAD_DATA %s' % url, flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'MSRC-reproduction/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
                length = response.headers.get('Content-Length')
                shutil.copyfileobj(response, output, length=1 << 20)
            if length is not None and temporary.stat().st_size != int(length):
                raise ValueError('incomplete HTTP body: %s' % url)
            if required and not valid(temporary):
                raise ValueError('public download differs from frozen source: %s' % url)
            temporary.replace(target)
            return target
        except (OSError, urllib.error.URLError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError('unreachable')


def install_model_recipe(assets):
    """Only small approved text overlays are bundled; all weights come from HF."""
    assets = inside(assets)
    frozen = json.loads((recipe_root() / 'models.json').read_text(encoding='utf-8'))
    for role, spec in frozen['models'].items():
        if len(spec['revision']) != 40:
            raise ValueError('public model revision must be a full commit')
    target = inside(assets / 'model_expectations.json')
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != frozen:
        raise ValueError('existing model expectations differ; use a fresh asset directory')
    save(target, frozen)
    overlays = json.loads((recipe_root() / 'model_overrides.json').read_text(encoding='utf-8'))
    manifest = {}
    for role, files in overlays.items():
        manifest[role] = {}
        for name, content in files.items():
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or relative.suffix not in {'.json', '.jinja', '.txt', '.md'}):
                raise ValueError('model recipe cannot contain weights or unsafe paths')
            data = content.encode('utf-8')
            digest = hashlib.sha256(data).hexdigest()
            if len(data) > 64 * 1024 or frozen['models'][role]['files'].get(name) != digest:
                raise ValueError('model overlay is not approved by the frozen file inventory')
            path = inside(assets / 'model_metadata' / role / name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest[role][name] = digest
    save(assets / 'model_metadata_manifest.json', manifest)
    print('PUBLIC_MODEL_RECIPE_OK: pinned revisions and non-weight compatibility files', flush=True)


def extract_textvqa(archive, target, names, expected):
    """Read original JPEG bytes with ZipFile; never transcode or extract all images."""
    target = inside(target)
    names = set(names)
    with zipfile.ZipFile(archive) as handle:
        selected = {}
        for info in handle.infolist():
            name = PurePosixPath(info.filename).name
            if name in names:
                if name in selected or info.is_dir():
                    raise ValueError('duplicate TextVQA image in the public ZIP')
                selected[name] = info
        if selected.keys() != names:
            raise ValueError('public TextVQA ZIP does not cover all 3166 validation images')
        for name, info in selected.items():
            destination = inside(target / name)
            data = handle.read(info)
            if hashlib.sha256(data).hexdigest() != expected['data/images/textvqa/' + name]:
                raise ValueError('TextVQA original JPEG differs: %s' % name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError('conflicting TextVQA original image')
            destination.write_bytes(data)


def verify_rebuilt(assets):
    files = expected_inputs()
    for name, digest in files.items():
        path = inside(Path(assets) / name)
        if not path.is_file() or sha(path) != digest:
            raise ValueError('rebuilt input differs from the frozen benchmark: %s' % name)
    print('PUBLIC_INPUTS_MATCH: %d original input files' % len(files), flush=True)


def prepare(assets, endpoint='https://huggingface.co'):
    from pipeline.build_benign import build
    from pipeline.prepare_data import prepare as rebuild
    assets = inside(assets)
    if (assets / 'data').exists() or (assets / 'raw').exists():
        raise ValueError('prepared data already exist; do not overwrite a frozen dataset')
    lock = json.loads((recipe_root() / 'data_sources.json').read_text(encoding='utf-8'))
    expected = expected_inputs()
    raw = inside(assets / 'downloads')
    pope = raw / 'pope'
    for setting in ('random', 'popular', 'adversarial'):
        name = 'coco_pope_%s.json' % setting
        fetch(lock['pope']['base_url'] + '/' + name, pope / name, expected['raw/pope/' + name])
    names = {json.loads(line)['image'] for line in (pope / 'coco_pope_random.json').read_text().splitlines() if line.strip()}
    if len(names) != 500:
        raise ValueError('POPE requires exactly 500 original COCO images')
    for name in sorted(names):
        fetch(lock['pope']['images_base_url'] + '/' + name, raw / 'pope_images' / name,
              expected['data/images/pope/' + name])
    annotation = fetch(lock['textvqa']['annotations_url'], raw / 'TextVQA_0.5.1_val.json',
                       expected['raw/textvqa/TextVQA_0.5.1_val.json'])
    rows = json.loads(annotation.read_text())['data']
    names = {row['image_id'] + '.jpg' for row in rows}
    if len(rows) != 5000 or len(names) != 3166:
        raise ValueError('TextVQA requires 5000 validation questions / 3166 images')
    archive = fetch(lock['textvqa']['images_url'], raw / 'train_val_images.zip',
                    md5=lock['textvqa']['images_md5'])
    extract_textvqa(archive, raw / 'textvqa_images', names, expected)
    for name, digest in sorted(expected.items()):
        prefix = 'data/MM-SafetyBench/'
        if name.startswith(prefix) and name.endswith('.parquet'):
            relative = name.removeprefix(prefix)
            url = '%s/datasets/%s/resolve/%s/%s?download=true' % (
                endpoint.rstrip('/'), lock['mmsafety']['repo'], lock['mmsafety']['revision'], relative)
            fetch(url, raw / 'MM-SafetyBench' / relative, digest)
    alpaca = lock['alpaca']
    url = '%s/datasets/%s/resolve/%s/%s?download=true' % (
        endpoint.rstrip('/'), alpaca['repo'], alpaca['revision'], alpaca['filename'])
    benign_expected = json.loads((recipe_root() / 'benign.expected.json').read_text())
    parquet = fetch(url, raw / 'alpaca.parquet', benign_expected['alpaca_sha256'])
    benign = build(parquet, raw / 'benign')
    sources = dict(pope_annotations=str(pope), pope_images=str(raw / 'pope_images'),
                   textvqa_annotations=str(annotation), textvqa_images=str(raw / 'textvqa_images'),
                   mmsafety_parquet=str(raw / 'MM-SafetyBench/data'), benign_annotations=str(benign))
    save(assets / 'public_sources.json', sources)
    rebuild(sources, assets)
    verify_rebuilt(assets)
    save(assets / 'public_reproduction.json', dict(
        status='PUBLIC_RAW_DOWNLOAD_AND_REBUILD_COMPLETE', sources=lock,
        external_delivery_package_used=False, pretrained_head_used=False,
        old_predictions_used=False, expected_inputs_sha256=sha(recipe_root() / 'inputs.sha256')))
    print('PUBLIC_DATA_READY: POPE=9000 TextVQA=5000 MM-SafetyBench=5040 benign=1680', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=common.ROOT / 'assets')
    parser.add_argument('--hf-endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    initialize()
    prepare(args.assets, args.hf_endpoint)


if __name__ == '__main__':
    main()
