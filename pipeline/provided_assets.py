"""Download hash-pinned project-fitted parameters for the supplied-state variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

from pipeline.common import ROOT, initialize, inside, sha


EXPECTED = {'factual.json', 'plugguard/head.pt', 'plugguard/threshold.json',
            'plugguard/protocol.json', 'plugguard/training_inputs.jsonl',
            'plugguard/input_hashes.json', 'plugguard/dataset_identity.json',
            'plugguard/development_generation.jsonl', 'plugguard/calibration_scores.jsonl'}


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != 'msrc-provided-parameters-v1'
            or manifest['contains_base_model_weights'] is not False
            or manifest['contains_test_predictions'] is not False or set(manifest['files']) != EXPECTED):
        raise ValueError('unexpected supplied-state package contents')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != EXPECTED | {'manifest.json'}:
        raise ValueError('extra/missing files in supplied-state package')
    for name, digest in manifest['files'].items():
        if sha(root / name) != digest:
            raise ValueError('fitted parameter hash mismatch: ' + name)
    return manifest


def unpack(archive, out):
    out = inside(out)
    if out.exists():
        raise ValueError('new parameter destination required')
    allowed = EXPECTED | {'manifest.json'}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != allowed:
            raise ValueError('unexpected/duplicate archive members')
        if any(not m.isfile() or '\\' in m.name or PurePosixPath(m.name).is_absolute()
               or '..' in PurePosixPath(m.name).parts for m in members):
            raise ValueError('unsafe fitted parameter archive')
        out.mkdir(parents=True)
        for member in members:
            path = inside(out / member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
    verify(out)


def prepare(check=False):
    root = inside(ROOT / 'assets/fitted')
    registry = json.loads((ROOT / 'resources/reproduction/fitted.json').read_text(encoding='utf-8'))
    if not root.exists():
        if check:
            raise ValueError('provided parameters not downloaded')
        cache = inside(ROOT / 'cache/fitted')
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / 'parameters.tar.gz'
        if archive.exists() and sha(archive) != registry['sha256']:
            raise ValueError('cached fitted parameters do not match the fixed release')
        if not archive.exists():
            partial = cache / 'parameters.download'
            if partial.exists():
                raise ValueError('incomplete parameter download exists; inspect before retrying')
            request = urllib.request.Request(registry['url'], headers={'User-Agent': 'MSRC-acceptance'})
            with urllib.request.urlopen(request, timeout=120) as source, partial.open('xb') as target:
                shutil.copyfileobj(source, target)
            if partial.stat().st_size != registry['bytes'] or sha(partial) != registry['sha256']:
                raise ValueError('downloaded fitted parameters do not match the fixed release')
            partial.rename(archive)
        unpack(archive, root)
    manifest = verify(root)
    if sha(root / 'manifest.json') != registry['manifest_sha256']:
        raise ValueError('supplied state belongs to a different published release')
    from pipeline.factual_parameters import fitted_collector, validate_code_contract
    validate_code_contract(json.loads((root / 'factual.json').read_text(encoding='utf-8')), fitted_collector())
    print('FITTED_SOURCE_OK ' + fitted_collector())
    print('FITTED_PARAMETERS_OK ' + str(root))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    initialize()
    prepare(args.check)


if __name__ == '__main__':
    main()
