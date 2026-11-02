from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'msrc-fast-review-evidence-v1'
SIGNALS = ['self_consistency', 'resample_consistency', 'typed_verification',
           'cross_model_agreement', 'grounding_check', 'object_support',
           'sequence_confidence', 'output_entropy']
FAMILIES = ('llava', 'qwen', 'internvl')
DATASETS = ('pope', 'textvqa')
DONORS = {'llava': 'qwen', 'qwen': 'llava', 'internvl': 'qwen'}
CORE = ('msrc/signals.py', 'msrc/types.py', 'msrc/object_support.py',
        'msrc/model.py', 'references/chair_synonyms.txt',
        'references/m4c_evaluator.py', 'evaluation/datasets/pope.py',
        'evaluation/groups.py', 'evaluation/split.py',
        'pipeline/safety_split.py',
        'extensions/multilayer/frozen/safety_lab/policy.py',
        'extensions/multilayer/frozen/safety_lab/multilayer_policy.py',
        'extensions/multilayer/frozen/safety_lab/prefix_data.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError('nonfinite JSON: ' + x))) for line in handle if line.strip()]


def indexed(values):
    result = {}
    for row in values:
        rid = row.get('id')
        require(isinstance(rid, str) and rid and rid not in result, 'missing/duplicate ID')
        result[rid] = row
    return result


def output(path):
    path = Path(path).expanduser().absolute()
    require(path.resolve() == path and path.is_relative_to(ROOT),
            'new output must stay inside the isolated repository; no symlinks')
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def relative_path(root, name):
    value = PurePosixPath(name)
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and not value.is_absolute() and all(p not in ('..', '.') for p in value.parts),
            'unsafe relative path')
    path = Path(root).joinpath(*value.parts)
    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(),
            'path escapes bundle')
    return path


def core_contract():
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in CORE}


def verify_bundle(root, expected=None):
    root = Path(root).expanduser().resolve(strict=True)
    require(root.is_dir(), 'evidence directory required')
    manifest = load(root / 'manifest.json')
    if expected:
        require(sha(root / 'manifest.json') == expected, 'release manifest hash mismatch')
    require(manifest['format'] == FORMAT and manifest['full_fresh_inference'] is False,
            'not a qualified historical evidence bundle')
    require(manifest['core_contract'] == core_contract(), 'scoring source differs from evidence release')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json'}, 'missing/extra evidence files')
    for name, item in manifest['files'].items():
        path = relative_path(root, name)
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'evidence changed: ' + name)
    return manifest


def ece(labels, risks, bins=15):
    import numpy as np
    y, p = np.asarray(labels), np.asarray(risks, dtype=float)
    require(len(y) > 0 and y.shape == p.shape and np.isin(y, [0, 1]).all()
            and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), 'invalid ECE inputs')
    assigned = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(float((y[assigned == i] - p[assigned == i]).sum()))
               for i in range(bins)) / len(y)
