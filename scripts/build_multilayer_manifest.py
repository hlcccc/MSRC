"""Maintainer-only: record byte identities, never run by acceptance testers."""
import hashlib
import json
from pathlib import Path


def build(root):
    core = root / 'extensions/multilayer/frozen'
    result = dict(format='msrc-multilayer-frozen-core-v1',
                  origin='3090 multilayer_001, verified again on HLC A100 2026-10-26',
                  files={p.relative_to(core).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(core.rglob('*.py'))})
    target = root / 'resources/multilayer/source_manifest.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
