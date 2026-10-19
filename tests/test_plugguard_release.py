"""Publication guards: preserve the pinned core and keep artifacts out of Git."""
import hashlib
import json
from pathlib import Path

import pytest

from pipeline import preflight

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_head_matches_the_validated_upstream():
    digest = hashlib.sha256((ROOT / 'references/plugguard/models.py').read_bytes()).hexdigest()
    assert digest == 'c16d237c5634bcaae24f6bedaf5075479fcf22e395c9ecfff081715fdf5d0479'


def test_paired_model_metadata_contains_no_machine_specific_paths():
    data = json.loads((ROOT / 'assets/plugguard/base_models.json').read_text(encoding='utf-8'))
    assert set(data) == {'generator', 'judge'}
    assert data['generator']['repo'] == 'Qwen/Qwen2.5-VL-7B-Instruct'
    for spec in data.values():
        assert set(spec) == {'repo', 'revision', 'files'}
        assert len(spec['revision']) == 40
        assert any(name.endswith('.safetensors') for name in spec['files'])
        assert all(not Path(name).is_absolute() and len(digest) == 64
                   for name, digest in spec['files'].items())


@pytest.mark.parametrize('backend', ['plugguard', 'singguard'])
def test_preflight_never_uses_singguard_as_plugguard_fallback(tmp_path, monkeypatch, backend):
    generator = tmp_path / 'generator'
    generator.mkdir()
    (generator / 'config.json').write_text('{}')
    (generator / 'model.safetensors').write_bytes(b'synthetic fixture')
    monkeypatch.setattr(preflight, 'model_paths', lambda: {
        'generator': str(generator), 'guard': str(tmp_path / 'missing_singguard')})
    monkeypatch.setenv('MSRC_SAFETY_BACKEND', backend)
    if backend == 'plugguard':
        assert set(preflight.check_models()) == {'generator'}
    else:
        with pytest.raises(SystemExit, match='guard has no config'):
            preflight.check_models()


def test_new_entry_point_is_exclusively_plugguard():
    source = (ROOT / 'scripts/04_score_safety.sh').read_text(encoding='utf-8')
    assert '-m pipeline.plugguard' in source
    assert '-m pipeline.safety' not in source
    assert '-m pipeline.delivery' not in source
    assert '--backend)' not in source
    assert 'SAFETY_SMOKE_COMPLETED_NOT_ACCEPTANCE' in source
