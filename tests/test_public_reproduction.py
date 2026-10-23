"""The public flow rebuilds inputs and fits new heads, not replayed evidence."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from pipeline import build_benign, common, prepare_assets, public_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    return tmp_path


def test_plain_recipe_is_seeded_and_does_not_use_alpaca_answers():
    rows = [{'instruction': 'Describe a flower.', 'output': 'NEVER READ'},
            {'instruction': 'Name three fruits'}, {'instruction': 'has 123 digits'},
            {'instruction': 'one'}, {'instruction': '!not a normal phrase'}]
    actual = build_benign.select_phrases(rows, want=2)
    assert set(actual) == {'describe a flower', 'name three fruits'}
    assert actual == build_benign.select_phrases(rows, want=2)
    assert len(build_benign.LURE) == 42
    assert len(build_benign.TEMPLATES) == 5


def test_record_identity_ignores_machine_paths_but_not_content():
    row = dict(id='benign:0000', question='question', phrase='phrase', tier='plain',
               label=0, category='benign_plain', image='/old/path.png')
    original = build_benign.records_digest([row])
    assert original == build_benign.records_digest([dict(row, image='/new/path.png')])
    assert original != build_benign.records_digest([dict(row, question='changed')])


def test_frozen_benign_assets_only_contain_raw_identity_checks():
    recipe = ROOT / 'resources/reproduction'
    expected = json.loads((recipe / 'benign.expected.json').read_text())
    assert expected['count'] == len(expected['images']) == 1680
    assert set(expected) == {'count', 'records_sha256', 'alpaca_sha256', 'font_sha256', 'images'}
    assert common.sha(recipe / 'DejaVuSans.ttf') == expected['font_sha256']
    assert all(name == 'benign_%04d.png' % i for i, name in enumerate(expected['images']))


def test_public_inputs_cover_all_raw_benchmarks_and_original_images():
    files = public_assets.expected_inputs()
    assert len([n for n in files if n.startswith('data/images/pope/')]) == 500
    assert len([n for n in files if n.startswith('data/images/textvqa/')]) == 3166
    assert len([n for n in files if n.startswith('data/images/mmsafety/')]) == 5040
    assert len([n for n in files if n.startswith('data/images/benign/')]) == 1680
    assert len([n for n in files if n.endswith('.parquet')]) == 39
    assert not any('prediction' in n or 'head.pt' in n or 'threshold' in n for n in files)


def test_model_recipe_has_no_paths_or_pretrained_guard_head(workspace, monkeypatch):
    monkeypatch.setattr(public_assets, 'recipe_root', lambda: ROOT / 'resources/reproduction')
    assets = workspace / 'assets'
    public_assets.install_model_recipe(assets)
    models = json.loads((assets / 'model_expectations.json').read_text())['models']
    assert set(models) == set(prepare_assets.REPOS)
    assert models['generator'] == models['qwen']
    for spec in models.values():
        assert set(spec) == {'repo', 'revision', 'files'}
        assert len(spec['revision']) == 40
    assert not list(assets.rglob('*.pt')) and not list(assets.rglob('*.safetensors'))
    for p in (assets / 'model_metadata').rglob('*'):
        if p.is_file():
            assert p.stat().st_size <= 64 * 1024


def test_conflicting_model_recipe_is_not_overwritten(workspace, monkeypatch):
    monkeypatch.setattr(public_assets, 'recipe_root', lambda: ROOT / 'resources/reproduction')
    assets = workspace / 'assets'
    assets.mkdir()
    (assets / 'model_expectations.json').write_text('{}')
    with pytest.raises(ValueError, match='differ'):
        public_assets.install_model_recipe(assets)


def test_fetch_reuses_only_verified_raw_files(workspace, monkeypatch):
    data = b'raw public annotation'
    digest = hashlib.sha256(data).hexdigest()
    class Response(io.BytesIO):
        headers = {'Content-Length': str(len(data))}
    calls = []
    def open_url(request, timeout):
        calls.append(request.full_url)
        return Response(data)
    monkeypatch.setattr(public_assets.urllib.request, 'urlopen', open_url)
    path = workspace / 'downloads/annotation.json'
    public_assets.fetch('https://example.invalid/raw', path, digest)
    public_assets.fetch('https://example.invalid/raw', path, digest)
    assert len(calls) == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='failed'):
        public_assets.fetch('https://example.invalid/raw', path, digest)


def test_textvqa_extracts_only_requested_original_bytes(workspace):
    path = workspace / 'images.zip'
    with zipfile.ZipFile(path, 'w') as handle:
        handle.writestr('train_images/keep.jpg', b'original JPEG bytes')
        handle.writestr('../../unrequested.jpg', b'not extracted')
    expected = {'data/images/textvqa/keep.jpg': hashlib.sha256(b'original JPEG bytes').hexdigest()}
    public_assets.extract_textvqa(path, workspace / 'selected', ['keep.jpg'], expected)
    assert (workspace / 'selected/keep.jpg').read_bytes() == b'original JPEG bytes'
    assert len(list((workspace / 'selected').iterdir())) == 1
    with pytest.raises(ValueError, match='cover'):
        public_assets.extract_textvqa(path, workspace / 'other', ['missing.jpg'], expected)


def test_no_public_preparation_over_a_frozen_input_tree(workspace):
    (workspace / 'assets/data').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exist'):
        public_assets.prepare(workspace / 'assets')


def test_full_safety_flow_has_no_default_pretrained_head():
    source = (ROOT / 'pipeline/plugguard.py').read_text(encoding='utf-8')
    assert "if args.mode == 'smoke' else None" in source
    wrapper = (ROOT / 'scripts/02_prepare_assets_data.sh').read_text(encoding='utf-8')
    assert '--download-data)' in wrapper and '-m pipeline.public_assets' in wrapper


def test_public_recipe_does_not_change_factual_or_safety_inference():
    source = (ROOT / 'pipeline/public_assets.py').read_text(encoding='utf-8')
    assert 'old_predictions_used=False' in source
    assert 'pretrained_head_used=False' in source
    assert '--head-package' not in source
    lock = json.loads((ROOT / 'resources/reproduction/data_sources.json').read_text())
    assert len(lock['textvqa']['images_md5']) == 32
    assert all(len(lock[key]['revision']) == 40 for key in ('pope', 'alpaca', 'mmsafety'))


def test_public_downloads_use_frozen_revisions_and_only_approved_files(workspace, monkeypatch):
    import sys
    from types import SimpleNamespace
    model = workspace / 'assets'
    model.mkdir()
    expected = {'models': {name: dict(repo=repo, revision='f' * 40, files={
        'config.json': hashlib.sha256(b'{}').hexdigest(),
        'model.safetensors': hashlib.sha256(b'fixture').hexdigest()})
        for name, repo in prepare_assets.REPOS.items()}}
    (model / 'model_expectations.json').write_text(json.dumps(expected))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs['revision'] == 'f' * 40
        assert kwargs['allow_patterns'] == ['config.json', 'model.safetensors']
        target = Path(kwargs['local_dir'])
        target.mkdir()
        (target / 'config.json').write_bytes(b'{}')
        (target / 'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=download, HfApi=lambda: pytest.fail('must not resolve latest revision')))
    prepare_assets.download_models(model, None)
    assert len(calls) == 5
