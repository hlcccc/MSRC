"""A netdisk mirror must not import predictions, splits, weights or private files."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipeline import common, raw_bundle


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    recipe = tmp_path / 'recipe'
    recipe.mkdir()
    for name in ('data_sources.json', 'inputs.sha256', 'benign.expected.json', 'raw_bundle.NOTICE.md'):
        (recipe / name).write_text('source and license notice', encoding='utf-8')
    monkeypatch.setattr(raw_bundle, 'recipe_root', lambda: recipe)
    payload = {
        'TextVQA_0.5.1_val.json': b'original annotation',
        'textvqa_images/original.jpg': b'original JPEG',
        'MM-SafetyBench/data/category/SD.parquet': b'original MM parquet',
        'alpaca.parquet': b'original instruction source',
    }
    digest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    groups = {
        'textvqa': {name: digest[name] for name in list(payload)[:2]},
        'mmsafety': {name: digest[name] for name in list(payload)[2:3]},
        'benign_source': {'alpaca.parquet': digest['alpaca.parquet']},
    }
    monkeypatch.setattr(raw_bundle, 'inventory', lambda: groups)
    source = tmp_path / 'source'
    for name, data in payload.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    # These must never be traversed or included by the allowlist-based packer.
    (source / 'prediction.jsonl').write_text('DO NOT DELIVER')
    (source / 'head.pt').write_bytes(b'DO NOT DELIVER')
    archive = tmp_path / 'delivery/raw.tar'
    raw_bundle.pack(source, archive)
    return tmp_path, source, archive, payload


def rewrite(source, target, change=None, extra=None):
    with tarfile.open(source) as old, tarfile.open(target, 'w') as new:
        for original in old.getmembers():
            data = old.extractfile(original).read()
            if change:
                data = change(original.name, data)
            info = tarfile.TarInfo(original.name)
            info.size = len(data)
            new.addfile(info, io.BytesIO(data))
        if extra:
            name, data, kind = extra
            info = tarfile.TarInfo(name)
            info.size, info.type = len(data), kind
            info.linkname = '../outside'
            new.addfile(info, io.BytesIO(data))


def test_roundtrip_only_restores_raw_originals(bundle):
    root, source, archive, payload = bundle
    target = root / 'fresh/assets'
    receipt = raw_bundle.restore(archive, target)
    assert receipt['files'] == 4 and receipt['predictions_restored'] is False
    assert receipt['archive_sha256'] == common.sha(archive)
    for name, data in payload.items():
        assert (target / 'downloads' / name).read_bytes() == data
    assert not (target / 'data').exists() and not (target / 'raw').exists()
    actual = {p.relative_to(target / 'downloads').as_posix()
              for p in (target / 'downloads').rglob('*') if p.is_file()}
    assert actual == set(payload)
    assert not (target / 'fitted').exists()
    assert raw_bundle.inspect(archive)['contains_fitted_parameters'] is False


def test_packing_is_deterministic_and_does_not_include_extra_files(bundle):
    root, source, archive, _ = bundle
    other = root / 'second.tar'
    raw_bundle.pack(source, other)
    assert archive.read_bytes() == other.read_bytes()
    with tarfile.open(archive) as handle:
        assert 'head.pt' not in handle.getnames() and 'prediction.jsonl' not in handle.getnames()


@pytest.mark.parametrize('name', ['prediction.jsonl', 'head.pt', '../escape', '/absolute', 'data/split.json'])
def test_no_extra_or_escaping_payload(bundle, name):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    rewrite(archive, bad, extra=(name, b'injected', tarfile.REGTYPE))
    with pytest.raises(ValueError, match='unapproved'):
        raw_bundle.restore(bad, root / 'not_created')
    assert not (root / 'not_created').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_no_links_directories_or_devices(bundle, kind):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    rewrite(archive, bad, extra=('link', b'', kind))
    with pytest.raises(ValueError, match='unsafe'):
        raw_bundle.inspect(bad)


def test_duplicate_members_are_rejected(bundle):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    rewrite(archive, bad, extra=('alpaca.parquet', b'original instruction source', tarfile.REGTYPE))
    with pytest.raises(ValueError, match='duplicate'):
        raw_bundle.inspect(bad)


def test_corrupt_bytes_are_rejected_before_writing(bundle):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    rewrite(archive, bad, change=lambda name, data: data[::-1] if name == 'alpaca.parquet' else data)
    with pytest.raises(ValueError, match='hash mismatch'):
        raw_bundle.restore(bad, root / 'not_created')
    assert not (root / 'not_created').exists()


def test_archive_cannot_self_authorize_a_different_hash(bundle):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    def change(name, data):
        if name == 'bundle.json':
            manifest = json.loads(data)
            manifest['files']['alpaca.parquet']['sha256'] = 'a' * 64
            return json.dumps(manifest).encode()
        return data
    rewrite(archive, bad, change=change)
    with pytest.raises(ValueError, match='pinned recipe'):
        raw_bundle.inspect(bad)


@pytest.mark.parametrize('key', ['recipe', 'contains_test_predictions', 'contains_fitted_parameters'])
def test_invalid_recipe_and_nonraw_mode_rejected(bundle, key):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    def change(name, data):
        if name == 'bundle.json':
            manifest = json.loads(data)
            manifest[key] = {} if key == 'recipe' else True
            return json.dumps(manifest).encode()
        return data
    rewrite(archive, bad, change=change)
    with pytest.raises(ValueError, match='different recipe'):
        raw_bundle.inspect(bad)


def test_notice_cannot_be_replaced(bundle):
    root, _, archive, _ = bundle
    bad = root / 'bad.tar'
    rewrite(archive, bad, change=lambda name, data: b'no credits' if name == 'NOTICE.md' else data)
    with pytest.raises(ValueError, match='notices'):
        raw_bundle.inspect(bad)


@pytest.mark.parametrize('existing', ['downloads', 'data', 'raw'])
def test_restore_never_overwrites_existing_materials(bundle, existing):
    root, _, archive, _ = bundle
    assets = root / 'existing'
    (assets / existing).mkdir(parents=True)
    with pytest.raises(ValueError, match='fresh'):
        raw_bundle.restore(archive, assets)


def test_partial_groups_remain_original_sources(bundle):
    root, source, _, _ = bundle
    archive = root / 'only_instructions.tar'
    raw_bundle.pack(source, archive, ['benign_source'])
    manifest = raw_bundle.inspect(archive)
    assert set(manifest['files']) == {'alpaca.parquet'}
    assert manifest['groups'] == ['benign_source']


def test_public_inventory_matches_frozen_original_counts():
    # fixture monkeypatching is deliberately absent in this test.
    groups = raw_bundle.inventory()
    assert len(groups['textvqa']) == 3167
    assert len(groups['mmsafety']) == 39
    assert len(groups['benign_source']) == 1
    assert len(set().union(*(set(files) for files in groups.values()))) == 3207
    assert not any('benign_' in name or 'fitted' in name or 'split' in name
                   for files in groups.values() for name in files)


def test_wrapper_keeps_raw_import_and_processing_in_one_step():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'scripts/02_prepare_assets_data.sh').read_text(encoding='utf-8')
    assert '--raw-package)' in text
    assert 'RAW_ARGS=(--raw-package "$RAW_PACKAGE")' in text
    public = (root / 'pipeline/public_assets.py').read_text(encoding='utf-8')
    assert 'bundle = restore(raw_package, assets)' in public
    assert 'benign = build(parquet, raw /' in public
    assert 'rebuild(sources, assets)' in public and 'verify_rebuilt(assets)' in public


def test_recipe_is_portable_across_windows_linux(bundle):
    recipe = raw_bundle.recipe_root()
    path = recipe / 'inputs.sha256'
    path.write_bytes(b'first\nsecond\n')
    expected = raw_bundle.recipe_identity()
    path.write_bytes(b'first\r\nsecond\r\n')
    assert raw_bundle.recipe_identity() == expected
    path.write_bytes(b'different\r\nsecond\r\n')
    assert raw_bundle.recipe_identity() != expected
