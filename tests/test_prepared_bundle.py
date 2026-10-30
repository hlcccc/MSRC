"""Prepared data may include fixed splits, never predictions or fitted states."""
import hashlib
import io
import json
import tarfile

import pytest

from pipeline import common, prepared_bundle as bundle


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    recipe = tmp_path / 'recipe'
    recipe.mkdir()
    for name in ('inputs.sha256', bundle.REGISTRY, bundle.NOTICE):
        (recipe / name).write_text('fixed public recipe', encoding='utf-8')
    monkeypatch.setattr(bundle, 'recipe_root', lambda: recipe)
    payload = {'data/factual/input.jsonl': b'public input',
               'partitions/train.jsonl': b'fixed split'}
    frozen = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    monkeypatch.setattr(bundle, 'inventory', lambda: frozen)
    monkeypatch.setattr(bundle, 'verify_partitions', lambda root: {'fixed': True})
    assets = tmp_path / 'source'
    for name, content in payload.items():
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (assets / 'head.pt').write_bytes(b'not data')
    (assets / 'prediction.jsonl').write_bytes(b'not an input')
    package = tmp_path / 'prepared.tar'
    bundle.pack(assets, package)
    return tmp_path, assets, package, payload


def rewrite(source, target, change=None, extra=None):
    with tarfile.open(source) as old, tarfile.open(target, 'w') as new:
        for item in old.getmembers():
            content = old.extractfile(item).read()
            content = change(item.name, content) if change else content
            entry = tarfile.TarInfo(item.name)
            entry.size = len(content)
            new.addfile(entry, io.BytesIO(content))
        if extra:
            name, kind = extra
            entry = tarfile.TarInfo(name)
            entry.type, entry.linkname = kind, '../outside'
            new.addfile(entry, io.BytesIO(b''))


def test_prepared_restore_and_verify(prepared):
    root, _, archive, payload = prepared
    target = root / 'new/assets'
    result = bundle.restore(archive, target)
    assert result['fixed_partitions_restored'] is True
    assert result['raw_data_processing_performed'] is False
    assert result['predictions_restored'] is False
    assert result['fitted_parameters_restored'] is False
    assert result['archive_sha256'] == common.sha(archive)
    assert bundle.verify(target) == {'fixed': True}
    for name, value in payload.items():
        assert (target / name).read_bytes() == value
    assert not (target / 'head.pt').exists() and not (target / 'prediction.jsonl').exists()


def test_deterministic_allowlist_pack(prepared):
    root, assets, archive, _ = prepared
    other = root / 'again.tar'
    bundle.pack(assets, other)
    assert archive.read_bytes() == other.read_bytes()


@pytest.mark.parametrize('name', ['prediction.jsonl', 'head.pt', '../escape', '/absolute',
                                 'data/unknown.jsonl', 'partitions/unknown.jsonl'])
def test_extra_inputs_results_or_escaping_members_rejected(prepared, name):
    root, _, archive, _ = prepared
    bad = root / 'bad.tar'
    rewrite(archive, bad, extra=(name, tarfile.REGTYPE))
    with pytest.raises(ValueError, match='unapproved'):
        bundle.restore(bad, root / 'never_created')
    assert not (root / 'never_created').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_special_archive_members_rejected(prepared, kind):
    root, _, archive, _ = prepared
    bad = root / 'bad.tar'
    rewrite(archive, bad, extra=('special', kind))
    with pytest.raises(ValueError, match='unsafe'):
        bundle.restore(bad, root / 'never_created')


def test_modified_partition_rejected_before_restore(prepared):
    root, _, archive, _ = prepared
    bad = root / 'bad.tar'
    rewrite(archive, bad, change=lambda name, value: value[::-1] if name.startswith('partitions/') else value)
    with pytest.raises(ValueError, match='hash mismatch'):
        bundle.restore(bad, root / 'never_created')
    assert not (root / 'never_created').exists()


def test_archive_cannot_authorize_its_own_revised_split(prepared):
    root, _, archive, _ = prepared
    bad = root / 'bad.tar'
    def change(name, value):
        if name == 'prepared_bundle.json':
            manifest = json.loads(value)
            manifest['files']['partitions/train.jsonl']['sha256'] = 'a' * 64
            return json.dumps(manifest).encode()
        return value
    rewrite(archive, bad, change=change)
    with pytest.raises(ValueError, match='pinned recipe'):
        bundle.restore(bad, root / 'never_created')


@pytest.mark.parametrize('name', ['data', 'raw', 'partitions'])
def test_does_not_overwrite_existing_data(prepared, name):
    root, _, archive, _ = prepared
    assets = root / 'existing'
    (assets / name).mkdir(parents=True)
    with pytest.raises(ValueError, match='fresh'):
        bundle.restore(archive, assets)


def test_separate_fitted_parameters_can_coexist(prepared):
    root, _, archive, _ = prepared
    assets = root / 'new/assets'
    (assets / 'fitted').mkdir(parents=True)
    (assets / 'fitted/head.pt').write_bytes(b'separate fitted delivery')
    bundle.restore(archive, assets)
    assert (assets / 'fitted/head.pt').read_bytes() == b'separate fitted delivery'


def test_modified_installed_input_rejected(prepared):
    root, _, archive, _ = prepared
    assets = root / 'new/assets'
    bundle.restore(archive, assets)
    (assets / 'partitions/train.jsonl').write_bytes(b'revised split')
    with pytest.raises(ValueError, match='changed'):
        bundle.verify(assets)


def test_extra_installed_input_rejected(prepared):
    root, _, archive, _ = prepared
    assets = root / 'new/assets'
    bundle.restore(archive, assets)
    (assets / 'data/prediction.jsonl').write_bytes(b'old result')
    with pytest.raises(ValueError, match='extra/missing'):
        bundle.verify(assets)


def test_recipe_identity_ignores_only_platform_line_endings(prepared):
    recipe = bundle.recipe_root()
    for name in ('inputs.sha256', bundle.REGISTRY):
        (recipe / name).write_bytes(b'first\nsecond\n')
    expected = bundle.recipe_identity()
    for name in ('inputs.sha256', bundle.REGISTRY):
        (recipe / name).write_bytes(b'first\r\nsecond\r\n')
    assert bundle.recipe_identity() == expected
    (recipe / bundle.REGISTRY).write_bytes(b'changed\r\nsecond\r\n')
    assert bundle.recipe_identity() != expected


def test_same_verified_import_can_resume_model_download(prepared):
    root, _, archive, _ = prepared
    assets = root / 'new/assets'
    first = bundle.restore(archive, assets)
    assert bundle.restore(archive, assets) == first


def transport(root, archive):
    folder = root / 'transport'
    folder.mkdir()
    data = archive.read_bytes()
    parts = []
    for index, start in enumerate(range(0, len(data), 4096), 1):
        path = folder / (archive.name + '.%03d' % index)
        payload = data[start:start + 4096]
        path.write_bytes(payload)
        parts.append(dict(name=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    value = dict(format='msrc-prepared-transport-v1', archive=archive.name,
                 sha256=common.sha(archive), parts=parts)
    (folder / 'parts.json').write_text(json.dumps(value), encoding='utf-8')
    return folder


def test_transport_parts_are_joined_and_restored_without_user_processing(prepared):
    root, _, archive, payload = prepared
    folder = transport(root, archive)
    assets = root / 'new/assets'
    bundle.restore(folder, assets)
    for name, content in payload.items():
        assert (assets / name).read_bytes() == content
    assert bundle.materialize(folder, assets).read_bytes() == archive.read_bytes()


def test_missing_or_corrupt_transport_part_fails_before_join(prepared):
    root, _, archive, _ = prepared
    folder = transport(root, archive)
    (folder / (archive.name + '.001')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='missing or corrupt'):
        bundle.restore(folder, root / 'never_created')
    assert not (root / 'never_created').exists()
