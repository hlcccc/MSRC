"""Resource recovery must not import a selected head, held-out scores or wrong features."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from pipeline import common, plugguard

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/04_recover_plugguard_development.py'
spec = importlib.util.spec_from_file_location('development_recovery', SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def failed(tmp_path):
    source = tmp_path / 'failed'
    source.mkdir()
    rows = [dict(id=str(i), split='dev', label=0, group=str(i), suite='benign') for i in range(3360)]
    (source / 'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    dump(source / 'input_hashes.json', {'images': {}})
    (source / 'inference_source.py').write_bytes(Path(plugguard.__file__).read_bytes())
    dump(source / 'protocol.json', dict(backend='plugguard', execution_kind='fresh_inference',
         training_mode='development_refit', source_inputs_sha256=common.sha(source / 'inputs.jsonl'),
         input_hashes_sha256=common.sha(source / 'input_hashes.json'),
         source_sha256=common.sha(source / 'inference_source.py'), created_ns=1))
    dump(source / 'status.json', dict(state='FAILED', error=recovery.RESOURCE_FAILURE, pid=99999999, failed_ns=5))
    dump(source / 'execution_claim.json', dict(pid=99999999, boot_id='not-current-boot'))
    dump(source / 'dataset_identity.json', {})
    dump(source / 'environment.json', {})
    record = dict(rows[0], threshold=None, warn=False, max_score=0, events=[], trigger_token=None,
                  generated_tokens=2, released_tokens=2, start_ns=2, end_ns=3)
    (source / 'development_generation.jsonl').write_text(json.dumps(record)+'\n', encoding='utf-8')
    return source


def test_only_valid_resource_failed_development_prefix_is_accepted(failed):
    protocol, records, failure = recovery.validate_source(failed, plugguard.__file__)
    assert len(records) == 1 and records[0]['id'] == '0'
    assert failure['state'] == 'FAILED' and protocol['training_mode'] == 'development_refit'


@pytest.mark.parametrize('artifact', recovery.FORBIDDEN)
def test_head_threshold_or_any_test_artifact_blocks_recovery(failed, artifact):
    (failed / artifact).write_bytes(b'not permitted, even if empty of results')
    with pytest.raises(ValueError, match='before head/threshold/test'):
        recovery.validate_source(failed, plugguard.__file__)


@pytest.mark.parametrize('mutation', ['wrong_failure', 'changed_source', 'changed_input',
                                    'out_of_order', 'test_row', 'partial_response', 'after_failure', 'scored'])
def test_invalid_source_cannot_be_recovered(failed, mutation):
    if mutation == 'wrong_failure':
        state = recovery.load(failed / 'status.json')
        state['error'] = 'out of memory, not an authorized foreign-owner recovery'
        dump(failed / 'status.json', state)
    elif mutation == 'changed_source':
        (failed / 'inference_source.py').write_bytes(b'changed algorithm')
    elif mutation == 'changed_input':
        with (failed / 'inputs.jsonl').open('a') as handle:
            handle.write('{}\n')
    else:
        rows = common.read_rows(failed / 'development_generation.jsonl')
        if mutation == 'out_of_order': rows[0]['id'] = '1'
        elif mutation == 'test_row': rows[0]['split'] = 'test'
        elif mutation == 'partial_response': rows[0]['released_tokens'] = 1
        elif mutation == 'after_failure': rows[0]['end_ns'] = 6
        else: rows[0]['threshold'] = .5
        (failed / 'development_generation.jsonl').write_text(json.dumps(rows[0])+'\n', encoding='utf-8')
    with pytest.raises(ValueError):
        recovery.validate_source(failed, plugguard.__file__)


def test_copied_feature_hash_and_original_bytes_are_preserved(failed, tmp_path, monkeypatch):
    _, records, _ = recovery.validate_source(failed, plugguard.__file__)
    root = tmp_path / 'new_run'
    (root / 'features').mkdir(parents=True)
    (failed / 'features').mkdir()
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(plugguard, 'ROOT', root)
    name = recovery.hashlib.sha256(b'0').hexdigest()+'.pt'
    path = failed / 'features' / name
    path.write_bytes(b'synthetic feature bytes, not a real checkpoint')
    feature = dict(id='0', label=0, input_sha256='input',
                   prefix=SimpleNamespace(shape=(1, 3, 3584)), response=SimpleNamespace(shape=(1, 2, 3584)))
    result = recovery.copy_features(failed, records, 'input', lambda p: feature)
    assert len(result) == 1
    assert path.read_bytes() == (root / 'features' / name).read_bytes()
    assert result[0]['sha256'] == common.sha(path)
    assert common.read_rows(root / 'development_generation.jsonl') == records
    with pytest.raises(ValueError, match='overwrite'):
        recovery.copy_features(failed, records, 'input', lambda p: feature)


@pytest.mark.parametrize('field,value', [('id', 'test-id'), ('label', 1), ('input_sha256', 'wrong'),
                                      ('response', SimpleNamespace(shape=(1, 1, 3584)))])
def test_wrong_or_partial_feature_cannot_be_copied(failed, tmp_path, monkeypatch, field, value):
    _, records, _ = recovery.validate_source(failed, plugguard.__file__)
    root = tmp_path / 'new_run'
    (root / 'features').mkdir(parents=True)
    (failed / 'features').mkdir()
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(plugguard, 'ROOT', root)
    name = recovery.hashlib.sha256(b'0').hexdigest()+'.pt'
    (failed / 'features' / name).write_bytes(b'fixture')
    feature = dict(id='0', label=0, input_sha256='input',
                   prefix=SimpleNamespace(shape=(1, 3, 3584)), response=SimpleNamespace(shape=(1, 2, 3584)))
    feature[field] = value
    with pytest.raises(ValueError):
        recovery.copy_features(failed, records, 'input', lambda p: feature)
    assert not list((root / 'features').glob('*.pt'))


def test_live_worker_is_not_recovered(failed, monkeypatch):
    if not Path('/proc/sys/kernel/random/boot_id').exists():
        pytest.skip('Linux process identity guard')
    dump(failed / 'execution_claim.json', dict(pid=os.getpid(),
         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    state = recovery.load(failed / 'status.json')
    state['pid'] = os.getpid()
    dump(failed / 'status.json', state)
    with pytest.raises(ValueError, match='still present'):
        recovery.validate_source(failed, plugguard.__file__)


@pytest.mark.parametrize('mutation', [None, 'copied_record', 'feature_bytes', 'origin_file', 'new_time'])
def test_recovery_chain_rechecks_exact_original_records_and_features(failed, tmp_path, monkeypatch, mutation):
    _, original, _ = recovery.validate_source(failed, plugguard.__file__)
    root = tmp_path / 'continued'
    origin = root / 'recovery_origin'
    origin.mkdir(parents=True)
    (root / 'features').mkdir()
    for name in recovery.ORIGIN_FILES:
        shutil.copyfile(failed / name, origin / name)
    old = recovery.load(failed / 'protocol.json')
    protocol = dict(old, created_ns=10)
    name = recovery.hashlib.sha256(b'0').hexdigest()+'.pt'
    feature = root / 'features' / name
    feature.write_bytes(b'unchanged fixture')
    (root / 'recovery_source.py').write_bytes(SCRIPT.read_bytes())
    manifest = dict(recovery_source_sha256=common.sha(SCRIPT),
        origin_files_sha256={name:common.sha(origin/name) for name in recovery.ORIGIN_FILES},
        recovered_development_rows=1, failed_source_run=str(failed),
        features=[dict(id='0', filename=name, sha256=common.sha(feature),
                       record_sha256=recovery.record_sha(original[0]))])
    dump(root / 'development_recovery.json', manifest)
    protocol['development_recovery_sha256'] = common.sha(root / 'development_recovery.json')
    dump(root / 'protocol.json', protocol)
    current = original + [dict(original[0], id='1', start_ns=11, end_ns=12)]
    if mutation == 'copied_record': current[0] = dict(current[0], answer='changed after copying')
    elif mutation == 'feature_bytes': feature.write_bytes(b'different bytes')
    elif mutation == 'origin_file': (origin/'inputs.jsonl').write_bytes(b'changed')
    elif mutation == 'new_time': current[1]['start_ns'] = 9
    (root / 'development_generation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current), encoding='utf-8')
    if mutation:
        with pytest.raises(ValueError):
            recovery.audit_recovery(root)
    else:
        result = recovery.audit_recovery(root)
        assert result['recovered_development_rows'] == result['newly_generated_development_rows'] == 1
        assert result['recovered_test_scores'] is False
