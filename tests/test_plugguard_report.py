"""CPU checks for grouping, the held-out metric, and actual online event evidence."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import acceptance_check, common, plugguard, plugguard_report as pg
from pipeline.safety_split import SPLIT_VERSION, source_sha256, validate_partition


def dump(root, name, value):
    (root / name).write_text(json.dumps(value), encoding='utf-8')


def jsonl(root, name, rows):
    (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


@pytest.fixture
def inputs():
    rows = []
    for side in ('dev', 'test'):
        for i in range(840):
            for label, suites in [(0, ['benign']), (1, ['SD', 'SD_TYPO', 'TYPO'])]:
                group = '%s:%d:%d' % (side, label, i)
                for suite in suites:
                    rows.append(dict(id=group + ':' + suite, group=group, label=label,
                                     split=side, suite=suite, question=group + ':' + suite,
                                     image=str(Path('/fixture/image.png').resolve()),
                                     image_sha256='a' * 64, split_group=group))
    return rows


@pytest.fixture
def evidence(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    root = tmp_path / 'safety'
    root.mkdir()
    jsonl(root, 'inputs.jsonl', inputs)
    roles = pg.assign_roles(inputs)
    dump(root, 'input_hashes.json', {'images': {inputs[0]['image']: 'a' * 64}})
    source = Path(plugguard.__file__).read_bytes()
    (root / 'inference_source.py').write_bytes(source)
    (root / 'head.pt').write_bytes(b'fixture, not a trained model')
    dump(root, 'preprocessing_protocol.json', {})
    dump(root, 'protocol.json', dict(source_inputs_sha256=common.sha(root / 'inputs.jsonl'),
         source_sha256=common.sha(root / 'inference_source.py'), generation_max_tokens=128,
         release_buffer_tokens=16, created_ns=90, execution_kind='fresh_inference',
         split_version=SPLIT_VERSION, split_source_sha256=source_sha256(),
         input_hashes_sha256=common.sha(root / 'input_hashes.json'),
         preprocessing_protocol_sha256=common.sha(root / 'preprocessing_protocol.json')))
    dump(root, 'dataset_identity.json', dict(test_used_for_selection=False, original_group_overlap=0,
         split_integrity=validate_partition(inputs, roles=roles),
         roles={k: sorted(v) for k, v in roles.items()}))
    jsonl(root, 'development_generation.jsonl', [r for r in inputs if r['split'] == 'dev'])
    jsonl(root, 'calibration_scores.jsonl', [dict(id=r['id'], label=r['label'],
          score=.9 if r['label'] else .1) for r in inputs if r['id'] in roles['calibration']])
    threshold = .1005
    dump(root, 'threshold.json', dict(threshold=threshold, grid_points=2001, tie_rule='lowest',
         head_sha256=common.sha(root / 'head.pt'), frozen_before_test_ns=100,
         test_used_for_fitting_or_selection=False))
    deliveries = []
    for r in inputs:
        if r['split'] == 'test':
            warn = bool(r['label'])
            deliveries.append(dict(r, threshold=threshold, warn=warn, start_ns=101, end_ns=103,
                reason='risk_gate' if warn else 'eos', trigger_token=1 if warn else None,
                generated_tokens=1, released_tokens=0 if warn else 1, max_score=.9 if warn else .1,
                events=[dict(token=1, risk=.9 if warn else .1, score_ns=102, warn=warn,
                             pending=1, released_before=0)]))
    jsonl(root, 'test_delivery.jsonl', deliveries)
    jsonl(root, 'answer_judgments.jsonl', [dict(id=r['id'], response_risk_label='Safe',
          offline_only_not_used_for_threshold=True) for r in deliveries])
    dump(root, 'primary_metric.json', dict(n=3360, tp=2520, tn=840, fp=0, fn=0,
         accuracy=1.0, primary_label=pg.PRIMARY_LABEL))
    return root


def test_metric_definition_matches_original():
    rows = [dict(label=1, warn=True), dict(label=0, warn=False),
            dict(label=0, warn=True), dict(label=1, warn=False)]
    result = pg.metrics(rows)
    assert result['accuracy'] == .5
    assert all(result[k] == 1 for k in ('TP', 'TN', 'FP', 'FN'))
    assert result['all_alarm'] == .5
    assert result['accuracy'] == plugguard.metrics(rows)['accuracy']


def test_roles_keep_original_variants_and_calibration_disjoint(inputs):
    roles = pg.assign_roles(inputs)
    assert {k: len(v) for k, v in roles.items()} == {'fit': 2688, 'calibration': 672, 'test': 3360}
    by_id = {r['id']: r for r in inputs}
    groups = [{by_id[rid]['group'] for rid in ids} for ids in roles.values()]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_cross_split_variant_rejected_without_changing_counts(inputs):
    first = next(r for r in inputs if r['split'] == 'dev' and r['suite'] == 'TYPO')
    other = next(r for r in inputs if r['split'] == 'test' and r['suite'] == 'TYPO')
    first['split'], other['split'] = other['split'], first['split']
    with pytest.raises(ValueError, match='crosses'):
        pg.assign_roles(inputs)


def test_checkpoint_identity_is_relocatable_but_content_sensitive(inputs):
    hashes = {inputs[0]['image']: 'a' * 64}
    before = pg.checkpoint_identity(inputs, hashes)
    relocated = [dict(r, image='/different/image.png') for r in inputs]
    assert before == pg.checkpoint_identity(relocated, {'/different/image.png': 'a' * 64})
    assert before != pg.checkpoint_identity(relocated, {'/different/image.png': 'b' * 64})


def test_complete_evidence_and_per_source_report(evidence):
    result = pg.write_report(evidence)
    assert result['primary']['accuracy'] == 1
    assert set(result['by_source']) == set(pg.SUITES)
    assert all(v['n'] == 840 for v in result['by_source'].values())
    assert result['warning_timing']['warned_cases_with_released_tokens'] == 0
    assert result['warning_timing']['scored_token_events'] == 3360
    acceptance_check.safety_check(evidence, result)


@pytest.mark.parametrize('mutation', ['missing_event', 'early_release', 'changed_label', 'freeze_after_test',
                                    'duplicate_id', 'different_threshold', 'judge_missing', 'future_score'])
def test_corrupt_or_incomplete_evidence_cannot_pass(evidence, mutation):
    rows = common.read_rows(evidence / 'test_delivery.jsonl')
    if mutation == 'missing_event':
        rows[0]['events'] = []
    elif mutation == 'early_release':
        rows[0]['events'][0]['release_ns'] = 101
    elif mutation == 'changed_label':
        rows[0]['label'] = 1 - rows[0]['label']
    elif mutation == 'freeze_after_test':
        rows[0]['start_ns'] = 99
    elif mutation == 'duplicate_id':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'different_threshold':
        rows[0]['threshold'] = .2
    elif mutation == 'judge_missing':
        judges = common.read_rows(evidence / 'answer_judgments.jsonl')
        jsonl(evidence, 'answer_judgments.jsonl', judges[:-1])
    else:
        rows[0]['events'][0]['score_ns'] = 104
    jsonl(evidence, 'test_delivery.jsonl', rows)
    with pytest.raises(ValueError):
        pg.audit(evidence)


def test_test_set_score_cannot_select_threshold(evidence):
    state = pg.load(evidence / 'threshold.json')
    state['threshold'] = .5
    dump(evidence, 'threshold.json', state)
    with pytest.raises(ValueError, match='development-only'):
        pg.audit(evidence)


def test_report_rejects_cross_id_identical_requests_even_if_original_ids_are_disjoint(inputs):
    dev = next(r for r in inputs if r['split'] == 'dev' and r['suite'] == 'TYPO')
    test = next(r for r in inputs if r['split'] == 'test' and r['suite'] == 'TYPO')
    test['question'] = dev['question']
    assert dev['group'] != test['group']
    with pytest.raises(ValueError, match='component crosses'):
        pg.assign_roles(inputs)


def test_report_does_not_trust_fabricated_zero_overlap_metadata(evidence):
    rows = common.read_rows(evidence / 'inputs.jsonl')
    dev = next(r for r in rows if r['split'] == 'dev' and r['suite'] == 'TYPO')
    test = next(r for r in rows if r['split'] == 'test' and r['suite'] == 'TYPO')
    test['question'] = dev['question']
    jsonl(evidence, 'inputs.jsonl', rows)
    protocol = pg.load(evidence / 'protocol.json')
    protocol['source_inputs_sha256'] = common.sha(evidence / 'inputs.jsonl')
    dump(evidence, 'protocol.json', protocol)
    with pytest.raises(ValueError, match='component crosses'):
        pg.audit(evidence)


def test_archived_evidence_cannot_be_presented_as_new_full_run(evidence):
    dump(evidence, 'evidence_origin.json', dict(execution_kind='archived_trial_report'))
    result = pg.write_report(evidence)
    assert result['old_scores_used'] is True
    with pytest.raises(ValueError, match='archived'):
        acceptance_check.safety_check(evidence, result)


def test_changed_base_weights_are_rejected(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    (tmp_path / 'assets/plugguard').mkdir(parents=True)
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'weights.safetensors').write_bytes(b'original')
    spec = dict(repo='fixture', revision='a' * 40,
                files={p.name: common.sha(p) for p in model.iterdir()})
    dump(tmp_path, 'assets/plugguard/base_models.json', dict(generator=spec, judge=spec))
    monkeypatch.setattr(plugguard, 'PROJECT', tmp_path)
    assert plugguard.verify_base_models(model, model)['generator']['revision'] == 'a' * 40
    (model / 'weights.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='paired model file'):
        plugguard.verify_base_models(model, model)
