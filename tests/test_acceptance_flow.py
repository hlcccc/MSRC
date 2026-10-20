"""Acceptance completion is not the same thing as meeting metric targets."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import acceptance_check, common, prepare_assets, preflight, report


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    return tmp_path


def factual_report():
    return dict(status='FULL_FACTUAL_FLOW_COMPLETE', mode='full', old_evidence_used=False,
                cells=[dict(cell=cell, n=9000 if cell.endswith('pope') else 5000,
                            n_dev=4500 if cell.endswith('pope') else 2504,
                            n_test=4500 if cell.endswith('pope') else 2496,
                            ece_before=.2, ece_after=.15, relative_improvement=.25,
                            minimum_actual_signals_per_test_sample=4)
                       for cell in sorted(acceptance_check.EXPECTED_CELLS)])


def safety_report():
    return dict(status='FULL_SAFETY_FLOW_COMPLETE', old_scores_used=False,
                all_test_delivery_and_judge=True, threshold=.27, passing=True,
                primary=dict(n=3360, TP=2400, TN=288, FP=552, FN=120, accuracy=.8))


def seed_reports(root, factual=None, safety=None):
    run = root / 'runs' / 'test_run'
    run.mkdir(parents=True)
    for name, value in [('factual', factual), ('safety', safety)]:
        if value is not None:
            (run / name).mkdir()
            (run / name / 'report.json').write_text(json.dumps(value))
    return run


def test_claimed_full_report_without_artifacts_is_incomplete(workspace):
    run = seed_reports(workspace, factual_report(), safety_report())
    result = report.build(run, 'milestone')
    assert result['verdict'] == 'INCOMPLETE'
    assert result['acceptance_pass'] is None
    assert len(result['incomplete_reasons']) == 2


def test_missing_module_cannot_pass(workspace):
    run = seed_reports(workspace, factual=factual_report())
    result = report.build(run, 'milestone')
    assert not result['complete']
    assert any('safety: missing report' in r for r in result['incomplete_reasons'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'smoke', 'old'])
def test_factual_coverage_status_and_provenance_are_required(workspace, mutation):
    factual = factual_report()
    if mutation == 'missing':
        factual['cells'].pop()
    if mutation == 'duplicate':
        factual['cells'][-1] = factual['cells'][0]
    if mutation == 'smoke':
        factual['status'], factual['mode'] = 'SMOKE_ONLY_NOT_FULL_ACCEPTANCE', 'smoke'
    if mutation == 'old':
        factual['old_evidence_used'] = True
    with pytest.raises(ValueError):
        acceptance_check.factual_check(workspace, factual)


def test_pending_safety_is_not_complete(workspace):
    safety = safety_report()
    safety['status'] = 'FRESH_INFERENCE_COMPLETE_DELIVERY_PENDING'
    with pytest.raises(ValueError, match='not fully complete'):
        acceptance_check.safety_check(workspace, safety)


def test_stage_targets_are_recomputed_not_copied(workspace, monkeypatch):
    run = seed_reports(workspace, factual_report(), safety_report())
    # Artifact validation has separate tests; isolate the stage-decision policy.
    monkeypatch.setattr(report, 'check', lambda *args: [])
    assert report.build(run, 'milestone')['verdict'] == 'PASS'
    completion = report.build(run, 'completion')
    assert completion['verdict'] == 'FAIL'
    assert completion['signals']['safety']['passing'] is False


def test_incomplete_exit_code_never_prints_run_completed(workspace, monkeypatch, capsys):
    seed_reports(workspace, factual=factual_report())
    monkeypatch.setattr(report, 'runs_dir', lambda: workspace / 'runs')
    monkeypatch.setattr(sys, 'argv', ['report', '--run-id', 'test_run'])
    assert report.main() == 2
    text = capsys.readouterr().out
    assert 'RUN_INCOMPLETE' in text
    assert 'RUN_COMPLETED' not in text


def test_jsonl_image_paths_are_relative_to_the_file(workspace):
    path = workspace / 'data' / 'input.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(id='a', image='../images/a.jpg')) + '\n')
    assert common.read_rows(path)[0]['image'] == str(workspace / 'images/a.jpg')


@pytest.mark.parametrize('text', ['', 'bad  data/a.jsonl\n', 'a' * 64 + '  ../outside\n'])
def test_manifest_empty_malformed_or_escaping_is_rejected(workspace, text):
    assets = workspace / 'assets'
    assets.mkdir()
    (assets / 'MANIFEST.sha256').write_text(text)
    with pytest.raises(ValueError):
        prepare_assets.verify(assets)


def test_manifest_must_cover_images_not_just_jsonl(workspace):
    assets = workspace / 'assets'
    (assets / 'data').mkdir(parents=True)
    path = assets / 'data/input.jsonl'
    path.write_text('{}\n')
    (assets / 'data/image.png').write_bytes(b'image')
    (assets / 'MANIFEST.sha256').write_text(common.sha(path) + '  data/input.jsonl\n')
    with pytest.raises(SystemExit, match='does not cover'):
        prepare_assets.verify(assets)


def test_cli_fit_groups_copies_by_original_image_identity():
    from msrc.cli import _split
    from evaluation.groups import image_group_id
    rows = [dict(dataset='POPE', id='%s:%s' % (setting, i),
                 image='/images/%s_COCO_val2014_%012d.jpg' % (setting, i))
            for i in range(4) for setting in ('random', 'popular', 'adversarial')]
    dev, test = _split(rows, .5, 20260920)
    assert dev.sum() == test.sum() == 6
    assert not ({image_group_id(r) for r, keep in zip(rows, dev) if keep}
                & {image_group_id(r) for r, keep in zip(rows, test) if keep})


def test_download_uses_modern_hub_api_and_reuses_duplicate_model(workspace, monkeypatch):
    calls = []
    def snapshot_download(**kwargs):
        assert set(kwargs) == {'repo_id', 'revision', 'local_dir', 'max_workers'}
        assert kwargs['revision'] == 'a' * 40
        calls.append(kwargs)
        folder = Path(kwargs['local_dir'])
        folder.mkdir()
        (folder / 'config.json').write_text('{}')
        (folder / 'model.safetensors').write_bytes(b'weight fixture')
    api = SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha='a' * 40))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=snapshot_download, HfApi=lambda: api))
    prepare_assets.download_models(workspace / 'assets', None)
    assert len(calls) == 5
    assert 'inclusionAI/SingGuard-4b' in {call['repo_id'] for call in calls}
    models = json.loads((workspace / 'assets/models.lock.json').read_text())['models']
    assert models['generator']['path'] == models['qwen']['path']
    prepare_assets.verify_model_inventory(workspace / 'assets')


def test_frozen_metadata_preserves_local_template_after_public_download(workspace):
    source = workspace / 'original_guard'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    (source / 'chat_template.jinja').write_text('frozen compatibility template')
    (source / 'model.safetensors').write_bytes(b'unchanged weights')
    files = {p.name: common.sha(p) for p in source.iterdir()}
    assets = workspace / 'assets'
    expected = {'guard': dict(path=str(source), files=files)}
    prepare_assets.export_model_metadata(assets, expected)
    target = assets / 'models/guard'
    target.mkdir(parents=True)
    (target / 'chat_template.jinja').write_text('different public template')
    (target / 'model.safetensors').write_bytes(b'unchanged weights')
    prepare_assets.apply_model_metadata(assets, {'guard': {'path': str(target)}}, expected)
    assert (target / 'chat_template.jinja').read_text() == 'frozen compatibility template'
    assert (target / 'model.safetensors').read_bytes() == b'unchanged weights'
    assert all(common.sha(target / name) == digest for name, digest in files.items())


def test_model_inventory_reads_utf8_even_with_a_different_system_default(workspace, monkeypatch):
    model = workspace / '\u6a21\u578b'
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    (model / 'model.safetensors').write_bytes(b'synthetic weights')
    assets = workspace / 'assets'
    prepare_assets.inventory_models(assets, {name: {'path': str(model)} for name in prepare_assets.REPOS})
    original = Path.read_text

    def non_utf8_default(path, encoding=None, **kwargs):
        return original(path, encoding=encoding or 'latin-1', **kwargs)

    monkeypatch.setattr(Path, 'read_text', non_utf8_default)
    prepare_assets.verify_model_inventory(assets)


def test_metadata_cannot_replace_weights_or_use_tampered_templates(workspace):
    assets = workspace / 'assets'
    source = assets / 'model_metadata/guard'
    source.mkdir(parents=True)
    target = assets / 'models/guard'
    target.mkdir(parents=True)
    (source / 'model.safetensors').write_bytes(b'not a permitted metadata overlay')
    digest = common.sha(source / 'model.safetensors')
    manifest = assets / 'model_metadata_manifest.json'
    manifest.write_text(json.dumps({'guard': {'model.safetensors': digest}}))
    expected = {'guard': {'files': {'model.safetensors': digest}}}
    with pytest.raises(ValueError, match='approved frozen'):
        prepare_assets.apply_model_metadata(assets, {'guard': {'path': str(target)}}, expected)
    (source / 'chat_template.jinja').write_text('tampered')
    manifest.write_text(json.dumps({'guard': {'chat_template.jinja': 'a' * 64}}))
    expected = {'guard': {'files': {'chat_template.jinja': 'a' * 64}}}
    with pytest.raises(ValueError, match='digest mismatch'):
        prepare_assets.apply_model_metadata(assets, {'guard': {'path': str(target)}}, expected)


def test_gpu_preflight_honors_command_line_override(monkeypatch):
    monkeypatch.setenv('MSRC_GPU', '1')
    monkeypatch.setattr(preflight, 'load_config', lambda: {'gpu': '0'})
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: '/usr/bin/nvidia-smi')
    monkeypatch.setattr(preflight.subprocess, 'check_output', lambda command, **kw:
                        '0, gpu-0, 80000, 10\n1, gpu-1, 80000, 10\n'
                        if '--query-gpu=index,uuid,memory.total,memory.used' in command else '')
    assert preflight.check_gpu()[0]['index'] == 1


def test_busy_gpu_is_waited_for_without_signalling_processes(monkeypatch, capsys):
    calls, sleeps = [], []
    def check(ignore_current_process=False):
        calls.append(ignore_current_process)
        if len(calls) == 1:
            raise preflight.GPUInUse('GPU 0 is occupied by another user pid 999')
        return [{'index': 0}]
    monkeypatch.setattr(preflight, 'check_gpu', check)
    monkeypatch.setattr(preflight.time, 'sleep', sleeps.append)
    assert preflight.wait_for_gpu() == [{'index': 0}]
    assert calls == [True, True]
    assert sleeps == [120]
    assert 'GPU_WAIT' in capsys.readouterr().out


def test_score_ids_must_not_silently_overwrite():
    with pytest.raises(ValueError, match='duplicate'):
        acceptance_check.unique([{'id': 'x'}, {'id': 'x'}])


def test_full_safety_output_is_independently_checked(workspace):
    import numpy as np
    from pipeline.safety_split import SPLIT_VERSION, source_sha256, validate_partition
    root = workspace / 'safety'
    root.mkdir()
    def dump(name, value):
        (root / name).write_text(json.dumps(value))
    def jsonl(name, rows):
        (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    rows = []
    for side in ('dev', 'test'):
        for i in range(840):
            rows.append(dict(id='%s:b:%d' % (side, i), label=0, split=side,
                             group='%s:b:%d' % (side, i)))
            for variant in ('SD', 'TYPO', 'SD_TYPO'):
                rows.append(dict(id='%s:h:%d:%s' % (side, i, variant), label=1,
                                 split=side, group='%s:h:%d' % (side, i)))
    for r in rows:
        r.update(question=r['id'], image=str(Path('/fixture/image.png').resolve()), image_sha256='a' * 64,
                 split_group=r['group'])
    scores = [dict(id=r['id'], p_harm=.9 if r['label'] else .1,
                   inference_start_ns=1, inference_end_ns=2) for r in rows]
    jsonl('inputs.jsonl', rows)
    jsonl('fresh_scores.jsonl', scores)
    dump('input_hashes.json', {'images': {rows[0]['image']: 'a' * 64}})
    dump('protocol.json', dict(inputs_sha256=common.sha(root / 'inputs.jsonl'), created_ns=0,
         split_version=SPLIT_VERSION, split_source_sha256=source_sha256(),
         input_hashes_sha256=common.sha(root / 'input_hashes.json')))
    threshold = float(np.linspace(0, 1, 2001)[201])
    state = dict(threshold=threshold, selected_on='dev only', n_dev=3360,
                 dev_correct=3360, protocol_sha256=common.sha(root / 'protocol.json'), frozen_ns=3)
    dump('operating_point.json', state)
    test = [r for r in rows if r['split'] == 'test']
    delivered = [dict(id=r['id'], threshold=threshold, warn=bool(r['label']),
                      guard_done_ns=2, decision_done_ns=4, sink_write_start_ns=7,
                      generation_start_ns=None if r['label'] else 5,
                      generation_done_ns=None if r['label'] else 6) for r in test]
    jsonl('delivery.jsonl', delivered)
    jsonl('answer_judgements.jsonl', [dict(id=r['id'], verdict='Safe') for r in test])
    dump('delivery_protocol.json', dict(operating_point_sha256=common.sha(root / 'operating_point.json'),
                                       fresh_scores_sha256=common.sha(root / 'fresh_scores.jsonl')))
    safety = safety_report()
    safety.update(threshold=threshold, original_group_overlap=0, valid_timestamp_order=True,
                  split_integrity=validate_partition(rows),
                  primary=dict(n=3360, TP=2520, TN=840, FP=0, FN=0, accuracy=1.0))
    acceptance_check.safety_check(root, safety)
    delivered[0]['decision_done_ns'] = 9
    jsonl('delivery.jsonl', delivered)
    with pytest.raises(ValueError, match='out of order'):
        acceptance_check.safety_check(root, safety)


def test_factual_gate_reproduces_fitted_outputs_and_rejects_fake_signal_counts(workspace, monkeypatch):
    import numpy as np
    from evaluation.groups import image_group_id
    from evaluation.split import grouped_split
    from msrc.model import RiskCalibrator
    root = workspace / 'factual'
    for name in ('inputs', 'evidence', 'calibrators', 'signal_usage'):
        (root / name).mkdir(parents=True)
    def dump(path, data):
        path.write_text(json.dumps(data))
    def jsonl(path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(acceptance_check, 'FACTUAL_COUNTS', {'pope': 8, 'textvqa': 8})
    protocol = dict(mode='full', bins=15, seed=20260920, inputs={})
    result = factual_report()
    result['cells'] = []
    for dataset in ('pope', 'textvqa'):
        raw = [dict(id='%s:%d' % (dataset, i), dataset=dataset, source_image_id=str(i)) for i in range(8)]
        dev, test = grouped_split([image_group_id(r) for r in raw], seed=protocol['seed'])
        for r, d in zip(raw, dev):
            r['acceptance_split'] = 'dev' if d else 'test'
        input_path = root / 'inputs' / (dataset + '.jsonl')
        jsonl(input_path, raw)
        protocol['inputs'][dataset] = dict(sha256=common.sha(input_path))
        x = np.random.default_rng(17).uniform(.1, .9, (8, 3))
        labels = np.zeros(8, dtype=int)
        labels[np.flatnonzero(dev)[::2]] = 1
        labels[np.flatnonzero(test)[::2]] = 1
        fit = RiskCalibrator(l2=.05).fit(x[dev], labels[dev])
        predictions = fit.predict_proba(x[test])
        names = ['sequence_confidence', 'self_consistency', 'output_entropy']
        state = dict(signals=names, live=[True] * 3, fill=[.5] * 3, coef=fit.coef_.tolist(),
                     mean=fit.mean_.tolist(), std=fit.std_.tolist(), fitting_scope='development only',
                     converged=True, train_ids=[r['id'] for r, d in zip(raw, dev) if d])
        detail = [dict(id=raw[i]['id'], label=int(labels[i]), risk=float(p), n_actual_signals=3,
                       signals=[dict(name=name, risk=float(x[i, j]), available=True,
                                     participates=True, weight=float(fit.coef_[j + 1]))
                                for j, name in enumerate(names)])
                  for i, p in zip(np.flatnonzero(test), predictions)]
        def ece(p):
            y = labels[test]
            bins = np.minimum((p * 15).astype(int), 14)
            return sum(abs(float((y[bins == b] - p[bins == b]).sum())) for b in range(15)) / len(y)
        before, after = ece(x[test, 0]), ece(predictions)
        for model in ('llava', 'qwen', 'internvl'):
            cell = model + '_' + dataset
            evidence = root / 'evidence' / (cell + '.jsonl')
            jsonl(evidence, raw)
            dump(root / 'calibrators' / (cell + '.json'), state)
            jsonl(root / 'signal_usage' / (cell + '.jsonl'), detail)
            result['cells'].append(dict(cell=cell, n=8, n_dev=4, n_test=4, group_overlap=0,
                                       new_evidence_sha256=common.sha(evidence),
                                       ece_before=before, ece_after=after,
                                       relative_improvement=(before - after) / before,
                                       minimum_actual_signals_per_test_sample=3))
    dump(root / 'protocol.json', protocol)
    acceptance_check.factual_check(root, result)
    path = root / 'signal_usage/llava_pope.jsonl'
    tampered = common.read_rows(path)
    tampered[0]['signals'][0]['available'] = False
    jsonl(path, tampered)
    with pytest.raises(ValueError, match='participation'):
        acceptance_check.factual_check(root, result)
