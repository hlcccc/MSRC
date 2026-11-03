"""CPU contract tests; these do not claim a physical three-5090 acceptance run."""
import io
import json
from pathlib import Path
import tarfile
import threading
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest

from pipeline import common, fast3_contract as c, fast3
from pipeline.fast3_safety import choose_judge, canonical


@pytest.mark.parametrize('options', [('--run-id', '../escape'), ('--gpu', '0,1')])
def test_invalid_cli_arguments_exit_two_before_initialization(options):
    result = subprocess.run([sys.executable, '-B', '-m', 'pipeline.fast3', '--phase', 'prepare',
        '--run-id', 'unused-validation-only', *options], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'error:' in result.stderr


@pytest.mark.parametrize('dataset,n', [('pope', 4500), ('textvqa', 2496)])
def test_partition_covers_every_fixed_test_once_and_preserves_tail(dataset, n):
    rows = [dict(id=str(i)) for i in range(n)]
    shards = [c.assigned(rows, dataset, rank) for rank in range(3)]
    assert c.merge_ordered(rows, shards) == rows
    assert max(map(len, shards)) - min(map(len, shards)) <= c.BLOCK[dataset]
    for rank in range(3):
        for index, block in c.blocks(rows, dataset, rank):
            assert index % 3 == rank
            assert block == rows[index * c.BLOCK[dataset]:(index + 1) * c.BLOCK[dataset]]
            assert len(block) % c.BATCH['internvl'][dataset] == 0


@pytest.mark.parametrize('rank', [-1, 3, True, None])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError):
        c.rank_root(common.ROOT, rank)


def test_empty_or_overlapping_or_extra_results_fail():
    expected = [dict(id='a'), dict(id='b')]
    for shards in ([], [[expected[0]], [expected[0], expected[1]]], [[dict(id='c')], expected]):
        with pytest.raises(ValueError):
            c.merge_ordered(expected, shards)


def test_global_ece_is_not_shard_average():
    from pipeline.factual import ece
    y, p = np.array([0, 1]), np.array([.5, .5])
    assert ece(y, p) == 0
    assert np.mean([ece(y[i:i + 1], p[i:i + 1]) for i in range(2)]) == .5


def test_block_rng_is_rank_and_resume_independent():
    assert c.seed_for('pope', 4) == c.seed_for('pope', 4)
    assert len({c.seed_for(d, i, s) for d in ('pope', 'textvqa') for i in range(4)
        for s in ('qwen', 'internvl')}) == 16


def test_fixed_split_is_never_halved_again():
    rows = [dict(id='d', acceptance_split='dev', acceptance_image_group='x'),
            dict(id='t', acceptance_split='test', acceptance_image_group='y')]
    with patch.dict(c.COUNTS, {'unit': (1, 1)}):
        assert c.fixed_tests(rows, 'unit') == [rows[1]]
        rows[1]['acceptance_image_group'] = 'x'
        with pytest.raises(ValueError, match='leakage'):
            c.fixed_tests(rows, 'unit')


def test_full_input_metadata_must_match_evidence():
    raw = [dict(id='a', question='q', acceptance_split='test')]
    row = dict(raw[0], answer='yes', donor_only=True, model_calls=1)
    c.aligned(raw, [row], donor=True)
    for edit in (dict(question='other'), dict(answer=''), dict(model_calls=11), dict(donor_only=False)):
        with pytest.raises(ValueError):
            c.aligned(raw, [dict(row, **edit)], donor=True)
    with pytest.raises(ValueError):
        c.aligned(raw, [])


def profile(times=None):
    times = times or {6: 12, 16: 8, 32: 6, 64: 7}
    return dict(complete=True, test_used_for_selection=False, measurements=[dict(
        batch_size=size, repeat=repeat, status='OK', rng_unchanged=True, raw=['Safety: Safe\nRefusal: No'],
        comparison=dict(safety_labels_equal=True), seconds=seconds)
        for repeat in (0, 1) for size, seconds in times.items()])


def test_fastest_judge_is_32_not_conservative_6_or_16():
    assert choose_judge(profile())['batch_size'] == 32


def test_faster_64_is_admitted_only_when_its_two_round_quality_passes():
    p = profile({6: 12, 16: 8, 32: 6, 64: 4})
    assert choose_judge(p)['batch_size'] == 64
    p['measurements'][-1]['comparison']['safety_labels_equal'] = False
    assert choose_judge(p)['batch_size'] == 32


@pytest.mark.parametrize('field,value', [('status', 'OOM'), ('rng_unchanged', False)])
def test_judge_failure_falls_back_to_valid_baseline(field, value):
    p = profile()
    for row in p['measurements']:
        if row['batch_size'] != 6:
            row[field] = value
    assert choose_judge(p)['batch_size'] == 6


def test_invalid_or_nonrepeatable_baseline_stops():
    p = profile()
    p['measurements'][4]['raw'] = ['Safety: Unsafe']
    with pytest.raises(ValueError):
        choose_judge(p)
    with pytest.raises(ValueError):
        choose_judge(dict(profile(), test_used_for_selection=True))


def test_small_speed_tie_uses_smaller_batch():
    assert choose_judge(profile({6: 10, 16: 6.1, 32: 6.0}))['batch_size'] == 16


@pytest.mark.parametrize('name,kind', [('rank-1/../../outside', 'file'),
    ('/rank-1/a', 'file'), ('rank-2/a', 'file'), ('rank-1/link', 'symlink'),
    ('rank-1\\a', 'file')])
def test_unsafe_worker_tar_is_rejected(tmp_path, name, kind):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        info = tarfile.TarInfo(name)
        if kind == 'symlink':
            info.type, info.linkname = tarfile.SYMTYPE, 'outside'
            out.addfile(info)
        else:
            info.size = 1
            out.addfile(info, io.BytesIO(b'x'))
    with patch.object(common, 'ROOT', tmp_path):
        with pytest.raises(ValueError):
            fast3.unpack(archive, tmp_path, 1)
    assert not (tmp_path / 'rank-1').exists()


def test_duplicate_worker_tar_rejected(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        for _ in range(2):
            info = tarfile.TarInfo('rank-1/a')
            info.size = 1
            out.addfile(info, io.BytesIO(b'x'))
    with patch.object(common, 'ROOT', tmp_path):
        with pytest.raises(ValueError):
            fast3.unpack(archive, tmp_path, 1)


def test_backward_and_calibrator_fit_are_blocked_without_training():
    import torch
    from msrc.model import RiskCalibrator
    with patch.object(torch.Tensor, 'backward'), patch.object(torch.autograd, 'backward'), \
            patch.object(torch.optim.Optimizer, 'step'), patch.object(RiskCalibrator, 'fit'):
        previous = torch.is_grad_enabled()
        try:
            c.forbid_training()
            assert not torch.is_grad_enabled()
            with pytest.raises(RuntimeError, match='forbidden'):
                RiskCalibrator().fit(None, None)
            with pytest.raises(RuntimeError, match='forbidden'):
                torch.tensor(1.).backward()
        finally:
            torch.set_grad_enabled(previous)


def test_plan_tampering_and_path_changes_fail(tmp_path):
    with patch.object(common, 'ROOT', tmp_path), patch.object(c, 'sources', return_value={'a': 'b'}):
        plan = dict(format=c.FORMAT, world_size=3, project_root=str(tmp_path), sources={'a': 'b'},
            onsite_training=False, batches=c.BATCH, blocks=c.BLOCK, immutable_files={})
        plan['plan_sha256'] = c.digest(plan)
        assert c.verify_plan(tmp_path, plan) == plan
        for key, value in [('world_size', 2), ('onsite_training', True), ('project_root', 'elsewhere')]:
            with pytest.raises(ValueError):
                c.verify_plan(tmp_path, dict(plan, **{key: value}))


def test_only_volatile_event_timestamps_removed():
    arms = [dict(start_ns=1, warn=True, events=[dict(score_ns=2, release_ns=3, risk=.4)])]
    assert canonical(arms) == [dict(warn=True, events=[dict(release_ns=True, risk=.4)])]
    assert arms[0]['start_ns'] == 1


def test_original_frozen_core_is_byte_identical():
    from extensions.multilayer.runner import verify_core
    assert verify_core()


def test_worker_archive_roundtrip_and_tamper_detection(tmp_path):
    source = tmp_path / 'source/rank-1'
    source.mkdir(parents=True)
    (source / 'factual').mkdir()
    (source / 'factual/a.json').write_text('{"value": 1}')
    (source / 'safety_audit.json').write_text('{"complete": true}')
    plan = dict(format=c.FORMAT, rank=1, experiment_sha256='unit-test-contract')
    (source / 'plan.json').write_text(json.dumps(plan))
    target = tmp_path / 'incoming'
    target.mkdir()
    archive = tmp_path / 'rank-1.tar.gz'
    with patch.object(common, 'ROOT', tmp_path), patch.object(fast3, 'verify', side_effect=lambda p: fast3.load(p / 'plan.json')):
        fast3.export(source, archive, 1)
        fast3.unpack(archive, target, 1)
        assert fast3.verify_receipt(target / 'rank-1', 1) == plan
        (target / 'rank-1/factual/a.json').write_text('{"value": 2}')
        with pytest.raises(ValueError, match='changed'):
            fast3.verify_receipt(target / 'rank-1', 1)


def test_three_worker_dispatch_is_concurrent_and_waits_for_all_on_failure(tmp_path):
    root = tmp_path / 'fast3'
    root.mkdir()
    hosts = tmp_path / 'hosts.json'
    hosts.write_text(json.dumps(dict(workers=[dict(rank=r, host='worker%d' % r, port=22) for r in (1, 2)])))
    barrier = threading.Barrier(3, timeout=5)
    completed = []

    def run(command, **kwargs):
        barrier.wait()
        completed.append(command[0])
        if command[0] == 'ssh' and 'worker1' in command:
            raise RuntimeError('intentional worker1 failure')
        if command[0] == 'scp':
            raise AssertionError('not expected here')
        if command[0] == 'ssh':
            # Finish remaining own task without running a real SCP in this failure test.
            raise RuntimeError('intentional worker2 failure')

    with patch.object(common, 'ROOT', tmp_path), patch.object(c, 'lock', return_value=nullcontext()), \
            patch.object(fast3, 'prepare', return_value=dict(experiment_sha256='unit')), \
            patch.object(fast3.subprocess, 'run', side_effect=run), patch.object(fast3, 'merge') as merged:
        with pytest.raises(ValueError, match='workers failed'):
            fast3.cluster(root, 'unit', hosts)
        assert len(completed) == 3
        merged.assert_not_called()
        assert len(fast3.load(root / 'cluster_timing.json')['errors']) == 2


def test_stage_exposes_no_training_phase():
    import inspect
    text = inspect.getsource(fast3.main)
    assert "'fit'" not in text and "'development'" not in text and "'label'" not in text
    assert 'c.forbid_training()' in inspect.getsource(fast3.stage)
