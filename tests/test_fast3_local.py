"""Local dispatch CPU tests; mocked GPUs are not physical acceptance evidence."""
from contextlib import nullcontext
import os
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from pipeline import common, fast3, fast3_contract as c, fast3_local as local


def specs():
    return [dict(rank=r, index=str(r), uuid='GPU-' + str(r) * 8,
        name='NVIDIA RTX 5090', total_mib=32607) for r in range(3)]


@pytest.mark.parametrize('value', ['0,0,1', '0,1', '0,1,2,3', '0,1,-2', '0, 1,2', '00,1,2', '0,1,a'])
def test_requires_exactly_three_distinct_canonical_indices(value):
    with pytest.raises(ValueError, match='three distinct'):
        local.parse_gpus(value)


def test_custom_physical_indices_preserve_rank_order():
    assert local.parse_gpus('4,2,5') == ['4', '2', '5']


def test_gpu_inventory_and_foreign_occupants_are_checked_without_signals():
    inventory = '\n'.join('%s, %s, %s, %s' % (d['index'], d['uuid'], d['name'], d['total_mib']) for d in specs())
    with patch.object(local.subprocess, 'check_output', side_effect=[inventory, '']):
        assert local.devices(['0', '1', '2']) == specs()
    with patch.object(local.subprocess, 'check_output', side_effect=[inventory, 'GPU-11111111, 999\n']):
        with pytest.raises(ValueError, match='occupied'):
            local.devices(['0', '1', '2'])
    with patch.object(local.subprocess, 'check_output', return_value=inventory):
        with pytest.raises(ValueError, match='not found'):
            local.devices(['0', '1', '4'])
    with patch.object(local.subprocess, 'check_output', return_value=inventory.replace('5090', 'A100')):
        with pytest.raises(ValueError, match='5090'):
            local.devices(['0', '1', '2'])


def test_uuid_binding_survives_repeated_common_select_device(monkeypatch):
    monkeypatch.setenv('MSRC_GPU', '1')
    monkeypatch.setenv('MSRC_CUDA_DEVICE_UUID', 'GPU-11111111')
    original = common.select_device
    monkeypatch.setattr(common, 'select_device', original)
    fast3.configure_device()
    for _ in range(2):
        assert common.select_device({'gpu': '0'}) == 'cuda:0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-11111111'
    monkeypatch.delenv('MSRC_CUDA_DEVICE_UUID')
    monkeypatch.setattr(common, 'select_device', original)
    common.select_device({'gpu': '0'})
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'


def test_supplied_parameter_source_contract_is_unchanged_by_local_gpu_binding(monkeypatch):
    from pipeline import factual_parameters as parameters
    collector = parameters.fitted_collector()
    sources = parameters.source_fingerprints(collector)
    assert common.sha(common.ROOT / 'pipeline/common.py') == 'c744ac684f0abe9311f77d98602239b700691a2a9a23a0459311e6045c024b72'
    package = dict(format=parameters.FORMAT, sources=sources,
        models_recipe_sha256=parameters.text_sha256(common.ROOT / 'resources/reproduction/models.json'))
    monkeypatch.setenv('MSRC_GPU', '2')
    monkeypatch.setenv('MSRC_CUDA_DEVICE_UUID', 'GPU-22222222')
    monkeypatch.setattr(common, 'select_device', common.select_device)
    fast3.configure_device()
    parameters.validate_code_contract(package, collector)
    assert parameters.source_fingerprints(collector) == sources
    common.select_device({'gpu': '0'})
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-22222222'


def dispatch(tmp_path, monkeypatch, run, merge=None):
    root = tmp_path / 'runs/unit/fast3'
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'lock', lambda *args: nullcontext())
    monkeypatch.setattr(c, 'sources', lambda: {'synthetic': 'CPU test'})
    monkeypatch.setattr(local, 'devices', lambda _: specs())
    monkeypatch.setattr(fast3, 'prepare', lambda *args: dict(experiment_sha256='unit'))
    monkeypatch.setattr(local.subprocess, 'run', run)
    if merge is not None:
        monkeypatch.setattr(fast3, 'merge', merge)
    return root


def test_three_local_workers_start_concurrently_use_uuid_and_do_not_transport(tmp_path, monkeypatch):
    barrier, completed = threading.Barrier(3, timeout=5), []

    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert '--no-archive' in command and '--expected-experiment' in command
        assert not any(x in command for x in ('ssh', 'scp', '--hosts'))
        rank = int(command[command.index('--rank') + 1])
        assert command[command.index('--gpu') + 1] == str(rank)
        assert kwargs['env']['MSRC_GPU'] == str(rank)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == specs()[rank]['uuid']
        assert kwargs['env']['MSRC_CUDA_DEVICE_UUID'] == specs()[rank]['uuid']
        barrier.wait()
        completed.append(rank)

    result = dict(gpu_uuids=[s['uuid'] for s in specs()], scoped_acceptance_pass=True)

    def merge(root, write):
        assert sorted(completed) == [0, 1, 2] and write is False
        return result

    def report(root, value):
        joint = root / 'joint'
        joint.mkdir()
        c.atomic(joint / 'outputs.json', dict(artifacts={}))

    root = dispatch(tmp_path, monkeypatch, run, merge)
    monkeypatch.setattr(fast3, 'write_report', report)
    assert local.local(root, 'unit', ['0', '1', '2']) is result
    timing = fast3.load(root / 'local3_timing.json')
    assert timing['complete'] and not timing['uses_ssh'] and not timing['uses_result_archives']
    assert len(timing['worker_executions']) == 3
    assert not list(root.glob('*.tar.gz')) and not (root / 'incoming').exists()
    assert result['topology'] == 'one_host_three_gpus'
    assert fast3.load(root / 'joint/outputs.json')['artifacts']['local3_timing.json'] == common.sha(root / 'local3_timing.json')


def test_failure_joins_all_own_workers_and_does_not_merge(tmp_path, monkeypatch):
    barrier, completed = threading.Barrier(3, timeout=5), []

    def run(command, **kwargs):
        rank = int(command[command.index('--rank') + 1])
        barrier.wait()
        completed.append(rank)
        if rank == 1:
            raise RuntimeError('intentional worker failure')

    def merge(*args, **kwargs):
        pytest.fail('incomplete results must not be merged')

    root = dispatch(tmp_path, monkeypatch, run, merge)
    with pytest.raises(ValueError, match='local workers failed'):
        local.local(root, 'unit', ['0', '1', '2'])
    assert sorted(completed) == [0, 1, 2]
    timing = fast3.load(root / 'local3_timing.json')
    assert not timing['complete'] and len(timing['errors']) == 1
    assert len(timing['worker_executions']) == 3
    first_started = timing['started_ns']
    # A retry keeps the original full-pipeline start, including failed attempts.
    with pytest.raises(ValueError, match='local workers failed'):
        local.local(root, 'unit', ['0', '1', '2'])
    timing = fast3.load(root / 'local3_timing.json')
    assert timing['started_ns'] == first_started and timing['includes_resume_downtime']


def test_preflight_failure_can_resume_without_resetting_clock(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: None)
    with patch.object(local, 'devices', side_effect=ValueError('busy')):
        with pytest.raises(ValueError, match='busy'):
            local.local(root, 'unit', ['0', '1', '2'])
    first = fast3.load(root / 'local3_timing.json')['started_ns']
    with patch.object(local, 'devices', side_effect=ValueError('busy again')):
        with pytest.raises(ValueError, match='busy again'):
            local.local(root, 'unit', ['0', '1', '2'])
    assert fast3.load(root / 'local3_timing.json')['started_ns'] == first


def test_resume_rejects_gpu_remap_and_three_machine_origin(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: None)
    origin = root / 'cluster_origin.json'
    c.atomic(origin, dict(format=c.FORMAT, run_id='unit', started_ns=1))
    with pytest.raises(ValueError, match='non-local'):
        local.local(root, 'unit', ['0', '1', '2'])
    c.atomic(origin, dict(format=c.FORMAT, run_id='unit', started_ns=1, topology='one_host_three_gpus'))
    with pytest.raises(ValueError, match='topology'):
        fast3.cluster_start(root, 'unit')
    c.atomic(root / 'local_dispatch.json', dict(topology='other'))
    with pytest.raises(ValueError, match='mapping'):
        local.local(root, 'unit', ['0', '1', '2'])


def test_workers_finalize_and_resume_without_tar(tmp_path, monkeypatch):
    root = tmp_path / 'rank-0'
    root.mkdir()
    calls = []
    monkeypatch.setattr(fast3, 'prepare', lambda *args: dict(experiment_sha256='unit'))
    monkeypatch.setattr(fast3.subprocess, 'run', lambda *args, **kwargs: calls.append(args[0]))
    monkeypatch.setattr(fast3, 'verify_models', lambda: None)
    with patch.object(fast3, 'finalize_receipt') as finalized, patch.object(fast3, 'export') as exported:
        fast3.worker(root, 'unit', 0, expected='unit', archive=False)
        assert len(calls) == 4
        finalized.assert_called_once_with(root, 0)
        exported.assert_not_called()
    (root / 'receipt.json').write_text('{}')
    with patch.object(fast3, 'verify_receipt') as verified, patch.object(fast3, 'export') as exported:
        fast3.worker(root, 'unit', 0, archive=False)
        verified.assert_called_once_with(root, 0)
        exported.assert_not_called()
    assert not list(tmp_path.glob('*.tar.gz'))


@pytest.mark.parametrize('args', [['--gpus', '0,0,1'], ['--hosts', 'unused.json'], ['--no-archive']])
def test_local_cli_rejects_invalid_options_before_initialization(args):
    result = subprocess.run([sys.executable, '-B', '-m', 'pipeline.fast3', '--phase', 'local',
        '--run-id', 'unused-local-validation-only', *args], capture_output=True, text=True)
    assert result.returncode == 2 and 'error:' in result.stderr


def test_mismatched_actual_gpu_evidence_does_not_publish_pass(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: None,
        lambda *args, **kwargs: dict(gpu_uuids=['wrong'] * 3))
    with patch.object(fast3, 'write_report') as report:
        with pytest.raises(ValueError, match='physical GPUs'):
            local.local(root, 'unit', ['0', '1', '2'])
        report.assert_not_called()


def test_preparation_failure_stops_before_any_gpu_worker(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: pytest.fail('must not dispatch'))
    with patch.object(fast3, 'prepare', side_effect=ValueError('bad assets')):
        with pytest.raises(ValueError, match='bad assets'):
            local.local(root, 'unit', ['0', '1', '2'])
    timing = fast3.load(root / 'local3_timing.json')
    assert not timing['complete'] and timing['worker_executions'] == {}


@pytest.mark.parametrize('passed,expected', [(True, 0), (False, 3), (None, 4)])
def test_local_entry_reports_pass_fail_or_incomplete_exit_status(tmp_path, monkeypatch, passed, expected):
    root = tmp_path / 'fast3'
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'root_for', lambda _: root)
    monkeypatch.setattr(common, 'initialize', lambda _: None)
    monkeypatch.setattr(common, 'select_device', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['fast3', '--phase', 'local', '--run-id', 'unit', '--gpus', '0,1,2'])
    if passed is None:
        def run(*args):
            raise ValueError('synthetic failed worker')
    else:
        def run(*args):
            return dict(scoped_acceptance_pass=passed)
    monkeypatch.setattr(local, 'local', run)
    assert fast3.main() == expected
    if passed is None:
        assert fast3.load(root / 'incomplete-rank-0.json')['verdict'] == 'INCOMPLETE'


def test_launch_script_help_enters_local_cli_without_gpu_work():
    import shutil
    shell = shutil.which('bash')
    if shell is None:
        pytest.skip('Bash is unavailable')
    env = dict(os.environ, MSRC_PYTHON=sys.executable)
    result = subprocess.run([shell, 'scripts/run_fast3_local.sh', '--help'], env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--gpus' in result.stdout and '--no-archive' in result.stdout
