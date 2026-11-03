"""CPU simulations of single-card scheduling; not GPU or Slurm acceptance evidence."""
from contextlib import nullcontext
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipeline import common, fast3, fast3_contract as c, fast_single as single

UUID = 'GPU-11111111-2222-3333-4444-555555555555'


def spec():
    return dict(index='5', uuid=UUID, name='NVIDIA RTX 5090', total_mib=32607,
        inherited_cuda_visible_devices='0', slurm_allocation=True)


def probe_outputs():
    return ['MSRC_DEVICE_JSON ' + json.dumps(dict(uuid=UUID, name='NVIDIA RTX 5090', total_bytes=32607 * 1024**2)),
        '5, ' + UUID + ', NVIDIA RTX 5090, 32607\n', '']


@pytest.mark.parametrize('raw_uuid', [UUID, UUID.removeprefix('GPU-')])
def test_probe_normalizes_actual_torch_27_uuid_string(raw_uuid, capsys):
    prop = SimpleNamespace(uuid=raw_uuid, name='NVIDIA RTX 5090', total_memory=32607 * 1024**2)
    fake = SimpleNamespace(__version__='2.7.1+cu128', cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_properties=lambda _: prop))
    with patch.dict(sys.modules, {'torch': fake}):
        exec(single.PROBE, {})
    payload = capsys.readouterr().out.strip().split('MSRC_DEVICE_JSON ', 1)[1]
    assert json.loads(payload)['uuid'] == UUID


def test_slurm_mask_is_probed_before_uuid_binding_and_nvml_uses_actual_index(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with patch.object(single.subprocess, 'check_output', side_effect=probe_outputs()) as query:
        assert single.device() == spec()
    calls = query.call_args_list
    assert calls[0].args[0][:3] == [sys.executable, '-B', '-c']
    assert calls[0].kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert calls[1].args[0][2] == UUID
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'


@pytest.mark.parametrize('mask', ['0', UUID, '5'])
def test_inherited_mask_is_never_replaced_by_explicit_gpu(monkeypatch, mask):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    with patch.object(single.subprocess, 'check_output') as query:
        with pytest.raises(ValueError, match='omit --gpu'):
            single.device('0')
        query.assert_not_called()


@pytest.mark.parametrize('mask', [None, ''])
def test_slurm_without_an_allocation_fails_before_any_cuda_probe(monkeypatch, mask):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    if mask is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    with patch.object(single.subprocess, 'check_output') as query:
        with pytest.raises(ValueError, match='Slurm must supply'):
            single.device()
        query.assert_not_called()


def test_explicit_physical_index_is_resolved_to_uuid_before_cuda_probe(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with patch.object(single.subprocess, 'check_output', side_effect=[UUID, *probe_outputs()]) as query:
        actual = single.device('5')
    assert actual['index'] == '5' and actual['uuid'] == UUID
    assert query.call_args_list[0].args[0][2] == '5'
    assert query.call_args_list[1].kwargs['env']['CUDA_VISIBLE_DEVICES'] == UUID
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ


@pytest.mark.parametrize('failure', ['occupied', 'mismatch', 'memory', 'missing_probe'])
def test_device_preflight_rejects_occupied_or_unmatched_devices(monkeypatch, failure):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    values = probe_outputs()
    if failure == 'occupied':
        values[2] = UUID + ', 999\n'
    elif failure == 'mismatch':
        values[1] = values[1].replace(UUID, 'GPU-00000000')
    elif failure == 'memory':
        values[1] = values[1].replace('32607', '16000')
    else:
        values[0] = '{}'
    with patch.object(single.subprocess, 'check_output', side_effect=values):
        with pytest.raises(ValueError):
            single.device()


def test_probe_rejects_multiple_cuda_devices_or_wrong_torch_before_worker(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    assert 'torch.cuda.device_count() == 1' in single.PROBE
    assert "torch.__version__ == '2.7.1+cu128'" in single.PROBE
    with patch.object(single.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, 'probe')):
        with pytest.raises(subprocess.CalledProcessError):
            single.device()


def dispatch(tmp_path, monkeypatch, run, merge=None):
    root = tmp_path / 'runs/unit/fast3'
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'lock', lambda *args: nullcontext())
    monkeypatch.setattr(c, 'sources', lambda: {'synthetic': 'CPU test'})
    monkeypatch.setattr(single, 'device', lambda _: spec())
    monkeypatch.setattr(fast3, 'prepare', lambda *args: dict(experiment_sha256='unit'))
    monkeypatch.setattr(single.subprocess, 'run', run)
    if merge is not None:
        monkeypatch.setattr(fast3, 'merge', merge)
    return root


def report(root, value):
    (root / 'joint').mkdir()
    c.atomic(root / 'joint/outputs.json', dict(artifacts={}))


def test_three_logical_shards_are_serial_on_same_uuid_without_transport(tmp_path, monkeypatch):
    completed = []

    def run(command, **kwargs):
        rank = int(command[command.index('--rank') + 1])
        assert rank == len(completed)
        assert command[0] == sys.executable and '--no-archive' in command
        assert command[command.index('--gpu') + 1] == '5'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == kwargs['env']['MSRC_CUDA_DEVICE_UUID'] == UUID
        assert kwargs['env']['MSRC_GPU'] == '5'
        assert not set(command) & {'ssh', 'scp', '--hosts'}
        completed.append(rank)

    result = dict(gpu_uuids=[UUID] * 3, scoped_acceptance_pass=True,
        physical_gpu_count=1, physical_three_gpu_run=False)

    def merge(root, write, physical_gpu_count):
        assert completed == [0, 1, 2] and not write and physical_gpu_count == 1
        fast3.verify_merge_devices(root, result['gpu_uuids'], physical_gpu_count)
        return result

    root = dispatch(tmp_path, monkeypatch, run, merge)
    monkeypatch.setattr(fast3, 'write_report', report)
    assert single.single(root, 'unit') is result
    timing = fast3.load(root / 'single_timing.json')
    assert timing['complete'] and timing['workers_serial'] and timing['logical_shards'] == 3
    assert not timing['uses_ssh'] and not timing['uses_result_archives']
    assert result['topology'] == 'one_host_one_gpu' and not result['physical_three_gpu_run']
    assert fast3.load(root / 'joint/outputs.json')['artifacts']['single_timing.json'] == common.sha(root / 'single_timing.json')
    assert not list(root.glob('*.tar.gz'))


def test_failure_stops_later_shards_without_merge_and_resume_keeps_origin(tmp_path, monkeypatch):
    called = []

    def run(command, **kwargs):
        rank = int(command[command.index('--rank') + 1])
        called.append(rank)
        if rank == 1:
            raise RuntimeError('intentional failure')

    root = dispatch(tmp_path, monkeypatch, run, lambda *args, **kwargs: pytest.fail('must not merge'))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='intentional'):
            single.single(root, 'unit')
        timing = fast3.load(root / 'single_timing.json')
        assert not timing['complete'] and timing['worker_executions']['0']['complete']
        assert not timing['worker_executions']['1']['complete'] and '2' not in timing['worker_executions']
        if len(called) == 2:
            first = timing['started_ns']
    assert called == [0, 1, 0, 1] and timing['started_ns'] == first and timing['includes_resume_downtime']


def test_preflight_failure_keeps_full_clock_and_never_starts_worker(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: pytest.fail('must not dispatch'))
    for _ in range(2):
        with patch.object(single, 'device', side_effect=ValueError('busy')):
            with pytest.raises(ValueError, match='busy'):
                single.single(root, 'unit')
        timing = fast3.load(root / 'single_timing.json')
        assert not timing['complete'] and timing['worker_executions'] == {}
        assert timing['started_ns'] == fast3.load(root / 'cluster_origin.json')['started_ns']


def test_other_topology_existing_results_and_remapped_gpu_cannot_resume(tmp_path, monkeypatch):
    root = dispatch(tmp_path, monkeypatch, lambda *args, **kwargs: pytest.fail('must not dispatch'))
    c.atomic(root / 'cluster_origin.json', dict(format=c.FORMAT, run_id='unit', started_ns=1, topology='one_host_three_gpus'))
    with pytest.raises(ValueError, match='other topology'):
        single.single(root, 'unit')
    c.atomic(root / 'cluster_origin.json', dict(format=c.FORMAT, run_id='unit', started_ns=1, topology=single.TOPOLOGY))
    c.atomic(root / 'single_dispatch.json', dict(device=spec(), different='identity'))
    with pytest.raises(ValueError, match='mapping'):
        single.single(root, 'unit')
    (root / 'merged').mkdir()
    with pytest.raises(ValueError, match='merged result exists'):
        single.single(root, 'unit')


@pytest.mark.parametrize('passed,expected', [(True, 0), (False, 3), (None, 4)])
def test_entry_exit_codes_and_incomplete_receipt(tmp_path, monkeypatch, passed, expected):
    root = tmp_path / 'fast3'
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'root_for', lambda _: root)
    monkeypatch.setattr(common, 'initialize', lambda _: None)
    monkeypatch.setattr(sys, 'argv', ['single', '--run-id', 'unit'])
    if passed is None:
        def execute(*args):
            raise ValueError('synthetic worker failure')
    else:
        def execute(*args):
            return dict(scoped_acceptance_pass=passed)
    monkeypatch.setattr(single, 'single', execute)
    assert single.main() == expected
    if passed is None:
        assert fast3.load(root / 'incomplete-single.json')['verdict'] == 'INCOMPLETE'


@pytest.mark.parametrize('args', [['--gpu', '0,1'], ['--gpu', '-1'], ['--run-id', '../escape']])
def test_invalid_cli_exits_before_initialization(args):
    result = subprocess.run([sys.executable, '-B', '-m', 'pipeline.fast_single', '--run-id', 'unused', *args],
        capture_output=True, text=True)
    assert result.returncode == 2 and 'error:' in result.stderr


def test_launch_help_and_slurm_script_syntax_without_gpu_work():
    shell = shutil.which('bash')
    if shell is None:
        pytest.skip('Bash unavailable')
    env = dict(os.environ, MSRC_PYTHON=sys.executable)
    result = subprocess.run([shell, 'scripts/run_fast_single.sh', '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and '--gpu' in result.stdout
    checked = subprocess.run([shell, '-n', 'scripts/run_fast_single_slurm.sh'], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    text = (common.ROOT / 'scripts/run_fast_single_slurm.sh').read_text()
    assert '#SBATCH --gres=gpu:1' in text and 'CUDA_VISIBLE_DEVICES=' not in text.replace(
        "printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\\n'", '')
    assert 'SLURM_SUBMIT_DIR' in text and 'scripts/01_setup_envs.sh --execute' in text


def test_slurm_wrapper_runs_setup_and_assets_inside_job_preserving_mask(tmp_path):
    shell = shutil.which('bash')
    if shell is None:
        pytest.skip('Bash unavailable')
    script = common.ROOT / 'scripts/run_fast_single_slurm.sh'
    project = tmp_path / 'MSRC'
    for folder in ('pipeline', 'scripts', '.venv/bin', 'cache/fitted'):
        (project / folder).mkdir(parents=True, exist_ok=True)
    for file in ('pipeline/fast_single.py', 'requirements-acceptance.txt', 'cache/fitted/parameters.tar.gz'):
        (project / file).write_text('fixture')
    stub = '#!/usr/bin/env bash\nset -eu\nprintf "%s|%s|%s\\n" "$0" "$*" "$CUDA_VISIBLE_DEVICES" >> "$MSRC_TEST_CALLS"\n'
    for file in ('scripts/01_setup_envs.sh', 'scripts/run_fast_single.sh', '.venv/bin/python'):
        target = project / file
        target.write_text(stub, newline='\n')
        target.chmod(0o755)
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=project.as_posix(),
        CUDA_VISIBLE_DEVICES=UUID, MSRC_TEST_CALLS=calls.as_posix())
    env.pop('MSRC_PROJECT_ROOT', None)
    result = subprocess.run([shell, str(script), '--run-id', 'unit'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4 and all(line.endswith('|' + UUID) for line in lines)
    assert '01_setup_envs.sh|--execute --offline-wheels ../env/wheels' in lines[0]
    assert '-m pipeline.provided_assets' in lines[1]
    assert '-m pipeline.fast3_assets' in lines[2]
    assert 'run_fast_single.sh|--run-id unit|' in lines[3]


def test_slurm_wrapper_rejects_direct_login_shell_before_setup():
    shell = shutil.which('bash')
    if shell is None:
        pytest.skip('Bash unavailable')
    env = dict(os.environ)
    env.pop('SLURM_JOB_ID', None)
    result = subprocess.run([shell, 'scripts/run_fast_single_slurm.sh', '--run-id', 'unused'],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'submit this script with sbatch' in result.stderr


def test_single_word_and_markdown_commands_and_report_paths_agree():
    import importlib.util
    script = common.ROOT / 'scripts/build_fast3_docs.py'
    spec = importlib.util.spec_from_file_location('single_doc_validation', script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    qa = builder.validate(common.ROOT / 'docs/FAST_SINGLE_FLOW.md',
        common.ROOT / 'docs/FAST_SINGLE_FLOW.docx', 'single_fitted_001', 'one_host_one_gpu')
    assert qa['steps'] == 5 and qa['command_blocks'] == 4
    assert qa['topology'] == 'one_host_one_gpu'
