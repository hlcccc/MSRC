"""Explicit sharing never disables owner, UUID, memory or frozen-policy checks."""

import json
import os
from types import SimpleNamespace

import pytest

from pipeline import gpu_sharing, preflight


@pytest.fixture
def policy(tmp_path, monkeypatch):
    path = tmp_path / 'shared.json'
    settings = dict(version=1, gpu_index=1, gpu_uuid='GPU-authorized-1',
                    allowed_other_users=['HLC'], allocator_cap_gib=24,
                    min_start_free_mib=32768, min_run_free_mib=8192,
                    target_duty_when_shared=0.5, check_interval_seconds=10,
                    authorization='Explicit synthetic user authorization')
    path.write_text(json.dumps(settings))
    monkeypatch.setattr(gpu_sharing, 'inside', lambda name: path)
    monkeypatch.setenv('MSRC_GPU_SHARE_POLICY', str(path))
    monkeypatch.setenv('MSRC_SAFETY_BACKEND', 'plugguard')
    monkeypatch.setenv('MSRC_GPU', '1')
    return path, settings


def mock_gpu(monkeypatch, owner='HLC', free=54000, index=1, uuid='GPU-authorized-1', own_mib=0):
    monkeypatch.setenv('MSRC_GPU', str(index))
    monkeypatch.setattr(preflight, 'load_config', lambda: {})
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: '/usr/bin/' + name)
    def output(command, **kwargs):
        if '--query-gpu=index,uuid,memory.total,memory.used' in command:
            return '%s, %s, 81920, 27000\n' % (index, uuid)
        if '--query-gpu=memory.free' in command:
            return str(free) + '\n'
        apps = '%s, 9999999, 27000\n' % uuid
        if own_mib:
            apps += '%s, %s, %s\n' % (uuid, os.getpid(), own_mib)
        return apps
    monkeypatch.setattr(preflight.subprocess, 'check_output', output)
    monkeypatch.setattr(preflight.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=owner))


def test_no_explicit_permission_keeps_the_busy_gpu_guard(monkeypatch):
    monkeypatch.delenv('MSRC_GPU_SHARE_POLICY', raising=False)
    mock_gpu(monkeypatch)
    with pytest.raises(preflight.GPUInUse, match='already in use'):
        preflight.check_gpu()


def test_explicit_permission_allows_only_the_named_gpu_and_owner(policy, monkeypatch):
    mock_gpu(monkeypatch)
    result = preflight.check_gpu()[0]
    assert result['index'] == 1 and result['shared_gpu_execution'] is True
    assert result['free_mib'] == 54000
    assert result['other_processes'][0]['owner'] == 'HLC'
    mock_gpu(monkeypatch, owner='someone_else')
    with pytest.raises(preflight.GPUInUse, match='unauthorized process owner'):
        preflight.check_gpu()


@pytest.mark.parametrize('kwargs', [{'index': 0}, {'uuid': 'GPU-another-card'}])
def test_sharing_permission_cannot_silently_switch_devices(policy, monkeypatch, kwargs):
    mock_gpu(monkeypatch, **kwargs)
    with pytest.raises(ValueError, match='authorized shared GPU'):
        preflight.check_gpu()


@pytest.mark.parametrize('free,own_mib', [(32000, 0), (16000, 16000), (8000, 24576)])
def test_memory_headroom_is_checked_before_and_during_the_run(policy, monkeypatch, free, own_mib):
    mock_gpu(monkeypatch, free=free, own_mib=own_mib)
    with pytest.raises(preflight.GPUInUse, match='reserve'):
        preflight.check_gpu(ignore_current_process=True)


@pytest.mark.parametrize('mutation', [
    {'allocator_cap_gib': 80}, {'target_duty_when_shared': 1.0},
    {'min_start_free_mib': 1024}, {'min_run_free_mib': 0},
    {'allowed_other_users': []}, {'authorization': ''}, {'ignored_option': True},
])
def test_unsafe_or_ambiguous_policy_is_rejected(policy, mutation):
    path, settings = policy
    path.write_text(json.dumps(dict(settings, **mutation)))
    with pytest.raises(ValueError):
        gpu_sharing.load_shared_policy()


def test_sharing_is_not_a_generic_uncapped_factual_override(policy, monkeypatch):
    monkeypatch.setenv('MSRC_SAFETY_BACKEND', 'singguard')
    with pytest.raises(ValueError, match='bounded PlugGuard'):
        gpu_sharing.load_shared_policy()


def test_shared_work_yields_and_rejects_permission_changes(policy, monkeypatch):
    from pipeline import plugguard
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(plugguard.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(plugguard.time, 'sleep', sleeps.append)
    monkeypatch.setattr(plugguard, 'save', lambda *args: None)
    monkeypatch.setattr(plugguard, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(plugguard, 'check_gpu', lambda **kw: [{'other_processes': [{'owner': 'HLC'}]}])
    budget = plugguard.WorkBudget()
    with budget.work('synthetic'):
        clock[0] += 2.0
    assert sleeps == [2.0]
    assert budget.snapshot()['voluntary_idle_seconds'] == 2.0
    path, settings = policy
    path.write_text(json.dumps(dict(settings, authorization='Changed after start')))
    clock[0] = 15.0
    with pytest.raises(ValueError, match='permission changed'):
        with budget.work('synthetic'):
            pass


def test_exclusive_use_resumes_when_no_other_job_is_present(policy, monkeypatch):
    from pipeline import plugguard
    clock, sleeps = [0.0], []
    monkeypatch.setattr(plugguard.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(plugguard.time, 'sleep', sleeps.append)
    monkeypatch.setattr(plugguard, 'save', lambda *args: None)
    monkeypatch.setattr(plugguard, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(plugguard, 'check_gpu', lambda **kw: [{'other_processes': []}])
    budget = plugguard.WorkBudget()
    with budget.work('synthetic'):
        clock[0] += 2.0
    assert budget.effective_duty == 1.0 and sleeps == []
