"""Explicit, run-local permission for bounded, cooperative GPU sharing."""

import json
import math
import os
import time

from pipeline.common import inside, save, sha


def load_shared_policy(*, workload=None):
    name = os.environ.get('MSRC_GPU_SHARE_POLICY')
    if not name:
        return None
    path = inside(name)
    if hasattr(os, 'getuid') and path.stat().st_uid != os.getuid():
        raise ValueError('GPU sharing policy must belong to the current user')
    policy = json.loads(path.read_text(encoding='utf-8'))
    expected = {'version', 'gpu_index', 'gpu_uuid', 'allowed_other_users',
                'allocator_cap_gib', 'min_start_free_mib', 'min_run_free_mib',
                'target_duty_when_shared', 'check_interval_seconds', 'authorization'}
    version = policy.get('version')
    if version == 2:
        expected.add('workload')
        if (policy.get('workload') != 'factual' or workload not in (None, 'factual')
                or os.environ.get('MSRC_SAFETY_BACKEND') == 'plugguard'):
            raise ValueError('Version 2 sharing is only for bounded factual collection')
        cap, minimum_start, minimum_run = 36, 49152, 12288
    elif version == 1:
        if os.environ.get('MSRC_SAFETY_BACKEND') != 'plugguard' or workload not in (None, 'plugguard'):
            raise ValueError('Version 1 sharing is only supported by the bounded PlugGuard backend')
        cap, minimum_start, minimum_run = 24, 32768, 8192
    else:
        raise ValueError('Unknown or incomplete GPU sharing policy')
    if set(policy) != expected or type(version) is not int:
        raise ValueError('Unknown or incomplete GPU sharing policy')
    if (type(policy['gpu_index']) is not int or policy['gpu_index'] < 0
            or not isinstance(policy['gpu_uuid'], str)
            or not policy['gpu_uuid'].startswith('GPU-')):
        raise ValueError('GPU sharing policy needs one exact index and UUID')
    users = policy['allowed_other_users']
    if (not isinstance(users, list) or not users
            or not all(isinstance(user, str) and user.strip() == user and user for user in users)
            or len(set(users)) != len(users)):
        raise ValueError('GPU sharing requires explicit other-process owners')
    if (type(policy['allocator_cap_gib']) is not int or policy['allocator_cap_gib'] != cap
            or not isinstance(policy['min_start_free_mib'], int)
            or not isinstance(policy['min_run_free_mib'], int)
            or policy['min_start_free_mib'] < minimum_start
            or policy['min_run_free_mib'] < minimum_run
            or policy['min_start_free_mib'] <= policy['min_run_free_mib']
            or policy['min_start_free_mib'] < cap * 1024 + policy['min_run_free_mib']
            or not 0 < policy['target_duty_when_shared'] <= 0.5
            or not 0 < policy['check_interval_seconds'] <= 10):
        raise ValueError('GPU sharing memory reserve or cooperative duty is unsafe')
    if not isinstance(policy['authorization'], str) or not policy['authorization'].strip():
        raise ValueError('GPU sharing needs a recorded user authorization')
    return dict(policy, source_path=str(path), source_sha256=sha(path))


class FactualWorkBudget:
    """Yield between complete samples without changing model calls or RNG state."""

    def __init__(self, output, memory_fraction):
        from pipeline.preflight import check_gpu
        import torch
        self.policy = load_shared_policy(workload='factual')
        if self.policy is None:
            raise ValueError('Factual sharing requires explicit permission')
        self.output = inside(output.with_name(output.name + '.resources.json'))
        self.torch = torch
        device = check_gpu(ignore_current_process=True)[0]
        expected = self.policy['allocator_cap_gib'] * 1024 / device['total_mib']
        if not math.isclose(memory_fraction, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Collector memory cap differs from the authorized factual policy')
        self.last_check = float('-inf')
        self.completed = 0
        self.work_seconds = self.idle_seconds = 0.0
        self.effective_duty = 1.0
        self.device = device
        self._save('model_loading')

    def before_record(self):
        from pipeline.preflight import check_gpu
        if time.monotonic() - self.last_check >= self.policy['check_interval_seconds']:
            current = load_shared_policy(workload='factual')
            if current['source_sha256'] != self.policy['source_sha256']:
                raise ValueError('GPU sharing permission changed during factual collection')
            self.device = check_gpu(ignore_current_process=True)[0]
            self.effective_duty = (self.policy['target_duty_when_shared']
                                   if self.device.get('other_processes') else 1.0)
            self.last_check = time.monotonic()
        self.torch.cuda.synchronize()
        self.started = time.monotonic()

    def after_record(self):
        self.torch.cuda.synchronize()
        elapsed = time.monotonic() - self.started
        self.completed += 1
        self.work_seconds += elapsed
        idle = elapsed * (1 / self.effective_duty - 1)
        if idle:
            time.sleep(idle)
            self.idle_seconds += idle
        self._save('collecting')

    def _save(self, stage):
        peak = self.torch.cuda.max_memory_reserved(0)
        if peak > self.policy['allocator_cap_gib'] * 1024**3:
            raise ValueError('Factual CUDA allocator exceeded its authorized memory cap')
        save(self.output, dict(
            stage=stage, complete=stage == 'complete', completed_samples=self.completed,
            shared_gpu_execution=True, shared_policy_sha256=self.policy['source_sha256'],
            allocator_cap_gib=self.policy['allocator_cap_gib'], peak_reserved_bytes=peak,
            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(0),
            target_duty_when_shared=self.policy['target_duty_when_shared'],
            effective_duty=self.effective_duty, measured_work_seconds=self.work_seconds,
            voluntary_idle_seconds=self.idle_seconds, device=self.device,
            hardware_compute_partition=False, zero_interference_guaranteed=False))

    def finish(self):
        self._save('complete')
