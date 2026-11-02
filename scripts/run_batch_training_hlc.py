"""Durable isolated HLC run, with reversible pauses of the user's named GPU0 job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temp.replace(path)


def process_identity(pid):
    path = Path('/proc') / str(pid)
    if not path.exists() or path.stat().st_uid != os.getuid():
        return None
    stat = (path / 'stat').read_text()
    tail = stat[stat.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=tail[19], state=tail[0],
        command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())


def pause_named_job(audit):
    target = '/mnt/data/HLC/tvv_multimodel_benchmarks_20261009/matrix_gpu0_20261028_v1/resume_gpu0.py'
    own = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                value = process_identity(int(path.name))
            except (ProcessLookupError, FileNotFoundError):
                continue
            if value and target in value['command'] and value['state'] != 'T':
                own.append(value)
    # Stop the dispatcher first so it cannot replace a worker while we pause it.
    own.sort(key=lambda value: (' worker ' in value['command'], value['pid']))
    receipt = dict(authorization='user explicitly permits pausing their other tasks for this highest-priority run',
        other_users_untouched=True, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        processes=[], created_ns=time.time_ns(), restores_after_this_run=True)
    save(audit / 'pause_receipt.json', receipt)
    for value in own:
        current = process_identity(value['pid'])
        if current and current['start_ticks'] == value['start_ticks']:
            os.kill(value['pid'], signal.SIGSTOP)
            receipt['processes'].append(value)
            save(audit / 'pause_receipt.json', receipt)
    return receipt


def restore(receipt, audit):
    if receipt['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('server rebooted; do not signal potentially reused process IDs')
    restored = []
    for value in reversed(receipt.get('processes', [])):
        current = process_identity(value['pid'])
        if current and current['start_ticks'] == value['start_ticks']:
            os.kill(value['pid'], signal.SIGCONT)
            restored.append(value['pid'])
    save(audit / 'restoration.json', dict(restored_pids=restored, at_ns=time.time_ns()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='batch_train_20261102_002')
    parser.add_argument('--restore-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (Path.home() != Path('/home/HLC') or not root.is_relative_to(Path('/mnt/data/HLC'))
            or root.stat().st_uid != os.getuid()):
        raise ValueError('this launcher is restricted to the isolated HLC workspace')
    audit = root / 'execution'
    audit.mkdir(exist_ok=True)
    if args.restore_only:
        restore(json.loads((audit / 'pause_receipt.json').read_text()), audit)
        return
    import fcntl
    with (audit / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (audit / 'status.json').exists():
            raise ValueError('do not launch a duplicate or overwrite a previous attempt')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        env = os.environ.copy()
        env.update(MSRC_GPU='0', CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   MSRC_JUDGE_BATCH_SIZE='4', HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1')
        for key, suffix in [('TMPDIR', 'tmp'), ('HF_HOME', 'cache/hf'), ('XDG_CACHE_HOME', 'cache'),
                            ('CUDA_CACHE_PATH', 'cache/cuda'), ('TORCH_HOME', 'cache/torch'),
                            ('TRITON_CACHE_DIR', 'cache/triton'), ('PIP_CACHE_DIR', 'cache/pip')]:
            folder = root / suffix
            folder.mkdir(parents=True, exist_ok=True)
            env[key] = str(folder)
        env.pop('MSRC_GPU_SHARE_POLICY', None)
        python = '/mnt/data/HLC/msrc_safety_no_fit_fresh_20261023/MSRC/.venv/bin/python'
        base = [python, '-B', '-u']
        steps = [(phase, base + ['-m', 'pipeline.batch_training', '--run-id', args.run_id, '--phase', phase])
                 for phase in ('prepare', 'profile', 'collect', 'evaluate')]
        steps += [('baseline_safety_training', base + ['-m', 'pipeline.plugguard',
                    '--out', 'runs/' + args.run_id + '/safety', '--refit'])]
        steps += [('multilayer_' + phase, base + ['-m', 'extensions.multilayer.batched',
                    '--run-id', args.run_id, '--phase', phase]) for phase in ('prepare', 'check', 'all')]
        steps += [('report', base + ['-m', 'pipeline.batch_training', '--run-id', args.run_id, '--phase', 'report'])]
        sharing = dict(version=3, workload='plugguard', gpu_index=0,
            gpu_uuid='GPU-a9817ccf-928c-e010-8b37-82c8628d9aba', allowed_other_users=['*'],
            allocator_cap_gib=24, min_start_free_mib=32768, min_run_free_mib=8192,
            target_duty_when_shared=.5, check_interval_seconds=5,
            authorization='User authorizes this highest-priority shared-server experiment, pausing only HLC own tasks')
        save(audit / 'shared_gpu_policy.json', sharing)
        state = dict(state='STARTING', run_id=args.run_id, started_ns=time.time_ns(),
            steps=steps, completed=[], pid=os.getpid(), gpu='A100 shared; not a 5090 timing measurement',
            environment_reused_readonly=True, old_predictions_or_fitted_parameters_used=False)
        save(audit / 'status.json', state)
        receipt = None
        process = None
        def interrupted(signum, frame):
            raise SystemExit('controller interrupted by signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        try:
            receipt = pause_named_job(audit)
            for number, (name, command) in enumerate(steps, 1):
                phase_env = dict(env)
                if number >= 5:
                    phase_env.update(MSRC_SAFETY_BACKEND='plugguard',
                        MSRC_GPU_SHARE_POLICY=str(audit / 'shared_gpu_policy.json'))
                state.update(state='RUNNING', phase=name, command=command, updated_ns=time.time_ns())
                save(audit / 'status.json', state)
                started = time.time_ns()
                with (audit / ('%02d_%s.log' % (number, name))).open('xb') as log:
                    process = subprocess.Popen(command, cwd=root, env=phase_env, stdout=log,
                                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    state['child_pid'] = process.pid
                    save(audit / 'status.json', state)
                    try:
                        code = process.wait(timeout=72 * 3600)
                    except subprocess.TimeoutExpired:
                        # This is only our immediate child; no bulk/user-wide signal.
                        process.terminate()
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise RuntimeError('own phase exceeded its 72-hour bound')
                process = None
                step = dict(phase=name, started_ns=started, ended_ns=time.time_ns(), returncode=code)
                save(audit / ('%02d_%s.json' % (number, name)), step)
                state['completed'].append(step)
                if code:
                    raise RuntimeError('phase failed: ' + name)
            state.update(state='COMPLETE', finished_ns=time.time_ns())
        except BaseException as error:
            state.update(state='FAILED', error=repr(error), traceback=traceback.format_exc(), finished_ns=time.time_ns())
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            save(audit / 'status.json', state)
            if receipt is None and (audit / 'pause_receipt.json').exists():
                receipt = json.loads((audit / 'pause_receipt.json').read_text())
            if receipt is not None:
                restore(receipt, audit)


if __name__ == '__main__':
    main()
