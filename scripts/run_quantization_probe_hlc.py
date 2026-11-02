"""Bounded, detached probe; no signals are sent to any existing workload."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    parser.add_argument('--cap-gib', type=int, choices=range(7, 20), default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowed = Path('/mnt/data/HLC').resolve(strict=True)
    if (root.resolve() != root or not root.is_relative_to(allowed)
            or root.stat().st_uid != os.getuid() or Path.home() != Path('/home/HLC')):
        raise ValueError('HLC-owned isolated workspace required')
    if not args.inputs.resolve(strict=True).is_relative_to(allowed):
        raise ValueError('read-only source inputs must be in the HLC workspace')
    audit = root / 'execution'
    audit.mkdir(exist_ok=True)
    import fcntl
    with (audit / 'probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (audit / 'status.json').exists():
            raise ValueError('preserve previous attempt; do not overwrite it')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        env = dict(os.environ, MSRC_GPU=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
            OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1',
            PYTHONPATH=str(root / 'deps/bnb_0_48_2') + os.pathsep + str(root))
        headers = root / 'deps/python_headers/usr/include'
        if headers.exists():
            if not headers.resolve().is_relative_to(allowed):
                raise ValueError('Python headers escape the HLC workspace')
            env['CPATH'] = str(headers / 'python3.10') + os.pathsep + str(headers)
        for key, folder in [('TMPDIR', 'tmp'), ('HF_HOME', 'cache/hf'), ('XDG_CACHE_HOME', 'cache'),
                            ('CUDA_CACHE_PATH', 'cache/cuda'), ('TORCH_HOME', 'cache/torch'),
                            ('TRITON_CACHE_DIR', 'cache/triton'), ('PIP_CACHE_DIR', 'cache/pip')]:
            target = root / folder
            if not target.resolve().is_relative_to(allowed):
                raise ValueError('cache path escapes the HLC workspace')
            target.mkdir(parents=True, exist_ok=True)
            env[key] = str(target)
        python = '/mnt/data/HLC/msrc_safety_no_fit_fresh_20261023/MSRC/.venv/bin/python'
        command = [python, '-B', '-u', '-m', 'pipeline.quantization_probe',
            '--run-id', args.run_id, '--precision', 'bnb_nf4', '--inputs', str(args.inputs),
            '--cap-gib', str(args.cap_gib)]
        state = dict(state='STARTING', pid=os.getpid(), command=command,
            started_ns=time.time_ns(), gpu_index=args.gpu, allocator_cap_gib=args.cap_gib,
            reserve_gib=8, duty=.5, original_run_untouched=True, acceptance_test=False,
            dependencies_isolated=True, base_environment_readonly=True)
        save(audit / 'status.json', state)
        with (audit / 'probe.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            state.update(state='RUNNING', child_pid=child.pid)
            save(audit / 'status.json', state)
            try:
                code = child.wait(timeout=90 * 60)
                state.update(state='COMPLETE' if code == 0 else 'FAILED', returncode=code)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                state.update(state='TIME_LIMIT', error='90-minute probe bound; partial measurements retained')
            except BaseException as error:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                state.update(state='FAILED', error=repr(error))
                raise
            finally:
                state['ended_ns'] = time.time_ns()
                save(audit / 'status.json', state)


if __name__ == '__main__':
    main()
