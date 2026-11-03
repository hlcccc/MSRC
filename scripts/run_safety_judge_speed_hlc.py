"""Bounded single-GPU research launch; never stops another HLC or group job."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowed = Path('/mnt/data/HLC').resolve(strict=True)
    for path in (root, args.source.resolve(strict=True), args.model.resolve(strict=True)):
        if not path.is_relative_to(allowed) or path.stat().st_uid != os.getuid():
            raise ValueError('HLC-owned root, source and model required')
    if re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id) is None:
        raise ValueError('simple run ID required')
    audit = root / 'execution' / args.run_id
    if not audit.resolve().is_relative_to(root):
        raise ValueError('execution path escapes the isolated root')
    audit.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(10)
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', MSRC_GPU='0', CUDA_VISIBLE_DEVICES='0',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for name, suffix in [('TMPDIR', 'tmp'), ('HF_HOME', 'cache/hf'), ('XDG_CACHE_HOME', 'cache'),
                         ('TORCH_HOME', 'cache/torch'), ('PIP_CACHE_DIR', 'cache/pip'),
                         ('CUDA_CACHE_PATH', 'cache/cuda'), ('TRITON_CACHE_DIR', 'cache/triton')]:
        path = (root / suffix).resolve()
        if not path.is_relative_to(root):
            raise ValueError('cache escapes the isolated root')
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    python = '/mnt/data/HLC/msrc_safety_no_fit_fresh_20261023/MSRC/.venv/bin/python'
    command = [python, '-B', '-u', '-m', 'pipeline.safety_judge_speed',
        '--source', str(args.source.resolve()), '--model', str(args.model.resolve()),
        '--run-id', args.run_id, '--gpu', '0', '--cap-gib', '22']
    state = dict(state='WAITING_FOR_GPU', run_id=args.run_id, pid=os.getpid(), gpu=0,
        command=command, cap_gib=22, reserve_gib=8, shared_compute_duty=.5,
        started_ns=time.time_ns(), formal_jobs_untouched=True, cloud_jobs_untouched=True,
        multi_gpu=False, full_acceptance=False)
    child = None
    try:
        deadline = time.monotonic() + 3600
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            state.update(checked_ns=time.time_ns(), free_mib=free)
            save(audit / 'status.json', state)
            if free >= 31 * 1024 and shutil.disk_usage(root).free >= 2 * 1024**3:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('one-hour bounded research resource wait expired')
            time.sleep(20)
        with (audit / 'benchmark.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            state.update(state='RUNNING', child_pid=child.pid)
            save(audit / 'status.json', state)
            code = child.wait(timeout=30 * 60)
            state.update(state='COMPLETE' if code == 0 else 'FAILED', returncode=code)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(state='FAILED', error=repr(error))
    finally:
        state['ended_ns'] = time.time_ns()
        save(audit / 'status.json', state)


if __name__ == '__main__':
    main()
