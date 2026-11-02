"""Prepare on CPU, wait for safe GPU headroom, and run an isolated bounded pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--dev-per-dataset',type=int,default=256)
    p.add_argument('--test-per-dataset',type=int,default=128)
    p.add_argument('--cap-gib',type=int,default=19)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    allowed=Path('/mnt/data/HLC').resolve(strict=True)
    if (not root.is_relative_to(allowed) or root.stat().st_uid!=os.getuid()
            or not args.source.resolve(strict=True).is_relative_to(allowed)):
        raise ValueError('HLC-owned isolated workspace and read-only HLC source required')
    audit=root/'execution';audit.mkdir(exist_ok=True)
    import fcntl
    with (audit/'entropy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (audit/'status.json').exists():raise ValueError('preserve the previous run')
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
        env=dict(os.environ,MSRC_GPU=str(args.gpu),CUDA_VISIBLE_DEVICES=str(args.gpu),
            PYTHONPATH=str(root),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
            PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        for key,sub in [('TMPDIR','tmp'),('HF_HOME','cache/hf'),('XDG_CACHE_HOME','cache'),
                        ('CUDA_CACHE_PATH','cache/cuda'),('TORCH_HOME','cache/torch'),
                        ('TRITON_CACHE_DIR','cache/triton'),('PIP_CACHE_DIR','cache/pip')]:
            path=(root/sub).resolve()
            if not path.is_relative_to(allowed):raise ValueError('cache path escapes HLC')
            path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
        python='/mnt/data/HLC/msrc_safety_no_fit_fresh_20261023/MSRC/.venv/bin/python'
        base=[python,'-B','-u','-m','pipeline.entropy_estimator','--source',str(args.source.resolve()),
            '--run-id',args.run_id,'--cap-gib',str(args.cap_gib),
            '--dev-per-dataset',str(args.dev_per_dataset),'--test-per-dataset',str(args.test_per_dataset)]
        state=dict(state='PREPARING',pid=os.getpid(),run_id=args.run_id,gpu=args.gpu,
            started_ns=time.time_ns(),original_jobs_untouched=True,base_environment_readonly=True,
            source_readonly=True,acceptance=False,cap_gib=args.cap_gib,reserve_gib=8)
        save(audit/'status.json',state)
        child=None
        try:
            with (audit/'prepare.log').open('xb') as log:
                subprocess.run(base+['--phase','prepare'],cwd=root,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            state.update(state='WAITING_FOR_GPU');save(audit/'status.json',state)
            limit=time.monotonic()+12*3600
            while True:
                line=subprocess.check_output(['nvidia-smi','--id='+str(args.gpu),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                state.update(free_mib=int(line.strip()),checked_ns=time.time_ns())
                save(audit/'status.json',state)
                if state['free_mib'] >= (args.cap_gib+9)*1024:break
                if time.monotonic()>limit:raise TimeoutError('12-hour bounded GPU wait ended')
                time.sleep(20)
            with (audit/'pilot.log').open('xb') as log:
                child=subprocess.Popen(base+['--phase','run'],cwd=root,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state.update(state='RUNNING',child_pid=child.pid);save(audit/'status.json',state)
                code=child.wait(timeout=3*3600)
                state.update(state='COMPLETE' if code==0 else 'FAILED',returncode=code)
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            state.update(state='FAILED',error=repr(error))
        finally:
            state['ended_ns']=time.time_ns();save(audit/'status.json',state)


if __name__=='__main__':main()
