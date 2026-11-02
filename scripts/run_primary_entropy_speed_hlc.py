"""Run a bounded isolated acceleration benchmark when safe GPU headroom exists."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parameters',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--gpu',type=int,choices=[0,1],default=0)
    p.add_argument('--cap-gib',type=int,default=19)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    allowed=Path('/mnt/data/HLC').resolve(strict=True)
    if (not root.is_relative_to(allowed) or root.stat().st_uid!=os.getuid()
            or not args.source.resolve(strict=True).is_relative_to(allowed)
            or not args.parameters.resolve(strict=True).is_relative_to(allowed)
            or re.fullmatch('[A-Za-z0-9_-]+',args.run_id) is None):
        raise ValueError('HLC-owned isolated workspace and a simple run ID are required')
    audit=root/'execution'/args.run_id
    if not audit.resolve().is_relative_to(root):raise ValueError('execution path escapes workspace')
    audit.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    os.nice(10)
    env=dict(os.environ,MSRC_GPU=str(args.gpu),CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONPATH=str(root),
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    for key,sub in [('TMPDIR','tmp'),('HF_HOME','cache/hf'),('XDG_CACHE_HOME','cache'),
                    ('CUDA_CACHE_PATH','cache/cuda'),('TORCH_HOME','cache/torch'),
                    ('TRITON_CACHE_DIR','cache/triton'),('PIP_CACHE_DIR','cache/pip')]:
        path=(root/sub).resolve()
        if not path.is_relative_to(root):raise ValueError('cache path escapes isolated workspace')
        path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    python='/mnt/data/HLC/msrc_safety_no_fit_fresh_20261023/MSRC/.venv/bin/python'
    command=[python,'-B','-u','-m','pipeline.primary_entropy_speed','--source',str(args.source.resolve()),
        '--parameters',str(args.parameters.resolve()),'--run-id',args.run_id,'--gpu',str(args.gpu),
        '--cap-gib',str(args.cap_gib)]
    state=dict(state='WAITING_FOR_GPU',pid=os.getpid(),run_id=args.run_id,gpu=args.gpu,
        started_ns=time.time_ns(),original_jobs_untouched=True,base_environment_readonly=True,
        fresh_inference_benchmark=True,full_acceptance=False,cap_gib=args.cap_gib,reserve_gib=8,
        command=command)
    child=None
    try:
        limit=time.monotonic()+6*3600
        while True:
            line=subprocess.check_output(['nvidia-smi','--id='+str(args.gpu),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
            state.update(free_mib=int(line.strip()),available_ram_kib=int(memory['MemAvailable'].split()[0]),
                checked_ns=time.time_ns())
            save(audit/'status.json',state)
            if (state['free_mib']>=(args.cap_gib+9)*1024 and state['available_ram_kib']>=32*1024**2
                    and shutil.disk_usage(allowed).free>2*1024**3):break
            if time.monotonic()>limit:raise TimeoutError('bounded six-hour resource wait ended')
            time.sleep(20)
        with (audit/'benchmark.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
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
