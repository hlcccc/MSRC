"""One allocated 5090, three unchanged logical shards, executed serially."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from pipeline import common, fast3, fast3_contract as c

TOPOLOGY = 'one_host_one_gpu'
PROBE = """
import json
import torch
assert torch.__version__ == '2.7.1+cu128', 'requires torch 2.7.1+cu128'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one visible CUDA GPU required'
p = torch.cuda.get_device_properties(0)
assert '5090' in p.name and p.total_memory >= 30000 * 1024**2, '32GB-class RTX 5090 required'
uuid = 'GPU-' + str(p.uuid).removeprefix('GPU-')
print('MSRC_DEVICE_JSON ' + json.dumps(dict(uuid=uuid, name=p.name, total_bytes=p.total_memory)))
"""


def device(gpu=None):
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    slurm = bool(os.environ.get('SLURM_JOB_ID'))
    c.require(not slurm or inherited not in (None, ''), 'Slurm must supply CUDA_VISIBLE_DEVICES; no device guessing')
    c.require(gpu is None or re.fullmatch(r'0|[1-9][0-9]*', gpu), '--gpu requires one physical index')
    c.require(gpu is None or inherited is None, 'with an inherited CUDA mask, omit --gpu; allocation must not be widened')
    env = dict(os.environ)
    if inherited is None:
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # Resolve an explicit/default NVML index to UUID before CUDA enumerates devices.
        requested = gpu or '0'
        listing = subprocess.check_output(['nvidia-smi', '-i', requested, '--query-gpu=uuid',
            '--format=csv,noheader,nounits'], text=True).strip()
        c.require(bool(re.fullmatch(r'GPU-[0-9a-fA-F-]+', listing)), 'one physical GPU UUID required')
        env['CUDA_VISIBLE_DEVICES'] = listing
    # The probe exits before the occupancy check; the coordinator never owns a CUDA context.
    output = subprocess.check_output([sys.executable, '-B', '-c', PROBE], env=env, text=True)
    records = [line[len('MSRC_DEVICE_JSON '):] for line in output.splitlines() if line.startswith('MSRC_DEVICE_JSON ')]
    c.require(len(records) == 1, 'missing CUDA-visible device identity')
    actual = json.loads(records[0])
    uuid = actual['uuid']
    c.require(bool(re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid)), 'invalid CUDA UUID')
    listing = subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(listing)) if row]
    c.require(len(rows) == 1 and len(rows[0]) == 4, 'one matching NVML device required')
    index, nvml_uuid, name, total = rows[0]
    c.require(index.isdigit() and nvml_uuid == uuid and '5090' in name and int(total) >= 30000,
        'CUDA/NVML identity or RTX 5090 memory differs')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    busy = [row for row in csv.reader(io.StringIO(apps)) if row and row[0].strip() == uuid]
    c.require(not busy, 'selected GPU is occupied; no processes were stopped: ' + repr(busy))
    return dict(index=index, uuid=uuid, name=name, total_mib=int(total),
        inherited_cuda_visible_devices=inherited, slurm_allocation=slurm)


def single(root, run_id, gpu=None):
    with ExitStack() as locks:
        locks.enter_context(c.lock(root, 'cluster.lock'))
        c.require(not (root / 'merged').exists(), 'merged result exists; use a new run ID')
        origin_path = root / 'cluster_origin.json'
        c.require(not origin_path.exists() or fast3.load(origin_path).get('topology') == TOPOLOGY,
            'existing other topology cannot be resumed as single GPU')
        started, first_started = fast3.cluster_start(root, run_id, topology=TOPOLOGY)
        timing = dict(topology=TOPOLOGY, physical_gpu_count=1, logical_shards=c.WORLD,
            started_ns=first_started, current_invocation_started_ns=started,
            includes_resume_downtime=first_started != started, includes_preparation=True,
            uses_ssh=False, uses_result_archives=False, workers_serial=True,
            excludes_initial_install_and_asset_transfer=True, complete=False, worker_executions={})
        report_written = False
        try:
            spec = device(gpu)
            # Same UUID lock namespace as the local-three-card dispatcher in this installation.
            locks.enter_context(c.lock(common.ROOT / 'runs/local3-gpu-locks', spec['uuid'] + '.lock'))
            dispatch_path = root / 'single_dispatch.json'
            dispatch = dict(format=c.FORMAT, topology=TOPOLOGY, run_id=run_id,
                hostname=socket.gethostname(), physical_gpu_count=1, logical_shards=c.WORLD,
                device=spec, sources=c.sources())
            if dispatch_path.exists():
                previous = fast3.load(dispatch_path)
                # Scheduler logical masks may change on reallocation; physical identity may not.
                previous['device'] = {k: v for k, v in previous['device'].items()
                    if k not in ('inherited_cuda_visible_devices', 'slurm_allocation')}
                identity = dict(dispatch, device={k: v for k, v in spec.items()
                    if k not in ('inherited_cuda_visible_devices', 'slurm_allocation')})
                c.require(previous == identity, 'single GPU mapping, host or source changed; use a new run ID')
            else:
                c.atomic(dispatch_path, dispatch)
            plans = [fast3.prepare(c.rank_root(root, rank), run_id, rank) for rank in range(c.WORLD)]
            c.require(len({p['experiment_sha256'] for p in plans}) == 1, 'serial worker contracts differ')
            for rank in range(c.WORLD):
                env = dict(os.environ, MSRC_GPU=spec['index'], MSRC_CUDA_DEVICE_UUID=spec['uuid'],
                    CUDA_VISIBLE_DEVICES=spec['uuid'])
                command = [sys.executable, '-B', '-u', '-m', 'pipeline.fast3', '--phase', 'worker',
                    '--run-id', run_id, '--rank', str(rank), '--gpu', spec['index'], '--no-archive',
                    '--expected-experiment', plans[0]['experiment_sha256']]
                record = dict(rank=rank, gpu_index=spec['index'], gpu_uuid=spec['uuid'],
                    started_ns=time.time_ns(), complete=False)
                timing['worker_executions'][str(rank)] = record
                print('SINGLE_GPU_SHARD_START %d/%d' % (rank + 1, c.WORLD), flush=True)
                try:
                    with (root / ('rank-%d-single.log' % rank)).open('ab') as log:
                        subprocess.run(command, cwd=common.ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    record['complete'] = True
                finally:
                    record['finished_ns'] = time.time_ns()
                    c.atomic(root / 'single_timing.json', timing)
                print('SINGLE_GPU_SHARD_COMPLETE %d/%d' % (rank + 1, c.WORLD), flush=True)
            result = fast3.merge(root, write=False, physical_gpu_count=1)
            c.require(result['gpu_uuids'] == [spec['uuid']] * c.WORLD,
                'worker evidence does not match the allocated physical GPU')
            result.update(topology=TOPOLOGY, logical_shards=c.WORLD, workers_serial=True,
                timing_file='single_timing.json',
                elapsed_through_global_scoring_seconds=(time.time_ns() - first_started) / 1e9,
                current_invocation_through_global_scoring_seconds=(time.time_ns() - started) / 1e9,
                timing_excludes_initial_install_and_asset_transfer=True)
            fast3.write_report(root, result)
            report_written = True
            timing.update(complete=True, includes_global_scoring_audit_and_report=True)
            return result
        except BaseException as error:
            timing['error'] = repr(error)
            raise
        finally:
            finished = time.time_ns()
            timing.update(finished_ns=finished, elapsed_seconds=(finished - first_started) / 1e9,
                current_invocation_elapsed_seconds=(finished - started) / 1e9)
            c.atomic(root / 'single_timing.json', timing)
            if report_written:
                manifest = root / 'joint/outputs.json'
                inventory = fast3.load(manifest)
                inventory['artifacts']['single_timing.json'] = common.sha(root / 'single_timing.json')
                c.atomic(manifest, inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', help='physical index on an exclusive server without an inherited CUDA mask')
    args = parser.parse_args()
    if Path.home() == Path('/home/HLC'):
        parser.error('this release is for allocated RTX 5090, not shared HLC GPUs')
    if args.gpu is not None and not re.fullmatch(r'0|[1-9][0-9]*', args.gpu):
        parser.error('--gpu requires one physical index')
    try:
        root = c.root_for(args.run_id)
    except ValueError as error:
        parser.error(str(error))
    common.initialize(root)
    try:
        result = single(root, args.run_id, args.gpu)
        return 0 if result['scoped_acceptance_pass'] else 3
    except Exception as error:
        c.atomic(root / 'incomplete-single.json', dict(complete=False, verdict='INCOMPLETE',
            error=repr(error), time_ns=time.time_ns()))
        print('INCOMPLETE ' + repr(error), file=sys.stderr, flush=True)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
