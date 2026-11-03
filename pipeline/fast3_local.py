"""One-host three-5090 dispatcher; reuse the frozen worker and global evaluator."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import io
import os
import re
import socket
import subprocess
import sys
import time

from pipeline import common, fast3, fast3_contract as c


def parse_gpus(value):
    parts = value.split(',')
    c.require(len(parts) == c.WORLD and all(re.fullmatch(r'0|[1-9][0-9]*', p) for p in parts)
        and len(set(parts)) == c.WORLD, 'local mode requires three distinct physical GPU indices, e.g. 0,1,2')
    return parts


def devices(gpus):
    listed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    inventory = {row[0].strip(): [v.strip() for v in row] for row in csv.reader(io.StringIO(listed)) if row}
    selected = []
    for rank, index in enumerate(gpus):
        c.require(index in inventory, 'physical GPU index not found: ' + index)
        _, uuid, name, total = inventory[index]
        c.require(re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid) and '5090' in name and int(total) >= 30000,
            'local mode requires three 32GB-class RTX 5090 cards')
        selected.append(dict(rank=rank, index=index, uuid=uuid, name=name, total_mib=int(total)))
    c.require(len({d['uuid'] for d in selected}) == c.WORLD, 'three distinct physical GPU UUIDs required')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    uuids = {d['uuid'] for d in selected}
    busy = [row for row in csv.reader(io.StringIO(apps)) if row and row[0].strip() in uuids]
    c.require(not busy, 'selected GPUs are occupied; no processes were stopped: ' + repr(busy))
    return selected


def local(root, run_id, gpus):
    gpus = parse_gpus(','.join(gpus))
    with ExitStack() as locks:
        locks.enter_context(c.lock(root, 'cluster.lock'))
        c.require(not (root / 'merged').exists(), 'merged result exists; use a new run ID')
        dispatch_path = root / 'local_dispatch.json'
        origin_path = root / 'cluster_origin.json'
        c.require(not origin_path.exists() or fast3.load(origin_path).get('topology') == 'one_host_three_gpus',
            'existing non-local run cannot be resumed as a local run')
        started, first_started = fast3.cluster_start(root, run_id, topology='one_host_three_gpus')
        errors, executions = [], {}
        timing = dict(topology='one_host_three_gpus', started_ns=first_started,
            current_invocation_started_ns=started, includes_resume_downtime=first_started != started,
            includes_preparation=True, uses_ssh=False, uses_result_archives=False,
            excludes_initial_install_and_asset_transfer=True, complete=False)
        report_written = False
        try:
            selected = devices(gpus)
            for spec in selected:
                locks.enter_context(c.lock(common.ROOT / 'runs/local3-gpu-locks', spec['uuid'] + '.lock'))
            dispatch = dict(format=c.FORMAT, topology='one_host_three_gpus', run_id=run_id,
                hostname=socket.gethostname(), workers=selected, sources=c.sources())
            if dispatch_path.exists():
                c.require(fast3.load(dispatch_path) == dispatch,
                    'local GPU mapping, host or source changed; preserve run and use a new ID')
            else:
                c.atomic(dispatch_path, dispatch)
            # Preparation is CPU-only and serialized; inference starts after all contracts agree.
            plans = [fast3.prepare(c.rank_root(root, rank), run_id, rank) for rank in range(c.WORLD)]
            c.require(len({p['experiment_sha256'] for p in plans}) == 1, 'local worker contracts differ')

            def execute(spec):
                rank = spec['rank']
                env = dict(os.environ, MSRC_GPU=spec['index'], MSRC_CUDA_DEVICE_UUID=spec['uuid'],
                    CUDA_VISIBLE_DEVICES=spec['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID')
                command = [sys.executable, '-B', '-u', '-m', 'pipeline.fast3', '--phase', 'worker',
                    '--run-id', run_id, '--rank', str(rank), '--gpu', spec['index'], '--no-archive',
                    '--expected-experiment', plans[0]['experiment_sha256']]
                record = dict(rank=rank, gpu_index=spec['index'], gpu_uuid=spec['uuid'],
                    started_ns=time.time_ns(), complete=False)
                executions[str(rank)] = record
                try:
                    with (root / ('rank-%d-local.log' % rank)).open('ab') as log:
                        subprocess.run(command, cwd=common.ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    record['complete'] = True
                finally:
                    record['finished_ns'] = time.time_ns()

            with ThreadPoolExecutor(max_workers=c.WORLD) as pool:
                tasks = [pool.submit(execute, spec) for spec in selected]
                # Always join every own subprocess, including when a peer fails.
                for task in tasks:
                    try:
                        task.result()
                    except Exception as error:
                        errors.append(repr(error))
            c.require(not errors, 'one or more local workers failed: ' + repr(errors))
            result = fast3.merge(root, write=False)
            c.require(result['gpu_uuids'] == [d['uuid'] for d in selected],
                'worker evidence does not match the selected physical GPUs')
            result.update(topology='one_host_three_gpus', timing_file='local3_timing.json',
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
                current_invocation_elapsed_seconds=(finished - started) / 1e9,
                errors=errors, worker_executions=executions)
            c.atomic(root / 'local3_timing.json', timing)
            if report_written:
                manifest = root / 'joint/outputs.json'
                inventory = fast3.load(manifest)
                inventory['artifacts']['local3_timing.json'] = common.sha(root / 'local3_timing.json')
                c.atomic(manifest, inventory)
