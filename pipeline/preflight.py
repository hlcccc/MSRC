"""Check the machine, the assets and the GPU before anything is generated.

Every check here fails loudly instead of degrading quietly: a missing model
shard or a GPU that another job already holds must stop the run, not shrink it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.common import (ROOT, inside, initialize, load_config, model_paths,
                             data_paths, sha, save, read_rows)

FACTUAL_COUNTS = {"pope.jsonl": 9000, "textvqa.jsonl": 5000}
SAFETY_FILES = ["mmsafety_all.jsonl", "mmsafety_sd.jsonl", "benign_native.jsonl",
                "matched_benign.jsonl"]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def check_packages() -> dict:
    import importlib.metadata as md
    wanted = ["numpy", "torch", "transformers", "accelerate", "pillow", "pandas",
              "pyarrow", "rapidocr-onnxruntime", "tqdm"]
    versions, missing = {}, []
    for name in wanted:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise SystemExit("missing packages: %s (run scripts/01_setup_envs.sh --execute)"
                         % ", ".join(missing))
    return versions


def check_models() -> dict:
    found = {}
    for name, path in model_paths().items():
        if name == 'guard' and os.environ.get('MSRC_SAFETY_BACKEND') == 'plugguard':
            continue
        path = Path(path)
        if not (path / "config.json").is_file():
            raise SystemExit("model %s has no config.json at %s" % (name, path))
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shards = sorted(set(json.loads(index.read_text(encoding='utf-8'))["weight_map"].values()))
        else:
            shards = sorted(p.name for p in path.glob("*") if p.suffix in WEIGHT_SUFFIXES)
        if not shards:
            raise SystemExit("model %s holds no weight shard at %s" % (name, path))
        absent = [shard for shard in shards if not (path / shard).is_file()]
        if absent:
            raise SystemExit("model %s is missing %d shard(s), first: %s"
                             % (name, len(absent), absent[0]))
        found[name] = dict(path=str(path), shards=len(shards))
    return found


def check_data() -> dict:
    data = data_paths()
    found = {}
    for name in ("factual", "safety"):
        if name not in data:
            raise SystemExit("configuration has no '%s' data directory" % name)
    for filename, expected in FACTUAL_COUNTS.items():
        path = Path(data["factual"]) / filename
        if not path.is_file():
            raise SystemExit("missing raw input: %s" % path)
        rows = read_rows(path)
        if len(rows) != expected:
            raise SystemExit("%s has %d rows, expected %d" % (filename, len(rows), expected))
        _check_inputs(rows, path)
        found[filename] = dict(rows=len(rows), sha256=sha(path))
    for filename in SAFETY_FILES:
        path = Path(data["safety"]) / filename
        if not path.is_file():
            raise SystemExit("missing raw input: %s" % path)
        rows = read_rows(path)
        expected = {"mmsafety_all.jsonl": 1680, "mmsafety_sd.jsonl": 3360,
                    "benign_native.jsonl": 1680, "matched_benign.jsonl": 1680}[filename]
        if len(rows) != expected:
            raise SystemExit("%s has %d rows, expected %d" % (filename, len(rows), expected))
        _check_inputs(rows, path)
        found[filename] = dict(rows=len(rows), sha256=sha(path))
    bench = Path(data["mmsafety"])
    if not bench.is_dir():
        raise SystemExit("missing MM-SafetyBench parquet tree: %s" % bench)
    found["mmsafety_parquet"] = dict(categories=len([p for p in bench.iterdir() if p.is_dir()]))
    assets = Path(data['factual']).parent.parent
    if (assets / 'prepared_bundle.json').is_file():
        from pipeline.prepared_bundle import verify
        found['fixed_partitions'] = verify(assets)
    return found


def _check_inputs(rows, path):
    if len({r['id'] for r in rows}) != len(rows):
        raise SystemExit("duplicate input IDs: %s" % path)
    for row in rows:
        if row.get('answer') or row.get('evidence'):
            raise SystemExit("old answers/evidence are not raw input: %s" % path)
        if not Path(row['image']).is_file():
            raise SystemExit("missing input image: %s" % row['image'])


class GPUInUse(SystemExit):
    pass


def check_gpu(ignore_current_process=False) -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found: this pipeline needs an NVIDIA GPU")
    cfg = load_config()
    index = str(os.environ.get("MSRC_GPU") or cfg.get("gpu", "0"))
    listed = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    target = None
    for line in listed.splitlines():
        if not line.strip():
            continue
        i, uuid, total, used = [part.strip() for part in line.split(",")]
        if i == index:
            target = dict(index=int(i), uuid=uuid, total_mib=int(total), used_mib=int(used))
    if target is None:
        raise SystemExit("GPU %s does not exist on this machine" % index)
    busy = []
    current_loaded = False
    current_memory_mib = 0
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"], text=True)
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [part.strip() for part in line.split(",")]
        if uuid == target["uuid"]:
            current_loaded |= int(pid) == os.getpid()
            if int(pid) == os.getpid():
                current_memory_mib += int(mem)
            if ignore_current_process and int(pid) == os.getpid():
                continue
            owner = subprocess.run(["ps", "-o", "user=", "-p", pid], text=True,
                                   capture_output=True).stdout.strip()
            busy.append(dict(pid=pid, owner=owner or "?", mib=mem))
    from pipeline.gpu_sharing import load_shared_policy
    policy = load_shared_policy()
    if policy is not None:
        if target['index'] != policy['gpu_index'] or target['uuid'] != policy['gpu_uuid']:
            raise ValueError('Requested GPU differs from the explicitly authorized shared GPU')
        unexpected = [p for p in busy if '*' not in policy['allowed_other_users']
                      and p['owner'] not in policy['allowed_other_users']]
        if unexpected:
            raise GPUInUse('Shared GPU contains an unauthorized process owner: %s' % unexpected)
        free = int(subprocess.check_output([
            'nvidia-smi', '--id=' + str(target['index']), '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        minimum = (max(policy['min_run_free_mib'],
                       policy['min_start_free_mib'] - current_memory_mib)
                   if current_loaded else policy['min_start_free_mib'])
        if free < minimum:
            raise GPUInUse('Shared GPU has %d MiB free; this stage requires %d MiB reserve'
                           % (free, minimum))
        target.update(free_mib=free, shared_gpu_execution=True,
                      shared_policy_sha256=policy['source_sha256'], other_processes=busy)
        return [target]
    if busy:
        raise GPUInUse("GPU %s is already in use (%s); choose another card with --gpus "
                         "or wait. This script never stops another user's job."
                         % (index, ", ".join("%s pid %s" % (b["owner"], b["pid"]) for b in busy)))
    return [target]


def wait_for_gpu() -> list[dict]:
    while True:
        try:
            return check_gpu(ignore_current_process=True)
        except GPUInUse as exc:
            print('GPU_WAIT: %s; checking again in 120s' % exc, flush=True)
            time.sleep(120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="run directory; environment.json is written there")
    ap.add_argument('--wait-for-gpu', action='store_true')
    args = ap.parse_args()
    initialize()
    cfg = load_config()
    if cfg.get('model_inventory'):
        from pipeline.prepare_assets import verify_model_inventory
        inventory = Path(cfg['model_inventory'])
        verify_model_inventory(inventory.parent)
        locked = json.loads(inventory.read_text(encoding='utf-8'))['models']
        if any(Path(path).resolve() != Path(locked[name]['path']).resolve()
               for name, path in model_paths().items()):
            raise ValueError('configured model path differs from frozen inventory')
    report = dict(project=str(ROOT), python=sys.version, packages=check_packages(),
                  models=check_models(), data=check_data(),
                  gpu=wait_for_gpu() if args.wait_for_gpu else check_gpu())
    report['dependency_freeze'] = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    if cfg.get('model_inventory'):
        report['model_inventory_sha256'] = sha(cfg['model_inventory'])
    if args.out:
        save(inside(args.out) / "environment.json", report)
    print(json.dumps({k: report[k] for k in ("packages", "gpu")}, indent=2, ensure_ascii=False))
    print("PREFLIGHT_OK: assets, packages and the requested GPU are usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
