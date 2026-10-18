"""Check the machine, the assets and the GPU before anything is generated.

Every check here fails loudly instead of degrading quietly: a missing model
shard or a GPU that another job already holds must stop the run, not shrink it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.common import (ROOT, inside, initialize, load_config, model_paths,
                             data_paths, sha, save)

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
        path = Path(path)
        if not (path / "config.json").is_file():
            raise SystemExit("model %s has no config.json at %s" % (name, path))
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
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
        rows = sum(1 for line in path.open(encoding="utf-8-sig") if line.strip())
        if rows != expected:
            raise SystemExit("%s has %d rows, expected %d" % (filename, rows, expected))
        found[filename] = dict(rows=rows, sha256=sha(path))
    for filename in SAFETY_FILES:
        path = Path(data["safety"]) / filename
        if not path.is_file():
            raise SystemExit("missing raw input: %s" % path)
        rows = sum(1 for line in path.open(encoding="utf-8-sig") if line.strip())
        found[filename] = dict(rows=rows, sha256=sha(path))
    bench = Path(data["mmsafety"])
    if not bench.is_dir():
        raise SystemExit("missing MM-SafetyBench parquet tree: %s" % bench)
    found["mmsafety_parquet"] = dict(categories=len([p for p in bench.iterdir() if p.is_dir()]))
    return found


def check_gpu() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found: this pipeline needs an NVIDIA GPU")
    cfg = load_config()
    index = str(cfg.get("gpu", "0"))
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
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"], text=True)
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [part.strip() for part in line.split(",")]
        if uuid == target["uuid"]:
            owner = subprocess.run(["ps", "-o", "user=", "-p", pid], text=True,
                                   capture_output=True).stdout.strip()
            busy.append(dict(pid=pid, owner=owner or "?", mib=mem))
    if busy:
        raise SystemExit("GPU %s is already in use (%s); choose another card with --gpus "
                         "or wait. This script never stops another user's job."
                         % (index, ", ".join("%s pid %s" % (b["owner"], b["pid"]) for b in busy)))
    return [target]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="run directory; environment.json is written there")
    args = ap.parse_args()
    initialize()
    report = dict(project=str(ROOT), python=sys.version, packages=check_packages(),
                  models=check_models(), data=check_data(), gpu=check_gpu())
    if args.out:
        save(inside(args.out) / "environment.json", report)
    print(json.dumps({k: report[k] for k in ("packages", "gpu")}, indent=2, ensure_ascii=False))
    print("PREFLIGHT_OK: assets, packages and the requested GPU are usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
