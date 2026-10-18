"""Turn a downloaded or delivered asset tree into configs/paths.json, then verify it.

Two things can arrive by network -- the six checkpoints and the public
benchmarks -- and two cannot: the curated MM-SafetyBench subset and the
1,680-image benign set are built here and ship inside the delivery package. The
script keeps that split explicit instead of pretending everything is a download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

from pipeline.common import ROOT, inside, root_path, sha, save

REPOS = {
    "llava": "llava-hf/llava-1.5-13b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "guard": "yingwenhao/SingGuard-4b",
    "generator": "Qwen/Qwen2.5-VL-7B-Instruct",
    "judge": "Qwen/Qwen3Guard-Gen-8B",
}
DATA_FILES = {
    "factual": ["pope.jsonl", "textvqa.jsonl"],
    "safety": ["mmsafety_all.jsonl", "mmsafety_sd.jsonl", "benign_native.jsonl",
               "matched_benign.jsonl"],
}


def write_config(assets: Path, gpu: str) -> Path:
    assets = inside(assets)
    cfg = {
        "gpu": gpu,
        "runs_dir": "runs",
        "models": {name: {"path": str(assets / "models" / name), "repo": repo}
                   for name, repo in REPOS.items()},
        "data": {
            "factual": str(assets / "data" / "factual"),
            "safety": str(assets / "data" / "safety"),
            "mmsafety": str(assets / "data" / "MM-SafetyBench" / "data"),
        },
    }
    target = ROOT / "configs" / "paths.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def unpack(package: Path, assets: Path) -> None:
    package = Path(package).expanduser().resolve()
    assets = inside(assets)
    if package.is_dir():
        log = shutil.copytree(package, assets, dirs_exist_ok=True)
        print("copied %s -> %s" % (package, log))
    elif package.is_file():
        with tarfile.open(package) as tar:
            tar.extractall(assets, filter="data")
        print("unpacked %s -> %s" % (package, assets))
    else:
        raise SystemExit("package not found: %s" % package)


def verify(assets: Path) -> None:
    assets = inside(assets)
    manifest = assets / "MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("no MANIFEST.sha256 in %s: cannot verify what was delivered" % assets)
    bad, checked = [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        path = assets / name
        checked += 1
        if not path.is_file():
            bad.append("%s missing" % name)
        elif sha(path) != digest:
            bad.append("%s sha256 mismatch" % name)
    if bad:
        raise SystemExit("asset verification failed:\n  " + "\n  ".join(bad))
    print("ASSETS_OK: %d files verified against MANIFEST.sha256" % checked)


def download_models(assets: Path, endpoint: str | None) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub missing: pip install huggingface_hub")
    dest = inside(assets / "models")
    dest.mkdir(parents=True, exist_ok=True)
    for name, repo in REPOS.items():
        print("DOWNLOAD %s <- %s" % (name, repo), flush=True)
        snapshot_download(repo_id=repo, local_dir=str(dest / name),
                          local_dir_use_symlinks=False, resume_download=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["unpack", "download-models", "write-config", "verify"])
    ap.add_argument("--package", type=Path, help="delivery tarball or directory")
    ap.add_argument("--assets", type=Path, default=ROOT / "assets")
    ap.add_argument("--gpu", default=os.environ.get("MSRC_GPU", "0"))
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    args = ap.parse_args()
    assets = args.assets
    if args.command == "unpack":
        unpack(args.package, assets)
    elif args.command == "download-models":
        download_models(assets, args.hf_endpoint)
    elif args.command == "verify":
        verify(assets)
        return 0
    if args.command in ("unpack", "download-models", "write-config"):
        print("config: %s" % write_config(assets, args.gpu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
