"""Select the pinned CUDA 12.8 wheels when the GPU requires Blackwell support."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
from pathlib import Path
import subprocess
import sys

from pipeline.env_check import pinned_versions

CUDA128_INDEX = "https://download.pytorch.org/whl/cu128"


def needs_cuda128():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    for row in csv.reader(result.stdout.splitlines()):
        capability = row[0].strip().split(".") if row else []
        if capability and capability[0].isdigit() and int(capability[0]) >= 10:
            return True
    return False


def prepare(offline_wheels=None):
    if not needs_cuda128():
        print("TORCH_RUNTIME_DEFAULT: retain the reference CUDA build", flush=True)
        return
    pins = pinned_versions()
    desired = {name: pins[name] + "+cu128" for name in ("torch", "torchvision")}
    installed = {}
    for name in desired:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    if installed == desired:
        print("TORCH_RUNTIME_CU128: pinned Blackwell-compatible wheels already installed", flush=True)
        return
    if offline_wheels:
        folder = Path(offline_wheels).resolve(strict=True)
        missing = [name for name, version in desired.items()
                   if not any(folder.glob(name + "-" + version + "-*.whl"))]
        if missing:
            raise RuntimeError("Blackwell requires the pinned +cu128 torch and torchvision wheels; "
                               "the offline wheel directory is missing: " + ", ".join(missing))
        source = ["--no-index", "--find-links", str(folder)]
    else:
        source = ["--index-url", CUDA128_INDEX]
    command = [sys.executable, "-m", "pip", "install",
               *(name + "==" + version for name, version in desired.items()), *source]
    print("TORCH_RUNTIME_CU128: installing the same pinned releases for Blackwell", flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-wheels", type=Path)
    args = parser.parse_args()
    prepare(args.offline_wheels)


if __name__ == "__main__":
    main()
