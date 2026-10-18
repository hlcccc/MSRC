"""Report the interpreter, the pinned packages and the GPU this machine offers.

Run by ``scripts/01_setup_envs.sh`` and again by the preflight check. The point
is that the acceptance record can name the exact versions the run used without
anyone retyping them.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys

REQUIRED = ["numpy", "torch", "transformers", "accelerate", "pillow", "tqdm"]
OPTIONAL = ["pandas", "pyarrow", "rapidocr-onnxruntime", "torchvision", "sentencepiece", "timm"]
REFERENCE = {
    "numpy": "2.2.6",
    "torch": "2.7.1+cu126",
    "transformers": "5.13.0",
    "accelerate": "1.14.0",
    "pillow": "12.3.0",
}


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_report() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        index, name, total, used, driver = [part.strip() for part in line.split(",")]
        rows.append(dict(index=int(index), name=name, total_mib=int(total),
                         used_mib=int(used), driver=driver))
    return rows


def main() -> int:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in REQUIRED + OPTIONAL},
        "reference_versions": REFERENCE,
        "gpu": gpu_report(),
    }
    missing = [name for name in REQUIRED if info["packages"][name] is None]
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if missing:
        print("ENV_INCOMPLETE: missing " + ", ".join(missing))
        return 2
    if not any("error" not in row for row in info["gpu"]):
        print("ENV_INCOMPLETE: nvidia-smi reported no usable GPU")
        return 2
    drift = {name: (info["packages"][name], want) for name, want in REFERENCE.items()
             if info["packages"][name] and info["packages"][name].split("+")[0] != want.split("+")[0]
             and info["packages"][name] != want}
    print("ENV_OK" + ("" if not drift else " (version drift: %s)" % json.dumps(drift, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
