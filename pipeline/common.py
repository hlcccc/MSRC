"""Shared plumbing for the acceptance pipeline.

Everything the five entry points need and nothing they do not: where the
project, the assets and a run directory live; how a JSONL row is written; how a
file is hashed. The paths are read from ``configs/paths.json`` -- written by
``scripts/02_prepare_assets_data.sh`` -- so that the same code runs on the
acceptance machine as on ours without editing constants.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SEED = 20260920

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("MSRC_CONFIG", ROOT / "configs" / "paths.json"))
EXAMPLE_CONFIG = ROOT / "configs" / "paths.example.json"


def load_config() -> dict:
    """Load configs/paths.json, falling back to the documented default layout."""
    path = Path(os.environ.get("MSRC_CONFIG", CONFIG_PATH))
    if path.is_file():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    elif EXAMPLE_CONFIG.is_file():
        cfg = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        print("[config] %s missing; using the default layout from %s"
              % (path, EXAMPLE_CONFIG.name), file=sys.stderr)
    else:
        raise SystemExit("no configuration: run scripts/02_prepare_assets_data.sh first")
    cfg.setdefault("runs_dir", "runs")
    return cfg


def root_path(value: str) -> Path:
    """Resolve a configured path; relative paths are relative to the project."""
    return (ROOT / value).expanduser().resolve() if not Path(value).is_absolute() \
        else Path(value).expanduser().resolve()


def model_paths() -> dict[str, str]:
    """Checkpoint directories as strings: they travel into JSON protocols as-is."""
    cfg = load_config()
    return {name: str(root_path(spec["path"] if isinstance(spec, dict) else spec))
            for name, spec in cfg["models"].items()}


def data_paths() -> dict[str, Path]:
    cfg = load_config()
    return {name: root_path(value) for name, value in cfg["data"].items()}


def runs_dir() -> Path:
    return root_path(load_config()["runs_dir"])


def inside(path) -> Path:
    """Resolve ``path`` and refuse anything that leaves the project directory.

    The pipeline writes only under the project root (assets, runs); a path that
    escapes it is a configuration mistake, not something to paper over.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("refusing to write outside the project tree: %s" % resolved)
    return resolved


def readable(path) -> Path:
    """Resolve a path that is only read (model weights, images), no containment rule."""
    return Path(path).expanduser().resolve()


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def read_rows(path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def emit(handle, row) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def initialize(base: Path | None = None) -> None:
    """Pin caches and thread counts inside the project before torch is imported."""
    base = inside(base or ROOT)
    for key, sub in (("TMPDIR", "tmp"), ("HF_HOME", "cache/hf"), ("HF_HUB_CACHE", "cache/hf/hub"),
                     ("TORCH_HOME", "cache/torch"), ("XDG_CACHE_HOME", "cache"),
                     ("PIP_CACHE_DIR", "cache/pip"), ("TRANSFORMERS_CACHE", "cache/hf")):
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "ARROW_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def select_device(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    index = str(os.environ.get("MSRC_GPU") or cfg.get("gpu", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    return "cuda:0"
