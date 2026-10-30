"""Turn a downloaded or delivered asset tree into configs/paths.json, then verify it.

Public weights and benchmarks can be downloaded. The public reproduction recipe
rebuilds the 1,680-image benign dataset from pinned public instructions. Six model
roles use five checkpoints because the generator is also the Qwen factual model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from pipeline.common import ROOT, inside, sha, save, initialize

REPOS = {
    "llava": "llava-hf/llava-1.5-13b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "guard": "inclusionAI/SingGuard-4b",
    "generator": "Qwen/Qwen2.5-VL-7B-Instruct",
    "judge": "Qwen/Qwen3Guard-Gen-8B",
}
# The multilayer delivery (indicators 1.11-1.13) runs with the PlugGuard backend and
# never loads the legacy SingGuard safety head, so `guard` is optional: an offline
# package may omit it, and download-models can skip it with --skip-optional-roles.
OPTIONAL_ROLES = {"guard"}
REQUIRED_ROLES = set(REPOS) - OPTIONAL_ROLES
DATA_FILES = {
    "factual": ["pope.jsonl", "textvqa.jsonl"],
    "safety": ["mmsafety_all.jsonl", "mmsafety_sd.jsonl", "benign_native.jsonl",
               "matched_benign.jsonl"],
}
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth'}
MAX_METADATA_BYTES = 64 * 1024**2


def write_config(assets: Path, gpu: str) -> Path:
    assets = inside(assets)
    lock = assets / "models.lock.json"
    models = json.loads(lock.read_text(encoding="utf-8"))["models"] if lock.exists() else {
        name: {"path": str(assets / "models" / name), "repo": repo}
        for name, repo in REPOS.items()}
    cfg = {
        "gpu": gpu,
        "runs_dir": "runs",
        "models": models,
        "model_inventory": str(lock),
        "data": {
            "factual": str(assets / "data" / "factual"),
            "safety": str(assets / "data" / "safety"),
            "mmsafety": str(assets / "data" / "MM-SafetyBench" / "data"),
        },
    }
    target = inside(ROOT / "configs" / "paths.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def unpack(package: Path, assets: Path) -> None:
    package = Path(package).expanduser().resolve()
    assets = inside(assets)
    if package.is_dir():
        for source in sorted(package.rglob('*')):
            if source.is_symlink():
                raise ValueError("delivery package must not contain symlinks: %s" % source)
            target = inside(assets / source.relative_to(package))
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        print("copied %s -> %s" % (package, assets))
    elif package.is_file():
        with tarfile.open(package) as tar:
            for member in tar.getmembers():
                target = inside(assets / member.name)
                if not target.is_relative_to(assets) or member.issym() or member.islnk():
                    raise ValueError("unsafe package member: %s" % member.name)
            tar.extractall(assets, filter="data")
        print("unpacked %s -> %s" % (package, assets))
    else:
        raise SystemExit("package not found: %s" % package)


def verify(assets: Path) -> None:
    assets = inside(assets)
    manifest = assets / "MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("no MANIFEST.sha256 in %s: cannot verify what was delivered" % assets)
    bad, listed = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        path = inside(assets / name)
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or not name or name in listed or Path(name).is_absolute()
                or not path.is_relative_to(assets)):
            raise ValueError("invalid manifest entry: %s" % line)
        listed.add(name)
        if not path.is_file():
            bad.append("%s missing" % name)
        elif sha(path) != digest:
            bad.append("%s sha256 mismatch" % name)
    if not listed:
        raise ValueError("empty asset manifest")
    required = {str(p.relative_to(assets)).replace('\\', '/')
                for sub in ('data', 'raw') for p in (assets / sub).rglob('*') if p.is_file()}
    if required - listed:
        bad.append("manifest does not cover %d data/raw file(s)" % len(required - listed))
    if bad:
        raise SystemExit("asset verification failed:\n  " + "\n  ".join(bad))
    print("ASSETS_OK: %d files verified against MANIFEST.sha256" % len(listed))


def inventory_models(assets: Path, models: dict) -> None:
    """Record the local model tree. Optional roles may be omitted from the delivery."""
    unknown = set(models) - set(REPOS)
    missing = REQUIRED_ROLES - set(models)
    if unknown or missing:
        raise ValueError('model roles must cover %s; missing %s; unknown %s'
                         % (sorted(REQUIRED_ROLES), sorted(missing), sorted(unknown)))
    memo, inventory = {}, {}
    for name, repo in REPOS.items():
        if name not in models:
            print('SKIP_OPTIONAL_MODEL %s (%s)' % (name, repo), flush=True)
            continue
        spec = models[name]
        path = Path(spec['path']).expanduser().resolve()
        if not (path / 'config.json').is_file():
            raise ValueError('model %s missing config.json: %s' % (name, path))
        if path not in memo:
            files = [p for p in path.rglob('*') if p.is_file()
                     and not any(part.startswith('.') for part in p.relative_to(path).parts)]
            if not any(p.suffix in ('.bin', '.safetensors') for p in files):
                raise ValueError('model %s contains no weights' % name)
            print('HASH_MODEL %s (%d files)' % (name, len(files)), flush=True)
            memo[path] = {str(p.relative_to(path)): sha(p) for p in sorted(files)}
        metadata = path / '.cache/huggingface/download/config.json.metadata'
        revision = spec.get('revision')
        if revision is None and metadata.is_file():
            revision = metadata.read_text(encoding='utf-8').splitlines()[0]
        inventory[name] = dict(path=str(path), repo=repo, revision=revision,
                               files=memo[path], identity='sha256 of local files')
    save(assets / 'models.lock.json', dict(models=inventory))
    save(assets / 'model_expectations.json', dict(models={
        name: {k: v for k, v in spec.items() if k != 'path'} for name, spec in inventory.items()}))
    export_model_metadata(assets, inventory)


def export_model_metadata(assets: Path, models: dict) -> None:
    """Deliver exact templates/tokenizers/configs, including local compatibility edits."""
    assets = inside(assets)
    metadata = {}
    for role, spec in models.items():
        base = Path(spec['path']).resolve()
        files = {}
        for name, digest in spec['files'].items():
            if Path(name).suffix in WEIGHT_SUFFIXES:
                continue
            source = (base / name).resolve()
            target_root = inside(assets / 'model_metadata' / role)
            target = inside(target_root / name)
            if not source.is_relative_to(base) or not target.is_relative_to(target_root):
                raise ValueError('model metadata path escapes its root')
            if source.stat().st_size > MAX_METADATA_BYTES or sha(source) != digest:
                raise ValueError('invalid frozen model metadata: %s/%s' % (role, name))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != digest:
                    raise ValueError('frozen metadata must not be overwritten: %s' % target)
            else:
                shutil.copy2(source, target)
            files[name] = digest
        metadata[role] = files
    save(assets / 'model_metadata_manifest.json', metadata)
    print('MODEL_METADATA_EXPORTED: %d role(s)' % len(metadata), flush=True)


def apply_model_metadata(assets: Path, models: dict, expected: dict) -> None:
    manifest = assets / 'model_metadata_manifest.json'
    if not manifest.is_file():
        return
    for role, files in json.loads(manifest.read_text(encoding='utf-8')).items():
        if role not in expected or role not in models:
            raise ValueError('metadata role has no frozen model expectation')
        source_root = inside(assets / 'model_metadata' / role)
        target_root = inside(models[role]['path'])
        for name, digest in files.items():
            source, target = inside(source_root / name), inside(target_root / name)
            if (not source.is_relative_to(source_root) or not target.is_relative_to(target_root)
                    or Path(name).suffix in WEIGHT_SUFFIXES
                    or expected[role].get('files', {}).get(name) != digest):
                raise ValueError('metadata is not an approved frozen model file')
            if source.stat().st_size > MAX_METADATA_BYTES or sha(source) != digest:
                raise ValueError('model metadata digest mismatch: %s/%s' % (role, name))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print('FROZEN_MODEL_METADATA_APPLIED', flush=True)


def verify_model_inventory(assets: Path) -> None:
    lock = json.loads((assets / 'models.lock.json').read_text(encoding='utf-8'))
    checked = set()
    roles = set(lock['models'])
    missing, unknown = REQUIRED_ROLES - roles, roles - set(REPOS)
    if missing or unknown:
        raise ValueError('model inventory must cover %s; missing %s; unknown %s'
                         % (sorted(REQUIRED_ROLES), sorted(missing), sorted(unknown)))
    for spec in lock['models'].values():
        base = Path(spec['path']).expanduser().resolve()
        for name, digest in spec['files'].items():
            path = (base / name).resolve()
            if not path.is_relative_to(base):
                raise ValueError('model inventory path escapes model root')
            if path in checked:
                continue
            if sha(path) != digest:
                raise ValueError('model changed: %s' % path)
            checked.add(path)
    print('MODEL_INVENTORY_OK: %d unique files' % len(checked), flush=True)


def download_models(assets: Path, endpoint: str | None, include_optional: bool = True) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ['HF_HUB_OFFLINE'] = '0'
    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        raise SystemExit("huggingface_hub missing: pip install huggingface_hub")
    dest = inside(assets / "models")
    dest.mkdir(parents=True, exist_ok=True)
    lock_path = assets / 'models.lock.json'
    if not lock_path.exists():
        lock_path = assets / 'model_expectations.json'
    if not lock_path.exists():
        from pipeline import common
        if (common.ROOT / 'resources/reproduction/models.json').is_file():
            from pipeline.public_assets import install_model_recipe
            install_model_recipe(assets)
    previous = json.loads(lock_path.read_text(encoding='utf-8'))['models'] if lock_path.exists() else {}
    models, downloaded = {}, {}
    for name, repo in REPOS.items():
        if name in OPTIONAL_ROLES and not include_optional:
            print("SKIP_OPTIONAL_MODEL %s (%s)" % (name, repo), flush=True)
            continue
        if repo in downloaded:
            models[name] = downloaded[repo]
            continue
        revision = (previous.get(name) or {}).get('revision') or HfApi().model_info(repo).sha
        print("DOWNLOAD %s <- %s" % (name, repo), flush=True)
        target = inside(dest / name)
        selection = {'allow_patterns': sorted(previous[name]['files'])} if previous.get(name, {}).get('files') else {}
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), max_workers=2, **selection)
        models[name] = downloaded[repo] = dict(path=str(target), repo=repo, revision=revision)
    apply_model_metadata(assets, models, previous)
    for name, expected in previous.items():
        if name not in models:
            continue
        for filename, digest in expected.get('files', {}).items():
            path = (Path(models[name]['path']) / filename).resolve()
            if not path.is_relative_to(Path(models[name]['path']).resolve()) or sha(path) != digest:
                raise ValueError('download differs from frozen model file: %s/%s' % (name, filename))
    inventory_models(assets, models)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["unpack", "download-models", "reuse-models", 'export-model-metadata',
                                        "verify-models", "write-config", "verify"])
    ap.add_argument("--package", type=Path, help="delivery tarball or directory")
    ap.add_argument("--assets", type=Path, default=ROOT / "assets")
    ap.add_argument("--gpu", default=os.environ.get("MSRC_GPU", "0"))
    ap.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    ap.add_argument('--local-models', type=Path, help='JSON mapping model roles to existing read-only paths')
    ap.add_argument('--skip-optional-roles', action='store_true',
                    help='download only the roles the multilayer delivery needs (skip SingGuard)')
    args = ap.parse_args()
    initialize()
    assets = inside(args.assets)
    if args.command == "unpack":
        unpack(args.package, assets)
    elif args.command == "download-models":
        download_models(assets, args.hf_endpoint, include_optional=not args.skip_optional_roles)
    elif args.command == 'reuse-models':
        models = json.loads(args.local_models.read_text(encoding='utf-8'))
        base = args.local_models.expanduser().resolve().parent
        resolved = {}
        for role, spec in models.items():
            spec = {'path': spec} if isinstance(spec, str) else dict(spec)
            if 'path' in spec and not Path(spec['path']).expanduser().is_absolute():
                # Relative entries resolve against the JSON file, so the offline
                # model folder can live anywhere the tester puts it.
                spec['path'] = str((base / spec['path']).resolve())
            resolved[role] = spec
        inventory_models(assets, resolved)
    elif args.command == 'export-model-metadata':
        export_model_metadata(assets, json.loads((assets / 'models.lock.json').read_text(encoding='utf-8'))['models'])
    elif args.command == 'verify-models':
        verify_model_inventory(assets)
        return 0
    elif args.command == "verify":
        verify(assets)
        return 0
    if args.command in ("unpack", "download-models", "reuse-models", "write-config"):
        print("config: %s" % write_config(assets, args.gpu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
