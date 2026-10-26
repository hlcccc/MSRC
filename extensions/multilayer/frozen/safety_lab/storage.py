"""Constrain every new server write to the isolated HLC-owned experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path('/mnt/data/HLC/msrc_safety_lab_20261023')
HLC_ROOT = Path('/mnt/data/HLC')
UUID = 'c6acedb3-3560-4539-a617-4c825a1f258c'


def checked_source(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('source must be absolute')
    if not path.is_relative_to(HLC_ROOT):
        raise ValueError('source outside HLC data tree')
    for parent in [path, *path.parents]:
        if parent == HLC_ROOT.parent:
            break
        if parent.is_symlink():
            raise ValueError('symlink source forbidden: ' + str(parent))
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(HLC_ROOT) or resolved.stat().st_uid != os.getuid():
        raise ValueError('non-HLC source')
    if not resolved.is_file():
        raise ValueError('expected a regular source file')
    return resolved


def validate_root(root=ROOT):
    root = Path(root)
    if root != ROOT or root.resolve() != ROOT:
        raise ValueError('unexpected isolated root')
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError('isolated root ownership differs')
    if Path.home() != Path('/home/HLC'):
        raise ValueError('this deployment is only for HLC')
    actual = subprocess.check_output(
        ['findmnt', '-n', '-o', 'UUID', '--target', str(HLC_ROOT)], text=True).strip()
    if actual != UUID:
        raise ValueError('data mount UUID differs')
    return root


def destination(path, root=ROOT):
    root = Path(root)
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError('output escapes isolated root')
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('symlink output forbidden')
        if current.exists() and current.stat().st_uid != os.getuid():
            raise ValueError('non-HLC output')
        current = current.parent
    return resolved


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, root=ROOT):
    target = destination(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination(target.with_suffix(target.suffix + '.tmp'), root)
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def json_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
