"""Install a hash-pinned private CPython when the host lacks Python 3.10/venv."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

from pipeline.common import ROOT, inside, save, sha


def prepare():
    recipe = json.loads((ROOT / 'resources/reproduction/python.json').read_text(encoding='utf-8'))
    if platform.system() != 'Linux' or platform.machine() not in recipe['linux']:
        raise RuntimeError('provide CPython 3.10 via MSRC_BASE_PYTHON on this platform')
    source = recipe['linux'][platform.machine()]
    destination = inside(ROOT / '.python' / recipe['version'])
    interpreter = destination / 'python/bin/python3.10'
    if interpreter.is_file():
        subprocess.run([str(interpreter), '-B', '-c', 'import ensurepip,sys; assert sys.version_info[:2] == (3,10)'], check=True)
        return interpreter
    if destination.exists():
        raise ValueError('partial private Python preserved; inspect ' + str(destination))
    cache = inside(ROOT / 'cache/python')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'cpython.tar.gz'
    if not archive.exists():
        partial = inside(cache / 'cpython.part')
        print('DOWNLOAD_PYTHON ' + source['url'], file=sys.stderr, flush=True)
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'MSRC-acceptance'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size != source['bytes'] or sha(partial) != source['sha256']:
            raise ValueError('private CPython archive differs from its published digest')
        partial.replace(archive)
    if sha(archive) != source['sha256']:
        raise ValueError('cached CPython archive hash differs')
    if shutil.disk_usage(ROOT).free < 512 * 1024**2:
        raise ValueError('insufficient project-disk space for private CPython')
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate CPython archive members')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or not name.parts or name.parts[0] != 'python'
                    or not (member.isfile() or member.isdir() or member.issym())):
                raise ValueError('unsafe CPython archive member: ' + member.name)
            target = inside(destination / member.name)
            if member.issym() and not (target.parent / member.linkname).resolve().is_relative_to(destination):
                raise ValueError('CPython symlink leaves private installation')
        destination.mkdir(parents=True)
        # Validated archive only; no device nodes, traversal or external symlinks.
        for member in members:
            target = inside(destination / member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source_file, target.open('xb') as output:
                    shutil.copyfileobj(source_file, output)
                target.chmod(member.mode & 0o777)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)
    subprocess.run([str(interpreter), '-B', '-c', 'import ensurepip,sys; assert sys.version_info[:2] == (3,10)'], check=True)
    save(ROOT / 'assets/python_runtime.json', dict(version=recipe['version'], source=source,
                                                  interpreter=str(interpreter), project_private=True))
    return interpreter


if __name__ == '__main__':
    print(prepare())
