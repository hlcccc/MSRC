"""Supply missing OpenCV/OCR system libraries inside this project, never via sudo."""
from __future__ import annotations

import argparse
import ctypes
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from pipeline.common import ROOT, inside, save, sha

LIBRARIES = ('libGL.so.1', 'libglib-2.0.so.0', 'libgthread-2.0.so.0')
PACKAGES = ('libgl1', 'libglx0', 'libglvnd0', 'libx11-6', 'libx11-xcb1',
            'libxext6', 'libxcb1', 'libxau6', 'libxdmcp6', 'libbsd0', 'libmd0',
            'libpcre2-8-0', 'libffi8', 'libmount1', 'libselinux1', 'libblkid1')


def library_paths():
    base = inside(ROOT / '.venv/system-libs')
    arch = {'x86_64': 'x86_64-linux-gnu', 'aarch64': 'aarch64-linux-gnu'}.get(platform.machine())
    return [base / p for p in (('usr/lib/' + arch, 'lib/' + arch, 'usr/lib', 'lib') if arch
                              else ('usr/lib', 'lib'))]


def missing_libraries():
    missing = []
    for name in LIBRARIES:
        try:
            ctypes.CDLL(name)
        except OSError as error:
            missing.append(dict(library=name, error=str(error)))
    return missing


def prepare(check=False):
    missing = missing_libraries()
    if not missing:
        print('SYSTEM_LIBRARIES_OK', flush=True)
        return []
    if check:
        raise RuntimeError('missing OCR libraries; rerun setup with --execute: ' + json.dumps(missing))
    if platform.system() != 'Linux' or any(shutil.which(name) is None for name in ('apt-get', 'apt-cache', 'dpkg-deb')):
        raise RuntimeError('automatic private OCR libraries require Debian/Ubuntu apt and dpkg-deb: ' + json.dumps(missing))
    cache = inside(ROOT / 'cache/runtime-libraries')
    cache.mkdir(parents=True, exist_ok=True)
    options = ['-o', 'Debug::NoLocking=1', '-o', 'APT::Sandbox::User=' + getpass.getuser(),
               '-o', 'Dir::Cache=' + str(cache / 'apt/cache'),
               '-o', 'Dir::State=' + str(cache / 'apt/state'),
               '-o', 'Dir::State::status=/var/lib/dpkg/status',
               '-o', 'Dir::Log=' + str(cache / 'apt/log'),
               '-o', 'Acquire::http::Timeout=30', '-o', 'Acquire::https::Timeout=30',
               '-o', 'Acquire::Retries=2',
               '-o', 'APT::Update::Post-Invoke::=', '-o', 'APT::Update::Post-Invoke-Success::=',
               '-o', 'APT::Update::Error-Mode=any']
    for relative in ('apt/cache/archives/partial', 'apt/state/lists/partial', 'apt/log', 'packages'):
        inside(cache / relative).mkdir(parents=True, exist_ok=True)
    # Read the host's repository configuration, but keep indexes, downloads and logs private.
    config = inside(cache / 'apt.conf')
    config.write_text('Dir::Etc::parts "-";\nDir::Etc::main "-";\n', encoding='utf-8')
    apt_env = dict(os.environ, APT_CONFIG=str(config))
    subprocess.run(['apt-get', *options, 'update'], cwd=cache, env=apt_env, check=True)
    def known(name):
        result = subprocess.run(['apt-cache', *options, 'show', name], cwd=cache,
                                capture_output=True, text=True, env=apt_env)
        return result.returncode == 0 and 'Package: ' + name + '\n' in result.stdout
    glib = 'libglib2.0-0t64' if known('libglib2.0-0t64') else 'libglib2.0-0'
    packages = (*PACKAGES, glib)
    subprocess.run(['apt-get', *options, 'download', *packages], cwd=cache / 'packages', env=apt_env, check=True)
    prefix = inside(ROOT / '.venv/system-libs')
    prefix.mkdir(parents=True, exist_ok=True)
    receipts = []
    for archive in sorted((cache / 'packages').glob('*.deb')):
        fields = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package', 'Version'], text=True)
        subprocess.run(['dpkg-deb', '-x', str(archive), str(prefix)], check=True)
        # ELF symlinks must not redirect loader searches outside the private prefix.
        for link in prefix.rglob('*'):
            if link.is_symlink() and not link.resolve().is_relative_to(prefix.resolve()):
                raise ValueError('system library symlink leaves project: ' + str(link))
        receipts.append(dict(archive=archive.name, sha256=sha(archive), package=fields.strip()))
    save(ROOT / 'assets/runtime_libraries.json', dict(
        private_prefix=str(prefix), packages=receipts, changed_system_packages=False))
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, library_paths()))
               + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
    subprocess.run([sys.executable, '-B', '-m', 'pipeline.runtime_libraries', '--check'],
                   cwd=ROOT, env=env, check=True)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    prepare(parser.parse_args().check)
