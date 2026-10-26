"""Hash-verified independent cloud assets; never write to an original project."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import uuid

ROOT = Path('/root/msrc_safety_lab_20261023')
OLD_IMAGES = '/mnt/data/HLC/msrc_plugguard_splitv2_shared_gpu1_20261022/MSRC/assets/data/images/'


def checked(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('destination escapes independent root')
    return path


def image_target(root, old):
    if not isinstance(old, str) or not old.startswith(OLD_IMAGES):
        raise ValueError('unexpected source image root')
    relative = PurePosixPath(old[len(OLD_IMAGES):])
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('unsafe image identity')
    if not relative.parts or relative.parts[0] not in ('benign', 'mmsafety'):
        raise ValueError('unexpected safety image family')
    return checked(root, Path(root) / 'images' / str(relative))


def validate_inventory(files):
    if not files:
        raise ValueError('empty file inventory')
    for name, digest in files.items():
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name or
                not re.fullmatch('[a-f0-9]{64}', digest)):
            raise ValueError('unsafe inventory entry')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(root, name, data):
    p = checked(root, Path(root) / name)
    temp = checked(root, p.with_suffix(p.suffix + '.tmp'))
    p.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    checked(root, temp).replace(checked(root, p))


def fetch(root, url, path, digest):
    root = Path(root).resolve()
    path = checked(root, path)
    if path.exists():
        if sha(path) != digest:
            raise ValueError('existing asset hash mismatch: ' + str(path))
        return path
    part = checked(root, path.with_name(path.name + '.part'))
    if part.is_file() and sha(part) == digest:
        part.replace(checked(root, path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise RuntimeError('insufficient independent disk budget')
    print('FETCH ' + str(path.relative_to(root)), flush=True)
    command = ['curl', '--fail', '--location', '--silent', '--show-error',
               '--retry', '4', '--retry-all-errors', '--retry-delay', '3', '--connect-timeout', '20',
               '--max-time', '7200']
    downloaded = part
    try:
        subprocess.run(command + ['--continue-at', '-', '--output', str(part), url],
                       check=True, cwd=root)
    except subprocess.CalledProcessError as error:
        if error.returncode != 33:
            raise
        # Preserve the rejected range-resume bytes for audit, never truncate them.
        downloaded = checked(root, path.with_name(path.name + '.full-' + uuid.uuid4().hex + '.part'))
        if shutil.disk_usage(root).free < 15 * 1024**3:
            raise RuntimeError('insufficient independent disk budget for full retry')
        subprocess.run(command + ['--output', str(downloaded), url], check=True, cwd=root)
    if sha(checked(root, downloaded)) != digest:
        raise ValueError('download SHA256 mismatch: ' + str(path))
    checked(root, downloaded).replace(checked(root, path))
    print('VERIFIED ' + str(path.relative_to(root)), flush=True)
    return path


def prepare_models(root):
    spec = json.loads((root / 'snapshot/source/assets/plugguard/base_models.json').read_text())
    overrides = json.loads((root / 'inbox/model_overrides.json').read_text())
    jobs = []
    for role, model in spec.items():
        validate_inventory(model['files'])
        for filename, digest in model['files'].items():
            target = root / 'models' / role / filename
            if filename in overrides.get(role, {}):
                data = overrides[role][filename].encode()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('unapproved model metadata overlay')
                target = checked(root, target)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError('existing model metadata differs')
                target.write_bytes(data)
            else:
                url = 'https://hf-mirror.com/%s/resolve/%s/%s?download=true' % (
                    model['repo'], model['revision'], filename)
                jobs.append((url, target, digest))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda x: fetch(root, *x), jobs))
    save(root, 'models_ready.json', dict(verified_ns=time.time_ns(), models=spec))


def prepare_raw(root):
    lock = json.loads((root / 'inbox/data_sources.json').read_text())['mmsafety']
    inventory = {}
    for line in (root / 'inbox/inputs.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name.startswith('data/MM-SafetyBench/') and name.endswith('.parquet'):
            inventory[name.removeprefix('data/MM-SafetyBench/')] = digest
    validate_inventory(inventory)
    if len(inventory) != 39:
        raise ValueError('all 39 frozen raw safety files required')
    def download(item):
        name, digest = item
        url = 'https://hf-mirror.com/datasets/%s/resolve/%s/%s?download=true' % (
            lock['repo'], lock['revision'], name)
        return fetch(root, url, root / 'raw' / name, digest)
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, sorted(inventory.items())))
    save(root, 'raw_ready.json', dict(files=[str(p) for p in files], inventory=inventory))
    return files, inventory


def prepare_images(root):
    import pyarrow.parquet as pq
    from PIL import Image
    files, inventory = prepare_raw(root)
    for file in files:
        category, variant = file.parent.name, file.stem
        rows = pq.read_table(file, columns=['image'], use_threads=False).to_pylist()
        for index, row in enumerate(rows):
            dest = checked(root, root / 'images/mmsafety' / category /
                           ('%s_%04d.png' % (variant, index)))
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = row['image']
            raw = raw['bytes'] if isinstance(raw, dict) else raw
            with Image.open(io.BytesIO(raw)) as im:
                im.convert('RGB').save(dest, format='PNG')
    expected = json.loads((root / 'snapshot/input_hashes.json').read_text())['images']
    mapping = {}
    for old, digest in expected.items():
        new = image_target(root, old)
        if not new.is_file() or sha(new) != digest:
            raise ValueError('exact image identity mismatch: ' + old)
        mapping[old] = str(new)
    save(root, 'image_mapping.json', mapping)
    save(root, 'images_ready.json', dict(count=len(mapping), verified_ns=time.time_ns(),
         original_manifest_sha256=sha(root / 'snapshot/input_hashes.json'),
         raw_inventory=inventory, all_original_image_bytes_match=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['models', 'raw', 'images'])
    args = parser.parse_args()
    root = ROOT
    if root.resolve() != root or root.stat().st_uid != os.getuid():
        raise ValueError('unsafe cloud root')
    os.chdir(root)
    {'models': prepare_models, 'raw': prepare_raw, 'images': prepare_images}[args.phase](root)


if __name__ == '__main__':
    main()
