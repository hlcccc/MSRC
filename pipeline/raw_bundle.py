"""Transport hash-pinned original inputs, never processed rows or fitted state."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from pipeline import common
from pipeline.common import inside, sha
from pipeline.public_assets import expected_inputs, recipe_root

FORMAT = 'msrc-original-inputs-v1'
GROUPS = ('textvqa', 'mmsafety', 'benign_source')
MAX_FILE_BYTES = 256 * 1024**2
MAX_MANIFEST_BYTES = 2 * 1024**2
NOTICE = 'NOTICE.md'


def inventory():
    """Only original, byte-identical public files are eligible for this mirror."""
    frozen = expected_inputs()
    groups = {name: {} for name in GROUPS}
    for name, digest in frozen.items():
        if name == 'raw/textvqa/TextVQA_0.5.1_val.json':
            groups['textvqa']['TextVQA_0.5.1_val.json'] = digest
        elif name.startswith('data/images/textvqa/'):
            groups['textvqa']['textvqa_images/' + name.rsplit('/', 1)[1]] = digest
        elif name.startswith('data/MM-SafetyBench/') and name.endswith('.parquet'):
            groups['mmsafety'][name.removeprefix('data/')] = digest
    benign = json.loads((recipe_root() / 'benign.expected.json').read_text(encoding='utf-8'))
    groups['benign_source']['alpaca.parquet'] = benign['alpaca_sha256']
    return groups


def recipe_identity():
    return {name: hashlib.sha256((recipe_root() / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in
            ('data_sources.json', 'inputs.sha256', 'benign.expected.json', 'raw_bundle.NOTICE.md')}


def selected_inventory(selected):
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(name, str) or name not in GROUPS for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('invalid raw bundle groups')
    groups = inventory()
    return {name: digest for group in selected for name, digest in groups[group].items()}


def regular_file(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('regular, non-symlink source file required: %s' % path)
    return path


def pack(source, output, selected=GROUPS):
    source = Path(source).expanduser().resolve(strict=True)
    output = inside(output)
    files = selected_inventory(list(selected))
    records = {}
    for name, digest in sorted(files.items()):
        path = regular_file(source / name)
        if (not path.is_relative_to(source) or path.stat().st_size > MAX_FILE_BYTES
                or sha(path) != digest):
            raise ValueError('original input differs from pinned recipe: %s' % name)
        records[name] = dict(sha256=digest, bytes=path.stat().st_size)
    manifest = dict(format=FORMAT, groups=list(selected), recipe=recipe_identity(), files=records,
                    contains_fitted_parameters=False, contains_test_predictions=False,
                    contains_processed_dataset=False, contains_base_model_weights=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive uncompressed TAR avoids recompressing JPEG/parquet and is deterministic.
    with tarfile.open(output, 'x', format=tarfile.PAX_FORMAT) as archive:
        def add(name, content):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
        add('bundle.json', json.dumps(manifest, sort_keys=True, indent=2).encode('utf-8'))
        add(NOTICE, (recipe_root() / 'raw_bundle.NOTICE.md').read_bytes())
        for name, record in sorted(records.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = record['bytes'], 0o644, 0
            with regular_file(source / name).open('rb') as handle:
                archive.addfile(info, handle)
    inspect(output)
    print('RAW_BUNDLE_CREATED: %d original files; %s' % (len(files), output), flush=True)
    return manifest


def _inspect(archive):
    members = archive.getmembers()
    names = [member.name for member in members]
    if (len(names) > 4000 or len(names) != len(set(names))
            or any(not member.isfile() or not 0 <= member.size <= MAX_FILE_BYTES for member in members)
            or 'bundle.json' not in names or NOTICE not in names):
        raise ValueError('unsafe or duplicate raw bundle members')
    member = archive.getmember('bundle.json')
    if member.size > MAX_MANIFEST_BYTES:
        raise ValueError('raw bundle manifest too large')
    with archive.extractfile(member) as handle:
        manifest = json.load(handle)
    if (manifest.get('format') != FORMAT or manifest.get('recipe') != recipe_identity()
            or any(manifest.get(name) is not False for name in (
                'contains_fitted_parameters', 'contains_test_predictions',
                'contains_processed_dataset', 'contains_base_model_weights'))):
        raise ValueError('raw bundle belongs to a different recipe or contains non-raw material')
    expected = selected_inventory(manifest.get('groups'))
    records = manifest.get('files')
    if (not isinstance(records, dict) or set(records) != set(expected)
            or set(names) != set(expected) | {'bundle.json', NOTICE}):
        raise ValueError('raw bundle has missing/unapproved original files')
    notice = (recipe_root() / 'raw_bundle.NOTICE.md').read_bytes()
    with archive.extractfile(NOTICE) as handle:
        if handle.read(len(notice) + 1) != notice:
            raise ValueError('original source/license notices must be retained')
    for name, digest in sorted(expected.items()):
        record = records[name]
        member = archive.getmember(name)
        if (not isinstance(record, dict) or record.get('sha256') != digest
                or type(record.get('bytes')) is not int or record['bytes'] != member.size):
            raise ValueError('raw bundle inventory differs from pinned recipe: %s' % name)
        actual = hashlib.sha256()
        with archive.extractfile(member) as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                actual.update(block)
        if actual.hexdigest() != digest:
            raise ValueError('raw bundle file hash mismatch: %s' % name)
    return manifest


def inspect(package):
    with tarfile.open(regular_file(package), 'r:') as archive:
        manifest = _inspect(archive)
    print('RAW_BUNDLE_OK: %d original files' % len(manifest['files']), flush=True)
    return manifest


def restore(package, assets):
    package = regular_file(package)
    assets = inside(assets)
    target = inside(assets / 'downloads')
    if any((assets / name).exists() for name in ('downloads', 'raw', 'data')):
        raise ValueError('fresh data destination required; existing inputs are not overwritten')
    # Validate every payload before creating a destination; never use extractall.
    with tarfile.open(package, 'r:') as archive:
        manifest = _inspect(archive)
        for name, record in sorted(manifest['files'].items()):
            destination = inside(target / name)
            if not destination.is_relative_to(target):
                raise ValueError('raw bundle path escapes download cache')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(name) as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            if sha(destination) != record['sha256']:
                raise ValueError('restored raw input changed: %s' % name)
    receipt = dict(archive_sha256=sha(package), format=FORMAT, groups=manifest['groups'],
                   files=len(manifest['files']), recipe=manifest['recipe'],
                   processed_inputs_restored=False, fitted_parameters_restored=False,
                   predictions_restored=False)
    common.save(assets / 'raw_bundle_receipt.json', receipt)
    print('RAW_BUNDLE_RESTORED: %d original files' % receipt['files'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pack', 'verify'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--source-downloads', type=Path)
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    if args.command == 'pack':
        if args.source_downloads is None:
            parser.error('pack requires --source-downloads')
        pack(args.source_downloads, args.package, args.groups)
    else:
        inspect(args.package)


if __name__ == '__main__':
    main()
