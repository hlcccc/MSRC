"""Deliver processed benchmark inputs and the audited, portable fixed partitions."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

from pipeline import common
from pipeline.common import inside, read_rows, save, sha
from pipeline.public_assets import expected_inputs, recipe_root

FORMAT = 'msrc-prepared-data-v1'
NOTICE = 'prepared_NOTICE.md'
REGISTRY = 'prepared_partitions.sha256'
MAX_FILE_BYTES = 256 * 1024**2


def partition_inventory():
    result = {}
    for line in (recipe_root() / REGISTRY).read_text(encoding='utf-8').splitlines():
        digest, separator, name = line.partition('  ')
        if (separator != '  ' or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or not name.startswith('partitions/') or name in result
                or Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name):
            raise ValueError('invalid prepared partition registry')
        result[name] = digest
    if len(result) != 10:
        raise ValueError('all ten fixed partition files are required')
    return result


def inventory():
    return {**expected_inputs(), **partition_inventory()}


def recipe_identity():
    return {name: hashlib.sha256((recipe_root() / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in
            ('inputs.sha256', REGISTRY, NOTICE)}


def regular(path):
    path = Path(os.path.abspath(Path(path).expanduser()))
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('regular non-symlink file required: %s' % path)
    return path


def materialize(package, assets):
    """Join browser-sized transport parts automatically, without changing data."""
    # Normalize the documented ../prepared_data path without following symlinks.
    package = Path(os.path.abspath(Path(package).expanduser()))
    if not package.is_dir():
        return regular(package)
    if package.resolve() != package or package.is_symlink():
        raise ValueError('regular prepared package directory required')
    description = regular(package / 'parts.json')
    if description.stat().st_size > 64 * 1024:
        raise ValueError('prepared transport manifest too large')
    value = json.loads(description.read_text(encoding='utf-8'))
    name, parts = value.get('archive'), value.get('parts')
    if (value.get('format') != 'msrc-prepared-transport-v1'
            or not isinstance(name, str) or Path(name).name != name
            or '\\' in name or not name.endswith('.tar')
            or not isinstance(parts, list) or not 1 <= len(parts) <= 16):
        raise ValueError('invalid prepared transport manifest')
    for index, part in enumerate(parts, 1):
        expected = name + '.%03d' % index
        if not isinstance(part, dict) or part.get('name') != expected:
            raise ValueError('prepared transport parts must be contiguous and ordered')
        path = regular(package / expected)
        if (type(part.get('bytes')) is not int or not 0 < part['bytes'] <= 2 * 1024**3
                or path.stat().st_size != part['bytes'] or sha(path) != part.get('sha256')):
            raise ValueError('prepared transport part missing or corrupt: ' + expected)
    target = inside(assets / 'downloads/prepared' / name)
    if target.exists():
        if sha(regular(target)) != value.get('sha256'):
            raise ValueError('assembled prepared archive changed')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.joining')
    with temporary.open('xb') as output:
        for part in parts:
            with regular(package / part['name']).open('rb') as source:
                shutil.copyfileobj(source, output, length=1 << 20)
    if sha(temporary) != value.get('sha256'):
        raise ValueError('assembled prepared archive hash mismatch')
    temporary.rename(target)
    print('PREPARED_PARTS_OK: %d parts joined automatically' % len(parts), flush=True)
    return target


def export_partitions(assets, output):
    """Publisher-only operation; use the unchanged acceptance preparation methods."""
    from pipeline.prepare_assets import write_config
    assets, output = inside(assets), inside(output)
    if (assets / 'partitions').exists() or output.exists():
        raise ValueError('new partition/output directories required')
    write_config(assets, '0')
    from pipeline import factual, safety
    from pipeline.factual_parameters import identity
    from pipeline.plugguard_report import assign_roles, checkpoint_identity
    # Imports normally read the same configuration at process start.
    factual.INPUTS = assets / 'data/factual'
    output.mkdir(parents=True)
    (output / 'factual').mkdir()
    (output / 'safety').mkdir()
    factual.prepare(output / 'factual', 'full')
    safety_rows = safety.prepare(output / 'safety', backend='plugguard')
    image_hashes = json.loads((output / 'safety/input_hashes.json').read_text())['images']
    roles = assign_roles(safety_rows, image_hashes)
    role_by_id = {rid: role for role, ids in roles.items() for rid in ids}
    summary = dict(format=FORMAT, seed=common.SEED, contains_predictions=False,
                   contains_fitted_parameters=False, factual={})

    def write(name, rows):
        target = inside(assets / 'partitions' / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                value = dict(row, image=Path(os.path.relpath(row['image'], target.parent)).as_posix())
                common.emit(handle, value)

    for dataset in ('pope', 'textvqa'):
        rows = read_rows(output / 'factual/inputs' / (dataset + '.jsonl'))
        write('factual/' + dataset + '.jsonl', rows)
        for split, name in [('dev', 'train'), ('test', 'test')]:
            write('factual/' + dataset + '_' + name + '.jsonl',
                  [row for row in rows if row['acceptance_split'] == split])
        summary['factual'][dataset] = dict(identity=identity(rows), total=len(rows),
            train=sum(row['acceptance_split'] == 'dev' for row in rows),
            test=sum(row['acceptance_split'] == 'test' for row in rows))
    safety_rows = [dict(row, role=role_by_id[row['id']]) for row in safety_rows]
    write('safety/all.jsonl', safety_rows)
    for split, name in [('dev', 'train'), ('test', 'test')]:
        write('safety/' + name + '.jsonl', [row for row in safety_rows if row['split'] == split])
    from pipeline.safety_split import validate_partition
    summary['safety'] = dict(total=6720, train=3360, test=3360,
        roles={role: len(ids) for role, ids in roles.items()},
        identity=checkpoint_identity(safety_rows, image_hashes),
        integrity=validate_partition(safety_rows, image_hashes, roles))
    save(assets / 'partitions/summary.json', summary)
    result = {path.relative_to(assets).as_posix(): sha(path)
              for path in sorted((assets / 'partitions').rglob('*')) if path.is_file()}
    print('FIXED_PARTITIONS_EXPORTED ' + json.dumps(summary, ensure_ascii=True), flush=True)
    return result


def verify_partitions(assets):
    """Check delivered halves against the unchanged deterministic split algorithms."""
    import numpy as np
    from evaluation.groups import image_group_id
    from evaluation.split import grouped_split
    from pipeline.factual_parameters import identity
    from pipeline.plugguard_report import assign_roles, checkpoint_identity
    from pipeline.safety_split import split_rows, validate_partition
    assets = Path(assets)
    root = assets / 'partitions'
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if (summary.get('format') != FORMAT or summary.get('seed') != common.SEED
            or summary.get('contains_predictions') is not False
            or summary.get('contains_fitted_parameters') is not False):
        raise ValueError('invalid fixed partition summary')
    for dataset, counts in [('pope', (4500, 4500)), ('textvqa', (2504, 2496))]:
        rows = read_rows(root / 'factual' / (dataset + '.jsonl'))
        source = read_rows(assets / 'data/factual' / (dataset + '.jsonl'))
        if (len(rows) != sum(counts) or len(source) != len(rows)
                or len({row['id'] for row in rows}) != len(rows)):
            raise ValueError('incomplete factual partition: ' + dataset)
        groups = np.asarray([image_group_id(row) for row in source])
        dev, test = grouped_split(groups, seed=common.SEED, dev_fraction=.5)
        expected = [dict(row, acceptance_split='dev' if dev[i] else 'test',
                         acceptance_image_group=str(groups[i])) for i, row in enumerate(source)]
        if rows != expected or (int(dev.sum()), int(test.sum())) != counts:
            raise ValueError('factual fixed partition differs from audited method')
        for split, suffix in [('dev', 'train'), ('test', 'test')]:
            if read_rows(root / 'factual' / (dataset + '_' + suffix + '.jsonl')) != [
                    row for row in rows if row['acceptance_split'] == split]:
                raise ValueError('factual train/test file differs from fixed partition')
        if summary['factual'][dataset] != dict(identity=identity(rows), total=len(rows),
                                               train=counts[0], test=counts[1]):
            raise ValueError('factual image/label/split identity changed')
    rows = read_rows(root / 'safety/all.jsonl')
    image_hashes = {row['image']: sha(row['image']) for row in rows}
    roles = assign_roles(rows, image_hashes)
    expected = split_rows(rows, image_hashes, common.SEED)
    if rows != expected or any(row['id'] not in roles.get(row['role'], set()) for row in rows):
        raise ValueError('safety fixed split/role differs from audited method')
    for split, suffix in [('dev', 'train'), ('test', 'test')]:
        if read_rows(root / 'safety' / (suffix + '.jsonl')) != [row for row in rows if row['split'] == split]:
            raise ValueError('safety train/test file differs from fixed partition')
    actual = dict(total=len(rows), train=sum(row['split'] == 'dev' for row in rows),
        test=sum(row['split'] == 'test' for row in rows),
        roles={role: len(ids) for role, ids in roles.items()},
        identity=checkpoint_identity(rows, image_hashes),
        integrity=validate_partition(rows, image_hashes, roles))
    if summary['safety'] != actual:
        raise ValueError('safety image/label/split identity changed')
    print('FIXED_SPLITS_OK: POPE=4500/4500 TextVQA=2504/2496 safety=3360/3360; overlap=0', flush=True)
    return summary


def pack(assets, package):
    assets, package = Path(assets).resolve(strict=True), inside(package)
    verify_partitions(assets)
    records = {}
    for name, digest in sorted(inventory().items()):
        path = regular(assets / name)
        if (not path.is_relative_to(assets) or path.stat().st_size > MAX_FILE_BYTES or sha(path) != digest):
            raise ValueError('prepared input differs from pinned recipe: ' + name)
        records[name] = dict(sha256=digest, bytes=path.stat().st_size)
    manifest = dict(format=FORMAT, recipe=recipe_identity(), files=records,
                    contains_fitted_parameters=False, contains_test_predictions=False,
                    contains_base_model_weights=False, contains_fixed_partitions=True)
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, 'x', format=tarfile.PAX_FORMAT) as archive:
        def add(name, content):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
        add('prepared_bundle.json', json.dumps(manifest, sort_keys=True, indent=2).encode())
        add(NOTICE, (recipe_root() / NOTICE).read_bytes())
        for name, record in sorted(records.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = record['bytes'], 0o644, 0
            with regular(assets / name).open('rb') as handle:
                archive.addfile(info, handle)
    print('PREPARED_BUNDLE_CREATED: %d files; %s' % (len(records), package), flush=True)
    return manifest


def inspect(archive):
    members = archive.getmembers()
    names = [member.name for member in members]
    expected = inventory()
    if (len(names) != len(set(names)) or set(names) != set(expected) | {'prepared_bundle.json', NOTICE}
            or any(not member.isfile() or not 0 <= member.size <= MAX_FILE_BYTES for member in members)):
        raise ValueError('unsafe, duplicate or unapproved prepared bundle members')
    if archive.getmember('prepared_bundle.json').size > 4 * 1024**2:
        raise ValueError('prepared bundle manifest too large')
    with archive.extractfile('prepared_bundle.json') as handle:
        manifest = json.load(handle)
    if (manifest.get('format') != FORMAT or manifest.get('recipe') != recipe_identity()
            or manifest.get('contains_fixed_partitions') is not True
            or any(manifest.get(key) is not False for key in (
                'contains_fitted_parameters', 'contains_test_predictions', 'contains_base_model_weights'))
            or not isinstance(manifest.get('files'), dict) or set(manifest['files']) != set(expected)):
        raise ValueError('prepared bundle belongs to a different method or contains results/weights')
    with archive.extractfile(NOTICE) as handle:
        if handle.read() != (recipe_root() / NOTICE).read_bytes():
            raise ValueError('source/license notice changed')
    for name, digest in sorted(expected.items()):
        member, record = archive.getmember(name), manifest['files'][name]
        if record != dict(sha256=digest, bytes=member.size):
            raise ValueError('prepared inventory differs from pinned recipe: ' + name)
        actual = hashlib.sha256()
        with archive.extractfile(member) as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                actual.update(block)
        if actual.hexdigest() != digest:
            raise ValueError('prepared file hash mismatch: ' + name)
    return manifest


def restore(package, assets):
    assets = inside(assets)
    package = materialize(package, assets)
    receipt_path = assets / 'prepared_bundle.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('archive_sha256') != sha(package):
            raise ValueError('existing data belongs to another prepared archive')
        verify(assets)
        return receipt
    if any((assets / name).exists() for name in ('data', 'raw', 'partitions', 'prepared_bundle.json', 'MANIFEST.sha256')):
        raise ValueError('fresh data destination required; existing data are not overwritten')
    with tarfile.open(package, 'r:') as archive:
        manifest = inspect(archive)
        for name in sorted(set(manifest['files']) | {NOTICE}):
            target = inside(assets / name)
            if not target.is_relative_to(assets):
                raise ValueError('prepared destination escapes assets')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(name) as source, target.open('xb') as handle:
                shutil.copyfileobj(source, handle, length=1 << 20)
    lines = [record['sha256'] + '  ' + name for name, record in sorted(manifest['files'].items())]
    (assets / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    summary = verify_partitions(assets)
    receipt = dict(format=FORMAT, recipe=recipe_identity(), archive_sha256=sha(package),
        files=len(manifest['files']), fixed_partitions_restored=True,
        raw_data_processing_performed=False, predictions_restored=False,
        fitted_parameters_restored=False, partitions=summary)
    save(assets / 'prepared_bundle.json', receipt)
    print('PREPARED_DATA_OK: %d verified input/partition files; no raw preprocessing' % receipt['files'], flush=True)
    return receipt


def verify(assets):
    assets = Path(assets)
    receipt = json.loads((assets / 'prepared_bundle.json').read_text(encoding='utf-8'))
    if receipt.get('format') != FORMAT or receipt.get('recipe') != recipe_identity():
        raise ValueError('prepared data receipt belongs to a different recipe')
    expected = inventory()
    actual = {path.relative_to(assets).as_posix() for sub in ('data', 'raw', 'partitions')
              for path in (assets / sub).rglob('*') if path.is_file()}
    if actual != set(expected):
        raise ValueError('extra/missing prepared inputs or partition files')
    for name, digest in expected.items():
        if sha(regular(assets / name)) != digest:
            raise ValueError('prepared data changed: ' + name)
    summary = verify_partitions(assets)
    if receipt.get('partitions') != summary:
        raise ValueError('installed fixed partitions differ from receipt')
    print('PREPARED_DATA_OK: %d verified input/partition files' % len(expected), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pack', 'restore', 'verify', 'export-partitions'))
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    common.initialize()
    if args.command == 'pack':
        if args.package is None:
            parser.error('pack requires --package')
        pack(args.assets, args.package)
    elif args.command == 'restore':
        if args.package is None:
            parser.error('restore requires --package')
        restore(args.package, args.assets)
    elif args.command == 'verify':
        verify(args.assets)
    else:
        if args.out is None:
            parser.error('export-partitions requires --out')
        result = export_partitions(args.assets, args.out)
        print('\n'.join(digest + '  ' + name for name, digest in sorted(result.items())), flush=True)


if __name__ == '__main__':
    main()
