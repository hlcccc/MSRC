"""Maintainer-only: hash staged netdisk payloads into Linux-compatible tables."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def payload(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts
            or any(char in name for char in '\\\r\n:')
            or relative.as_posix() != name):
        raise ValueError('unsafe payload path: ' + name)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('payload escapes delivery directory: ' + name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def table(root, names):
    if len(names) != len(set(names)):
        raise ValueError('duplicate payload path')
    lines = [digest(payload(root, name)) + '  ' + name for name in sorted(names)]
    return ('\n'.join(lines) + '\n').encode('utf-8')


def build(root):
    root = root.resolve(strict=True)
    recipe = payload(root, 'prepared_data/parts.json')
    parts = json.loads(recipe.read_text(encoding='utf-8'))
    if parts.get('format') != 'msrc-prepared-transport-v1' or not parts.get('parts'):
        raise ValueError('unsupported or empty prepared-data recipe')
    names = ['MSRC-code-netdisk-20261030.zip', 'prepared_data/parts.json']
    for part in parts['parts']:
        name = part['name']
        if PurePosixPath(name).name != name:
            raise ValueError('part name must be a basename')
        names.append('prepared_data/' + name)
    outputs = {root / 'SHA256SUMS.txt': table(root, names)}
    if (root / 'fitted_only').exists():
        fitted = (root / 'fitted_only').resolve()
        if not fitted.is_relative_to(root):
            raise ValueError('fitted directory escapes delivery directory')
        outputs[fitted / 'SHA256SUMS.txt'] = table(fitted, ['parameters-20261023.tar.gz'])
    for destination in outputs:
        if destination.is_symlink():
            raise ValueError('checksum destination must not be a symlink')
    # Binary writes bypass Windows newline translation and never add a BOM.
    for destination, content in outputs.items():
        destination.write_bytes(content)
    return [str(path) for path in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    for directory in args.directories:
        for output in build(directory):
            print('CHECKSUM_LF_OK ' + output)


if __name__ == '__main__':
    main()
