"""The Windows release builder must emit tables accepted by GNU sha256sum."""
import hashlib
import json
import shutil
import subprocess

import pytest

from scripts import build_netdisk_checksums as checksums


def staged(tmp_path, fitted=False):
    root = tmp_path / ('MSRC_fitted' if fitted else 'MSRC_no_fit')
    prepared = root / 'prepared_data'
    prepared.mkdir(parents=True)
    (root / 'MSRC-code-netdisk-20261030.zip').write_bytes(b'code archive')
    (prepared / 'data.tar.001').write_bytes(b'prepared data')
    (prepared / 'parts.json').write_text(json.dumps(dict(
        format='msrc-prepared-transport-v1', parts=[dict(name='data.tar.001')])))
    if fitted:
        (root / 'fitted_only').mkdir()
        (root / 'fitted_only/parameters-20261023.tar.gz').write_bytes(b'parameters')
    return root


@pytest.mark.parametrize('fitted', [False, True])
def test_tables_are_lf_without_bom_and_contain_real_payload_hashes(tmp_path, fitted):
    root = staged(tmp_path, fitted)
    (root / 'SHA256SUMS.txt').write_bytes(b'old windows table\r\n')
    outputs = checksums.build(root)
    assert len(outputs) == (2 if fitted else 1)
    for output in outputs:
        path = checksums.Path(output)
        content = path.read_bytes()
        assert b'\r' not in content and not content.startswith(b'\xef\xbb\xbf')
        assert content.endswith(b'\n')
        for line in content.decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            assert digest == hashlib.sha256((path.parent / name).read_bytes()).hexdigest()


@pytest.mark.skipif(not shutil.which('sha256sum'), reason='GNU sha256sum not installed')
def test_native_linux_command_accepts_generated_tables(tmp_path):
    root = staged(tmp_path, fitted=True)
    for output in checksums.build(root):
        path = checksums.Path(output)
        result = subprocess.run(['sha256sum', '-c', path.name], cwd=path.parent,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside', 'data\\bad', 'data\rname'])
def test_unsafe_names_cannot_enter_a_checksum_table(tmp_path, name):
    with pytest.raises(ValueError, match='unsafe payload'):
        checksums.payload(tmp_path.resolve(), name)


def test_missing_parameter_does_not_partially_replace_tables(tmp_path):
    root = staged(tmp_path, fitted=True)
    target = root / 'SHA256SUMS.txt'
    target.write_bytes(b'previous release\n')
    (root / 'fitted_only/parameters-20261023.tar.gz').unlink()
    with pytest.raises(FileNotFoundError):
        checksums.build(root)
    assert target.read_bytes() == b'previous release\n'
