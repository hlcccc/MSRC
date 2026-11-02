"""Verify a release archive before extracting regular files to a new directory."""
import argparse
import tarfile
from pathlib import Path

from fast_review.common import load, output, relative_path, require, sha, verify_bundle


def restore(archive, root, digest):
    archive, root = Path(archive).expanduser().resolve(strict=True), output(root)
    require(len(digest) == 64 and sha(archive) == digest, 'release archive SHA256 mismatch')
    require(not root.exists(), 'refusing to overwrite evidence directory')
    with tarfile.open(archive, 'r:gz') as tar:
        members, seen, total = tar.getmembers(), set(), 0
        for member in members:
            relative_path(root, member.name)
            require(member.isfile() and member.name not in seen, 'only unique regular files allowed')
            seen.add(member.name)
            total += member.size
        require(total < 4 * 1024**3, 'unreasonable evidence archive size')
        root.mkdir(parents=True)
        for member in members:
            target = relative_path(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as destination:
                while block := source.read(1 << 20):
                    destination.write(block)
    verify_bundle(root)
    print('EVIDENCE_OK ' + str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('delivery/MSRC-fast-evidence-20261102.tar.gz'))
    parser.add_argument('--release', type=Path, default=Path('resources/fast_review/release.json'))
    parser.add_argument('--out', type=Path, default=Path('assets/fast_review'))
    args = parser.parse_args()
    release = load(args.release)
    require(release['format'] == 'msrc-fast-review-release-v1', 'unknown release')
    if args.out.exists():
        require(sha(args.archive) == release['archive_sha256'], 'release archive SHA256 mismatch')
        verify_bundle(args.out, release['manifest_sha256'])
        print('EVIDENCE_OK ' + str(args.out))
    else:
        restore(args.archive, args.out, release['archive_sha256'])
    verify_bundle(args.out, release['manifest_sha256'])


if __name__ == '__main__':
    main()
