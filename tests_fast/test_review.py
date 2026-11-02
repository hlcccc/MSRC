import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from fast_review.common import ROOT, ece, indexed, relative_path
from fast_review.review import predict
from fast_review.finish import finish


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'tmp').mkdir(exist_ok=True)

    def test_ece_boundaries(self):
        self.assertEqual(ece([0, 1], [0, 1]), 0)
        self.assertEqual(ece([0, 0], [1, 1]), 1)

    def test_complete_cpu_imports(self):
        from references.m4c_evaluator import TextVQAAccuracyEvaluator
        from extensions.multilayer.runner import audit_events, parse_response_safety
        self.assertTrue(callable(audit_events))
        self.assertTrue(callable(TextVQAAccuracyEvaluator))

    def test_ece_rejects_bad_input(self):
        for labels, risks in [([], []), ([2], [.5]), ([1], [float('nan')]), ([1], [1.01])]:
            with self.subTest(labels=labels, risks=risks), self.assertRaises(ValueError):
                ece(labels, risks)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            indexed([{'id': 'a'}, {'id': 'a'}])

    def test_archive_paths(self):
        for name in ('../x', '/absolute', 'C:/windows', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative_path(ROOT, name)

    def test_missing_is_imputed_not_counted(self):
        from fast_review.common import SIGNALS
        state = dict(signals=SIGNALS, live=[True] + [False] * 7, fill=[.25] * 8,
                     mean=[0.], std=[1.], coef=[0., 1.], converged=True,
                     fitting_scope='development only')
        actual = predict(state, np.ones((1, 8)), np.zeros((1, 8), dtype=bool))
        self.assertAlmostEqual(float(actual[0]), 1 / (1 + np.exp(-.25)))

    def test_bad_training_scope_refused(self):
        with self.assertRaises(ValueError):
            predict(dict(signals=[], live=[True] * 8), np.ones((1, 8)), np.ones((1, 8), dtype=bool))

    def test_finish_cannot_skip_demo(self):
        from fast_review.common import save
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            save(root / 'review/summary.json', dict(format='msrc-fast-review-result-v1'))
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                finish(root, True)

    def test_wrong_archive_hash(self):
        from fast_review.prepare import restore
        from fast_review.common import sha
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            source = Path(directory) / 'bundle.tar.gz'
            source.write_bytes(b'not an archive')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                restore(source, Path(directory) / 'restored', '0' * 64)

    def test_symlink_archive_refused(self):
        from fast_review.prepare import restore
        from fast_review.common import sha
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            source = Path(directory) / 'bundle.tar.gz'
            with tarfile.open(source, 'w:gz') as archive:
                member = tarfile.TarInfo('escape')
                member.type, member.linkname = tarfile.SYMTYPE, '/etc/passwd'
                archive.addfile(member)
            with self.assertRaisesRegex(ValueError, 'regular files'):
                restore(source, Path(directory) / 'restored', sha(source))


if __name__ == '__main__':
    (ROOT / 'tmp').mkdir(exist_ok=True)
    unittest.main()
