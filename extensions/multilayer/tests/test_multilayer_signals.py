import copy
import unittest

import numpy as np

from safety_lab.multilayer_signals import fit_directions, project, selected_layers


class MultilayerSignalTests(unittest.TestCase):
    def fixture(self):
        records, values = [], []
        for label in (0, 1):
            for i in range(3):
                for position in (8, 16):
                    records.append(dict(id=f'{label}-{i}', group=f'g{label}-{i}',
                                        role='fit', position=position, label=label,
                                        label_source='response_prefix_judge'))
                    vector = [-1.0 if label == 0 else 1.0, .1 * (i + 1)]
                    values.append([vector, vector, vector])
        return records, np.asarray(values)

    def test_fixed_depths_and_invalid_small_decoder(self):
        self.assertEqual(selected_layers(28), (14, 21, 28))
        self.assertEqual(selected_layers(31), (16, 24, 31))
        for value in (0, 2, True, 3.5):
            with self.assertRaises(ValueError):
                selected_layers(value)

    def test_fit_separates_supervised_direction_without_pca(self):
        records, x = self.fixture()
        model = fit_directions(records, x, min_groups=3)
        z = project(x, model)
        self.assertEqual(z.shape, (12, 3))
        self.assertTrue(np.isfinite(z).all())
        self.assertGreater(z[6:].mean(), z[:6].mean())
        self.assertEqual(model['fit_groups'], {'0': 3, '1': 3})
        self.assertGreaterEqual(model['hold_threshold'], 0)
        self.assertEqual(model['weight'].shape, (3, 2))

    def test_roles_duplicates_nonfinite_and_insufficient_groups_rejected(self):
        records, x = self.fixture()
        for role in ('calibration', 'test'):
            bad = copy.deepcopy(records)
            bad[-1]['role'] = role
            with self.assertRaisesRegex(ValueError, 'fit'):
                fit_directions(bad, x, min_groups=3)
        bad = copy.deepcopy(records)
        bad[-1] = dict(bad[-2])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fit_directions(bad, x, min_groups=3)
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            fit_directions(records, x, min_groups=3)
        records, x = self.fixture()
        with self.assertRaisesRegex(ValueError, 'groups'):
            fit_directions(records, x, min_groups=4)

    def test_request_weights_do_not_multiply_long_request(self):
        records, x = self.fixture()
        model = fit_directions(records, x, min_groups=3)
        # Repeat the identical two checkpoints with new positions for one request.
        extra = [dict(records[i], position=32 + i) for i in (0, 1)]
        duplicated = fit_directions(records + extra, np.concatenate([x, x[:2]]), min_groups=3)
        np.testing.assert_allclose(model['weight'], duplicated['weight'], rtol=1e-9)
        np.testing.assert_allclose(model['bias'], duplicated['bias'], rtol=1e-9)

    def test_constant_safe_variance_and_zero_vector_are_finite(self):
        records, x = self.fixture()
        x[:6] = 0
        model = fit_directions(records, x, min_groups=3)
        self.assertTrue(np.isfinite(project(x, model)).all())
        self.assertTrue(np.isfinite(model['weight']).all())

    def test_corrupt_shapes_labels_and_model_rejected(self):
        records, x = self.fixture()
        with self.assertRaises(ValueError):
            fit_directions(records, x[:, :2], min_groups=3)
        bad = copy.deepcopy(records)
        bad[0]['label'] = True
        with self.assertRaises(ValueError):
            fit_directions(bad, x, min_groups=3)
        model = fit_directions(records, x, min_groups=3)
        model['weight'][0, 0] = np.inf
        with self.assertRaises(ValueError):
            project(x, model)


if __name__ == '__main__':
    unittest.main()
