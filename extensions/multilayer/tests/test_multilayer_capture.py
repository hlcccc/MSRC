from types import SimpleNamespace
import unittest

import torch

from safety_lab.multilayer_capture import LayerCapture


class Block(torch.nn.Module):
    def forward(self, x):
        return (x + 1,)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(4)])
        self.norm = torch.nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return self.norm(x)


class LayerCaptureTests(unittest.TestCase):
    def test_preserves_outputs_and_captures_only_current_vectors(self):
        decoder = Decoder()
        x = torch.zeros(1, 5, 2)
        with LayerCapture(decoder, (2, 3, 4)) as capture:
            capture.start(prefill=True)
            output = decoder(x)
            values, final = capture.result()
            self.assertEqual(values.shape, (3, 2))
            self.assertEqual(final.shape, (1, 5, 2))
            self.assertTrue(torch.equal(output, x + 4))
            self.assertEqual(values[:, 0].tolist(), [2., 3., 4.])
            capture.start(prefill=False)
            decoder(x)
            self.assertEqual(capture.result()[1].shape, (1, 1, 2))
        self.assertTrue(all(not layer._forward_hooks for layer in decoder.layers))
        self.assertFalse(decoder.norm._forward_hooks)

    def test_missing_or_stale_forward_cannot_be_consumed(self):
        decoder = Decoder()
        with LayerCapture(decoder, (2, 3, 4)) as capture:
            capture.start(prefill=False)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                capture.result()
            decoder(torch.zeros(1, 1, 2))
            capture.result()
            capture.start(prefill=False)
            decoder.layers[1](torch.zeros(1, 1, 2))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                capture.result()

    def test_hooks_removed_after_exception(self):
        decoder = Decoder()
        with self.assertRaises(RuntimeError):
            with LayerCapture(decoder, (2, 3, 4)):
                raise RuntimeError('synthetic failure')
        self.assertFalse(decoder.norm._forward_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in decoder.layers))

    def test_final_normalization_is_distinct_from_last_block(self):
        decoder = Decoder()
        decoder.norm = torch.nn.LayerNorm(2)
        with LayerCapture(decoder, (2, 3, 4)) as capture:
            capture.start(prefill=False)
            decoder(torch.tensor([[[0., 1.]]]))
            values, final = capture.result()
            self.assertEqual(values[-1].tolist(), [4., 5.])
            self.assertNotEqual(final[0, 0].tolist(), values[-1].tolist())


if __name__ == '__main__':
    unittest.main()
