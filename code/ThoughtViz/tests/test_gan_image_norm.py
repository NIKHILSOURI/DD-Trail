"""Unit tests for GAN image range helpers (stdlib only; no pytest required)."""
import unittest

import numpy as np

from utils.gan_image_norm import batch_to_preview_uint8, tanh_to_rgb_unit, tensor_to_image_uint8
from utils.image_utils import combine_rgb_preview_grid


class TestGanImageNorm(unittest.TestCase):
    def test_tanh_to_rgb_unit(self):
        self.assertTrue(
            np.allclose(tanh_to_rgb_unit(np.array([-1.0, 0.0, 1.0])), [0.0, 0.5, 1.0])
        )

    def test_tensor_to_image_uint8_extremes(self):
        u = tensor_to_image_uint8(np.full((2, 2, 3), -1.0))
        self.assertEqual(u.shape, (2, 2, 3))
        self.assertEqual(int(u.min()), 0)
        u2 = tensor_to_image_uint8(np.full((2, 2, 3), 1.0))
        self.assertEqual(int(u2.max()), 255)

    def test_batch_preview_vectorized(self):
        b = np.array([[[[-1.0, 0.0, 1.0]]]], dtype=np.float32)
        out = batch_to_preview_uint8(b, from_tanh=True)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, b.shape)

    def test_combine_rgb_preview_grid_uint8(self):
        x = np.zeros((2, 64, 64, 3), dtype=np.float32)
        grid = combine_rgb_preview_grid(x, from_tanh=True)
        self.assertEqual(grid.dtype, np.uint8)
        self.assertEqual(grid.shape[-1], 3)
        self.assertGreater(grid.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
