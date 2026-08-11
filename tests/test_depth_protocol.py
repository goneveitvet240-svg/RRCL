import unittest

import numpy as np

from datasets_depth import _depth_to_patches


class DepthProtocolTest(unittest.TestCase):
    def test_invalid_zero_depth_does_not_bias_patch_mean(self):
        depth = np.zeros((28, 28), dtype=np.float32)
        depth[:, :14] = 2.0
        values, valid = _depth_to_patches(
            depth, img_size=28, n_patches=4, min_valid_fraction=0.49
        )
        self.assertEqual(values.shape, (4,))
        self.assertEqual(valid.shape, (4,))
        self.assertTrue(np.allclose(values[valid], 2.0, atol=1e-5))

    def test_patch_with_too_little_valid_depth_is_removed(self):
        depth = np.zeros((28, 28), dtype=np.float32)
        depth[:2, :2] = 3.0
        _, valid = _depth_to_patches(
            depth, img_size=28, n_patches=4, min_valid_fraction=0.5
        )
        self.assertFalse(np.any(valid))


if __name__ == "__main__":
    unittest.main()
