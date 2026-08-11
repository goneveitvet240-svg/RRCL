import unittest

from vff_baselines import paleologu_vff_factor


class PaleologuVFFTest(unittest.TestCase):
    def test_uses_absolute_memory_at_noise_floor(self):
        self.assertEqual(
            paleologu_vff_factor(1.0, 1.0, 1.0, gamma=1.5),
            1.0,
        )

    def test_forgets_after_large_innovation(self):
        factor = paleologu_vff_factor(100.0, 1.0, 1.0, gamma=1.5)
        self.assertLess(factor, 1.0)
        self.assertGreaterEqual(factor, 0.05)


if __name__ == "__main__":
    unittest.main()
