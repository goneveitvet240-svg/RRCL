import unittest

import numpy as np

from analytic_forgetting_baselines import SIFtRLS, dos_elm_factor_update


class DOSFactorUpdateTest(unittest.TestCase):
    def test_degradation_reduces_factor_and_uses_squared_statistic_weight(self):
        update = dos_elm_factor_update(0.8, previous_score=0.9, current_score=0.4)
        self.assertLess(update.factor, 0.8)
        self.assertAlmostEqual(update.statistic_factor, update.factor**2)

    def test_improvement_increases_factor_but_caps_at_one(self):
        update = dos_elm_factor_update(0.99, previous_score=0.1, current_score=100.0)
        self.assertEqual(update.factor, 1.0)

    def test_one_step_change_is_bounded_by_point_one(self):
        update = dos_elm_factor_update(0.5, previous_score=1e100, current_score=-1e100)
        self.assertLessEqual(abs(update.factor - 0.5), 0.1 + 1e-12)

    def test_nonfinite_score_is_rejected(self):
        with self.assertRaises(ValueError):
            dos_elm_factor_update(1.0, previous_score=np.nan, current_score=0.0)


class SIFtRLSTest(unittest.TestCase):
    def test_full_rank_measurement_equals_uniform_exponential_forgetting(self):
        X = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        y = np.array([[1.0], [2.0], [3.0]])
        ridge = 4.0
        factor = 0.7
        model = SIFtRLS(
            2, ridge=ridge, forgetting=factor, epsilon=1e-12, bias=False
        )

        info = model.update(X, y)
        expected_R = factor * ridge * np.eye(2) + X.T @ X
        expected_W = np.linalg.solve(expected_R, X.T @ y)

        self.assertTrue(info.uniform_forgetting_equivalent)
        np.testing.assert_allclose(model.R, expected_R, atol=1e-10)
        np.testing.assert_allclose(model.W, expected_W, atol=1e-10)

    def test_unexcited_information_direction_is_not_forgotten(self):
        model = SIFtRLS(
            2, ridge=1.0, forgetting=0.5, epsilon=1e-12, bias=False
        )
        model.R = np.diag([4.0, 9.0])

        info = model.update(np.array([[1.0, 0.0]]), np.array([[1.0]]))

        self.assertFalse(info.uniform_forgetting_equivalent)
        np.testing.assert_allclose(model.R, np.diag([3.0, 9.0]), atol=1e-10)

    def test_small_singular_direction_is_filtered(self):
        model = SIFtRLS(
            2, ridge=1.0, forgetting=0.5, epsilon=1e-6, bias=False
        )
        info = model.update(
            np.diag([1.0, 1e-4]),
            np.array([[1.0], [1.0]]),
        )
        self.assertEqual(info.filtered_rank, 1)
        self.assertFalse(info.uniform_forgetting_equivalent)

    def test_information_matrix_remains_symmetric_positive_definite(self):
        rng = np.random.default_rng(7)
        model = SIFtRLS(
            5, ridge=0.01, forgetting=0.1, epsilon=1e-10, bias=True
        )
        for _ in range(8):
            X = rng.standard_normal((3, 5))
            y = rng.standard_normal((3, 1))
            model.update(X, y)
            np.testing.assert_allclose(model.R, model.R.T, atol=1e-10)
            self.assertGreater(np.linalg.eigvalsh(model.R).min(), 0.0)


if __name__ == "__main__":
    unittest.main()
