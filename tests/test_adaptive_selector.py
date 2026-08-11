import unittest

import numpy as np

from adaptive_selector import BalancedSufficientStatsSelector


def stats(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return [x.T @ x, x.T @ y, float(np.sum(y * y)), x.shape[0]]


class BoundedSelectorTest(unittest.TestCase):
    def test_state_size_is_constant_across_domains(self):
        selector = BalancedSufficientStatsSelector(1, lam=0.0)
        initial_bytes = selector.state_bytes
        for _ in range(20):
            selector.select_and_update(stats([1, 1], [1, 1]), stats([1], [1]))
            self.assertEqual(selector.state_bytes, initial_bytes)
        self.assertEqual(selector.n_domains, 20)

    def test_abstains_without_mapping_drift(self):
        selector = BalancedSufficientStatsSelector(1, lam=0.0)
        selector.select_and_update(stats(np.ones(100), np.ones(100)), stats([1], [1]))
        result = selector.select_and_update(stats(np.ones(10), np.ones(10)), stats([1], [1]))
        self.assertEqual(result.factor, 1.0)
        self.assertEqual(result.drift_score, 0.0)

    def test_forgets_when_old_domain_mass_biases_balanced_validation(self):
        selector = BalancedSufficientStatsSelector(1, lam=0.0, f_min=0.01)
        selector.select_and_update(
            stats(np.ones(100), np.full(100, 10.0)), stats([1], [10.0])
        )
        result = selector.select_and_update(stats([1], [0.0]), stats([1], [0.0]))
        self.assertLess(result.factor, 1.0)
        self.assertGreater(result.drift_score, 0.0)

    def test_component_safety_gate_can_abstain(self):
        selector = BalancedSufficientStatsSelector(
            1, lam=0.0, f_min=0.01, max_component_relative_harm=0.0
        )
        selector.select_and_update(
            stats(np.ones(100), np.full(100, 10.0)), stats([1], [10.0])
        )
        result = selector.select_and_update(stats([1], [0.0]), stats([1], [0.0]))
        self.assertEqual(result.factor, 1.0)
        self.assertTrue(result.safety_abstained)


if __name__ == "__main__":
    unittest.main()
