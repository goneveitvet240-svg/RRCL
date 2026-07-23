import unittest

from evaluation_metrics import scalar_regression_metrics


class ScalarRegressionMetricsTest(unittest.TestCase):
    def test_perfect_monotonic_prediction(self):
        metrics = scalar_regression_metrics([2, 4, 6], [1, 2, 3])
        self.assertAlmostEqual(metrics["plcc"], 1.0)
        self.assertAlmostEqual(metrics["srcc"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 2.0)

    def test_reverse_order_has_negative_rank_correlation(self):
        metrics = scalar_regression_metrics([3, 2, 1], [1, 2, 3])
        self.assertAlmostEqual(metrics["srcc"], -1.0)

    def test_constant_prediction_reports_undefined_correlations(self):
        metrics = scalar_regression_metrics([1, 1, 1], [1, 2, 3])
        self.assertIsNone(metrics["plcc"])
        self.assertIsNone(metrics["srcc"])


if __name__ == "__main__":
    unittest.main()
