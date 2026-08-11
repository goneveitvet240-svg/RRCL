import unittest

import numpy as np

from analytic_scalar_protocol import (
    default_validation_masks,
    run_dos_elm_style,
    run_sift,
)


def _toy_domains():
    rng = np.random.default_rng(3)
    train, test = {}, {}
    for domain, slope in enumerate((1.0, 1.0, -1.0)):
        X = rng.normal(size=(30, 3))
        y = 5.0 + slope * X[:, 0]
        train[domain] = (X[:24], y[:24])
        test[domain] = (X[24:], y[24:])
    return train, test


class AnalyticScalarProtocolTest(unittest.TestCase):
    def test_dos_factor_update_affects_only_the_following_domain(self):
        train, test = _toy_domains()
        result = run_dos_elm_style(train, test, ridge=1.0)
        records = result["records"]

        self.assertEqual(records[0]["design_factor_used"], 1.0)
        self.assertEqual(records[1]["design_factor_used"], 1.0)
        self.assertIsNotNone(records[1]["next_factor_update"])
        self.assertEqual(
            records[2]["design_factor_used"],
            records[1]["next_factor_update"]["factor"],
        )

    def test_sift_selection_and_evaluation_never_require_test_for_selection(self):
        train, test = _toy_domains()
        masks = default_validation_masks(train, val_every=4)
        result = run_sift(
            train,
            test,
            masks,
            ridge=1.0,
            factors=[1.0, 0.5],
            epsilon=1e-10,
        )
        self.assertIn(result["selected_factor"], (1.0, 0.5))
        self.assertEqual(len(result["train_only_selection_curve"]), 2)
        self.assertEqual(np.asarray(result["matrix"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
