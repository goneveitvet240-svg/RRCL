import math
import unittest

from scripts.summarize_confirmatory import _record, mean_ci


class ConfirmatorySummaryTest(unittest.TestCase):
    def test_five_seed_interval_uses_four_degrees_of_freedom(self):
        mean, low, high = mean_ci([0, 1, 2, 3, 4])
        expected_half_width = 2.776 * math.sqrt(2.5) / math.sqrt(5)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(low, mean - expected_half_width)
        self.assertAlmostEqual(high, mean + expected_half_width)

    def test_false_positive_is_conditioned_on_negative_control(self):
        payload = {
            "rel_f1": 1.0,
            "adaptive_rel": 1.1,
            "oracle_rel": 0.999,
            "f_used": [1.0, 0.5],
            "verdict": {
                "minimum_oracle_relative_gain": 0.005,
                "oracle_gain_recovery": None,
            },
            "adaptive_evidence": {"selections": []},
        }
        record = _record(payload, "qnrf_sha_shb")
        self.assertTrue(record["negative_control"])
        self.assertTrue(record["harmful_false_positive"])

    def test_kadid_schema_is_supported(self):
        payload = {
            "f1": 0.4,
            "adaptive": 0.3,
            "opt_rel_mean": 0.3,
            "f_used": [1.0, 0.5],
            "verdict": {
                "minimum_oracle_relative_gain": 0.005,
                "oracle_gain_recovery": 1.0,
            },
            "selection_records": [{"safety_abstained": False}],
        }
        record = _record(payload, "kadid")
        self.assertAlmostEqual(record["gain_pct"], 25.0)
        self.assertAlmostEqual(record["recovery_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
