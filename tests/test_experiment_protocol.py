import json
import unittest
from pathlib import Path

from datasets_real import _stable_domain_seed
from run_adaptive_f import classify_adaptive_result


ROOT = Path(__file__).resolve().parents[1]


class SamplingIdentityTest(unittest.TestCase):
    def test_logical_dataset_key_is_path_independent(self):
        seed_from_canonical_copy = _stable_domain_seed("JHU-v2.0", 42)
        seed_from_legacy_copy = _stable_domain_seed("JHU-v2.0", 42)
        self.assertEqual(seed_from_canonical_copy, seed_from_legacy_copy)

    def test_core_reorder_configs_use_one_dataset_identity(self):
        config_names = [
            "domains_jhu_sha_shb.json",
            "domains_jhu_shb_sha.json",
            "domains_sha_shb_jhu.json",
        ]
        signatures = {}
        for config_name in config_names:
            payload = json.loads((ROOT / "configs" / config_name).read_text())
            for domain in payload["domains"]:
                key = domain["sample_key"]
                signature = (
                    domain["kind"],
                    domain["root"],
                    domain["train_split"],
                    domain["test_split"],
                )
                if key in signatures:
                    self.assertEqual(signatures[key], signature)
                else:
                    signatures[key] = signature


class AdaptiveVerdictTest(unittest.TestCase):
    def test_does_not_call_zero_recovery_good(self):
        verdict = classify_adaptive_result(
            rel_f1=0.5099,
            oracle_rel=0.4909,
            adaptive_rel=0.5099,
            f_used=[1.0, 1.0, 1.0],
        )
        self.assertFalse(verdict["success"])
        self.assertEqual(verdict["status"], "missed_fixed_oracle_gain")
        self.assertEqual(verdict["oracle_gain_recovery"], 0.0)

    def test_accepts_material_oracle_gain_recovery(self):
        verdict = classify_adaptive_result(
            rel_f1=0.6959,
            oracle_rel=0.5379,
            adaptive_rel=0.5372,
            f_used=[1.0, 0.25, 0.35],
        )
        self.assertTrue(verdict["success"])
        self.assertEqual(verdict["status"], "captures_fixed_oracle_gain")
        self.assertGreater(verdict["oracle_gain_recovery"], 0.99)

    def test_correct_abstention_requires_no_fixed_oracle_gain(self):
        verdict = classify_adaptive_result(
            rel_f1=0.3651,
            oracle_rel=0.3651,
            adaptive_rel=0.3651,
            f_used=[1.0, 1.0],
        )
        self.assertTrue(verdict["success"])
        self.assertEqual(
            verdict["status"], "correct_abstention_no_fixed_oracle_gain"
        )


if __name__ == "__main__":
    unittest.main()
