"""Tests for the frozen C4 material-decision rule."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_gamma_c4 import (
    EXPECTED_SEEDS,
    FAMILIES,
    material_decision,
    summarize,
    validate_order_pair,
)


def _payload(train_curve, test_curve, train_gamma, test_gamma):
    return {
        "diagnostic_only": True,
        "sample_seed": 42,
        "split_seed": 42,
        "gammas": [0.0, 0.5, 1.0],
        "val_balanced_fused_proxy": train_curve,
        "test_balanced_rel_mae_DIAGNOSTIC": test_curve,
        "argmin": {
            "train_fused_gamma": train_gamma,
            "test_gamma_DIAGNOSTIC": test_gamma,
        },
        "data_manifest": {
            "domains": [
                {
                    "name": "A",
                    "sample_key": "A",
                    "train": {"count": 10, "ids_sha256": "train"},
                    "test": {"count": 5, "ids_sha256": "test"},
                }
            ]
        },
        "holdout_split": {
            "domains": [
                {
                    "sample_key": "A",
                    "roles": {
                        "fit": {"count": 8, "ids_sha256": "fit"},
                        "precision_val": {"count": 1, "ids_sha256": "p"},
                        "gamma_val": {"count": 1, "ids_sha256": "g"},
                    },
                }
            ]
        },
        "equivalence": {
            "gamma0_equals_f1": True,
            "gamma1_equals_three_way_precision_endpoint": True,
        },
        "_provenance": {
            "git_commit": "same-clean-commit",
            "git_dirty": False,
        },
    }


class MaterialDecisionTest(unittest.TestCase):

    def test_material_jhu_like_gain_is_reweighting(self):
        payload = _payload(
            train_curve=[1.0, 0.8, 0.81],
            test_curve=[0.58, 0.52, 0.49],
            train_gamma=0.5,
            test_gamma=1.0,
        )
        result = material_decision(payload)
        self.assertTrue(result["oracle_material_reweight"])
        self.assertTrue(result["train_predicts_reweight"])
        self.assertTrue(result["correct_material_decision"])

    def test_tiny_qnrf_like_numerical_gain_is_no_signal(self):
        payload = _payload(
            train_curve=[1.0, 1.01, 1.02],
            test_curve=[0.38044, 0.38043, 0.381],
            train_gamma=0.0,
            test_gamma=0.5,
        )
        result = material_decision(payload)
        self.assertLess(result["oracle_gain_relative"], 0.005)
        self.assertFalse(result["oracle_material_reweight"])
        self.assertFalse(result["train_predicts_reweight"])
        self.assertTrue(result["correct_material_decision"])

    def test_wrong_train_direction_fails(self):
        payload = _payload(
            train_curve=[1.0, 0.9, 0.8],
            test_curve=[0.4, 0.41, 0.42],
            train_gamma=1.0,
            test_gamma=0.0,
        )
        self.assertFalse(material_decision(payload)["correct_material_decision"])

    def test_stored_argmin_must_match_curve(self):
        payload = _payload(
            train_curve=[1.0, 0.8, 0.9],
            test_curve=[0.4, 0.39, 0.41],
            train_gamma=1.0,
            test_gamma=0.5,
        )
        with self.assertRaisesRegex(ValueError, "stored fused train argmin"):
            material_decision(payload)


class OrderPairTest(unittest.TestCase):

    def test_matching_order_pair_passes(self):
        payload = _payload(
            train_curve=[1.0, 0.8, 0.9],
            test_curve=[0.4, 0.38, 0.39],
            train_gamma=0.5,
            test_gamma=0.5,
        )
        validate_order_pair(payload, copy.deepcopy(payload))

    def test_manifest_mismatch_fails(self):
        left = _payload([1.0, 0.8, 0.9], [0.4, 0.38, 0.39], 0.5, 0.5)
        right = copy.deepcopy(left)
        right["data_manifest"]["domains"][0]["train"]["ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "sampled data manifests"):
            validate_order_pair(left, right)

    def test_curve_mismatch_fails(self):
        left = _payload([1.0, 0.8, 0.9], [0.4, 0.38, 0.39], 0.5, 0.5)
        right = copy.deepcopy(left)
        right["test_balanced_rel_mae_DIAGNOSTIC"][1] += 0.01
        with self.assertRaisesRegex(ValueError, "curve mismatch"):
            validate_order_pair(left, right)


class FullSummaryTest(unittest.TestCase):

    def _write_tree(self, root, wrong_qnrf_seeds=(), mixed_commit=False):
        root = Path(root)
        for seed in EXPECTED_SEEDS:
            for family, tasks in FAMILIES.items():
                if family == "jhu":
                    payload = _payload(
                        [1.0, 0.8, 0.9],
                        [0.58, 0.52, 0.49],
                        0.5,
                        1.0,
                    )
                elif seed in wrong_qnrf_seeds:
                    payload = _payload(
                        [1.0, 0.9, 0.8],
                        [0.38, 0.39, 0.40],
                        1.0,
                        0.0,
                    )
                else:
                    payload = _payload(
                        [1.0, 1.1, 1.2],
                        [0.38044, 0.38043, 0.381],
                        0.0,
                        0.5,
                    )
                payload["sample_seed"] = seed
                payload["split_seed"] = seed
                for task_index, task in enumerate(tasks):
                    artifact = copy.deepcopy(payload)
                    if mixed_commit and seed == EXPECTED_SEEDS[-1] and task_index == 1:
                        artifact["_provenance"]["git_commit"] = "different-commit"
                    destination = (
                        root
                        / f"seed_{seed}"
                        / f"shrinkage_diag_{task}"
                        / "shrinkage_diagnostic.json"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(artifact))

    def test_four_of_five_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp, wrong_qnrf_seeds={42})
            result = summarize(tmp)
            self.assertTrue(result["passed"])
            self.assertEqual(result["families"]["qnrf"]["correct_seeds"], 4)

    def test_three_of_five_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp, wrong_qnrf_seeds={42, 43})
            result = summarize(tmp)
            self.assertFalse(result["passed"])
            self.assertEqual(result["families"]["qnrf"]["correct_seeds"], 3)

    def test_mixed_source_commits_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp, mixed_commit=True)
            with self.assertRaisesRegex(ValueError, "one exact commit"):
                summarize(tmp)


if __name__ == "__main__":
    unittest.main()
