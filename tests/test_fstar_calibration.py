"""Tests for the formal f* prediction/oracle calibration protocol."""

from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_theory_predict import (
    FACTOR_GRID,
    _select_minimum,
    oracle_factor,
    predicted_factor,
)
from scripts.summarize_fstar_calibration import (
    CONFIG_FAMILIES,
    EXPECTED_SEEDS,
    summarize,
)
from scripts.validate_release import FSTAR_CALIBRATION_CONFIGS


def _artifact(config, seed, f_pred=1.0, f_oracle=1.0):
    prediction = []
    oracle = []
    for factor in FACTOR_GRID:
        prediction.append(
            {
                "factor": factor,
                "bias_term": 0.0,
                "variance_term": abs(factor - f_pred),
                "objective": abs(factor - f_pred),
            }
        )
        oracle.append(
            {
                "factor": factor,
                "balanced_normalized_mse": 0.5
                + 0.1 * abs(factor - f_oracle),
                "per_domain_normalized_mse": [
                    0.5 + 0.1 * abs(factor - f_oracle)
                ],
                "balanced_rel_mae": 0.4 + 0.1 * abs(factor - f_oracle),
                "per_domain_rel_mae": [0.4 + 0.1 * abs(factor - f_oracle)],
            }
        )
    mse_f1 = next(
        row["balanced_normalized_mse"]
        for row in oracle
        if row["factor"] == 1.0
    )
    mse_pred = next(
        row["balanced_normalized_mse"]
        for row in oracle
        if row["factor"] == f_pred
    )
    mse_oracle = next(
        row["balanced_normalized_mse"]
        for row in oracle
        if row["factor"] == f_oracle
    )
    rel_f1 = next(
        row["balanced_rel_mae"] for row in oracle if row["factor"] == 1.0
    )
    rel_pred = next(
        row["balanced_rel_mae"] for row in oracle if row["factor"] == f_pred
    )
    rel_oracle = next(
        row["balanced_rel_mae"] for row in oracle if row["factor"] == f_oracle
    )
    sample_key = f"dataset-{config}"
    return {
        "protocol": "theory-f-calibration-v1",
        "diagnostic_only": True,
        "test_data_used_only_for_oracle_diagnostic": True,
        "config_name": f"domains_{config}.json",
        "task": "crowd",
        "sample_seed": seed,
        "split_seed": seed,
        "factor_grid": list(FACTOR_GRID),
        "prediction_curve_train_only": prediction,
        "oracle_curve_DIAGNOSTIC": oracle,
        "f_pred": f_pred,
        "f_oracle_DIAGNOSTIC": f_oracle,
        "f_oracle_relmae_DIAGNOSTIC": f_oracle,
        "test_mse_f1_DIAGNOSTIC": mse_f1,
        "test_mse_at_f_pred_DIAGNOSTIC": mse_pred,
        "test_mse_oracle_DIAGNOSTIC": mse_oracle,
        "test_rel_f1_DIAGNOSTIC": rel_f1,
        "test_rel_at_f_pred_DIAGNOSTIC": rel_pred,
        "test_rel_oracle_DIAGNOSTIC": rel_oracle,
        "selected_rel_gain_over_f1_DIAGNOSTIC": (
            (rel_f1 - rel_pred) / rel_f1
        ),
        "oracle_rel_regret_at_f_pred_DIAGNOSTIC": (
            (rel_pred - rel_oracle) / rel_oracle
        ),
        "data_manifest": {
            "domains": [
                {
                    "name": sample_key,
                    "sample_key": sample_key,
                    "train": {"count": 10, "ids_sha256": f"train-{seed}"},
                    "test": {"count": 5, "ids_sha256": "test"},
                }
            ]
        },
        "holdout_split": {
            "domains": [
                {
                    "sample_key": sample_key,
                    "split_seed": seed,
                    "fit_count": 8,
                    "fit_ids_sha256": f"fit-{seed}",
                    "val_count": 2,
                    "val_ids_sha256": f"val-{seed}",
                    "algorithm_version": "holdout-hash-v1",
                    "id_source": "image_basename",
                }
            ]
        },
        "_provenance": {
            "git_commit": "one-clean-commit",
            "git_dirty": False,
            "config_sha256": f"config-hash-{config}",
            "arguments": {
                "task": "crowd",
                "lam": 100.0,
                "img_size": 518,
                "backbone": "vit_base_patch14_dinov2.lvd142m",
                "max_per_domain": 400,
                "sample_seed": seed,
                "split_seed": seed,
                "val_every": 5,
                "allow_dirty": False,
            },
        },
    }


class TheoryCoreTest(unittest.TestCase):

    def test_numerical_tie_prefers_larger_factor(self):
        rows = [
            {"factor": 0.5, "objective": 1.0},
            {"factor": 1.0, "objective": 1.0 + 1e-14},
        ]
        self.assertEqual(_select_minimum(rows, "objective")["factor"], 1.0)

    def test_no_drift_equal_information_predicts_absolute_memory(self):
        quantities = [
            {
                "w": np.asarray([[2.0]]),
                "mu": 10.0,
                "r": 1.0,
                "rho": 1.0,
            }
            for _ in range(3)
        ]
        factor, _ = predicted_factor(
            quantities, d_aug=1, lam=0.0, grid=FACTOR_GRID
        )
        self.assertEqual(factor, 1.0)

    def test_oracle_uses_the_supplied_grid(self):
        train = {
            0: {
                "X": np.asarray([[0.0], [1.0], [2.0]]),
                "y": np.asarray([1.0, 2.0, 3.0]),
            },
            1: {
                "X": np.asarray([[0.0], [1.0], [2.0]]),
                "y": np.asarray([1.0, 2.0, 3.0]),
            },
        }
        test = copy.deepcopy(train)
        factor, _, curve = oracle_factor(
            train, test, grid=(0.5, 1.0), lam=1.0
        )
        self.assertIn(factor, (0.5, 1.0))
        self.assertEqual([row["factor"] for row in curve], [0.5, 1.0])

    def test_formal_runner_and_summarizer_share_scenarios(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_fstar_calibration_multiseed.sh"
        ).read_text()
        block = re.search(r"CONFIGS=\(\n(.*?)\n\)", script, re.S)
        self.assertIsNotNone(block)
        shell_configs = [
            line.strip()
            for line in block.group(1).splitlines()
            if line.strip()
        ]
        seed_loop = re.search(r"for seed in ([0-9 ]+); do", script)
        self.assertIsNotNone(seed_loop)
        shell_seeds = tuple(int(value) for value in seed_loop.group(1).split())
        self.assertEqual(shell_configs, list(CONFIG_FAMILIES))
        self.assertEqual(shell_configs, FSTAR_CALIBRATION_CONFIGS)
        self.assertEqual(shell_seeds, EXPECTED_SEEDS)


class TheorySummaryTest(unittest.TestCase):

    def _write_tree(self, root, mixed_commit=False, manifest_mismatch=False):
        root = Path(root)
        for seed in EXPECTED_SEEDS:
            for index, config in enumerate(CONFIG_FAMILIES):
                f_pred = 0.8 if (seed + index) % 2 else 1.0
                f_oracle = 0.75 if (seed + index) % 3 else 1.0
                payload = _artifact(config, seed, f_pred, f_oracle)
                if mixed_commit and seed == 46 and index == 0:
                    payload["_provenance"]["git_commit"] = "other-commit"
                if manifest_mismatch and seed == 42 and index in (0, 1):
                    payload["data_manifest"]["domains"][0][
                        "sample_key"
                    ] = "shared"
                    payload["holdout_split"]["domains"][0][
                        "sample_key"
                    ] = "shared"
                    if index == 1:
                        payload["data_manifest"]["domains"][0]["train"][
                            "ids_sha256"
                        ] = "different"
                destination = (
                    root
                    / f"seed_{seed}"
                    / f"theory_{config}"
                    / "fstar_calibration.json"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload))

    def test_complete_tree_produces_40_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp)
            result = summarize(tmp)
            self.assertEqual(result["overall"]["n_points"], 40)
            self.assertEqual(len(result["rows"]), 40)
            self.assertTrue(result["development_data_only"])

    def test_mixed_commits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp, mixed_commit=True)
            with self.assertRaisesRegex(ValueError, "one commit"):
                summarize(tmp)

    def test_same_dataset_seed_manifest_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_tree(tmp, manifest_mismatch=True)
            with self.assertRaisesRegex(ValueError, "sample manifests"):
                summarize(tmp)


if __name__ == "__main__":
    unittest.main()
