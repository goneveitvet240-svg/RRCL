import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from development_baselines import (
    METHOD_SAMPLE_MEAN_POOLED,
    objective_weights,
    solve_ridge,
    weighted_sufficient_statistics,
)
from kadid_group_cv import assign_group_folds, fold_manifest, paired_group_bootstrap


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script("run_kadid_group_cv_v4")
VALIDATOR = _load_script("validate_kadid_group_cv_v4")


class KADIDGroupCVV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = ROOT / "configs" / "kadid_group_cv_v4.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))

    def _toy_payload(self):
        domains = RUNNER._ToyGroupCVDomains()
        args = SimpleNamespace(
            domain_config=None,
            lambdas=RUNNER.protocol_lambda_grid(self.protocol),
            reference_lambda=self.protocol["reference_normalized_lambda"],
            projection_dim=8,
            projection_seed=self.protocol["projection"]["seed"],
            skip_projection=False,
        )
        payload = RUNNER.run_audit(
            domains,
            self.protocol,
            args,
            domains.data_manifest(),
            {"device": "none-toy", "cache_dir": None},
        )
        payload["selftest"] = True
        return payload

    def test_fold_assignment_is_deterministic_balanced_and_domain_independent(self):
        groups = [f"ref-{index:02d}" for index in range(64)]
        first = assign_group_folds(groups, "shared-content", 42, 5)
        second = assign_group_folds(reversed(groups), "shared-content", 42, 5)
        self.assertEqual(first, second)
        manifest = fold_manifest(first, "shared-content", 42, 5)
        self.assertLessEqual(max(manifest["fold_counts"]) - min(manifest["fold_counts"]), 1)
        self.assertEqual(sum(manifest["fold_counts"]), 64)
        self.assertNotIn("domain", json.dumps(manifest).lower())

    def test_post_selection_bootstrap_is_deterministic_and_qualified(self):
        baseline = {f"g{i}": float(i + 2) for i in range(8)}
        candidate = {f"g{i}": float(i + 1) for i in range(8)}
        first = paired_group_bootstrap(baseline, candidate, seed=7, resamples=200)
        second = paired_group_bootstrap(baseline, candidate, seed=7, resamples=200)
        self.assertEqual(first, second)
        self.assertEqual(first["point_mean_baseline_minus_candidate"], 1.0)
        self.assertIn("not selection-adjusted", first["qualification"])

    def test_toy_audit_validates_and_covers_all_methods(self):
        payload = self._toy_payload()
        self.assertEqual(set(payload["methods"]), set(RUNNER.METHOD_ROSTER))
        self.assertTrue(VALIDATOR.validate_payload(payload, self.protocol))
        group_count = payload["fold_manifest"]["group_count"]
        for comparison in payload["paired_oof_bootstrap_vs_linear_sample_mean_pooled"].values():
            self.assertEqual(comparison["common_groups"], group_count)

    def test_test_metric_uses_one_prediction_per_image_without_broadcasting(self):
        domains = RUNNER._ToyGroupCVDomains()
        records, dimension = RUNNER.collect_arrays(domains)
        view = RUNNER.LinearView(dimension)
        contexts = RUNNER.build_cv_contexts(
            records,
            assign_group_folds(
                {group for record in records for group in record["train"]["groups"]},
                self.protocol["group_cv"]["sample_key"],
                self.protocol["group_cv"]["seed"],
                self.protocol["group_cv"]["folds"],
            ),
            self.protocol["group_cv"]["folds"],
            view,
        )
        result = RUNNER.run_shared_cv(
            records,
            contexts,
            view,
            RUNNER.METHOD_LINEAR_POOLED,
            RUNNER.protocol_lambda_grid(self.protocol),
            self.protocol["reference_normalized_lambda"],
        )
        selected = [
            row["selected_lambda"]
            for row in result["cv_selected"]["lambda_selection"]
        ]
        scales = [float(np.mean(record["train"]["y"])) for record in records]
        blocks = [
            RUNNER._stats(RUNNER._aug(record["train"]["X"]), record["train"]["y"] / scale)
            for record, scale in zip(records, scales)
        ]
        weights_by_domain = objective_weights(
            [block["n"] for block in blocks], METHOD_SAMPLE_MEAN_POOLED
        )
        R, C = weighted_sufficient_statistics(blocks, weights_by_domain)
        weights = solve_ridge(R, C, selected[-1])
        normalized = (RUNNER._aug(records[0]["test"]["X"]) @ weights).reshape(-1)
        expected = float(
            np.mean(
                np.abs(np.maximum(normalized, 0.0) * scales[0] - records[0]["test"]["y"])
            )
            / np.mean(np.abs(records[0]["test"]["y"]))
        )
        actual = result["cv_selected"]["development_test"]["relative_mae"]["matrix"][-1][0]
        self.assertEqual(normalized.shape, records[0]["test"]["y"].shape)
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))

    def test_validator_rejects_forged_fold_and_bootstrap(self):
        payload = self._toy_payload()
        forged_fold = copy.deepcopy(payload)
        forged_fold["fold_manifest"]["assignments"][0]["fold"] = (
            forged_fold["fold_manifest"]["assignments"][0]["fold"] + 1
        ) % self.protocol["group_cv"]["folds"]
        with self.assertRaisesRegex(AssertionError, "forged fold assignment"):
            VALIDATOR.validate_payload(forged_fold, self.protocol)

        forged_bootstrap = copy.deepcopy(payload)
        forged_bootstrap["paired_oof_bootstrap_vs_linear_sample_mean_pooled"][
            RUNNER.METHOD_INDEPENDENT
        ]["point_mean_baseline_minus_candidate"] += 1.0
        with self.assertRaisesRegex(AssertionError, "forged bootstrap summary"):
            VALIDATOR.validate_payload(forged_bootstrap, self.protocol)

    def test_natural_arguments_reject_projection_override(self):
        protocol = self.protocol
        args = SimpleNamespace(
            backbone=protocol["backbone"],
            img_size=protocol["image_size"],
            max_per_domain=protocol["max_train_per_domain"],
            sample_seed=protocol["sample_seed"],
            lambdas=RUNNER.protocol_lambda_grid(protocol),
            reference_lambda=protocol["reference_normalized_lambda"],
            projection_dim=17,
            projection_seed=protocol["projection"]["seed"],
            skip_projection=False,
        )
        with self.assertRaisesRegex(RuntimeError, "differ from frozen"):
            RUNNER.require_frozen_arguments(args, protocol)

    def test_selftest_file_provenance_validates(self):
        payload = self._toy_payload()
        protocol_bytes = self.protocol_path.read_bytes()
        payload["_provenance"] = {
            "config_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "config_snapshot": self.protocol,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            result.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(VALIDATOR.validate_file(result, self.protocol_path))


if __name__ == "__main__":
    unittest.main()
