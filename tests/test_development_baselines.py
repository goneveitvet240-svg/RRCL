import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from development_baselines import (
    METHOD_DOMAIN_BALANCED,
    METHOD_DOMAIN_BALANCED_MASS_MATCHED,
    METHOD_POOLED,
    accumulate_stats,
    baseline_gain_decomposition,
    empty_stats,
    objective_weights,
    ridge_path,
    select_scalar_lambda,
    select_shared_lambda,
    state_cost_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_new_method_baselines",
    ROOT / "scripts" / "run_new_method_baselines.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_new_method_baselines",
    ROOT / "scripts" / "validate_new_method_baselines.py",
)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def _dual_record(x_fit, y_fit, x_val, y_val, alpha=0.5):
    x_fit = np.asarray(x_fit, dtype=np.float64).reshape(-1, 1)
    y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1, 1)
    x_val = np.asarray(x_val, dtype=np.float64).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
    fit_patch = empty_stats(1)
    fit_image = empty_stats(1)
    full_patch = empty_stats(1)
    full_image = empty_stats(1)
    fused = empty_stats(2)
    accumulate_stats(fit_patch, x_fit, y_fit)
    accumulate_stats(fit_image, x_fit, y_fit)
    accumulate_stats(full_patch, np.vstack([x_fit, x_val]), np.vstack([y_fit, y_val]))
    accumulate_stats(full_image, np.vstack([x_fit, x_val]), np.vstack([y_fit, y_val]))
    fused_rows = np.concatenate([alpha * x_val, (1.0 - alpha) * x_val], axis=1)
    accumulate_stats(fused, fused_rows, y_val)
    return {
        "fit": {"patch": fit_patch, "image": fit_image},
        "full": {"patch": full_patch, "image": full_image},
        "val": {"fused": fused},
    }


class ObjectiveWeightsTest(unittest.TestCase):
    def test_pooled_weights_are_one(self):
        np.testing.assert_allclose(objective_weights([2, 5], METHOD_POOLED), [1, 1])

    def test_canonical_balancing_gives_equal_domain_mass(self):
        counts = np.asarray([2.0, 8.0, 10.0])
        weights = objective_weights(counts, METHOD_DOMAIN_BALANCED)
        np.testing.assert_allclose(counts * weights, np.full(3, 1.0 / 3.0))
        self.assertAlmostEqual(float(np.sum(counts * weights)), 1.0)

    def test_mass_matched_balancing_preserves_total_observation_mass(self):
        counts = np.asarray([2.0, 8.0, 10.0])
        weights = objective_weights(counts, METHOD_DOMAIN_BALANCED_MASS_MATCHED)
        self.assertAlmostEqual(float(np.sum(counts * weights)), float(np.sum(counts)))
        np.testing.assert_allclose(counts * weights, np.full(3, counts.mean()))

    def test_rejects_zero_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            objective_weights([2, 0], METHOD_DOMAIN_BALANCED)


class LambdaSelectionTest(unittest.TestCase):
    def test_ridge_path_matches_direct_solves(self):
        R = np.asarray([[4.0, 1.0], [1.0, 2.0]])
        C = np.asarray([[1.0], [3.0]])
        path = ridge_path(R, C, [0.1, 1.0, 10.0])
        for lam, weights in path.items():
            expected = np.linalg.solve(R + lam * np.eye(2), C)
            np.testing.assert_allclose(weights, expected, atol=1e-12)

    def test_selection_is_based_on_validation_trace(self):
        record = _dual_record([1, 2, 3], [1, 2, 3], [4, 5], [4, 5])
        selected = select_shared_lambda([record], METHOD_POOLED, [0.01, 1.0, 100.0])
        risks = {item["lambda"]: item["validation_risk"] for item in selected.candidates}
        self.assertEqual(selected.lam, min(risks, key=risks.get))
        self.assertEqual(len(selected.candidates), 3)

    def test_exact_tie_chooses_larger_lambda(self):
        record = _dual_record([0, 0], [0, 0], [0, 0], [0, 0])
        selected = select_shared_lambda([record], METHOD_POOLED, [0.1, 1.0, 10.0])
        self.assertEqual(selected.lam, 10.0)

    def test_scalar_selection_uses_balanced_domain_validation(self):
        records = []
        for slope in (1.0, 2.0):
            fit = empty_stats(2)
            val = empty_stats(2)
            Xf = np.asarray([[1.0, 1.0], [2.0, 1.0]])
            Xv = np.asarray([[3.0, 1.0], [4.0, 1.0]])
            accumulate_stats(fit, Xf, slope * Xf[:, :1])
            accumulate_stats(val, Xv, slope * Xv[:, :1])
            records.append({"fit": fit, "full": fit, "val": val})
        selected = select_scalar_lambda(records, METHOD_POOLED, [0.01, 1.0, 100.0])
        self.assertEqual(len(selected.candidates), 3)
        self.assertTrue(np.isfinite(selected.validation_risk))


class CostAndDecompositionTest(unittest.TestCase):
    def test_cost_separates_update_and_deployment(self):
        cost = state_cost_bytes(5, heads=2, trajectories=3)
        self.assertEqual(cost["online_update_state_bytes"], 2 * 3 * (25 + 5) * 8)
        self.assertEqual(cost["deployment_weights_bytes"], 2 * 3 * 5 * 8)

    def test_decomposition_does_not_invent_later_gaps(self):
        result = baseline_gain_decomposition(
            {
                "pooled_reference_lambda": 1.0,
                "pooled_tuned_lambda": 0.8,
                "domain_balanced_tuned": 0.7,
                "mass_matched_domain_balanced_tuned": 0.75,
                "single_domain_tuned": 0.6,
            }
        )
        self.assertAlmostEqual(result["regularization_gain_vs_reference"], 0.2)
        self.assertEqual(result["candidate_family_gap"]["status"], "not_measured_in_batch1")


class ProtocolGuardTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "configs" / "new_method_development_v1.json").read_text()
        )

    def test_rejects_fdst_before_loading_data(self):
        payload = {"domains": [{"name": "FDST", "root": "/tmp/fdst"}]}
        with self.assertRaisesRegex(RuntimeError, "not allow-listed|forbidden"):
            RUNNER.assert_development_domain_config(
                "configs/domains_fdst.json", payload, self.protocol
            )

    def test_accepts_allowlisted_development_config(self):
        path = ROOT / "configs" / "domains_sha_shb.json"
        payload = json.loads(path.read_text())
        self.assertTrue(
            RUNNER.assert_development_domain_config(path, payload, self.protocol)
        )

    def test_rejects_allowlisted_filename_with_unknown_domain_identity(self):
        payload = {
            "domains": [
                {
                    "name": "unknown",
                    "kind": "jhu",
                    "root": "/tmp/unknown",
                    "sample_key": "unknown-source",
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "identities are not allow-listed"):
            RUNNER.assert_development_domain_config(
                "configs/domains_sha_shb.json", payload, self.protocol
            )

    def test_output_must_stay_in_development_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "output must stay"):
                RUNNER.require_development_output(Path(tmp) / "result")

    def test_natural_run_cannot_override_frozen_lambda_grid(self):
        args = SimpleNamespace(
            val_every=self.protocol["val_every"],
            patch_target=self.protocol["patch_target"],
            alpha=self.protocol["alpha_patch"],
            lambdas=[1.0],
            reference_lambda=self.protocol["reference_lambda"],
            projection_dim=self.protocol["projection"]["dimension"],
            projection_seed=self.protocol["projection"]["seed"],
            backbone=self.protocol["backbone"],
            img_size=self.protocol["image_size"],
            max_per_domain=self.protocol["max_per_domain"],
            skip_ranpac=False,
        )
        with self.assertRaisesRegex(RuntimeError, "differ from frozen"):
            RUNNER.require_frozen_method_arguments(args, self.protocol)

    def test_toy_audit_has_all_first_batch_methods(self):
        args = SimpleNamespace(
            patch_target="dct5",
            alpha=0.25,
            split_seed=42,
            val_every=5,
            reference_lambda=100.0,
            lambdas=[0.1, 1.0, 10.0],
            skip_ranpac=False,
            projection_dim=8,
            projection_seed=0,
        )
        domains = RUNNER._ToyDomains()
        result = RUNNER.run_audit(
            domains, self.protocol, args, domains.data_manifest()
        )
        self.assertFalse(result["confirmation_data_used"])
        self.assertEqual(set(result["shared_methods"]), set(RUNNER.SHARED_METHODS))
        self.assertIn("single_domain", result)
        self.assertIn("aligned_fit_only", result["ranpac_style_image"])
        self.assertTrue(VALIDATOR.validate_payload(result))


if __name__ == "__main__":
    unittest.main()
