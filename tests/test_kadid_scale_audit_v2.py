import importlib.util
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from development_baselines import (
    METHOD_DOMAIN_BALANCED,
    METHOD_SAMPLE_MEAN_POOLED,
    objective_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script("run_kadid_scale_audit_v2")
VALIDATOR = _load_script("validate_kadid_scale_audit_v2")


class KADIDScaleAuditV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = ROOT / "configs" / "new_method_kadid_scale_audit_v2.json"
        cls.protocol = json.loads(cls.protocol_path.read_text())

    def test_normalized_objectives_have_unit_total_mass(self):
        counts = np.asarray([427, 379, 401, 392, 383])
        for method in (METHOD_SAMPLE_MEAN_POOLED, METHOD_DOMAIN_BALANCED):
            weights = objective_weights(counts, method)
            self.assertTrue(math.isclose(float(weights @ counts), 1.0))

    def test_grid_is_frozen_quarter_decades(self):
        grid = RUNNER.protocol_lambda_grid(self.protocol)
        self.assertEqual(len(grid), 37)
        self.assertAlmostEqual(grid[0], 1e-5)
        self.assertAlmostEqual(grid[-1], 1e4)
        ratios = np.asarray(grid[1:]) / np.asarray(grid[:-1])
        self.assertTrue(np.allclose(ratios, 10 ** 0.25))

    def test_natural_arguments_reject_grid_override(self):
        args = SimpleNamespace(
            backbone=self.protocol["backbone"],
            img_size=self.protocol["image_size"],
            max_per_domain=self.protocol["max_train_per_domain"],
            sample_seed=self.protocol["sample_seed"],
            split_seed=self.protocol["selector_split_seed"],
            val_every=self.protocol["selector_val_every"],
            lambdas=[0.1, 1.0],
        )
        with self.assertRaisesRegex(RuntimeError, "differ from frozen"):
            RUNNER.require_frozen_arguments(args, self.protocol)

    def test_toy_audit_covers_equivalence_and_paired_comparisons(self):
        args = SimpleNamespace(
            domain_config=None,
            split_seed=42,
            val_every=5,
            lambdas=RUNNER.protocol_lambda_grid(self.protocol),
        )
        domains = RUNNER._ToyScalarDomains()
        payload = RUNNER.run_audit(
            domains,
            self.protocol,
            args,
            domains.data_manifest(),
            {"device": "none-toy", "cache_dir": None},
        )
        self.assertEqual(set(payload["normalized_methods"]), set(RUNNER.NORMALIZED_METHODS))
        self.assertEqual(set(payload["same_lambda_pairs"]), set(RUNNER.NORMALIZED_METHODS))
        for result in payload["normalized_methods"].values():
            for selection in result["lambda_selection"]:
                audit = selection["equivalence_audit"]
                self.assertLess(audit["weights_relative_l2"], 1e-9)
                self.assertLess(audit["validation_risk_absolute_difference"], 1e-9)
        self.assertTrue(
            VALIDATOR.validate_payload(
                payload, self.protocol, allow_invalidated=True
            )
        )
        with self.assertRaisesRegex(AssertionError, "invalidated protocol"):
            VALIDATOR.validate_payload(payload, self.protocol)

    def test_cli_selftest_artifact_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "runs_real" / "new_method_kadid_scale_audit_v2" / "main"
            # require_output is intentionally rooted in the repository, so the
            # CLI path guard itself is covered elsewhere; exercise provenance
            # validation here with a direct toy payload.
            args = SimpleNamespace(
                domain_config=None,
                split_seed=42,
                val_every=5,
                lambdas=RUNNER.protocol_lambda_grid(self.protocol),
            )
            domains = RUNNER._ToyScalarDomains()
            payload = RUNNER.run_audit(
                domains,
                self.protocol,
                args,
                domains.data_manifest(),
                {"device": "none-toy", "cache_dir": None},
            )
            payload["selftest"] = True
            payload["_provenance"] = {
                "config_sha256": __import__("hashlib").sha256(
                    self.protocol_path.read_bytes()
                ).hexdigest(),
                "config_snapshot": self.protocol,
            }
            output.mkdir(parents=True)
            result = output / "kadid_scale_audit_v2.json"
            result.write_text(json.dumps(payload))
            self.assertTrue(
                VALIDATOR.validate_file(
                    result, self.protocol_path, allow_invalidated=True
                )
            )


class KADIDScaleAuditV3IsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs" / "new_method_kadid_scale_audit_v3.json").read_text()
        )

    def _payload(self):
        args = SimpleNamespace(
            domain_config=None,
            split_seed=self.protocol["selector_split_seed"],
            val_every=self.protocol["selector_val_every"],
            selector_sample_key=self.protocol["selector_sample_key"],
            lambdas=RUNNER.protocol_lambda_grid(self.protocol),
        )
        domains = RUNNER._ToyScalarDomains()
        return RUNNER.run_audit(
            domains,
            self.protocol,
            args,
            domains.data_manifest(),
            {"device": "none-toy", "cache_dir": None},
        )

    def test_shared_selector_key_eliminates_cross_domain_role_overlap(self):
        payload = self._payload()
        audit = payload["selector_isolation_audit"]
        self.assertEqual(audit["scope"], "global_across_domains")
        self.assertEqual(audit["cross_role_overlap_count"], 0)
        role_by_group = {}
        for domain in audit["domains"]:
            for role in ("fit_groups", "validation_groups"):
                for group in domain[role]:
                    role_by_group.setdefault(group, set()).add(role)
        self.assertTrue(role_by_group)
        self.assertTrue(all(len(roles) == 1 for roles in role_by_group.values()))
        self.assertTrue(VALIDATOR.validate_payload(payload, self.protocol))

    def test_domain_specific_selector_key_reproduces_cross_domain_leakage(self):
        domains = RUNNER._ToyScalarDomains()
        old_records, _ = RUNNER.collect_domain_records(domains, 42, 5)
        old_fit = set().union(*(record["_fit_groups"] for record in old_records))
        old_validation = set().union(
            *(record["_validation_groups"] for record in old_records)
        )
        self.assertTrue(old_fit & old_validation)

        corrected_records, _ = RUNNER.collect_domain_records(
            domains, 42, 5, self.protocol["selector_sample_key"]
        )
        corrected_fit = set().union(
            *(record["_fit_groups"] for record in corrected_records)
        )
        corrected_validation = set().union(
            *(record["_validation_groups"] for record in corrected_records)
        )
        self.assertFalse(corrected_fit & corrected_validation)

    def test_validator_rejects_forged_positive_isolation_field(self):
        payload = self._payload()
        forged = copy.deepcopy(payload)
        validation_group = forged["selector_isolation_audit"]["domains"][0][
            "validation_groups"
        ][0]
        forged["selector_isolation_audit"]["domains"][1]["fit_groups"].append(
            validation_group
        )
        forged["selector_isolation_audit"]["cross_role_overlap_count"] = 0
        with self.assertRaisesRegex(
            AssertionError, "invalid role groups|manifest mismatch|forged fit role|overlap"
        ):
            VALIDATOR.validate_payload(forged, self.protocol)


if __name__ == "__main__":
    unittest.main()
