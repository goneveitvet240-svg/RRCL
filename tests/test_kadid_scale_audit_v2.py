import importlib.util
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
        self.assertTrue(VALIDATOR.validate_payload(payload, self.protocol))

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
            self.assertTrue(VALIDATOR.validate_file(result, self.protocol_path))


if __name__ == "__main__":
    unittest.main()
