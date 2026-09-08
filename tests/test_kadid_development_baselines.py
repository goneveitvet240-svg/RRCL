import importlib.util
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_script("run_new_method_kadid_baselines")
VALIDATOR = _load_script("validate_new_method_kadid_baselines")


class KADIDDevelopmentProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs" / "new_method_kadid_development_v1.json").read_text()
        )

    def test_portable_config_is_present_and_allowlisted(self):
        path = ROOT / "configs" / "domains_iqa_kadid.json"
        payload = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kadid_root = Path(tmp)
            (kadid_root / "images").mkdir()
            (kadid_root / "dmos.csv").touch()
            with mock.patch.dict(os.environ, {"RRCL_KADID_ROOT": str(kadid_root)}):
                self.assertTrue(
                    RUNNER.assert_kadid_config(path, payload, self.protocol)
                )

    def test_fdst_token_is_rejected(self):
        payload = {
            "domains": [
                {
                    "name": name,
                    "kind": "csv",
                    "root": "/tmp/fdst",
                    "csv": "/tmp/fdst.csv",
                }
                for name in self.protocol["expected_domains"]
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            RUNNER.assert_kadid_config(
                "domains_iqa_kadid.json", payload, self.protocol
            )

    def test_frozen_natural_arguments_reject_projection_override(self):
        args = SimpleNamespace(
            backbone=self.protocol["backbone"],
            img_size=self.protocol["image_size"],
            max_per_domain=self.protocol["max_train_per_domain"],
            sample_seed=self.protocol["sample_seed"],
            split_seed=self.protocol["selector_split_seed"],
            val_every=self.protocol["selector_val_every"],
            lambdas=[float(value) for value in self.protocol["lambda_grid"]],
            reference_lambda=self.protocol["reference_lambda"],
            projection_dim=16,
            projection_seed=self.protocol["projection"]["seed"],
            skip_projection=False,
        )
        with self.assertRaisesRegex(RuntimeError, "differ from frozen"):
            RUNNER.require_frozen_arguments(args, self.protocol)

    def test_toy_run_covers_first_batch_and_group_splits(self):
        args = SimpleNamespace(
            domain_config=None,
            split_seed=42,
            val_every=5,
            reference_lambda=100.0,
            lambdas=[0.1, 1.0, 10.0],
            skip_projection=False,
            projection_dim=8,
            projection_seed=0,
        )
        domains = RUNNER._ToyScalarDomains()
        payload = RUNNER.run_audit(
            domains,
            self.protocol,
            args,
            domains.data_manifest(),
            {"device": "none-toy", "cache_dir": None},
        )
        self.assertEqual(set(payload["shared_methods"]), set(RUNNER.SHARED_METHODS))
        self.assertIn("aligned_fit_only", payload["ranpac_style_projection"])
        for record in payload["domain_records"]:
            self.assertGreater(record["split_manifest"]["fit_groups"]["count"], 0)
            self.assertGreater(
                record["split_manifest"]["validation_groups"]["count"], 0
            )
        self.assertTrue(VALIDATOR.validate_payload(payload, allow_invalidated=True))
        with self.assertRaisesRegex(AssertionError, "invalidated protocol"):
            VALIDATOR.validate_payload(payload)

    def test_validator_accepts_relocated_domain_config_with_matching_hash(self):
        protocol_path = ROOT / "configs" / "new_method_kadid_development_v1.json"
        domain_path = ROOT / "configs" / "domains_iqa_kadid.json"
        args = SimpleNamespace(
            domain_config=None,
            split_seed=42,
            val_every=5,
            reference_lambda=100.0,
            lambdas=[0.1, 1.0, 10.0],
            skip_projection=False,
            projection_dim=8,
            projection_seed=0,
        )
        domains = RUNNER._ToyScalarDomains()
        payload = RUNNER.run_audit(
            domains,
            self.protocol,
            args,
            domains.data_manifest(),
            {"device": "none-toy", "cache_dir": None},
        )
        protocol_bytes = protocol_path.read_bytes()
        payload["selftest"] = False
        payload["domain_config"] = {
            "path": "/released/server/path/domains_iqa_kadid.json",
            "sha256": hashlib.sha256(domain_path.read_bytes()).hexdigest(),
        }
        payload["_provenance"] = {
            "config_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "config_snapshot": json.loads(protocol_bytes),
        }
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(payload))
            self.assertTrue(
                VALIDATOR.validate_file(
                    result_path,
                    protocol_path,
                    domain_path,
                    allow_invalidated=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
