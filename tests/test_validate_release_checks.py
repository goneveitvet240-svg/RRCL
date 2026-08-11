"""Negative tests for the release validator's split/coverage strictness.

Codex review 2026-07-26 (round 4): the new validator logic itself needs
negative coverage — wrong nested seeds, duplicated or missing sample_keys,
wrong VFF boundary counts and failed equivalence flags must all FAIL.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_release import (
    _holdout_split_errors,
    optional_artifact_errors,
    validate_json,
)


def _prov():
    return {
        "schema_version": "rrcl-result-v2",
        "git_commit": "deadbeef",
        "config_sha256": "ff",
        "git_dirty": False,
    }


def _data_manifest(keys=("A", "B", "C")):
    return {
        "domains": [
            {"sample_key": key, "name": key, "kind": "mock",
             "train": {"count": 10, "ids_sha256": "t"},
             "test": {"count": 5, "ids_sha256": "e"}}
            for key in keys
        ]
    }


def _holdout_domains(keys=("A", "B", "C"), seed=42, fit="aa", val="bb"):
    return [
        {"sample_key": key, "split_seed": seed,
         "fit_ids_sha256": fit, "val_ids_sha256": val}
        for key in keys
    ]


def _payload(seed=42, holdout_keys=("A", "B", "C"), nested_seed=None):
    nested = seed if nested_seed is None else nested_seed
    return {
        "split_seed": seed,
        "data_manifest": _data_manifest(),
        "adaptive_evidence": {
            "M_rel": [[0.1]],
            "holdout_split": {
                "split_seed": nested,
                "domains": _holdout_domains(holdout_keys, seed=nested),
            },
        },
        "verdict": {"status": "x"},
        "_provenance": _prov(),
    }


class TestHoldoutSplitErrors(unittest.TestCase):

    def _errors(self, payload):
        return _holdout_split_errors(payload, payload.get("adaptive_evidence"))

    def test_valid_payload_passes(self):
        self.assertEqual(self._errors(_payload()), [])

    def test_missing_split_seed_fails(self):
        payload = _payload()
        del payload["split_seed"]
        self.assertTrue(any("split_seed" in e for e in self._errors(payload)))

    def test_nested_block_seed_mismatch_fails(self):
        payload = _payload(seed=42, nested_seed=43)
        self.assertTrue(
            any("split_seed differs" in e for e in self._errors(payload))
        )

    def test_duplicated_sample_key_fails(self):
        payload = _payload(holdout_keys=("A", "A", "C"))
        self.assertTrue(
            any("missing or duplicated" in e for e in self._errors(payload))
        )

    def test_coverage_mismatch_fails(self):
        payload = _payload(holdout_keys=("A", "B"))
        self.assertTrue(
            any("does not cover" in e for e in self._errors(payload))
        )

    def test_incomplete_hashes_fail(self):
        payload = _payload()
        payload["adaptive_evidence"]["holdout_split"]["domains"][1][
            "val_ids_sha256"] = ""
        self.assertTrue(
            any("incomplete holdout" in e for e in self._errors(payload))
        )


class TestVffBoundaryChecks(unittest.TestCase):

    def _vff_payload(self, diagnostics, seed=42):
        return {
            "split_seed": seed,
            "data_manifest": _data_manifest(),
            "paleologu_batch_vff": {"score": 0.4, "diagnostics": diagnostics},
            "rrcl_bounded": {
                "evidence": {
                    "holdout_split": {
                        "split_seed": seed,
                        "domains": _holdout_domains(),
                    },
                },
            },
            "_provenance": _prov(),
        }

    def _validate(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vff_baselines.json"
            path.write_text(json.dumps(payload))
            return validate_json(path, "vff_x/vff_baselines.json")

    def _boundary(self, seed=42):
        return {"error_power": 1.0,
                "holdout_split": {"sample_key": "B", "split_seed": seed,
                                  "fit_ids_sha256": "aa",
                                  "val_ids_sha256": "bb"}}

    def test_correct_boundary_count_passes(self):
        payload = self._vff_payload([None, self._boundary(), self._boundary()])
        self.assertEqual(self._validate(payload), [])

    def test_wrong_boundary_count_fails(self):
        payload = self._vff_payload([None, self._boundary()])
        errors = self._validate(payload)
        self.assertTrue(any("boundary records" in e for e in errors))

    def test_boundary_seed_mismatch_fails(self):
        payload = self._vff_payload(
            [None, self._boundary(seed=41), self._boundary()]
        )
        errors = self._validate(payload)
        self.assertTrue(any("split_seed differs" in e for e in errors))

    def test_missing_boundary_manifest_fails(self):
        payload = self._vff_payload([None, {"error_power": 1.0},
                                     self._boundary()])
        errors = self._validate(payload)
        self.assertTrue(any("boundary holdout manifest" in e for e in errors))


class TestOptionalArtifactScan(unittest.TestCase):

    def _shrinkage_payload(self, equiv1=True, seed=42, role_count=3):
        roles = {"fit": {"count": 8, "ids_sha256": "f"},
                 "precision_val": {"count": 1, "ids_sha256": "p"},
                 "gamma_val": {"count": 1, "ids_sha256": "g"}}
        return {
            "diagnostic_only": True,
            "split_seed": seed,
            "data_manifest": _data_manifest(),
            "equivalence": {
                "gamma0_equals_f1": True,
                "gamma1_equals_three_way_precision_endpoint": equiv1,
            },
            "holdout_split": {
                "split_seed": seed,
                "domains": [
                    {"sample_key": key, "split_seed": seed, "roles": roles}
                    for key in ("A", "B", "C")[:role_count]
                ],
            },
            "_provenance": _prov(),
        }

    def _run(self, name, payload, nested=True):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            base = runs / "confirmatory" / "seed_42" / "x" if nested else runs / "x"
            base.mkdir(parents=True)
            (base / name).write_text(json.dumps(payload))
            return optional_artifact_errors(runs)

    def test_nested_shrinkage_artifact_is_discovered(self):
        failures = self._run(
            "shrinkage_diagnostic.json", self._shrinkage_payload(equiv1=False)
        )
        self.assertTrue(
            any("three-way precision endpoint" in f for f in failures)
        )

    def test_valid_shrinkage_artifact_passes(self):
        failures = self._run(
            "shrinkage_diagnostic.json", self._shrinkage_payload()
        )
        self.assertEqual(failures, [])

    def test_shrinkage_coverage_mismatch_fails(self):
        failures = self._run(
            "shrinkage_diagnostic.json", self._shrinkage_payload(role_count=2)
        )
        self.assertTrue(any("does not cover" in f for f in failures))

    def test_precision_missing_split_seed_fails(self):
        payload = {
            "data_manifest": _data_manifest(),
            "holdout_split": {"split_seed": 42, "domains": _holdout_domains()},
            "_provenance": _prov(),
        }
        failures = self._run("precision_weighted.json", payload)
        self.assertTrue(any("missing split_seed" in f for f in failures))

    def test_valid_precision_artifact_passes(self):
        payload = {
            "split_seed": 42,
            "data_manifest": _data_manifest(),
            "holdout_split": {"split_seed": 42, "domains": _holdout_domains()},
            "_provenance": _prov(),
        }
        failures = self._run("precision_weighted.json", payload)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
