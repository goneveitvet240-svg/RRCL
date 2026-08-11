"""Negative tests for the confirmatory summarizer's split/manifest gates.

Codex review 2026-07-26: the split_seed and holdout-manifest checks added to
scripts/summarize_confirmatory.py must demonstrably FAIL on wrong seeds,
missing hashes and mismatched manifests — otherwise an invalid multi-seed run
could still summarize as PASSED.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_confirmatory import (
    _holdout_by_key,
    _validate_method_identity,
    _validate_order_identity,
    _validate_seed,
)


def _payload(seed, split_seed=None, holdout=True, fit_hash="aa", val_hash="bb",
             rel_f1=0.5, train_hash="t1"):
    body = {
        "rel_f1": rel_f1,
        "data_manifest": {
            "sample_seed": seed,
            "domains": [
                {"sample_key": "A", "train": {"ids_sha256": train_hash},
                 "test": {"ids_sha256": "te"}},
            ],
        },
    }
    if split_seed is not None:
        body["split_seed"] = split_seed
    if holdout:
        body["adaptive_evidence"] = {
            "holdout_split": {
                "split_seed": split_seed,
                "domains": [
                    {"sample_key": "A", "fit_ids_sha256": fit_hash,
                     "val_ids_sha256": val_hash},
                ],
            },
        }
    else:
        body["adaptive_evidence"] = {}
    return body


class TestValidateSeed(unittest.TestCase):

    def test_correct_payload_passes(self):
        failures = []
        _validate_seed(_payload(43, split_seed=43), "p", 43, "jhu_sha_shb", failures)
        self.assertEqual(failures, [])

    def test_wrong_split_seed_fails(self):
        failures = []
        _validate_seed(_payload(43, split_seed=42), "p", 43, "jhu_sha_shb", failures)
        self.assertTrue(any("split seed" in f for f in failures))

    def test_missing_split_seed_fails(self):
        failures = []
        _validate_seed(_payload(43, split_seed=None), "p", 43, "jhu_sha_shb", failures)
        self.assertTrue(any("split seed" in f for f in failures))

    def test_missing_holdout_manifest_fails(self):
        failures = []
        _validate_seed(
            _payload(43, split_seed=43, holdout=False), "p", 43, "jhu_sha_shb",
            failures,
        )
        self.assertTrue(any("holdout" in f for f in failures))

    def test_incomplete_holdout_hashes_fail(self):
        failures = []
        _validate_seed(
            _payload(43, split_seed=43, val_hash=""), "p", 43, "jhu_sha_shb",
            failures,
        )
        self.assertTrue(any("incomplete holdout" in f for f in failures))

    def test_kadid_branch_ignores_crowd_split_checks(self):
        failures = []
        payload = {"data_manifest": {"split_seed": [43, 43]}}
        _validate_seed(payload, "p", 43, "kadid", failures)
        self.assertEqual(failures, [])


class TestOrderAndMethodIdentity(unittest.TestCase):

    def test_matching_orders_pass(self):
        loaded = {
            (43, "original", "a"): _payload(43, split_seed=43),
            (43, "original", "b"): _payload(43, split_seed=43),
        }
        failures = []
        _validate_order_identity(loaded, [43], ["original"], ["a", "b"], failures)
        self.assertEqual(failures, [])

    def test_holdout_mismatch_across_orders_fails(self):
        loaded = {
            (43, "original", "a"): _payload(43, split_seed=43, fit_hash="aa"),
            (43, "original", "b"): _payload(43, split_seed=43, fit_hash="zz"),
        }
        failures = []
        _validate_order_identity(loaded, [43], ["original"], ["a", "b"], failures)
        self.assertTrue(any("holdout split manifest differs" in f for f in failures))

    def test_sample_manifest_mismatch_still_fails(self):
        loaded = {
            (43, "original", "a"): _payload(43, split_seed=43, train_hash="t1"),
            (43, "original", "b"): _payload(43, split_seed=43, train_hash="t2"),
        }
        failures = []
        _validate_order_identity(loaded, [43], ["original"], ["a", "b"], failures)
        self.assertTrue(any("sample manifest differs" in f for f in failures))

    def test_original_safe_holdout_mismatch_fails(self):
        loaded = {
            (43, "original", "jhu_sha_shb"): _payload(43, split_seed=43,
                                                      fit_hash="aa"),
            (43, "safe", "jhu_sha_shb"): _payload(43, split_seed=43,
                                                  fit_hash="zz"),
        }
        failures = []
        _validate_method_identity(loaded, [43], ["jhu_sha_shb"], failures)
        self.assertTrue(any("holdout splits differ" in f for f in failures))

    def test_holdout_by_key_shape(self):
        payload = _payload(43, split_seed=43)
        self.assertEqual(_holdout_by_key(payload), {"A": ("aa", "bb")})


if __name__ == "__main__":
    unittest.main()
