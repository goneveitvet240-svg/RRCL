"""Unit tests for the stable-hash fit/validation holdout split.

Pilot audit finding (2026-07-26): the previous ``i % 5`` stream-order rule
gave SHA (300 train images) and SHB (exactly 400) an identical validation set
for every sample seed, so multi-seed runs never re-partitioned them.  These
tests pin the replacement contract.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_split import (
    ID_SOURCE_IMAGE,
    ID_SOURCE_INDEX,
    ROLE_FIT,
    ROLE_GAMMA_VAL,
    ROLE_PRECISION_VAL,
    SPLIT_ALGORITHM_VERSION,
    domain_sample_key,
    holdout_role,
    is_validation,
    iter_stream_with_ids,
    require_both_partitions,
    require_roles_nonempty,
    roles_manifest,
    split_manifest,
)


def _sha_like_ids(n=300, stem="IMG"):
    """Fixed-filename id lists that model SHA/SHB (no subsampling ever)."""
    return [f"{stem}_{i:04d}.jpg" for i in range(1, n + 1)]


def _assignment(ids, key, seed, val_every=5):
    return {i for i in ids if is_validation(key, i, seed, val_every)}


class TestStableAssignment(unittest.TestCase):

    def test_deterministic_same_seed_same_partition(self):
        ids = _sha_like_ids()
        a = _assignment(ids, "ShanghaiTech-A", 42)
        b = _assignment(ids, "ShanghaiTech-A", 42)
        self.assertEqual(a, b)

    def test_different_split_seed_changes_fixed_dataset_partition(self):
        """SHA/SHB regression: fixed id lists MUST re-partition across seeds."""
        for key, n in (("ShanghaiTech-A", 300), ("ShanghaiTech-B", 400)):
            ids = _sha_like_ids(n, key)
            partitions = {frozenset(_assignment(ids, key, seed))
                          for seed in (42, 43, 44, 45, 46)}
            self.assertGreater(
                len(partitions), 1,
                f"{key}: seeds 42-46 produced identical validation sets",
            )

    def test_order_independent(self):
        ids = _sha_like_ids()
        rng = np.random.default_rng(0)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        self.assertEqual(
            _assignment(ids, "JHU-v2.0", 42),
            _assignment(shuffled, "JHU-v2.0", 42),
        )

    def test_expected_fraction(self):
        ids = [f"img_{i}.jpg" for i in range(5000)]
        val = _assignment(ids, "QNRF", 42)
        fraction = len(val) / len(ids)
        self.assertGreater(fraction, 0.15)
        self.assertLess(fraction, 0.25)

    def test_domain_and_seed_enter_hash(self):
        ids = _sha_like_ids(200)
        self.assertNotEqual(_assignment(ids, "A", 42), _assignment(ids, "B", 42))
        self.assertNotEqual(_assignment(ids, "A", 42), _assignment(ids, "A", 43))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            is_validation("", "img.jpg", 42)
        with self.assertRaises(ValueError):
            is_validation("A", "", 42)
        with self.assertRaises(ValueError):
            is_validation("A", "img.jpg", 42, val_every=1)


class TestThreeWayRoles(unittest.TestCase):
    """Diagnostic three-way split (fixes the validation double-dipping)."""

    def test_roles_partition_the_two_way_validation_set(self):
        """role != fit  <=>  is_validation; the two val roles are disjoint."""
        ids = _sha_like_ids(2000)
        for seed in (42, 43):
            for image_id in ids:
                role = holdout_role("ShanghaiTech-A", image_id, seed)
                two_way = is_validation("ShanghaiTech-A", image_id, seed)
                self.assertEqual(role != ROLE_FIT, two_way)

    def test_all_three_roles_populated_and_deterministic(self):
        ids = _sha_like_ids(2000)
        roles = {}
        for image_id in ids:
            roles.setdefault(holdout_role("QNRF", image_id, 42), []).append(image_id)
        self.assertEqual(
            set(roles), {ROLE_FIT, ROLE_PRECISION_VAL, ROLE_GAMMA_VAL}
        )
        # ~10% each validation role
        for role in (ROLE_PRECISION_VAL, ROLE_GAMMA_VAL):
            fraction = len(roles[role]) / len(ids)
            self.assertGreater(fraction, 0.06)
            self.assertLess(fraction, 0.14)
        roles_again = {}
        for image_id in ids:
            roles_again.setdefault(
                holdout_role("QNRF", image_id, 42), []
            ).append(image_id)
        self.assertEqual(roles, roles_again)

    def test_roles_manifest_disjointness_and_gate(self):
        manifest = roles_manifest(
            "A", 42, 5,
            {ROLE_FIT: ["a.jpg"], ROLE_PRECISION_VAL: ["b.jpg"],
             ROLE_GAMMA_VAL: ["c.jpg"]},
            ID_SOURCE_IMAGE,
        )
        require_roles_nonempty(
            manifest, (ROLE_FIT, ROLE_PRECISION_VAL, ROLE_GAMMA_VAL), "here"
        )
        with self.assertRaises(ValueError):
            roles_manifest(
                "A", 42, 5,
                {ROLE_FIT: ["a.jpg"], ROLE_PRECISION_VAL: ["a.jpg"]},
                ID_SOURCE_IMAGE,
            )
        empty = roles_manifest(
            "A", 42, 5,
            {ROLE_FIT: ["a.jpg"], ROLE_PRECISION_VAL: [], ROLE_GAMMA_VAL: ["c.jpg"]},
            ID_SOURCE_IMAGE,
        )
        with self.assertRaises(RuntimeError):
            require_roles_nonempty(
                empty, (ROLE_FIT, ROLE_PRECISION_VAL, ROLE_GAMMA_VAL), "here"
            )


class TestManifest(unittest.TestCase):

    def test_manifest_records_partition(self):
        ids = _sha_like_ids(50)
        val = sorted(_assignment(ids, "A", 42))
        fit = sorted(set(ids) - set(val))
        manifest = split_manifest("A", 42, 5, fit, val, ID_SOURCE_IMAGE)
        self.assertEqual(manifest["fit_count"], len(fit))
        self.assertEqual(manifest["val_count"], len(val))
        self.assertEqual(manifest["algorithm_version"], SPLIT_ALGORITHM_VERSION)
        self.assertEqual(manifest["id_source"], ID_SOURCE_IMAGE)
        # Hashes are membership-sensitive
        other = split_manifest("A", 42, 5, fit[1:], val + [fit[0]], ID_SOURCE_IMAGE)
        self.assertNotEqual(manifest["val_ids_sha256"], other["val_ids_sha256"])

    def test_manifest_rejects_overlap(self):
        with self.assertRaises(ValueError):
            split_manifest("A", 42, 5, ["x.jpg", "y.jpg"], ["y.jpg"], ID_SOURCE_IMAGE)

    def test_require_both_partitions(self):
        good = split_manifest("A", 42, 5, ["a.jpg"], ["b.jpg"], ID_SOURCE_IMAGE)
        require_both_partitions(good, "here")
        empty_val = split_manifest("A", 42, 5, ["a.jpg"], [], ID_SOURCE_IMAGE)
        with self.assertRaises(RuntimeError):
            require_both_partitions(empty_val, "here")


class TestStreamAdapters(unittest.TestCase):

    def test_with_ids_stream_is_used_when_available(self):
        class _Dom:
            def stream(self, split, t, with_ids=False):
                assert with_ids
                yield "X", "Y", 4, "img_0007.jpg"

        rows = list(iter_stream_with_ids(_Dom(), "train", 0))
        self.assertEqual(rows[0][3], "img_0007.jpg")
        self.assertEqual(rows[0][4], ID_SOURCE_IMAGE)

    def test_fallback_to_positional_ids(self):
        class _Dom:
            def stream(self, split, t):
                yield "X", "Y", 4
                yield "X2", "Y2", 4

        rows = list(iter_stream_with_ids(_Dom(), "train", 0))
        self.assertEqual([r[3] for r in rows], ["index:0", "index:1"])
        self.assertEqual(rows[0][4], ID_SOURCE_INDEX)

    def test_domain_sample_key_prefers_spec_key(self):
        class _Spec:
            sample_key = "JHU-v2.0"
            name = "JHU"

        class _Dom:
            domains = [_Spec()]

        self.assertEqual(domain_sample_key(_Dom(), 0), "JHU-v2.0")

        class _Bare:
            pass

        self.assertEqual(domain_sample_key(_Bare(), 3), "domain:3")


class TestRealDatasetStreamSignature(unittest.TestCase):

    def test_real_stream_accepts_with_ids(self):
        """RealCountingDomains.stream must expose the with_ids parameter."""
        import inspect
        try:
            from datasets_real import RealCountingDomains
        except Exception as exc:  # torch-free dev boxes
            self.skipTest(f"datasets_real unavailable here: {exc}")

        signature = inspect.signature(RealCountingDomains.stream)
        self.assertIn("with_ids", signature.parameters)


if __name__ == "__main__":
    unittest.main()
