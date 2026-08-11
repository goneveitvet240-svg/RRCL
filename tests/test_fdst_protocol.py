"""Torch-free tests for the pre-result C7-fdst-v3 protocol."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_fdst import (
    C7_PROTOCOL,
    enumerate_video_pairs,
    frame_sort_key,
    read_points,
)
from scripts.preflight_fdst import (
    audit_labels,
    choose_domains,
    fixed_temporal_groups,
    load_scene_map,
    main as preflight_main,
    order_hash,
    temporal_blocks_grouped,
)


class TestFDSTDataHelpers(unittest.TestCase):

    def test_protocol_id_is_new_version(self):
        self.assertEqual(C7_PROTOCOL, "C7-fdst-v3")

    def test_frame_sort_key_is_natural(self):
        names = ["train_data/v/10.jpg", "train_data/v/2.jpg", "train_data/v/1.jpg"]
        self.assertEqual(
            sorted(names, key=frame_sort_key),
            ["train_data/v/1.jpg", "train_data/v/2.jpg", "train_data/v/10.jpg"],
        )

    def test_via_region_dict_and_list_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"regions": {
                "0": {"shape_attributes": {"x": 3, "y": 4}},
                "1": {"shape_attributes": {"cx": 5, "cy": 6}},
            }}))
            np.testing.assert_allclose(read_points(path), [[3, 4], [5, 6]])
            path.write_text(json.dumps({"regions": [
                {"shape_attributes": {"x": 7, "y": 8}}
            ]}))
            np.testing.assert_allclose(read_points(path), [[7, 8]])

    def test_empty_regions_are_valid_zero_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text('{"regions": {}}')
            self.assertEqual(read_points(path).shape, (0, 2))

    def test_nested_via_object_is_supported_but_unknown_schema_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"image-key": {"regions": {
                "0": {"shape_attributes": {"x": 1, "y": 2}}
            }}}))
            np.testing.assert_allclose(read_points(path), [[1, 2]])
            path.write_text('{"unexpected": 1}')
            with self.assertRaises(RuntimeError):
                read_points(path)

    def test_pairing_is_same_directory_same_stem_and_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FDST"
            video = root / "train_data" / "v1"
            video.mkdir(parents=True)
            (root / "test_data" / "hidden").mkdir(parents=True)
            for stem in ("1", "2"):
                (video / f"{stem}.jpg").touch()
                (video / f"{stem}.json").write_text('{"regions": {}}')
            (video / "3.jpg").touch()  # no JSON
            (root / "test_data" / "hidden" / "9.jpg").touch()
            (root / "test_data" / "hidden" / "9.json").write_text('{}')
            videos = enumerate_video_pairs(root)
            self.assertEqual(list(videos), ["v1"])
            self.assertEqual(len(videos["v1"]), 2)

    def test_frames_directly_under_train_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FDST"
            (root / "train_data").mkdir(parents=True)
            (root / "train_data" / "1.jpg").touch()
            (root / "train_data" / "1.json").write_text('{}')
            with self.assertRaises(RuntimeError):
                enumerate_video_pairs(root)


class TestFDSTProtocolSelection(unittest.TestCase):

    def test_fixed_groups_and_split_are_contiguous(self):
        ids = [f"train_data/v/{index:04d}.jpg" for index in range(50)]
        groups = fixed_temporal_groups(ids, 10)
        self.assertEqual([len(group) for group in groups], [10] * 5)
        fit, val, test = temporal_blocks_grouped(groups)
        self.assertEqual(fit + val + test, ids)
        self.assertEqual((len(fit), len(val), len(test)), (30, 10, 10))

    def test_fixed_groups_reject_cross_video_and_short_tail(self):
        with self.assertRaises(RuntimeError):
            fixed_temporal_groups(["train_data/a/1.jpg", "train_data/b/2.jpg"], 10)
        ids = [f"train_data/a/{index:04d}.jpg" for index in range(24)]
        with self.assertRaises(RuntimeError):
            fixed_temporal_groups(ids, 10)

    def test_choose_one_video_per_distinct_scene(self):
        def pairs(video, count):
            return [(f"train_data/{video}/{i:04d}.jpg", f"train_data/{video}/{i:04d}.json")
                    for i in range(count)]
        videos = {"a": pairs("a", 30), "b": pairs("b", 40), "c": pairs("c", 35)}
        scene_map = {"videos": {"a": "s1", "b": "s1", "c": "s2"}}
        selected = choose_domains(videos, scene_map, min_frames=20, k=2)
        self.assertEqual({row["video_id"] for row in selected}, {"b", "c"})
        self.assertEqual(len({row["scene_id"] for row in selected}), 2)

    def test_scene_map_must_cover_disk_exactly(self):
        videos = {"a": [("x", "y")], "b": [("z", "q")]}
        with self.assertRaises(SystemExit):
            choose_domains(videos, {"videos": {"a": "s1"}}, 1, 1)

    def test_scene_map_rejects_unreviewed_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"videos": {"v1": "REVIEW_REQUIRED"}}))
            with self.assertRaises(SystemExit):
                load_scene_map(path)

    def test_scene_map_requires_evidence_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            path.write_text(json.dumps({"videos": {"v1": "s1"}}))
            with self.assertRaises(SystemExit):
                load_scene_map(path)

    def test_order_hash_is_deterministic_and_identity_sensitive(self):
        self.assertEqual(order_hash("s1", "v1"), order_hash("s1", "v1"))
        self.assertNotEqual(order_hash("s1", "v1"), order_hash("s2", "v1"))


class TestFDSTPreflight(unittest.TestCase):

    def _fake_root(self, tmp, counts):
        root = Path(tmp) / "FDST"
        for video, count in counts.items():
            directory = root / "train_data" / video
            directory.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (16, 12)).save(directory / f"{index:04d}.jpg")
                annotation = {"regions": {
                    "0": {"shape_attributes": {"x": 4, "y": 5}}
                }}
                (directory / f"{index:04d}.json").write_text(json.dumps(annotation))
        return root

    def _run(self, tmp, root, mapping, testing=True, skip=True):
        scene_map = Path(tmp) / "scene_map.json"
        scene_map.write_text(json.dumps({
            "evidence": "unit-test fixture",
            "videos": mapping,
        }))
        argv = [
            "preflight_fdst.py", "--root", str(root), "--scene-map", str(scene_map),
            "--min-frames", "50", "--k", "2", "--block-size", "10",
            "--splits-dir", str(Path(tmp) / "splits"),
            "--config-out", str(Path(tmp) / "domains_fdst.json"),
        ]
        if testing:
            argv.append("--testing")
        if skip:
            argv.append("--skip-label-audit")
        with mock.patch.object(sys, "argv", argv):
            preflight_main()

    def test_testing_preflight_materializes_video_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_root(tmp, {"v1": 50, "v2": 60, "v3": 20})
            self._run(tmp, root, {"v1": "s1", "v2": "s2", "v3": "s3"})
            manifest = json.loads((Path(tmp) / "splits" / "manifest.json").read_text())
            self.assertEqual(manifest["protocol"], "C7-fdst-testing")
            self.assertEqual(len(manifest["domains"]), 2)
            self.assertEqual(len({row["scene_id"] for row in manifest["domains"]}), 2)
            self.assertEqual(manifest["rules"]["official_test_partition"], "excluded")
            config = json.loads((Path(tmp) / "domains_fdst.json").read_text())
            self.assertTrue(all(row["kind"] == "fdst_video" for row in config["domains"]))

    def test_label_audit_parses_images_and_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_root(tmp, {"v1": 50, "v2": 50})
            self._run(tmp, root, {"v1": "s1", "v2": "s2"}, skip=False)
            manifest = json.loads((Path(tmp) / "splits" / "manifest.json").read_text())
            self.assertEqual(manifest["label_audit"]["frames_checked"], 100)
            self.assertEqual(manifest["label_audit"]["total_points"], 100)
            self.assertEqual(manifest["label_audit"]["clipped_points"], 0)

    def test_v3_label_audit_records_near_frame_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FDST"
            video = root / "train_data" / "v1"
            video.mkdir(parents=True)
            Image.new("RGB", (1000, 500)).save(video / "0001.jpg")
            (video / "0001.json").write_text(json.dumps({"regions": {
                "0": {"shape_attributes": {"x": -8, "y": 20}},
                "1": {"shape_attributes": {"x": 50, "y": 60}},
            }}))
            domains = [{"pairs": [
                ("train_data/v1/0001.jpg", "train_data/v1/0001.json")
            ]}]
            audit = audit_labels(root, domains)
            self.assertEqual(audit["clipped_points"], 1)
            self.assertEqual(audit["frames_with_clipped_points"], 1)
            self.assertEqual(audit["max_overflow_x_px"], 8.0)
            self.assertEqual(audit["hard_out_of_bounds_points"], 0)

    def test_v3_label_audit_rejects_beyond_relative_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FDST"
            video = root / "train_data" / "v1"
            video.mkdir(parents=True)
            Image.new("RGB", (1000, 500)).save(video / "0001.jpg")
            (video / "0001.json").write_text(json.dumps({"regions": {
                "0": {"shape_attributes": {"x": -11, "y": 20}},
            }}))
            domains = [{"pairs": [
                ("train_data/v1/0001.jpg", "train_data/v1/0001.json")
            ]}]
            with self.assertRaises(SystemExit):
                audit_labels(root, domains)

    def test_non_default_parameters_without_testing_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_root(tmp, {"v1": 50, "v2": 50})
            with self.assertRaises(SystemExit):
                self._run(tmp, root, {"v1": "s1", "v2": "s2"}, testing=False)

    def test_regeneration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fake_root(tmp, {"v1": 50, "v2": 50})
            mapping = {"v1": "s1", "v2": "s2"}
            self._run(tmp, root, mapping)
            with self.assertRaises(SystemExit):
                self._run(tmp, root, mapping)


class TestFDSTRunnerGates(unittest.TestCase):

    def test_repository_appendix_is_accepted(self):
        from scripts.run_fdst_c7 import load_appendix
        repo = Path(__file__).resolve().parents[1]
        appendix, digest = load_appendix(repo)
        self.assertEqual(appendix["protocol"], "C7-fdst-v3")
        self.assertEqual(len(digest), 64)

    def test_testing_manifest_is_rejected(self):
        from scripts.run_fdst_c7 import check_manifest
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "splits").mkdir()
            (repo / "splits" / "manifest.json").write_text(
                json.dumps({"protocol": "C7-fdst-testing"})
            )
            with self.assertRaises(SystemExit):
                check_manifest(repo, "splits")

    def test_v3_runner_rejects_missing_label_audit_before_model_access(self):
        from scripts.run_fdst_c7 import check_manifest
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "splits").mkdir()
            (repo / "splits" / "manifest.json").write_text(json.dumps({
                "protocol": "C7-fdst-v3",
                "rules": {
                    "eligible_partition": "train_data only",
                    "official_test_partition": "excluded",
                    "domain_unit": "one video; six distinct scenes",
                    "target_scope": "full_frame",
                    "min_frames": 120,
                    "k": 6,
                    "block_size": 10,
                    "temporal_blocks": "~60/20/20 contiguous, fixed-block-aligned",
                    "order_salt": "order-v3",
                    "point_coordinate_policy": "finite; clip to frame if within 1% per axis; reject otherwise",
                    "point_tolerance_fraction_per_axis": 0.01,
                },
                "label_audit": {},
            }))
            with self.assertRaisesRegex(SystemExit, "label audit"):
                check_manifest(repo, "splits")

    def test_single_shot_result_and_lock_are_enforced(self):
        from scripts.run_fdst_c7 import acquire_single_shot_lock, enforce_single_shot
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            enforce_single_shot(repo)
            lock = acquire_single_shot_lock(repo, commit="abc")
            self.assertTrue(lock.exists())
            with self.assertRaises(SystemExit):
                acquire_single_shot_lock(repo, commit="abc")
            (repo / "runs_real" / "fdst_c7" / "fdst_c7.json").write_text("{}")
            with self.assertRaises(SystemExit):
                enforce_single_shot(repo)

    def test_fdst_bootstrap_uses_fixed_blocks(self):
        from scripts.run_fdst_c7 import require_fdst_groups
        ids = [f"train_data/v/{index:04d}.jpg" for index in range(30)]
        groups = require_fdst_groups(ids, "test")
        self.assertEqual([len(group) for group in groups], [10, 10, 10])


if __name__ == "__main__":
    unittest.main()
