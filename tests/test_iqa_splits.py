import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets_iqa import (
    IQADomains,
    IQASpec,
    _load_csv_pairs,
    _train_test_split,
    build_iqa_config,
)
from run_iqa_cl import grouped_validation_mask


class KADIDSplitTest(unittest.TestCase):
    def test_config_paths_expand_environment_variables(self):
        with mock.patch.dict(os.environ, {"RRCL_DATA_ROOT": "/tmp/rrcl-data"}):
            specs = build_iqa_config(
                {
                    "domains": [
                        {
                            "name": "portable",
                            "kind": "csv",
                            "root": "${RRCL_DATA_ROOT}/images",
                            "csv": "${RRCL_DATA_ROOT}/scores.csv",
                        }
                    ]
                }
            )
        self.assertEqual(specs[0].root, "/tmp/rrcl-data/images")
        self.assertEqual(specs[0].csv, "/tmp/rrcl-data/scores.csv")

    def test_feature_cache_can_live_under_persistent_root(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "datasets_iqa.DinoFeatureExtractor"
        ):
            persistent = Path(tmp) / "persistent-cache"
            with mock.patch.dict(
                os.environ, {"RRCL_FEATURE_CACHE_ROOT": str(persistent)}
            ):
                domains = IQADomains([], backbone="toy", img_size=32)
            self.assertEqual(
                Path(domains.cache_dir), persistent / "iqa" / "toy_img32"
            )
            self.assertTrue(Path(domains.cache_dir).is_dir())

    def test_official_csv_derives_type_and_keeps_references_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            csv_path = root / "dmos.csv"
            rows = []
            for ref in range(1, 6):
                for distortion in range(1, 4):
                    name = f"I{ref:02d}_{distortion:02d}_01.png"
                    (images / name).touch()
                    rows.append(
                        {"dist_img": name, "ref_img": f"I{ref:02d}.png", "dmos": "3.0"}
                    )
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dist_img", "ref_img", "dmos"])
                writer.writeheader()
                writer.writerows(rows)

            spec = IQASpec(
                name="KADID-Test",
                kind="csv",
                root=str(images),
                csv=str(csv_path),
                img_col="dist_img",
                mos_col="dmos",
                filter_col="dist_type",
                filter_val="1,2",
                group_col="ref_img",
                train_ratio=0.8,
                split_seed=42,
            )
            pairs = _load_csv_pairs(spec)
            self.assertEqual(len(pairs), 10)
            train, test = _train_test_split(pairs, spec.train_ratio, spec.split_seed)
            train_refs = {record[2] for record in train}
            test_refs = {record[2] for record in test}
            self.assertFalse(train_refs & test_refs)
            self.assertEqual(len(train_refs), 4)
            self.assertEqual(len(test_refs), 1)

    def test_selector_validation_holds_out_whole_reference_groups(self):
        groups = ["I01"] * 3 + ["I02"] * 3 + ["I03"] * 3 + ["I04"] * 3
        mask = grouped_validation_mask(groups, val_every=3)
        fit_groups = {group for group, held_out in zip(groups, mask) if not held_out}
        validation_groups = {
            group for group, held_out in zip(groups, mask) if held_out
        }
        self.assertTrue(fit_groups)
        self.assertTrue(validation_groups)
        self.assertFalse(fit_groups & validation_groups)


if __name__ == "__main__":
    unittest.main()
