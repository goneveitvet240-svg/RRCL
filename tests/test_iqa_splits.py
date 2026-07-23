import csv
import tempfile
import unittest
from pathlib import Path

from datasets_iqa import IQASpec, _load_csv_pairs, _train_test_split


class KADIDSplitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
