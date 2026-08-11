import tempfile
import unittest
from pathlib import Path

from datasets_age import AgeDomains, AgeSpec


class AgeSplitTest(unittest.TestCase):
    def test_agedb_subjects_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for identity in range(10):
                for age in (20, 30):
                    (root / f"{identity}_person{identity}_{age}_m.jpg").touch()
            domains = object.__new__(AgeDomains)
            domains.domains = [
                AgeSpec(name="AgeDB", kind="agedb", root=str(root))
            ]
            domains.test_frac = 0.2
            domains.max_per_domain = None
            train = domains._selected_pairs("train", 0)
            test = domains._selected_pairs("test", 0)
            train_people = {Path(path).stem.split("_")[1] for path, _ in train}
            test_people = {Path(path).stem.split("_")[1] for path, _ in test}
            self.assertFalse(train_people & test_people)
            self.assertEqual(len(test_people), 2)


if __name__ == "__main__":
    unittest.main()
