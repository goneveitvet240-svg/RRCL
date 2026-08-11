import json
import math
import tempfile
import unittest
from pathlib import Path

from result_io import dump_result


class StrictResultIOTest(unittest.TestCase):
    def test_nan_matrix_slots_become_standard_json_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            dump_result(path, {"matrix": [[1.0, math.nan]]})
            raw = path.read_text()
            self.assertNotIn("NaN", raw)
            self.assertEqual(json.loads(raw), {"matrix": [[1.0, None]]})

    def test_infinity_is_rejected_without_partial_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                dump_result(path, {"bad": math.inf})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
