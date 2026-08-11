"""Regression tests for crash-safe generated feature caches."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cache_io import atomic_save_array, load_array_cache


class AtomicArrayCacheTest(unittest.TestCase):

    def test_round_trip_and_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "nested" / "feature.npy"
            expected = np.arange(12, dtype=np.float32).reshape(3, 4)

            atomic_save_array(destination, expected)

            np.testing.assert_array_equal(load_array_cache(destination), expected)
            self.assertEqual(
                [path.name for path in destination.parent.iterdir()],
                ["feature.npy"],
            )

    def test_failed_write_never_exposes_partial_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "feature.npy"

            def fail_after_partial_write(stream, _array):
                stream.write(b"partial")
                raise OSError("simulated disk full")

            with mock.patch("cache_io.np.save", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_save_array(destination, np.ones((2, 2)))

            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failed_replacement_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "feature.npy"
            original = np.asarray([1.0, 2.0])
            atomic_save_array(destination, original)

            with mock.patch("cache_io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    atomic_save_array(destination, np.asarray([9.0]))

            np.testing.assert_array_equal(load_array_cache(destination), original)
            self.assertEqual(
                [path.name for path in Path(tmp).iterdir()],
                ["feature.npy"],
            )

    def test_truncated_cache_is_removed_and_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "feature.npy"
            destination.write_bytes(b"\x93NUMPY")

            self.assertIsNone(load_array_cache(destination))
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
