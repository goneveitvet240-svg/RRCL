import os
import unittest
from unittest import mock

from features import resolve_device


class DeviceResolutionTest(unittest.TestCase):
    def test_explicit_device_wins(self):
        with mock.patch.dict(os.environ, {"RRCL_DEVICE": "cpu"}):
            self.assertEqual(resolve_device("mps"), "mps")

    def test_environment_device_is_supported(self):
        with mock.patch.dict(os.environ, {"RRCL_DEVICE": "cpu"}):
            self.assertEqual(resolve_device(), "cpu")

    def test_auto_prefers_cuda(self):
        with mock.patch.dict(os.environ, {"RRCL_DEVICE": "auto"}), mock.patch(
            "features.torch.cuda.is_available", return_value=True
        ):
            self.assertEqual(resolve_device(), "cuda")

    def test_auto_uses_mps_when_cuda_is_absent(self):
        with mock.patch.dict(os.environ, {"RRCL_DEVICE": "auto"}), mock.patch(
            "features.torch.cuda.is_available", return_value=False
        ), mock.patch("features.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device(), "mps")

    def test_auto_falls_back_to_cpu(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "features.torch.cuda.is_available", return_value=False
        ), mock.patch("features.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(resolve_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
