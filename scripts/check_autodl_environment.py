#!/usr/bin/env python3
"""Report whether the current Python environment is ready for RRCL on AutoDL."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


PACKAGES = ("numpy", "PIL", "torch", "torchvision", "timm", "h5py", "scipy")


def package_version(name):
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-kadid", action="store_true")
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "errors": [],
    }
    for name in PACKAGES:
        try:
            report["packages"][name] = package_version(name)
        except Exception as exc:
            report["errors"].append(f"cannot import {name}: {exc}")

    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_device_count"] = int(torch.cuda.device_count())
        report["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if args.require_cuda and not report["cuda_available"]:
            report["errors"].append("CUDA is required but torch.cuda.is_available() is false")
    except Exception:
        report["cuda_available"] = False

    kadid_value = os.environ.get("RRCL_KADID_ROOT", "")
    kadid_root = Path(kadid_value).expanduser() if kadid_value else None
    report["kadid_root"] = str(kadid_root) if kadid_root else None
    report["kadid_images_present"] = bool(kadid_root and (kadid_root / "images").is_dir())
    report["kadid_csv_present"] = bool(kadid_root and (kadid_root / "dmos.csv").is_file())
    if args.require_kadid and not (
        report["kadid_images_present"] and report["kadid_csv_present"]
    ):
        report["errors"].append(
            "RRCL_KADID_ROOT must contain images/ and dmos.csv"
        )

    cache_value = os.environ.get("RRCL_FEATURE_CACHE_ROOT", "")
    cache_root = Path(cache_value).expanduser() if cache_value else None
    report["feature_cache_root"] = str(cache_root) if cache_root else None
    if cache_root:
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            probe = cache_root / ".rrcl_write_probe"
            probe.touch()
            probe.unlink()
            report["feature_cache_writable"] = True
        except OSError as exc:
            report["feature_cache_writable"] = False
            report["errors"].append(f"feature cache is not writable: {exc}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
