#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${RRCL_PYTHON:-python3}"

echo "[1/4] Checking the AutoDL base image"
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is absent. Start from an AutoDL PyTorch/CUDA image, then rerun this script."
    ) from exc
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("The installed PyTorch cannot access a CUDA GPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[2/4] Installing RRCL Python dependencies without replacing CUDA PyTorch"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements-autodl.txt

echo "[3/4] Running the environment report"
"$PYTHON_BIN" scripts/check_autodl_environment.py --require-cuda

echo "[4/4] Running the repository test suite"
"$PYTHON_BIN" -m pytest -q

echo "AutoDL environment is ready."
