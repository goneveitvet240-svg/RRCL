#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RRCL_KADID_ROOT="${RRCL_KADID_ROOT:-/root/autodl-tmp/datasets/KADID10k/extracted/kadid10k}"
export RRCL_FEATURE_CACHE_ROOT="${RRCL_FEATURE_CACHE_ROOT:-/root/autodl-tmp/rrcl_feature_cache}"
PYTHON_BIN="${RRCL_PYTHON:-python3}"

"$PYTHON_BIN" scripts/check_autodl_environment.py --require-cuda --require-kadid

echo "[1/3] Precomputing or resuming KADID DINOv2 features"
"$PYTHON_BIN" -u scripts/precompute_kadid_features.py --device cuda

echo "[2/3] Running the frozen KADID development-v1 baseline audit"
"$PYTHON_BIN" -u scripts/run_new_method_kadid_baselines.py --device cuda

echo "[3/3] Validating the result artifact"
"$PYTHON_BIN" scripts/validate_new_method_kadid_baselines.py \
  runs_real/new_method_kadid_development_v1/main/kadid_baseline_audit.json

echo "KADID development run completed and validated."
