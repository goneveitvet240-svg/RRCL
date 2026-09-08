#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RRCL_KADID_ROOT="${RRCL_KADID_ROOT:-/root/autodl-tmp/datasets/KADID10k/extracted/kadid10k}"
export RRCL_FEATURE_CACHE_ROOT="${RRCL_FEATURE_CACHE_ROOT:-/root/autodl-tmp/rrcl_feature_cache}"
PYTHON_BIN="${RRCL_PYTHON:-python3}"

"$PYTHON_BIN" scripts/check_autodl_environment.py --require-cuda --require-kadid

echo "[1/3] Reusing or completing the frozen KADID feature cache"
"$PYTHON_BIN" -u scripts/precompute_kadid_features.py --device cuda

echo "[2/3] Running the frozen KADID scale-equivalence audit v2"
"$PYTHON_BIN" -u scripts/run_kadid_scale_audit_v2.py --device cuda

echo "[3/3] Validating the scale-audit artifact"
"$PYTHON_BIN" scripts/validate_kadid_scale_audit_v2.py \
  runs_real/new_method_kadid_scale_audit_v2/main/kadid_scale_audit_v2.json

echo "KADID scale audit v2 completed and validated."
