#!/usr/bin/env bash
# Compatibility entry point for the frozen multi-seed calibration protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

exec bash scripts/run_fstar_calibration_multiseed.sh
