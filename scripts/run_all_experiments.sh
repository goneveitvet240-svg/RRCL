#!/usr/bin/env bash
# Full RRCL experiment run. Use this when you want a complete, reproducible
# result archive for paper tables/figures.
#
# Example:
#   nohup bash scripts/run_all_experiments.sh \
#     > runs_real/run_all_experiments.nohup 2>&1 &
#
# The script runs sequentially to avoid IO/GPU contention.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs_real

echo "===== [0/11] preflight ====="
python scripts/preflight_experiments.py --check-data
python -m unittest discover -s tests -v

echo "===== [1/11] main norm-ablation ====="
bash scripts/run_normabl_main_all.sh

echo "===== [2/11] adaptive f ====="
bash scripts/run_adaptf2_all.sh

echo "===== [3/11] baselines ====="
bash scripts/run_baselines_all.sh

echo "===== [4/11] VFF-RLS baselines ====="
bash scripts/run_vff_all.sh

echo "===== [5/11] QNRF negative cases ====="
bash scripts/run_qnrf_all.sh

echo "===== [6/11] QNRF capacity projection ====="
bash scripts/run_proj_qnrf.sh

echo "===== [7/11] age transfer ====="
bash scripts/run_age_all.sh

echo "===== [8/11] theory diagnostic ====="
bash scripts/run_theory_all.sh

echo "===== [9/11] extended regression tasks ====="
bash scripts/run_new_tasks.sh

echo "===== [10/11] DOS-ELM and SIFt-RLS analytic baselines ====="
bash scripts/run_analytic_all.sh

echo "===== [11/11] release validation ====="
python scripts/validate_release.py

echo "ALL EXPERIMENT RUNS DONE"
