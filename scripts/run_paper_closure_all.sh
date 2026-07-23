#!/usr/bin/env bash
# Full RRCL paper-closure run. Use this when you want a complete, reproducible
# result archive for paper tables/figures.
#
# Recommended launch on AutoDL:
#   cd /root/recursive_ridge_cl_autodl_min
#   nohup bash run_paper_closure_all.sh > runs_real/run_paper_closure_all.nohup 2>&1 &
#
# The script runs sequentially to avoid IO/GPU contention.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs_real

echo "===== [0/10] preflight ====="
python scripts/preflight_experiments.py --check-data
python -m unittest discover -s tests -v

echo "===== [1/10] main norm-ablation ====="
bash scripts/run_normabl_main_all.sh

echo "===== [2/10] adaptive f ====="
bash scripts/run_adaptf2_all.sh

echo "===== [3/10] baselines ====="
bash scripts/run_baselines_all.sh

echo "===== [4/10] VFF-RLS baselines ====="
bash scripts/run_vff_all.sh

echo "===== [5/10] QNRF negative cases ====="
bash scripts/run_qnrf_all.sh

echo "===== [6/10] QNRF capacity projection ====="
bash scripts/run_proj_qnrf.sh

echo "===== [7/10] age transfer ====="
bash scripts/run_age_all.sh

echo "===== [8/10] theory diagnostic ====="
bash scripts/run_theory_all.sh

echo "===== [9/10] extended regression tasks ====="
bash scripts/run_new_tasks.sh

echo "===== [10/10] release validation ====="
python scripts/validate_release.py

echo "ALL PAPER-CLOSURE RUNS DONE"
