#!/usr/bin/env bash
# Full RRCL paper-closure run. Use this when you want a complete, reproducible
# result archive for paper tables/figures.
#
# Recommended launch on AutoDL:
#   cd /root/recursive_ridge_cl_autodl_min
#   nohup bash run_paper_closure_all.sh > runs_real/run_paper_closure_all.nohup 2>&1 &
#
# The script runs sequentially to avoid IO/GPU contention.
set -u
cd "$(dirname "$0")"
mkdir -p runs_real

echo "===== [1/7] main norm-ablation ====="
bash run_normabl_main_all.sh

echo "===== [2/7] adaptive f ====="
bash run_adaptf2_all.sh

echo "===== [3/7] baselines ====="
bash run_baselines_all.sh

echo "===== [4/7] QNRF negative cases ====="
bash run_qnrf_all.sh

echo "===== [5/7] QNRF capacity projection ====="
bash run_proj_qnrf.sh

echo "===== [6/7] age transfer ====="
bash run_age_all.sh

echo "===== [7/7] theory diagnostic ====="
bash run_theory_all.sh

echo "ALL PAPER-CLOSURE RUNS DONE"
