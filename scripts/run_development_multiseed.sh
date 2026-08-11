#!/usr/bin/env bash
# DEVELOPMENT multi-seed runs for RRCL (renamed from run_confirmatory_multiseed.sh).
#
# JHU/SHA/SHB/QNRF and KADID are
# DEVELOPMENT data — their test sets have repeatedly informed mechanism
# analysis and method design.  Runs from this script support development
# decisions only; they are NOT the unseen confirmation required by
# the frozen FDST boundary.
#
# Each seed drives BOTH --sample-seed and --split-seed so training subsets
# and selector holdout partitions re-randomize together.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="${RRCL_BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
SEEDS="${RRCL_SEEDS:-42 43 44 45 46}"
SAFETY_HARM="${RRCL_SAFETY_HARM:-0.0}"

python scripts/preflight_experiments.py

for seed in $SEEDS; do
  echo "===== crowd seed=$seed: original and Pareto-safe selector ====="
  for order in jhu_sha_shb jhu_shb_sha sha_shb_jhu; do
    python run_adaptive_f.py \
      --config "configs/domains_${order}.json" \
      --backbone "$BACKBONE" --img-size 518 --lam 100 \
      --max-per-domain 400 --sample-seed "$seed" --split-seed "$seed" \
      --selector-search grid \
      --out "runs_real/development/seed_${seed}/original_${order}"
    python run_adaptive_f.py \
      --config "configs/domains_${order}.json" \
      --backbone "$BACKBONE" --img-size 518 --lam 100 \
      --max-per-domain 400 --sample-seed "$seed" --split-seed "$seed" \
      --selector-search grid \
      --max-component-relative-harm "$SAFETY_HARM" \
      --out "runs_real/development/seed_${seed}/safe_${order}"
  done

  echo "===== raw precision comparator seed=$seed ====="
  for order in jhu_sha_shb sha_shb_jhu qnrf_sha_shb qnrf_shb_sha; do
    python scripts/run_precision_weighted.py \
      --config "configs/domains_${order}.json" \
      --backbone "$BACKBONE" --img-size 518 --lam 100 \
      --max-per-domain 400 --sample-seed "$seed" --split-seed "$seed" \
      --out "runs_real/development/seed_${seed}/precision_${order}"
  done
  # NOTE: shrinkage-selector runs are added here ONLY after the C4 verdict
  # on the seed-42 gamma diagnostics authorizes a selector design.

  echo "===== QNRF negative controls seed=$seed ====="
  for order in qnrf_sha_shb qnrf_shb_sha sha_shb_qnrf; do
    python run_adaptive_f.py \
      --config "configs/domains_${order}.json" \
      --backbone "$BACKBONE" --img-size 518 --lam 100 \
      --max-per-domain 400 --sample-seed "$seed" --split-seed "$seed" \
      --selector-search grid \
      --out "runs_real/development/seed_${seed}/original_${order}"
    python run_adaptive_f.py \
      --config "configs/domains_${order}.json" \
      --backbone "$BACKBONE" --img-size 518 --lam 100 \
      --max-per-domain 400 --sample-seed "$seed" --split-seed "$seed" \
      --selector-search grid \
      --max-component-relative-harm "$SAFETY_HARM" \
      --out "runs_real/development/seed_${seed}/safe_${order}"
  done

  echo "===== KADID reference-group split seed=$seed ====="
  python run_iqa_cl.py \
    --config configs/domains_iqa_kadid.json \
    --backbone "$BACKBONE" --img-size 518 --lam 100 \
    --max-per-domain 500 --split-seed "$seed" \
    --selector-search grid \
    --out "runs_real/development/kadid_seed_${seed}"
  python run_iqa_cl.py \
    --config configs/domains_iqa_kadid.json \
    --backbone "$BACKBONE" --img-size 518 --lam 100 \
    --max-per-domain 500 --split-seed "$seed" \
    --selector-search grid \
    --max-component-relative-harm "$SAFETY_HARM" \
    --out "runs_real/development/kadid_safe_seed_${seed}"
done

echo "Development runs complete."
