#!/usr/bin/env bash
# Development-only first-batch baseline audit. FDST is intentionally absent.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKBONE="${RRCL_BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
SEEDS="${RRCL_SEEDS:-42}"
CONFIGS="${RRCL_CONFIGS:-jhu_sha_shb jhu_shb_sha sha_jhu_shb sha_shb sha_shb_jhu qnrf_sha_shb qnrf_shb_sha sha_shb_qnrf}"
PROJECTION_DIM="${RRCL_PROJECTION_DIM:-2000}"

for seed in $SEEDS; do
  for config_name in $CONFIGS; do
    output="runs_real/new_method_development_v1/seed_${seed}/${config_name}"
    python -u scripts/run_new_method_baselines.py \
      --domain-config "configs/domains_${config_name}.json" \
      --backbone "$BACKBONE" \
      --img-size 518 \
      --max-per-domain 400 \
      --sample-seed "$seed" \
      --split-seed "$seed" \
      --projection-dim "$PROJECTION_DIM" \
      --out "$output"
    python scripts/validate_new_method_baselines.py "$output/baseline_audit.json"
  done
done
