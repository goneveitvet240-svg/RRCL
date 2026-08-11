#!/usr/bin/env bash
# Matched analytic forgetting baselines. These runs reuse cached DINOv2
# features when the main paper-closure experiments have already run.
set -euo pipefail
cd "$(dirname "$0")/.."

for name in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu; do
  python run_analytic_scalar_baselines.py \
    --task crowd_image \
    --config "configs/domains_${name}.json" \
    --img-size 518 --lam 100 --max-per-domain 400 \
    --out "runs_real/analytic_crowd_${name}"
done

for name in age_utk_young_old age_agedb_utk age_utk_agedb; do
  python run_analytic_scalar_baselines.py \
    --task age \
    --config "configs/domains_${name}.json" \
    --img-size 518 --lam 100 --max-per-domain 400 \
    --out "runs_real/analytic_${name}"
done

python run_analytic_scalar_baselines.py \
  --task ava \
  --config configs/domains_ava.json \
  --img-size 518 --lam 100 --max-per-domain 400 \
  --out runs_real/analytic_ava

python run_analytic_scalar_baselines.py \
  --task iqa \
  --config configs/domains_iqa_kadid.json \
  --img-size 518 --lam 100 --max-per-domain 400 \
  --out runs_real/analytic_kadid
