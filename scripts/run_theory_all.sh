#!/usr/bin/env bash
# Theory validation: f*_pred (train-only) vs f*_oracle (test sweep) on ALL
# crowd + age sequences, image-level scalar regression. Reuses cached features.
# Outputs one CSV: runs_real/theory_points.csv
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CSV=runs_real/theory_points.csv
rm -f "$CSV"
COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 --max-per-domain 400 --sample-seed 42 --csv $CSV"

for cfg in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu qnrf_sha_shb qnrf_shb_sha sha_shb_qnrf; do
  echo "==== crowd $cfg ===="
  python -u run_theory_predict.py --config configs/domains_${cfg}.json --task crowd $COMMON 2>&1 | tail -4
done
for cfg in age_utk_young_old age_agedb_utk age_utk_agedb; do
  echo "==== age $cfg ===="
  python -u run_theory_predict.py --config configs/domains_${cfg}.json --task age $COMMON 2>&1 | tail -4
done
echo "==== CSV ===="
cat "$CSV"
echo "ALL DONE"
