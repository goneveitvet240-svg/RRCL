#!/usr/bin/env bash
# Age-estimation transfer test on the 3 current orders.
# Outputs -> runs_real/age_<cfg>/age_result.json
set -u
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--max-per-domain 400 --ranpac-dim 2000 \
--forgets 1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.15,0.1"

mkdir -p runs_real
for cfg in age_utk_young_old age_agedb_utk age_utk_agedb; do
  echo "==== age $cfg ===="
  python -u run_age_cl.py --config domains_${cfg}.json $COMMON \
    --out runs_real/${cfg} > runs_real/${cfg}.log 2>&1
  echo "==== done $cfg ===="
done
echo "ALL DONE"
