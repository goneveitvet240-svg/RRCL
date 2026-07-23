#!/usr/bin/env bash
# Main crowd-counting magnitude-vs-drift norm-ablation on the 4 core orders.
# Outputs -> runs_real/normabl_<cfg>/norm_ablation.json
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--patch-target dct5 --alpha 0.25 \
--forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 --max-per-domain 400 \
--sample-seed 42 --target-norms none,mean"

mkdir -p runs_real
for cfg in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu; do
  echo "==== norm-ablation $cfg ===="
  python -u run_real_norm_ablation.py --config configs/domains_${cfg}.json $COMMON \
    --out runs_real/normabl_${cfg} > runs_real/normabl_${cfg}.log 2>&1
  echo "==== done $cfg ===="
done
echo "ALL DONE"
