#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs_real

COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 --patch-target dct5 --alpha 0.25 --max-per-domain 400 --sample-seed 42 --selector-search grid"
for cfg in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu; do
  echo "===== VFF baseline: $cfg ====="
  python -u run_vff_baselines.py \
    --config "configs/domains_${cfg}.json" $COMMON \
    --out "runs_real/vff_${cfg}"
done
