#!/usr/bin/env bash
# Run the sufficient-statistic adaptive-f selector on all 4 orders, sequentially.
# Disconnect-safe when launched with nohup. Outputs -> runs_real/adaptf2_<cfg>/.
set -u
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--patch-target dct5 --alpha 0.25 \
--forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 --max-per-domain 400"
for cfg in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu; do
  echo "==== running $cfg ===="
  python -u run_adaptive_f.py --config domains_${cfg}.json $COMMON \
    --out runs_real/adaptf2_${cfg} > runs_real/adaptf2_${cfg}.log 2>&1
  echo "==== done $cfg ===="
done
echo "ALL DONE"
