#!/usr/bin/env bash
# Capacity check: re-run the magnitude/drift norm-ablation with a fixed random
# projection (RanPAC-style, RRCL_PROJ_DIM) to rule out "the linear head underfits
# dense crowds so drift is invisible". Uses cached DINOv2 features (projection is
# applied at accumulation time, no re-extraction). Results -> runs_real/normabl_proj_*.
# Disconnect-safe with nohup.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RRCL_PROJ_DIM=4096          # fixed random ReLU features -> closed-form capacity
COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--patch-target dct5 --alpha 0.25 \
--forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 --max-per-domain 400 \
--sample-seed 42 --target-norms none,mean"
for cfg in qnrf_sha_shb qnrf_shb_sha jhu_sha_shb; do
  echo "==== proj4096 norm-ablation $cfg ===="
  python -u run_real_norm_ablation.py --config configs/domains_${cfg}.json $COMMON \
    --out runs_real/normabl_proj_${cfg} > runs_real/normabl_proj_${cfg}.log 2>&1
  echo "==== done $cfg ===="
done
echo "ALL DONE"
