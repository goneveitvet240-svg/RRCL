#!/usr/bin/env bash
# QNRF cross-dataset runs: norm-ablation + adaptive-f on 3 QNRF orders.
# First QNRF stream extracts+caches DINOv2 features (one-time, slow); rest reuse cache.
# Disconnect-safe with nohup.
set -u
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--patch-target dct5 --alpha 0.25 \
--forgets 1.0,0.8,0.6,0.4,0.35,0.3,0.25,0.2,0.15,0.1 --max-per-domain 400"
for cfg in qnrf_sha_shb qnrf_shb_sha sha_shb_qnrf; do
  echo "==== norm-ablation $cfg ===="
  python -u run_real_norm_ablation.py --config domains_${cfg}.json $COMMON \
    --target-norms none,mean --out runs_real/normabl_${cfg} \
    > runs_real/normabl_${cfg}.log 2>&1
  echo "==== adaptive $cfg ===="
  python -u run_adaptive_f.py --config domains_${cfg}.json $COMMON \
    --out runs_real/adaptf2_${cfg} > runs_real/adaptf2_${cfg}.log 2>&1
done
echo "ALL DONE"
