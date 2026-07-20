#!/usr/bin/env bash
# SGD / single-domain / f1(=joint) baselines on the 4 main orders.
# Fast: reuses cached DINOv2 features; closed-form + light SGD. Disconnect-safe with nohup.
set -u
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
COMMON="--img-size 518 --backbone vit_base_patch14_dinov2.lvd142m --lam 100 \
--patch-target dct5 --alpha 0.25 --max-per-domain 400 --sgd-epochs 10 --sgd-lr 0.01"
for cfg in jhu_sha_shb jhu_shb_sha sha_shb sha_shb_jhu; do
  echo "==== baselines $cfg ===="
  python -u run_baselines.py --config domains_${cfg}.json $COMMON \
    --out runs_real/base_${cfg} > runs_real/base_${cfg}.log 2>&1
  echo "==== done $cfg ===="
done
echo "ALL DONE"
