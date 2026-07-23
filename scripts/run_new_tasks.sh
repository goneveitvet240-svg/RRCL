#!/bin/bash
# Run RRCL experiments on three new B-level tasks (KADID / NYU / AVA).
# Execute on AutoDL after syncing this repo to /root/autodl-tmp/RRCL/.
#
# Usage:
#   bash run_new_tasks.sh [--smoke]    # --smoke uses max-per-domain=50 for quick sanity check
#
# Prerequisites:
#   - Datasets at /root/autodl-tmp/datasets/  (see RRCL_B_TASKS_MANIFEST.md)
#   - NYU split files at  datasets/NYU_Depth_V2/processed/splits/  (nyu_d{0,1,2}_{train,test}.txt)
#   - pip install timm torch torchvision Pillow numpy h5py

set -euo pipefail
# cd to repo root (one level up from scripts/)
cd "$(dirname "$0")/.."

MAX=500
if [[ "${1:-}" == "--smoke" ]]; then MAX=50; echo "[smoke mode] max_per_domain=$MAX"; fi

echo "===  KADID-10k IQA (5 distortion-family domains)  ==="
python run_iqa_cl.py \
  --config configs/domains_iqa_kadid.json \
  --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
  --lam 1e2 --max-per-domain "$MAX" --selector-search grid \
  --out runs_real/kadid

echo ""
echo "===  NYU Depth V2 (3 source-batch proxy domains; not natural scene domains)  ==="
python run_depth_cl.py \
  --config configs/domains_depth_nyu.json \
  --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
  --lam 1e2 --sample-seed 42 --selector-search grid \
  --max-per-domain "$MAX" \
  --out runs_real/nyu

echo ""
echo "===  AVA Aesthetics 10% (3 score-tier domains)  ==="
python run_ava_cl.py \
  --config configs/domains_ava.json \
  --img-size 518 --backbone vit_base_patch14_dinov2.lvd142m \
  --lam 1e2 --max-per-domain "$MAX" --selector-search grid \
  --out runs_real/ava

echo ""
echo "=== All three tasks done. Results in runs_real/{kadid,nyu,ava}/ ==="
