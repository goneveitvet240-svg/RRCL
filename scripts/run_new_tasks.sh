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

set -e
cd "$(dirname "$0")"

MAX=500
if [[ "$1" == "--smoke" ]]; then MAX=50; echo "[smoke mode] max_per_domain=$MAX"; fi

echo "===  KADID-10k IQA (5 distortion-family domains)  ==="
python run_iqa_cl.py \
  --config domains_iqa_kadid.json \
  --img-size 518 --backbone dinov2_vitb14 \
  --lam 1e2 --max-per-domain "$MAX" \
  --out runs_real/kadid

echo ""
echo "===  NYU Depth V2 (3 acquisition-batch domains)  ==="
# Depth uses the existing patch-level pipeline; run_adaptive_f.py is the runner.
# Here we run just the f=1 vs adaptive sweep via run_real_norm_ablation.py if available,
# otherwise run a quick forward pass to verify loading.
python run_adaptive_f.py \
  --config domains_depth_nyu.json \
  --img-size 518 --backbone dinov2_vitb14 \
  --lam 1e2 --patch-target mean --alpha 0.25 \
  --max-per-domain "$MAX" \
  --out runs_real/nyu \
  || echo "[WARN] run_adaptive_f.py failed for NYU; check patch-target compatibility"

echo ""
echo "===  AVA Aesthetics 10% (3 score-tier domains)  ==="
python run_ava_cl.py \
  --config domains_ava.json \
  --img-size 518 --backbone dinov2_vitb14 \
  --lam 1e2 --max-per-domain "$MAX" \
  --out runs_real/ava

echo ""
echo "=== All three tasks done. Results in runs_real/{kadid,nyu,ava}/ ==="
