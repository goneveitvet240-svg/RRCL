#!/usr/bin/env bash
# Frozen f*_pred vs f*_oracle development diagnostic:
# 5 seeds x 8 natural crowd-counting orderings/configurations.
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/preflight_experiments.py

if [[ -n "$(git status --porcelain)" ]]; then
  echo "FSTAR CALIBRATION REFUSED: git worktree is dirty" >&2
  exit 1
fi

OUT_ROOT="runs_real/fstar_calibration"
if [[ -e "${OUT_ROOT}/fstar_calibration_summary.json" ]]; then
  echo "FSTAR CALIBRATION REFUSED: completed summary already exists" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

COMMON=(
  --task crowd
  --backbone vit_base_patch14_dinov2.lvd142m
  --img-size 518
  --lam 100
  --max-per-domain 400
  --val-every 5
)

CONFIGS=(
  jhu_sha_shb
  jhu_shb_sha
  sha_jhu_shb
  sha_shb_jhu
  qnrf_sha_shb
  qnrf_shb_sha
  sha_shb_qnrf
  sha_shb
)

for seed in 42 43 44 45 46; do
  for config in "${CONFIGS[@]}"; do
    destination="${OUT_ROOT}/seed_${seed}/theory_${config}"
    artifact="${destination}/fstar_calibration.json"
    if [[ -e "${artifact}" ]]; then
      echo "FSTAR CALIBRATION REFUSED: pre-existing artifact ${artifact}" >&2
      exit 1
    fi
    echo "===== f* calibration seed=${seed}: ${config} ====="
    python -u run_theory_predict.py \
      --config "configs/domains_${config}.json" \
      "${COMMON[@]}" \
      --sample-seed "${seed}" \
      --split-seed "${seed}" \
      --out "${destination}"
  done
done

python scripts/summarize_fstar_calibration.py --runs "${OUT_ROOT}"
python scripts/plot_fstar_calibration.py \
  --summary "${OUT_ROOT}/fstar_calibration_summary.json" \
  --out "${OUT_ROOT}/fstar_calibration_scatter.pdf"

echo "ALL FSTAR CALIBRATION RUNS DONE"
