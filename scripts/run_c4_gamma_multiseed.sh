#!/usr/bin/env bash
# Frozen C4 material-decision run: 5 seeds x 2 dataset families x 2 orders.
# JHU/SHA/SHB/QNRF are development data; this is not independent confirmation.
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/preflight_experiments.py

if [[ -n "$(git status --porcelain)" ]]; then
  echo "C4 REFUSED: git worktree is dirty" >&2
  exit 1
fi

OUT_ROOT="runs_real/c4_gamma"
if [[ -e "${OUT_ROOT}/c4_gamma_summary.json" ]]; then
  echo "C4 REFUSED: completed summary already exists at ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

COMMON=(
  --backbone vit_base_patch14_dinov2.lvd142m
  --img-size 518
  --lam 100
  --alpha 0.25
  --patch-target dct5
  --max-per-domain 400
)

for seed in 42 43 44 45 46; do
  for cfg in jhu_sha_shb sha_shb_jhu qnrf_shb_sha qnrf_sha_shb; do
    destination="${OUT_ROOT}/seed_${seed}/shrinkage_diag_${cfg}"
    result="${destination}/shrinkage_diagnostic.json"
    if [[ -e "${result}" ]]; then
      echo "C4 REFUSED: partial/pre-existing result at ${result}" >&2
      exit 1
    fi
    echo "===== C4 gamma seed=${seed}: ${cfg} ====="
    python -u scripts/run_precision_shrinkage_diagnostic.py \
      --config "configs/domains_${cfg}.json" \
      "${COMMON[@]}" \
      --sample-seed "${seed}" \
      --split-seed "${seed}" \
      --out "${destination}"
  done
done

python scripts/summarize_gamma_c4.py --runs "${OUT_ROOT}"
echo "ALL C4 GAMMA RUNS DONE"
