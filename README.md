# RRCL: Adaptive Forgetting for Analytic Continual Visual Regression

**Paper**: *Adaptive Forgetting for Analytic Continual Visual Regression* (in preparation, targeting ICPR 2026 / PRCV 2026)

**Author**: Pang Wei (庞惟), Qingdao University

---

## Overview

This project investigates whether analytic continual learning (ACL) methods should always retain full historical information. Existing ACL methods (ACIL, RanPAC, DS-AL) default to absolute memory (forgetting factor f = 1, equivalent to joint training) for classification tasks. We show this assumption fails for **dense visual regression under domain shift**.

**Core findings:**

1. **Magnitude–Drift Decomposition**: The benefit of forgetting can be decomposed into two distinct sources — *target-scale mismatch* (eliminated by per-domain mean normalization) and *genuine cross-domain mapping drift* (persists after normalization). This directly addresses whether forgetting gains are artifacts of scale imbalance.

2. **Test-Free Adaptive Forgetting Selector**: At each domain boundary, we select the optimal forgetting factor by minimizing a balanced held-out validation error computed from sufficient statistics of the current domain's training data — no test set, no replay, no SGD.

3. **Qualitative Drift Criterion**: A criterion based solely on training statistics correctly predicts "should we forget?" in 9/10 evaluated sequences across two task types.

---

## Results (Crowd Counting, Mean-Normalized, Relative MAE ↓)

| Order | f=1 (Joint) | Oracle f | **Adaptive f** | vs f=1 |
|-------|------------|---------|---------------|--------|
| JHU→SHA→SHB | 0.511 | 0.482 (f=0.60) | **0.482** | **+5.7%** |
| JHU→SHB→SHA | 0.511 | 0.482 (f=0.60) | **0.482** | **+5.6%** |
| SHA→SHB | 0.365 | 0.365 (f=1) | 0.365 | ±0% |
| SHA→SHB→JHU | 0.511 | 0.511 (f=1) | 0.511 | ±0% |

Age estimation (UTK Young→Mid→Old): adaptive f improves ~+4.8% over f=1; cross-dataset controls correctly return f=1.

---

## Method

- **Backbone**: Frozen DINOv2 ViT-B/14, never fine-tuned
- **Head**: Recursive Ridge Regression (RLS) with forgetting factor f
- **Update rule** (domain t):
  ```
  R_t = f · R_{t-1} + X_t^T X_t
  C_t = f · C_{t-1} + X_t^T y_t
  W_t = (R_t + λI)^{-1} C_t
  ```
- Single-pass, replay-free, SGD-free, no old image storage

---

## Repository Structure

```
RRCL/
├── rls_head.py              # Core forgetting-RLS head
├── features.py              # Frozen DINOv2 feature extractor
├── metrics.py               # MAE / relative MAE metrics
├── datasets_real.py         # Crowd counting domain loader (JHU, SHA, SHB, QNRF)
├── datasets_age.py          # Age estimation domain loader (UTKFace, AgeDB)
├── datasets_depth.py        # Depth estimation domain loader (NYU Depth V2)
├── datasets_iqa.py          # IQA domain loader (KADID-10k)
├── run_adaptive_f.py        # Main: adaptive f selector + full evaluation
├── run_age_cl.py            # Age estimation continual learning
├── run_baselines.py         # SGD / Single-domain / f=1 / RanPAC baselines
├── run_real_norm_ablation.py # Magnitude–drift decomposition ablation
├── run_theory_predict.py    # Qualitative drift criterion evaluation
├── configs/                 # JSON domain configs (paths, splits)
├── scripts/                 # Shell wrappers for batch experiments
└── paper/                   # LaTeX source (main.tex + refs.bib)
```

---

## Quick Start

### Requirements
```bash
pip install torch torchvision timm numpy pillow
# HuggingFace mirror (for DINOv2 on servers without direct HF access):
export HF_ENDPOINT=https://hf-mirror.com
```

### Run norm ablation (JHU→SHA→SHB)
```bash
python run_real_norm_ablation.py \
  --config configs/domains_jhu_sha_shb.json \
  --img-size 518 --lam 100 --max-per-domain 400 \
  --out runs/normabl_jhu_sha_shb
```

### Run adaptive f + baselines
```bash
python run_adaptive_f.py \
  --config configs/domains_jhu_sha_shb.json \
  --img-size 518 --lam 100 --max-per-domain 400 \
  --out runs/adaptive_jhu_sha_shb
```

### Age estimation
```bash
python run_age_cl.py \
  --config configs/domains_age_utk_young_old.json \
  --img-size 518 --lam 100 --max-per-domain 400 \
  --out runs/age_utk
```

---

## Datasets

| Dataset | Task | Domain split |
|---------|------|-------------|
| ShanghaiTech A/B | Crowd counting | Two venues (sparse/dense) |
| JHU-CROWD++ | Crowd counting | Cross-dataset |
| UCF-QNRF | Crowd counting | Cross-dataset |
| UTKFace | Age estimation | Age bands (Young/Mid/Old) |
| AgeDB | Age estimation | Cross-dataset control |

Download links and path configuration: edit the relevant `configs/domains_*.json` files.

---

## Citation

```bibtex
@article{pangwei2026rrcl,
  title   = {Adaptive Forgetting for Analytic Continual Visual Regression},
  author  = {Pang Wei},
  year    = {2026},
  note    = {In preparation}
}
```
