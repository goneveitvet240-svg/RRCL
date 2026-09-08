# RRCL KADID new-method development protocol v1

> **Invalidated on 2026-09-08:** the implementation hashed the selector split
> with `(domain_name, reference_id)`, so the same pristine reference content
> could be fit data in one distortion family and validation data in another.
> Train/test isolation was intact, but cross-domain fit/validation isolation
> was not. Numerical results produced under v1 are retained only as an audit
> trail and must not be used as paper evidence. See the corrected v3 protocol.

Status: **development-only（仅开发）**
Protocol id: `rrcl-new-method-kadid-development-v1`
Frozen draft date: 2026-09-08

Before the first natural-data execution, the dataset config was made portable:
`RRCL_KADID_ROOT` now supplies the machine-specific extracted-data directory.
This deployment amendment changes no domain, split, method, hyperparameter, or
evaluation choice.

## Purpose

This protocol ports the first-batch analytic baseline audit to the locally
available KADID-10k data. It evaluates whether regularization, domain-balanced
training, independent task-aware heads, or fixed nonlinear capacity explain
performance differences before an automatic forgetting selector is designed.

KADID is reused development data. It cannot replace a future untouched natural
confirmation source, and it does not establish a crowd-counting result.

## Domain and split definition

The five domains are fixed distortion families: Noise, Blur, Color,
Compression, and Spatial. The dataset loader first creates an 80/20
train/test split over pristine reference contents. No distorted version of one
reference image crosses this boundary.

The training side is split again into fit and selector-validation roles by a
stable hash of the complete reference-content group. Individual distorted
images are never independently scattered across these roles. Domain target
normalization uses the fit-role mean DMOS only.

## Methods

The shared-head objectives are:

```text
pooled_f1:                       w_j = 1
domain_balanced:                 w_j = 1 / (t n_j)
domain_balanced_mass_matched:    w_j = mean(n_1,...,n_t) / n_j
```

The task-aware independent-head reference selects lambda separately for each
domain. The nonlinear reference applies one fixed Gaussian projection and
ReLU before the analytic image-level ridge head; it is RanPAC-style, not the
complete RanPAC classifier and not a new-method claim.

At each boundary, lambda is selected from the frozen grid using balanced
domain-mean, unclipped, normalized validation MSE. Final utility is balanced
domain-mean clipped relative MAE. The proxy and downstream metric remain
separate in every artifact.

`aligned_fit_only` is primary: validation samples never enter its sufficient
statistics. `full_refit_ablation` reuses the selected lambda and trains on all
training samples, but is labelled uncertified with respect to fit-only
validation risk.

## Compute and cache policy

The frozen natural run uses DINOv2 ViT-B/14 at 518 pixels, at most 500 sampled
training images per domain, and every reference-disjoint test image. Device
selection prefers CUDA, then MPS when actually available, then CPU. An explicit
device or `RRCL_DEVICE` overrides automatic selection.

Feature files are written atomically. Set `RRCL_FEATURE_CACHE_ROOT` to a
persistent disk or synchronized directory; the KADID cache is created below
`$RRCL_FEATURE_CACHE_ROOT/iqa/`. A released compute instance must therefore not
destroy the only copy of extracted features.

## Evidence boundary and gates

- FDST is forbidden for tuning, ablation, or threshold selection.
- The local KADID result remains development evidence.
- A positive random-projection result is capacity evidence only.
- Candidate-family, selection, and deployment-adoption gaps remain explicitly
  unmeasured in this batch.
- The automatic-selector stage begins only if the first batch leaves material,
  reproducible opportunity after tuned and domain-balanced controls.
