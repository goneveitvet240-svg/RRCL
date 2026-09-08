# RRCL KADID scale-equivalence audit v2

Status: **development-only（仅开发）**

Protocol id: `rrcl-new-method-kadid-scale-audit-v2`

Frozen date: 2026-09-08

## Why this batch exists

KADID development-v1 compared the following two domain-balanced objectives
using the same numerical ridge parameter:

```text
canonical:     w_j = 1 / (t n_j)
mass-matched:  w_j = mean(n_1,...,n_t) / n_j
```

At boundary `t`, their data statistics differ by the total fit count
`N_t = sum_j n_j`. Ridge regression is not invariant to that multiplication
unless lambda is multiplied by the same `N_t`. Consequently, the first-batch
gap mixes domain weighting（域权重）with effective regularization scale
（有效正则化尺度）. It cannot by itself be attributed to domain balancing.

This batch removes that confound before any automatic forgetting-factor
selector is designed.

## Frozen objectives

Both objectives have total observation weight one:

```text
sample_mean_pooled:  w_j = 1 / N_t
domain_balanced:     w_j = 1 / (t n_j)
```

They differ only in how the unit mass is distributed across domains. Both use
the same 37-point quarter-decade grid from `10^-5` through `10^4`, selected at
every boundary using balanced domain-mean validation MSE. Exact ties choose
the larger lambda. Test data never selects lambda, an objective, or an anchor.

For each boundary the implementation must also verify:

```text
sample_mean_pooled(lambda)
  == pooled_f1(N_t * lambda)

domain_balanced(lambda)
  == domain_balanced_mass_matched(N_t * lambda)
```

The check covers sufficient statistics, fitted weights, and validation risk.
It is an algebraic audit, not a new baseline.

## Required comparisons

The artifact reports:

1. independently validation-selected normalized pooled and domain-balanced
   trajectories;
2. paired evaluations of both objectives at the pooled-selected lambda;
3. paired evaluations of both objectives at the balanced-selected lambda;
4. raw/normalized equivalence errors at every boundary;
5. full lambda traces, matrices, data manifests, configuration hash, and code
   provenance.

The paired comparisons answer a narrower causal question: when normalized
regularization is held fixed, does reallocating equal total mass from samples
to domains help? The independently selected comparison answers the practical
question after each objective receives equal train-side tuning opportunity.

## Evidence and scope boundary

- Data, target normalization, reference-group splits, DINOv2 backbone, image
  size, sample cap, and seeds are unchanged from KADID development-v1.
- Only `aligned_fit_only` is run. The v1 full-refit ablation changed the fitted
  object and is not needed to resolve this scale confound.
- Projection and independent-domain heads are not rerun; their v1 evidence is
  retained rather than silently discarded.
- Fixed trajectory banks, risk-controlled hard selection, analytic shrinkage,
  nonlinear heads, projection layers, and automatic `f` remain in the project
  plan. This audit does not authorize or reject any of them.
- KADID remains reused development evidence. FDST is forbidden and no result
  from this batch is independent confirmation.

## Decision use

If the practical and same-lambda comparisons both favor domain balancing, the
effect is no longer explainable only by global scale. If the effect disappears,
the v1 canonical gain is reclassified mainly as regularization-grid mismatch.
Either result is followed by the already planned synthetic-v2 fixed
counterfactual trajectory-bank gate; it is not itself evidence for an
automatic-`f` contribution.
