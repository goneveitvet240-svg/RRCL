# RRCL KADID group-CV baseline restoration v4

Status: **development-only（仅开发）**

Protocol id: `rrcl-kadid-group-cv-v4`

Frozen date: 2026-09-09

## Purpose

V1 and v2 are invalid because their inner selector split leaked pristine
reference content across distortion domains. V3 repaired one split and found
normalized pooled and domain-balanced heads effectively tied, but one split
cannot establish selection stability. V4 restores the invalidated baseline
roster under globally shared five-fold group cross-validation（分组交叉验证）.

This is an evidence-repair batch, not the automatic-`f` method batch.

## Fold construction

The 64 development-train pristine reference ids are ranked by SHA-256 of:

```text
KADID-10k-reference-content | reference_id | seed=42
```

Ranked groups are assigned round-robin to five folds, producing fold sizes
that differ by at most one. Domain name, target, stream order, and test data
do not enter the assignment. The same reference id has the same fold in all
five distortion families.

At each continual boundary and fold:

- four folds fit the candidate;
- the held-out fold evaluates it;
- target normalization uses only that fold's fit labels;
- OOF losses are accumulated per domain and then averaged across domains.

Lambda is selected by balanced domain-mean OOF normalized MSE. Exact ties
choose the larger lambda. After selection, the algorithm is refit on all 64
development-train groups and evaluated descriptively on the already-seen
KADID test split. This final refit is a standard CV hyperparameter workflow;
it is not the same fitted parameter evaluated in each fold.

## Restored baselines

1. normalized sample-mean pooled linear head;
2. normalized domain-mean balanced linear head;
3. independent task-aware linear head per distortion family;
4. fixed 2,000-dimensional Gaussian-ReLU projection of the already
   mean-pooled DINOv2 image descriptor, followed by normalized pooled ridge.

Every shared baseline reports both OOF-selected lambda and the frozen
normalized reference lambda `1.0`. The independent heads select lambda per
domain. All use the same 37-point normalized lambda grid.

The projection baseline is explicitly image-level and post-pooling. It does
not test the proposed patch-level construction
`mean_p ReLU(P^T x_p + b)` and is not full RanPAC.

## Uncertainty and evidence limits

The runner stores selected OOF loss for every pristine reference group.
Pairwise empirical bootstrap intervals resample those groups and keep method
losses paired. This diagnoses instability; it is not a distribution-free
guarantee because squared losses are unbounded and reference images need not
be identically distributed. It is also a **post-selection** summary: the same
OOF predictions were used to choose lambda. The intervals are therefore not
selection-adjusted confidence intervals and must not be presented as formal
inferential evidence.

KADID test results have already been inspected repeatedly and remain
development-only. They cannot select methods or provide final confirmation.
FDST is forbidden. Fixed trajectory banks, risk-controlled selection,
analytic shrinkage, patch-level nonlinearity, and automatic `f` remain in
scope for later frozen protocols.
