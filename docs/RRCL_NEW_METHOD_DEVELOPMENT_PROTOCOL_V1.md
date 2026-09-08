# RRCL new-method development protocol v1

Status: **development-only（仅开发）**
Protocol id: `rrcl-new-method-development-v1`
Frozen draft date: 2026-09-08

## 1. Objective

This protocol tests the cheapest explanations for apparent scalar-forgetting
benefits before constructing a more complex selector. It separates:

1. regularization selection（正则化选择）;
2. global sample-mass scaling（全局样本质量级）;
3. domain-balanced training（域平衡训练）;
4. the value or cost of sharing one head across domains;
5. fixed nonlinear random-feature capacity（固定非线性随机特征容量）.

It does **not** test a new automatic forgetting-factor selector. Candidate
trajectories, shrinkage, and risk-controlled selection belong to later gates
and are not authorized by a positive baseline result alone.

## 2. Evidence boundary

- SHA, SHB, JHU, and QNRF remain development data.
- FDST must not be loaded, inspected, tuned on, or used by this protocol.
- The runner rejects domain configurations outside the allow-list in
  `configs/new_method_development_v1.json` and rejects any configuration whose
  path or serialized contents contain a forbidden dataset token.
- Outputs must be written below `runs_real/new_method_development_v1/`.
- Results from this protocol cannot be described as independent confirmation.
- Frozen synthetic-v1, selector-v2, and FDST artifacts are not overwritten or
  incorporated into the output.
- Natural-data runs use frozen DINOv2 ViT-B/14 features at image size 518 and
  at most 400 sampled training images per domain. Sample and split seeds vary
  only as explicit replication indices.

## 3. Data roles

Each current domain's training images are deterministically divided by
`holdout-hash-v1` into approximately 80% fit and 20% validation observations.
Complete images, not patch tokens, are assigned to a role.

The domain target scale is the mean image count of the **fit partition only**.
The same fixed scale is used for fit, validation, optional refit, and test
evaluation for that domain.

Primary protocol: `aligned_fit_only`.

- candidate fitting uses fit observations only;
- lambda selection uses validation observations only;
- the evaluated model is the same fit-only model scored during selection;
- validation observations never enter the fitted sufficient statistics.

Secondary ablation: `full_refit_ablation`.

- reuse the lambda selected by the fit/validation procedure;
- refit the same weighting rule on all training observations;
- label the result as a refit ablation because validation risk does not
  certify the refitted parameter value.

## 4. Compared analytic objectives

For each head and domain, let `(R_j, C_j, n_j)` denote sufficient statistics
and the number of observations in that head's objective.

### Pooled f=1

```text
w_j = 1
```

This is the ordinary shared-head cumulative ridge solution.

### Canonical domain-balanced

```text
w_j = 1 / (t n_j)
```

Each domain contributes `1/t` times its mean loss, exactly matching the stated
average-domain ridge objective. This changes both domain weighting and the
total scale of the data term.

### Mass-matched domain-balanced

```text
w_j = mean(n_1, ..., n_t) / n_j
```

Every domain has equal total weight while the sum of observation weights
matches the pooled objective. This controls the global mass confound but does
not equalize covariance spectra or condition numbers.

Patch and image heads use their own observation counts. A patch observation
and an image observation are not treated as the same statistical unit.

### Independent per-domain heads

Each domain receives its own dual analytic head and its own train-selected
lambda. This is a task-aware reference, not a member of the shared-head family.
Update-state and deployment-state memory are reported separately.

### RanPAC-style image head

The existing fixed Gaussian projection plus ReLU implementation is evaluated
as an image-only capacity reference. It is explicitly labelled `RanPAC-style`:
it is not the complete RanPAC classifier and is not claimed as a new method.

## 5. Lambda selection

The frozen grid is:

```text
0.01, 0.1, 1, 10, 100, 1000, 10000
```

At every domain boundary, one common lambda for the dual head is selected by
the balanced mean, across seen domains, of **unclipped fused normalized MSE**.
The fused design is the exact stacked linear design before non-negative
clipping. Exact ties choose the larger lambda.

The downstream outcome remains balanced mean clipped relative MAE. These two
metrics are reported separately; improvement in the quadratic selection proxy
must not be described as automatic downstream improvement.

## 6. Required outputs

Every run records:

- data and split manifests;
- lambda risks and selected lambda at every boundary;
- aligned-fit and full-refit result matrices;
- unclipped normalized MSE and clipped relative MAE;
- matrix condition numbers for the selected final-boundary systems;
- online update-state and deployment-state memory estimates;
- an explicit gain decomposition with unavailable later-stage gaps marked
  `not_measured_in_batch1` rather than filled using test-selected values.

## 7. Decision gates

The numerical values below are prospective planning defaults and may be
changed only in a new protocol version before inspecting the corresponding
result.

1. If tuned pooled `f=1` removes most apparent gain, treat regularization as
   the leading explanation.
2. If either domain-balanced baseline recovers at least 80% of the material
   fixed-factor Oracle opportunity, do not begin a complex selector without a
   separate residual-opportunity justification.
3. If independent heads dominate, any shared-head method claim must name the
   constraint that makes sharing necessary: unknown domain id, fixed model
   count, transfer, or strict update memory.
4. A nonlinear feature result is an engineering baseline unless it changes
   opportunity, identifiability, or safe selection—not merely average error.
5. FDST remains unavailable for all decisions above.

## 8. Next gate

Only after this baseline audit should synthetic-v2 and a fixed counterfactual
trajectory bank be built. Beam search or directional forgetting requires its
own Oracle-headroom gate and prior-art comparison.
