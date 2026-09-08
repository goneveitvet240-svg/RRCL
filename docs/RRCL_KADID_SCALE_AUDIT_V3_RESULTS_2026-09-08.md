# RRCL KADID leakage-corrected scale audit v3 result

Formal run date: 2026-09-08 (Asia/Shanghai)

Pre-result freeze commit: `08e0caa3b66ef1380887befba45f35837a779979`

Configuration SHA-256: `40bc1f601ad29098b8d33d6aaa1707a77192223cc1fe775604ed6d88b2a39cf6`

Result JSON SHA-256: `9a08c83f39f5206016589a3a615931f71fea86728262bd573e5dc3bb7d858a8a`

Downloaded archive SHA-256: `e172fb97332a1cb03f1a05a492ee663d2a8486e25b6d0add96c37df807901513`

## Correction outcome

V1 and v2 are invalidated because their domain-specific selector hash allowed
the same pristine reference content to cross fit and validation roles across
distortion families. The audit reconstructed 46 global validation groups in
v2; all 46 also appeared in another domain's fit role.

V3 uses one shared selector key for every distortion family. Its recorded and
independently recomputed isolation result is:

| Global role | Reference groups |
|---|---:|
| fit | 51 |
| selector validation | 13 |
| fit/validation intersection | **0** |

The validator reads every literal group id, recomputes its `holdout-hash-v1`
role, checks per-domain and global hashes, and rejects a forged zero-overlap
claim.

## Corrected numerical result

Both objectives selected the same normalized lambda at all five boundaries:

```text
1.7782794, 1.7782794, 0.3162278, 0.5623413, 1.0
```

| Objective | Final selected lambda | Validation MSE | Test relative MAE | Test normalized MSE |
|---|---:|---:|---:|---:|
| sample-mean pooled | 1.0 | 0.15168549 | **0.26854490** | **0.11818845** |
| domain-mean balanced | 1.0 | 0.15173116 | 0.26855214 | 0.11818936 |

The relative-MAE change from domain balancing is `-0.00270%` (negative means
worse). Its absolute change is only `+0.00000725`. Four domain values improve
by between `0.0055%` and `0.0175%`, while Color worsens by `0.0611%`; the
balanced mean is effectively tied at the displayed precision.

This corrected run provides no material domain-balancing gain. It does not
prove that domain balancing is universally useless: this is one KADID sample
seed and one selector split, without a confidence interval.

## Independent reproduction and numerical checks

Before the formal AutoDL run, the same v3 computation was independently
reconstructed on the local machine from the backed-up feature archive and
local KADID CSV. It independently rebuilt dataset sampling and group roles
rather than reading the v3 result. The local and AutoDL final MAEs agree within
`4e-16`.

At every boundary, the raw/normalized ridge reparameterization also passed.
The maximum relative parameter discrepancy was below `9.62e-14`, and the
maximum validation-risk discrepancy was below `2.59e-15`.

The formal artifact records `git_dirty=false` at the frozen commit. It passed
the server validator and a separate validator run after download. All 4,625
DINOv2 feature files were cache hits; the formal audit took 26.75 seconds.

## What may and may not be claimed

Supported for this exact development run:

- v1/v2 selector isolation was invalid;
- v3 has zero cross-domain reference-content overlap between fit and
  validation;
- normalized pooled and domain-balanced objectives are numerically tied on
  this v3 run;
- the ridge scale-equivalence identity is implemented correctly.

Not supported:

- the old `11.041%`, `0.560%`, or `0.588%` effects as valid domain-balancing
  evidence;
- a general claim that domain balancing never helps;
- a stable automatic-lambda or automatic-`f` claim from one validation split;
- any v1 projection or independent-head number, because those selections used
  the invalid split;
- independent confirmation: KADID test results have been inspected repeatedly
  during development.

The next reliability requirement is a frozen multi-split or nested-validation
analysis before treating validation-selected hyperparameters as stable. The
nonlinear-head, projection, independent-head, trajectory-bank, shrinkage, and
automatic-`f` directions remain in scope, but their earlier KADID evidence
must not be reused without the corrected global split.
