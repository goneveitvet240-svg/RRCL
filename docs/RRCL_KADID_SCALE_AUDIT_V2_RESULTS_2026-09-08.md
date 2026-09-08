# RRCL KADID scale-equivalence audit v2 result

Formal run date: 2026-09-08 (Asia/Shanghai)

Pre-result freeze commit: `dc2b4a8fe7e42433cc7268b2767deac0fe6e96ae`

Result JSON SHA-256: `57d29c514af5efb84fd011486377d384107587ffcc8c15a6396562ea8ce44e38`

Downloaded archive SHA-256: `ef56b71318747b27a286b2c24d742a61f5c473d32c44b5ccbae222fdacf52a71`

## Result first

After both objectives were normalized to total observation weight one and
given the same 37-point train-side lambda search, domain balancing improved
final balanced relative MAE by only `0.588%`:

| Objective | Validation-selected lambda | Final relative MAE | Final normalized MSE |
|---|---:|---:|---:|
| sample-mean pooled | 0.0316228 | 0.346242 | 0.195150 |
| domain-mean balanced | 0.0316228 | 0.344208 | 0.192561 |

Both objectives selected exactly the same lambda at all five boundaries:

```text
1.0, 0.0316228, 0.0562341, 0.0562341, 0.0316228
```

Because the selected paths are identical, the independently selected result
is also an exact same-lambda paired comparison in this run. Domain balancing
helped all five final-domain relative-MAE values, but only by approximately
`0.42%` to `0.79%` per domain.

## What this changes about the first batch

KADID development-v1 reported:

| First-batch method | Numeric lambda | Final relative MAE |
|---|---:|---:|
| raw pooled | 100 | 0.314835 |
| canonical domain-balanced | 0.1 | 0.280074 |
| raw mass-matched domain-balanced | 100 | 0.313071 |

The apparent canonical gain over tuned pooled was `11.041%`, whereas the two
scale-controlled estimates of domain weighting are now consistent:

- v1 raw mass-matched versus raw pooled at lambda 100: `0.560%`;
- v2 normalized domain-balanced versus normalized pooled at lambda 0.0316228:
  `0.588%`.

Therefore, the approximately 11% number must not be described as a domain-
balancing gain. Most of it was associated with effective ridge scale and the
coarse validation grid. The residual domain-weighting effect on this run is
small but directionally consistent across every final domain.

## Newly exposed selection problem

For the unchanged canonical normalized objective, v1's coarse grid selected
lambda `0.1`, whose test relative MAE was `0.280074`. The frozen denser v2 grid
selected `0.0316228`: its validation risk was `3.212%` lower than the risk at
`0.1`, yet its test relative MAE was `22.899%` higher (`0.344208`).

This is direct development evidence of selector instability or validation-to-
test proxy mismatch. It is not evidence that lambda `0.1` should now be chosen:
that choice would use the observed test result. Instead, it strengthens the
need for the already planned selection-reliability stage—risk control,
shrinkage, and stability diagnostics—before claiming an automatic `f`
algorithm.

## Algebraic and provenance checks

At every boundary the run verified:

```text
sample_mean_pooled(lambda)
  == pooled_f1(N_t * lambda)

domain_balanced(lambda)
  == domain_balanced_mass_matched(N_t * lambda)
```

The maximum relative parameter discrepancy was below `4.13e-13`; the maximum
absolute validation-risk discrepancy was below `3.46e-15`. The result passed
both the remote validator and a separate validation after download. Its
provenance records the frozen commit above with `git_dirty=false`. The 4,625
feature files were all cache hits, and the formal audit took 26.42 seconds
after cache verification.

## Scope and next gate

This result is reused KADID development evidence, not independent
confirmation. It does not test or eliminate nonlinear heads, projection
layers, independent heads, fixed trajectory banks, shrinkage, or automatic
`f`. The next planned gate remains synthetic-v2 plus a fixed counterfactual
trajectory bank, with selection reliability evaluated separately from Oracle
candidate-family headroom. FDST remains forbidden for development decisions.
