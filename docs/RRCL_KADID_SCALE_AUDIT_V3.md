# RRCL KADID leakage-corrected scale audit v3

Status: **development-only（仅开发）**

Protocol id: `rrcl-new-method-kadid-scale-audit-v3`

Frozen date: 2026-09-08

## Correction

The post-v2 audit found that v1/v2 called the split hash with
`(domain_name, reference_id)`. KADID's five distortion-family domains reuse
the same pristine reference identities, so one reference could be validation
data in one domain and fit data in another. Across v2, all 46 groups that were
validation data somewhere also occurred as fit data elsewhere.

Train/test reference isolation remained intact. The defect affects the inner
fit/selector-validation isolation and invalidates v1/v2 model-selection and
test-comparison evidence.

V3 changes exactly one experimental degree of freedom:

```text
old split key: domain_name + reference_id
new split key: "KADID-10k-reference-content" + reference_id
```

Therefore, one reference id receives the same role in every domain. Objective
definitions, the 37-point lambda grid, backbone, image size, sample cap,
target normalization, metrics, and seeds remain identical to v2.

## Mandatory isolation evidence

The result records the literal fit and validation group lists for every
domain. The validator must independently:

1. recompute each role with `holdout-hash-v1`;
2. compare group counts and hashes with each domain manifest;
3. form global fit and validation unions;
4. require their intersection to be empty;
5. reject a forged positive isolation field.

The formal run is invalid if any global cross-role overlap remains.

## Methods and evidence boundary

The two total-mass-one objectives remain:

```text
sample_mean_pooled:  w_j = 1 / N_t
domain_balanced:     w_j = 1 / (t n_j)
```

Lambda is selected only by balanced validation MSE. Test relative MAE is
reported after selection. Raw/normalized ridge equivalence is checked at each
boundary. KADID remains reused development data; FDST is forbidden.

V3 still does not test a nonlinear head, projection layer, fixed trajectory
bank, shrinkage rule, or automatic `f`. Those project directions remain
unchanged and require later protocols. The v1 projection and independent-head
numbers inherit the invalid selector split, so those baselines must also be
rerun before use; their directions are retained, but their old evidence is not.
