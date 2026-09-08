# RRCL experimental record

RRCL separates opportunity measurement from deployable selection. A
fixed-factor Oracle may inspect held-out risk only to determine whether the
predeclared scalar family contains a useful factor. Every deployable rule uses
training-side information only.

## 1. Development diagnostics

The crowd-counting studies evaluate frozen analytic and reliability-based
rules over multiple domain orders and seeds. They are development evidence,
not independent confirmation. Their role is to expose the difference between
Oracle opportunity, factor prediction accuracy, and downstream decision
utility.

Key entry points:

```bash
bash scripts/run_c4_gamma_multiseed.sh
bash scripts/run_fstar_calibration_multiseed.sh
python3 scripts/validate_tmlr_claim_set.py
```

## 2. Controlled synthetic boundary

The preregistered `2^4` study varies label scale, domain reliability, mapping,
and covariance anisotropy. Ten seeds and two orders produce 320 retained
scenarios. The formal result is immutable; validation recomputes its summaries
from the raw records.

```bash
python3 scripts/validate_tmlr_synthetic_v1.py
```

Protocol and result:

- `RRCL_TMLR_SYNTHETIC_PROTOCOL.md`
- `RRCL_TMLR_SYNTHETIC_RESULTS_2026-08-07.md`

## 3. FDST single-shot boundary

FDST is an external, class-conditional boundary using six reviewed scenes from
the official training partition. The v3 protocol transparently records the
pre-result coordinate-tolerance amendment. The split manifest, scene mapping,
and validation logic are retained; raw FDST data are not distributed.

```bash
python3 scripts/validate_fdst_c7_result.py
```

Protocol and result:

- `C7_FDST_PROTOCOL_V2.md`
- `C7_FDST_PROTOCOL_V3.md`
- `RRCL_TMLR_FDST_RESULTS_2026-08-09.md`

## 4. Held-out learned-selector study

The selector study uses disjoint simulator seeds for meta-training,
meta-validation, and meta-test. It evaluates a KNN curve selector alongside
the frozen analytic rule, SIFt-RLS, and DOS-ELM-style regression adaptation.
The formal outcome is reported against the complete preregistered four-part
criterion, including the failed criterion.

```bash
python3 scripts/validate_tmlr_selector_v2.py
```

Protocol and result:

- `RRCL_TMLR_SELECTOR_V2_PROTOCOL.md`
- `RRCL_TMLR_SELECTOR_V2_RESULTS_2026-08-10.md`

## 5. Post-freeze exploratory diagnostics

Two deterministic diagnostics are kept outside the frozen claim-set bundle:

```bash
python3 scripts/history_dominance_phase.py
python3 scripts/explore_candidate_family.py
python3 scripts/validate_exploratory_diagnostics.py
```

The history-dominance study distinguishes the closed-form minimizer of risk
averaged over training noise from a conditional finite-sample Oracle that
selects after each realization. The latter may select on estimation noise even
when mapping mismatch is zero.

The candidate-family study reproduces the deployed scalar family, then uses a
separate strictly nested, mean-normalized attribution track. This prevents
constraint attribution from being confounded by changes in total weight and
effective ridge scale. Both studies are exploratory, use population-risk
Oracles, and define no deployable selector.

## Evidence rules

- Diagnostic Oracles are not deployable methods.
- Development evidence is not described as independent confirmation.
- A null opportunity result cannot establish selector impossibility.
- Scalar-grid findings do not generalize to directional or matrix-valued
  weighting.
- Failed gates and harmful baselines remain in the record.
