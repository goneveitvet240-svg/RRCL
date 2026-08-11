# RRCL: Scalar Forgetting in Analytic Continual Visual Regression

This repository contains the code, frozen protocols, validation utilities, and
paper source for:

> *Opportunity Does Not Imply Reliable Selection: Preregistered Tests of
> Scalar Forgetting in Analytic Continual Visual Regression*

RRCL studies a narrow question: when a recursive ridge head is updated over
frozen visual features, does a scalar forgetting-factor family contain useful
historical weighting, and can that factor be selected using training-side
information alone? Diagnostic Oracles are used only to measure opportunity;
they are never presented as deployable methods.

## Evidence at a glance

The repository retains positive, null, and harmful outcomes under their
predeclared evidence roles:

| Study | Role | Main outcome |
|---|---|---|
| Crowd-counting diagnostics | Development evidence | Oracle opportunity exists in selected settings, but the frozen train-only rules fail their decision gates. |
| Controlled synthetic study | Preregistered mechanism boundary | All 16 cells remain below the prespecified 0.5% scalar-grid opportunity threshold. |
| FDST confirmation | Frozen single-shot external boundary | The fixed-factor Oracle selects `f=1`; no material scalar-grid opportunity is found. |
| Held-out selector study | Preregistered learned-selector test | KNN has positive utility but passes only 3/4 composite criteria; matched SIFt-RLS and DOS-ELM-style results are retained. |

These findings bound the tested scalar factor grids, selectors, and task
distributions. They do not establish a universal impossibility result and do
not rule out directional, spectral, or matrix-valued weighting.

## Method

For domain `t`, RRCL updates the sufficient statistics of a ridge head as

```text
R_t = f R_{t-1} + X_t^T X_t
C_t = f C_{t-1} + X_t^T y_t
W_t = (R_t + lambda I)^{-1} C_t
```

The natural-image experiments use a frozen DINOv2 backbone. The analytic head
is replay-free and SGD-free after feature extraction.

## Repository structure

```text
RRCL/
├── adaptive_selector.py              # Train-only scalar selection rules
├── analytic_forgetting_baselines.py  # Analytic baseline implementations
├── rls_head.py                       # Recursive ridge head
├── tmlr_synthetic.py                 # Controlled synthetic generator
├── tmlr_selector_v2.py               # Held-out learned-selector study
├── datasets_*.py                     # Natural regression data adapters
├── configs/                          # Domain and frozen protocol configs
├── scripts/                          # Runners, builders, and validators
├── tests/                            # Protocol and implementation tests
├── docs/                             # Protocols and result records
├── paper/                            # Anonymous TMLR LaTeX source
└── release/                          # Deterministic submission supplement
```

See [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) for a component-level map and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experimental sequence.

## Reproduction

Install the Python dependencies required by the experiment being reproduced.
The core test suite is intentionally torch-free where possible:

```bash
python3 -m pytest -q
```

Validate the retained formal studies without rerunning them:

```bash
python3 scripts/validate_tmlr_synthetic_v1.py
python3 scripts/validate_fdst_c7_result.py
python3 scripts/validate_tmlr_selector_v2.py
python3 scripts/validate_tmlr_claim_set.py
```

The formal runners enforce frozen configuration hashes, a clean Git worktree,
and single-shot locks. Do not invoke a formal runner merely to test the code;
use its documented `--testing` path or the test suite instead.

## Data

Raw datasets are not redistributed. Dataset roots are supplied locally through
the configuration files. The retained natural tasks include crowd counting,
age estimation, aesthetic assessment, image-quality assessment, depth
estimation, and the FDST confirmation boundary.

## Paper and supplement

- Paper source: `paper/main.tex`
- Bibliography: `paper/refs.bib`
- Submission supplement: `release/rrcl_tmlr_submission_supplement_2026-08-10.zip`
- Supplement checksum: adjacent `.sha256` file

The paper source remains anonymous for double-blind review. Generated LaTeX
artifacts and raw experiment outputs are intentionally excluded from Git.
