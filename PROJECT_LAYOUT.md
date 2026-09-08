# RRCL repository map

This file maps the public research artifact by function. It contains no
machine-specific paths; dataset and output locations are supplied at runtime.

## Core methods

- `rls_head.py`: recursive ridge regression with scalar forgetting.
- `adaptive_selector.py`: held-out, training-only factor selection.
- `analytic_forgetting_baselines.py`: DOS-ELM-style and SIFt-RLS cores.
- `vff_baselines.py`: variable-forgetting baselines.
- `precision_shrinkage.py`: precision and shrinkage diagnostics.
- `tmlr_synthetic.py`: controlled synthetic study.
- `tmlr_selector_v2.py`: held-out learned-selector and matched-baseline study.

## Natural-data adapters

- `datasets_real.py`: crowd-counting datasets.
- `datasets_fdst.py`: frozen FDST confirmation layout.
- `datasets_age.py`: age-estimation datasets.
- `datasets_ava.py`: aesthetic assessment.
- `datasets_iqa.py`: image-quality assessment.
- `datasets_depth.py`: depth estimation.

## Experiment entry points

- Development diagnostics: `run_adaptive_f.py`, `run_theory_predict.py`,
  `run_real_norm_ablation.py`, and the matching scripts under `scripts/`.
- Controlled study: `scripts/run_tmlr_synthetic_v1.py`.
- FDST boundary: `scripts/preflight_fdst.py` and `scripts/run_fdst_c7.py`.
- Held-out selector study: `scripts/run_tmlr_selector_v2.py`.
- Post-freeze exploratory diagnostics: `scripts/history_dominance_phase.py`
  and `scripts/explore_candidate_family.py`.

Formal runners are guarded by frozen hashes and single-shot locks. Their
validators are the normal public verification entry points.
The exploratory diagnostics write to separate ignored directories and do not
modify any frozen formal artifact.

## Protocols and results

- Synthetic: `docs/RRCL_TMLR_SYNTHETIC_PROTOCOL.md` and
  `docs/RRCL_TMLR_SYNTHETIC_RESULTS_2026-08-07.md`.
- FDST: `docs/C7_FDST_PROTOCOL_V2.md`, `docs/C7_FDST_PROTOCOL_V3.md`, and
  `docs/RRCL_TMLR_FDST_RESULTS_2026-08-09.md`.
- Learned selector: `docs/RRCL_TMLR_SELECTOR_V2_PROTOCOL.md` and
  `docs/RRCL_TMLR_SELECTOR_V2_RESULTS_2026-08-10.md`.
- Unified experiment guide: `docs/EXPERIMENTS.md`.

## Reproducibility components

- `configs/`: frozen JSON configs and FDST split manifests.
- `scripts/validate_*.py`: independent result recomputation and checks.
- `scripts/setup_autodl.sh`: CUDA-safe dependency setup and repository tests.
- `scripts/run_kadid_autodl.sh`: resumable KADID development run on AutoDL.
- `scripts/package_autodl_results.sh`: validated-result export archive.
- `tests/`: implementation and protocol regression tests.
- `paper/figs/`: tables and figures bound to retained evidence.
- `release/`: deterministic anonymous submission supplement and checksum.

Generated results, feature caches, datasets, and LaTeX build files are ignored
by Git. No raw third-party dataset is distributed in this repository.
