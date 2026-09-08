# RRCL KADID runbook: local CPU and AutoDL GPU

This runbook restores the new-method development path after the previous
AutoDL instance and its ephemeral datasets were released.

## Available local assets

- KADID-10k: `/Users/pangwei/Documents/ai/datasets/KADID10k/`
- Frozen DINOv2 model weights: present in the local Hugging Face cache
- KADID feature cache: initially absent
- CUDA: unavailable
- MPS: PyTorch is built with MPS support, but MPS is unavailable on this host
- Effective accelerator for this run: CPU

The FDST dataset and cache remain excluded from method development.

## Measured feasibility

On 2026-09-08, one uncached KADID image at 518x518 produced a `(1369, 768)`
DINOv2 patch-token array. Model construction took approximately 12.5 seconds
and one CPU inference took approximately 2.36 seconds.

Protocol v1 uses at most 500 training images per each of five domains and all
reference-disjoint test images, approximately 4,625 unique images. At the
measured seconds-per-image rate, about 3.0 hours is therefore a lower-bound
estimate for feature extraction;
the complete audit also performs analytic solves and a 2,000-dimensional
random-feature comparison.

## Resumable precomputation

Feature writes are atomic. Re-running the command loads completed `.npy`
files and continues missing images.

The tracked dataset config is portable. Set `RRCL_KADID_ROOT` to the extracted
KADID directory containing `images/` and `dmos.csv`.

Default repository-local cache:

```bash
export RRCL_KADID_ROOT=/path/to/KADID10k/extracted/kadid10k
python3 scripts/precompute_kadid_features.py
```

Persistent external or server storage:

```bash
RRCL_FEATURE_CACHE_ROOT=/path/to/persistent/rrcl_cache \
python3 scripts/precompute_kadid_features.py
```

Short non-evidential pipeline check:

```bash
python3 scripts/precompute_kadid_features.py --limit 10 --progress-every 1
```

The cache manifest is written beside the feature arrays. An interrupted or
limited run is explicitly marked incomplete and is not an experiment result.

## Baseline audit

After feature precomputation:

```bash
python3 scripts/run_new_method_kadid_baselines.py
python3 scripts/validate_new_method_kadid_baselines.py \
  runs_real/new_method_kadid_development_v1/main/kadid_baseline_audit.json
```

The primary result is `aligned_fit_only`; `full_refit_ablation` is separately
labelled and is not certified by the fit-only validation objective.

## Scale-equivalence audit v2

After the first-batch result exposed a global regularization-scale confound,
run the frozen normalized-objective follow-up with the same feature cache:

```bash
python3 scripts/run_kadid_scale_audit_v2.py
python3 scripts/validate_kadid_scale_audit_v2.py \
  runs_real/new_method_kadid_scale_audit_v2/main/kadid_scale_audit_v2.json
```

The v2 protocol compares sample-mean pooled and domain-mean balanced
objectives with total observation weight one and the same 37-point normalized
lambda grid. It also verifies the corresponding raw-scale solution at every
domain boundary. It does not rerun the projection or independent-head
baselines and does not yet test a candidate trajectory bank or automatic `f`.

After moving an artifact to another machine, the validator falls back to the
tracked domain config with the recorded basename and still requires its
SHA-256 to match. A relocated config can also be supplied explicitly:

```bash
python3 scripts/validate_new_method_kadid_baselines.py /path/to/kadid_baseline_audit.json \
  --domain-config configs/domains_iqa_kadid.json
```

## Future GPU migration

Use `RRCL_FEATURE_CACHE_ROOT` on persistent storage rather than the instance's
ephemeral system disk. Copy the whole cache root back to local storage before
releasing the instance. Cache keys include the absolute source path, backbone,
and image size; if dataset paths change across machines, precomputed caches do
not currently deduplicate automatically, so keep a stable mounted dataset path
or add a path-independent cache manifest before migration.

## Fresh AutoDL instance

Clone below `/root/autodl-tmp`, which is the intended persistent data volume,
then run:

```bash
bash scripts/setup_autodl.sh
export RRCL_KADID_ROOT=/root/autodl-tmp/datasets/KADID10k/extracted/kadid10k
bash scripts/run_kadid_autodl.sh
bash scripts/package_autodl_results.sh

# Follow-up scale audit, reusing the completed feature cache
bash scripts/run_kadid_scale_audit_v2.sh
bash scripts/package_kadid_scale_audit_v2.sh
```

The setup script deliberately keeps the CUDA-enabled PyTorch supplied by the
AutoDL base image. The run script fails before feature extraction if CUDA or
the KADID layout is unavailable. The final packaging command writes a result
archive and SHA-256 checksum under `/root/autodl-tmp/rrcl_exports`.
