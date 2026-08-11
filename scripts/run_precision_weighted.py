#!/usr/bin/env python3
"""Causal residual-precision weighted analytic continual regression baseline.

Motivation
----------
The confirmatory analysis showed that the benefit of f < 1 on JHU sequences is
consistent with *domain quality heterogeneity* rather than mapping drift.
Under heteroscedastic noise the classical optimal domain weighting is
inverse-variance (GLS / Aitken weighting)

    w_k  ~  1 / sigma_k^2,

which depends on measured domain noise rather than on recency.

This script is a *required* baseline before any multi-factor forgetting
extension.  It is CAUSAL: every domain's precision is estimated from only
its own data and domains processed so far; no future-domain statistics leak
into early decisions.

Noise estimate
--------------
For domain k, using only its training set:
  1. Split by image index (every 5th image → held-out validation).
  2. Fit a domain-local image-level ridge head on the fit partition.
  3. Compute heldout_domain_local_mse on the validation partition.
  4. residual_precision = 1 / (heldout_domain_local_mse + epsilon).

Causal accumulation
-------------------
At boundary k (after processing domain k):
  P_R(k) = sum_{j=1..k} p_j * R_j       (weighted patch-head R)
  P_C(k) = sum_{j=1..k} p_j * C_j       (weighted patch-head C)
  bar_p(k) = (1/k) * sum_{j=1..k} p_j   (running mean precision)

  R_eff(k) = P_R(k) / bar_p(k)           (normalized patch R)
  C_eff(k) = P_C(k) / bar_p(k)           (normalized patch C)

The same weighted accumulation is done separately for the image head.
Weights are normalized to mean 1 at each boundary so lambda keeps its
usual scale.  No clipping is applied (weight_ratio is reported as
diagnostic only).

The image-level residual precision is shared by both patch and image
heads.  This is an ASSUMPTION recorded in the output JSON, not a
proven mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_real import RealCountingDomains, build_from_config
from holdout_split import (
    domain_sample_key,
    is_validation,
    iter_stream_with_ids,
    require_both_partitions,
    split_manifest,
)
from rls_head import ForgettingRidgeRLS
from run_real_image_aux import _image_feature, _image_target, transform_patch_target
from run_real_norm_ablation import domain_scale, eval_domain

# Shared oracle grid (matches run_adaptive_f.py)
_ORACLE_FORGETS = [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]


def _aug(mat):
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat, np.ones((mat.shape[0], 1))], axis=1)


def _estimate_domain_noise(domains, t, scale, lam, epsilon=1e-8, val_every=5,
                           split_seed=42):
    """Held-out image-level residual MSE for domain t ONLY (causal).

    The fit/validation partition uses the stable hash split shared by every
    crowd runner (holdout_split.py), so it is order-independent and controlled
    by ``split_seed``.

    Returns (heldout_domain_local_mse, n_val, residual_precision, manifest).
    """
    if not (np.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"epsilon={epsilon} must be finite and positive")
    sample_key = domain_sample_key(domains, t)
    fit_R = fit_C = None
    val_rows, val_targets = [], []
    fit_ids, val_ids = [], []
    id_source = None
    for X, Y, _, image_id, source in iter_stream_with_ids(domains, "train", t):
        id_source = source
        row = _aug(_image_feature(X))
        target = _image_target(Y) / scale
        if is_validation(sample_key, image_id, split_seed, val_every):
            val_rows.append(row)
            val_targets.append(target)
            val_ids.append(image_id)
            continue
        fit_ids.append(image_id)
        if fit_R is None:
            d_aug = row.shape[1]
            fit_R = np.zeros((d_aug, d_aug), dtype=np.float64)
            fit_C = np.zeros((d_aug, 1), dtype=np.float64)
        fit_R += row.T @ row
        fit_C += row.T @ target

    manifest = require_both_partitions(
        split_manifest(sample_key, split_seed, val_every, fit_ids, val_ids, id_source),
        f"precision noise estimate for domain {t} ({sample_key})",
    )
    w = np.linalg.solve(lam * np.eye(fit_R.shape[0]) + fit_R, fit_C)
    rows = np.concatenate(val_rows, axis=0)
    targets = np.concatenate(val_targets, axis=0)
    residual = targets - rows @ w
    mse = float(np.mean(residual * residual))
    precision = 1.0 / max(mse, epsilon)
    return mse, len(val_ids), precision, manifest


def run_causal_precision(domains, patch_target, alpha, lam, epsilon=1e-8,
                         split_seed=42):
    """Causal residual-precision weighting.

    The image-level residual precision is estimated for each domain from
    only that domain's training data.  Both patch and image heads share
    the same image-level precision estimate.
    """
    if not (np.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"epsilon={epsilon} must be finite and positive")
    T = domains.n_domains()
    d_in = None
    for X, _, _ in domains.stream("train", 0):
        d_in = np.asarray(X).shape[1]
        break

    scales = [domain_scale(domains, t, "mean") for t in range(T)]

    # Per-domain diagnostics
    heldout_mse: list[float] = []
    residual_precisions: list[float] = []
    n_validation_images: list[int] = []
    normalized_weights_per_boundary: list[list[float]] = []
    split_manifests: list[dict] = []

    # Patch-head weighted accumulators
    P_R_patch = np.zeros((d_in + 1, d_in + 1), dtype=np.float64)
    P_C_patch = np.zeros((d_in + 1, 1), dtype=np.float64)
    # Image-head weighted accumulators
    P_R_image = np.zeros((d_in + 1, d_in + 1), dtype=np.float64)
    P_C_image = np.zeros((d_in + 1, 1), dtype=np.float64)

    # Running sum of precisions
    sum_p = 0.0

    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)

    M_rel = np.full((T, T), np.nan)
    weight_ratios: list[float] = []

    for t in range(T):
        # --- causal noise estimate ---
        mse, n_val, p_t, manifest = _estimate_domain_noise(
            domains, t, scales[t], lam, epsilon, split_seed=split_seed
        )
        if not (np.isfinite(mse) and mse >= 0):
            raise ValueError(f"domain {t}: heldout_domain_local_mse={mse} is invalid")
        if not (np.isfinite(p_t) and p_t > 0):
            raise ValueError(f"domain {t}: residual_precision={p_t} is invalid")
        heldout_mse.append(mse)
        residual_precisions.append(p_t)
        n_validation_images.append(n_val)
        split_manifests.append(manifest)

        # --- accumulate p_t * R_t, p_t * C_t (NOT p_t/bar_p * R_t) ---
        sum_p += p_t
        bar_p = sum_p / (t + 1)  # running mean precision

        # Record normalized weights (p_j / bar_p) — these ARE the model weights
        norm_ws = [residual_precisions[j] / max(bar_p, 1e-12) for j in range(t + 1)]
        normalized_weights_per_boundary.append(norm_ws)
        weight_ratios.append(
            max(norm_ws) / max(min(norm_ws), 1e-12) if len(norm_ws) > 1 else 1.0
        )

        for X, Y, _ in domains.stream("train", t):
            py = transform_patch_target(Y, patch_target) / scales[t]
            iy = _image_target(Y) / scales[t]

            # patch accumulation:  p_t · R_t,  p_t · C_t
            Xa_p = np.concatenate([np.asarray(X, dtype=np.float64),
                                   np.ones((np.asarray(X).shape[0], 1))], axis=1)
            P_R_patch += p_t * (Xa_p.T @ Xa_p)
            P_C_patch += p_t * (Xa_p.T @ py)

            # image accumulation
            Xa_i = _aug(_image_feature(X))
            P_R_image += p_t * (Xa_i.T @ Xa_i)
            P_C_image += p_t * (Xa_i.T @ iy)

        # --- normalize by bar_p before solving ---
        patch_head.R = P_R_patch / max(bar_p, 1e-12)
        patch_head.C = P_C_patch / max(bar_p, 1e-12)
        image_head.R = P_R_image / max(bar_p, 1e-12)
        image_head.C = P_C_image / max(bar_p, 1e-12)
        patch_head.solve()
        image_head.solve()

        for i in range(t + 1):
            _, M_rel[t, i] = eval_domain(
                domains, i, patch_head, image_head, alpha, scales[i]
            )

    final_rel = float(np.nanmean(M_rel[T - 1, :T]))
    return (
        final_rel, heldout_mse, residual_precisions, n_validation_images,
        normalized_weights_per_boundary, weight_ratios, M_rel, scales,
        split_manifests,
    )


def run_f1_reference(domains, patch_target, alpha, lam):
    """Absolute memorization (f=1) reference."""
    T = domains.n_domains()
    d_in = None
    for X, _, _ in domains.stream("train", 0):
        d_in = np.asarray(X).shape[1]
        break
    scales = [domain_scale(domains, t, "mean") for t in range(T)]

    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    M_rel = np.full((T, T), np.nan)
    for t in range(T):
        patch_head.begin_task(1.0)
        image_head.begin_task(1.0)
        for X, Y, _ in domains.stream("train", t):
            py = transform_patch_target(Y, patch_target) / scales[t]
            iy = _image_target(Y) / scales[t]
            patch_head.accumulate(X, py)
            image_head.accumulate(_image_feature(X), iy)
        patch_head.solve()
        image_head.solve()
        for i in range(t + 1):
            _, M_rel[t, i] = eval_domain(
                domains, i, patch_head, image_head, alpha, scales[i]
            )
    return float(np.nanmean(M_rel[T - 1, :T])), M_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--lam", type=float, default=1e2)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--patch-target", default="dct5")
    ap.add_argument("--max-per-domain", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="independent seed for the stable-hash fit/val split")
    ap.add_argument("--epsilon", type=float, default=1e-8,
                    help="floor for residual-precision denominator")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    with open(a.config) as handle:
        cfg = json.load(handle)
    domains = RealCountingDomains(
        build_from_config(cfg), backbone=a.backbone, img_size=a.img_size,
        max_per_domain=a.max_per_domain, sample_seed=a.sample_seed,
    )
    data_manifest = domains.data_manifest()

    # epsilon validation (ValueError, not assert)
    if not (np.isfinite(a.epsilon) and a.epsilon > 0):
        raise ValueError(f"epsilon={a.epsilon} must be finite and positive")

    t0 = time.time()

    # f=1 reference
    rel_f1, f1_M_rel = run_f1_reference(domains, a.patch_target, a.alpha, a.lam)

    # Causal precision weighting
    (rel_pw, heldout_mse, residual_precisions, n_val_images,
     norm_weights, weight_ratios, precision_M_rel, scales,
     split_manifests) = run_causal_precision(
        domains, a.patch_target, a.alpha, a.lam, a.epsilon,
        split_seed=a.split_seed,
    )

    # Fixed-f oracle sweep
    from run_real_norm_ablation import train_eval
    oracle_curve = []
    best_f, best_rel = None, float("inf")
    for f in _ORACLE_FORGETS:
        _, rel, _, _, _ = train_eval(domains, a.patch_target, f, a.alpha, a.lam, "mean")
        oracle_curve.append({"factor": f, "rel_MAE": rel})
        if rel < best_rel:
            best_f, best_rel = f, rel

    names = [spec.name for spec in domains.domains]
    print(f"\nconfig={a.config}  seed={a.sample_seed}")
    print(f"{'domain':<12}{'mse':>14}{'precision':>14}{'weight':>10}")
    for name, mse, prec, w in zip(names, heldout_mse, residual_precisions,
                                  norm_weights[-1] if norm_weights else [1.0]):
        print(f"{name:<12}{mse:>14.6g}{prec:>14.6g}{w:>10.3f}")
    print(f"  weight_ratio(final) = {weight_ratios[-1]:.3f}" if weight_ratios else "")

    gain_f1 = (rel_f1 - rel_pw) / max(rel_f1, 1e-12) * 100.0
    print(f"\n  f=1 (absolute memory)     rel_MAE = {rel_f1:.5f}")
    print(f"  oracle fixed f={best_f:<5g}     rel_MAE = {best_rel:.5f}")
    print(f"  precision-weighted        rel_MAE = {rel_pw:.5f}")
    print(f"  -> vs f=1: {gain_f1:+.2f}%")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        from result_io import dump_result
        from run_provenance import with_provenance

        payload = dict(
            config=a.config,
            sample_seed=a.sample_seed,
            split_seed=a.split_seed,
            holdout_split={"split_seed": a.split_seed, "domains": split_manifests},
            domain_names=names,
            data_manifest=data_manifest,
            heldout_domain_local_mse=heldout_mse,
            residual_precision=residual_precisions,
            n_validation_images=n_val_images,
            normalized_weights_at_each_boundary=norm_weights,
            weight_ratio=weight_ratios[-1] if weight_ratios else None,
            rel_f1=rel_f1,
            rel_precision_weighted=rel_pw,
            oracle_curve=oracle_curve,
            oracle_f=best_f,
            oracle_rel=best_rel,
            f1_M_rel=f1_M_rel.tolist(),
            precision_M_rel=precision_M_rel.tolist(),
            scales=scales,
            complete_cli_arguments=sys.argv,
            runtime=time.time() - t0,
            image_head_proxy_shared_by_patch_and_image_heads=(
                "image-level residual precision is shared by patch and image heads"
            ),
        )
        dump_result(
            out / "precision_weighted.json",
            with_provenance(payload, a.config, vars(a)),
        )
        print(f"\nsaved -> {out / 'precision_weighted.json'}")


if __name__ == "__main__":
    main()
