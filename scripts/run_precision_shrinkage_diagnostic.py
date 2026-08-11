#!/usr/bin/env python3
"""Precision-shrinkage DIAGNOSTIC: can a train-only objective pick gamma?

This script does NOT implement a shrinkage selector.  It sweeps

    w_k(gamma) = (1 - gamma) + gamma * p_k / mean(p),   gamma in [0, 1],

and answers, per sequence:

  1. Equivalence  — gamma=0 reproduces the f=1 model, and gamma=1 reproduces
     the THREE-WAY precision endpoint: causal precision weighting whose
     precisions are estimated on the precision_val role (10%).  This is NOT
     numerically identical to the production raw-precision method
     (scripts/run_precision_weighted.py), whose precisions use the full
     two-way validation set (20%).  The two are separate comparators and
     must be reported as such; only the weighted-accumulation MACHINERY is
     shared (and is unit-tested against production with shared precisions).
  2. Representativeness — does a train-only balanced validation objective
     have its argmin at the same gamma as the TEST balanced rel-MAE curve?
     Three PROXY objectives are reported separately: patch MSE, image MSE
     and fused MSE (stacked [W_patch; W_image] against the fused design
     row).  ALL are unclipped quadratic proxies: the deployed metric is
     non-negative-clipped fused relative MAE, which cannot be computed
     exactly from O(d^2) second-order statistics — no "exact alignment"
     claim is permitted.  The fused proxy is the headline curve.  The test
     curve is a diagnostic readout only and must never feed a selection rule.
  3. Safety structure — per-domain validation deltas vs per-domain test
     deltas (sign agreement, fused proxy).  If validation reproduces the
     sign structure (e.g. QNRF sacrifices two domains, JHU improves all
     three), a train-only per-domain no-harm gate is viable; if not, the
     failure is representativeness and no gate computed from this split can
     fix it.
  4. Outlier audit — top precision-partition images per domain by absolute
     residual of the domain-local fit, with their share of the partition's
     held-out MSE (is JHU "unreliable" a heavy-tail-few-images story?).

Validation reuse: the former two-way split let the same held-out images
both estimate residual precision and score gamma.  This runner uses the
three-way hash roles (fit / precision_val / gamma_val) from holdout_split;
the two validation roles are disjoint and their union equals the two-way
selector validation set.

Interpretation guard (pre-registered): if the QNRF train-only objective
still prefers gamma > 0 while the test optimum is gamma = 0, a bootstrap
gate would only certify that the WRONG signal is statistically stable;
representativeness, not variance, is then the blocker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdout_split import (
    ROLE_FIT,
    ROLE_GAMMA_VAL,
    ROLE_PRECISION_VAL,
    domain_sample_key,
    holdout_role,
    iter_stream_with_ids,
    require_roles_nonempty,
    roles_manifest,
)
from precision_shrinkage import (
    causal_boundary_reference,
    causal_final_weights,
    ensure_clean_worktree,
    gamma_sweep,
    grid_argmin,
    quadratic_mse,
    sign_agreement,
    solve_ridge,
)
from result_io import dump_result
from rls_head import ForgettingRidgeRLS
from run_provenance import with_provenance
from run_real_image_aux import _image_feature, _image_target, transform_patch_target
from run_real_norm_ablation import domain_scale, eval_domain


def _aug(mat):
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat, np.ones((mat.shape[0], 1))], axis=1)


def _zero(d_aug):
    return {
        "R": np.zeros((d_aug, d_aug)),
        "C": np.zeros((d_aug, 1)),
        "S": 0.0,
        "n": 0,
    }


def collect_domain(domains, t, scale, patch_target, alpha, d_aug, val_every, split_seed):
    """One pass over domain t's training stream with the THREE-WAY split.

    Roles (holdout_split.holdout_role, pairwise disjoint):
      * fit           -> candidate-model statistics;
      * precision_val -> residual-precision estimation rows only;
      * gamma_val     -> gamma-selection quadratics only.
    Their union of the two validation roles equals the two-way selector
    validation set, and no image is used both to estimate a precision and to
    score gamma (pilot-audit double-dipping fix).  The TEST stream is never
    touched here.
    """
    sample_key = domain_sample_key(domains, t)
    full = {"patch": [np.zeros((d_aug, d_aug)), np.zeros((d_aug, 1))],
            "image": [np.zeros((d_aug, d_aug)), np.zeros((d_aug, 1))]}
    fit = {"patch": [np.zeros((d_aug, d_aug)), np.zeros((d_aug, 1))],
           "image": [np.zeros((d_aug, d_aug)), np.zeros((d_aug, 1))]}
    val = {"patch": _zero(d_aug), "image": _zero(d_aug), "fused": _zero(2 * d_aug)}
    precision_quad = _zero(d_aug)
    prec_rows, prec_targets, prec_ids_order, prec_raw_counts = [], [], [], []
    role_ids = {ROLE_FIT: [], ROLE_PRECISION_VAL: [], ROLE_GAMMA_VAL: []}
    id_source = None
    for X, Y, _, image_id, source in iter_stream_with_ids(domains, "train", t):
        id_source = source
        X = np.asarray(X, dtype=np.float64)
        Xp = _aug(X)
        yp = transform_patch_target(Y, patch_target) / scale
        mean_token = X.mean(axis=0, keepdims=True)
        Xi = _aug(mean_token)
        yi = _image_target(Y) / scale

        full["patch"][0] += Xp.T @ Xp
        full["patch"][1] += Xp.T @ yp
        full["image"][0] += Xi.T @ Xi
        full["image"][1] += Xi.T @ yi

        role = holdout_role(sample_key, image_id, split_seed, val_every)
        role_ids[role].append(image_id)
        if role == ROLE_PRECISION_VAL:
            precision_quad["R"] += Xi.T @ Xi
            precision_quad["C"] += Xi.T @ yi
            precision_quad["S"] += float(np.sum(yi * yi))
            precision_quad["n"] += Xi.shape[0]
            prec_rows.append(Xi)
            prec_targets.append(yi)
            prec_ids_order.append(image_id)
            prec_raw_counts.append(float(np.asarray(Y).sum()))
        elif role == ROLE_GAMMA_VAL:
            for space, rows, targets in (("patch", Xp, yp), ("image", Xi, yi)):
                val[space]["R"] += rows.T @ rows
                val[space]["C"] += rows.T @ targets
                val[space]["S"] += float(np.sum(targets * targets))
                val[space]["n"] += rows.shape[0]
            patch_block = np.concatenate([X.sum(axis=0), [float(X.shape[0])]])
            image_block = np.concatenate([mean_token[0], [1.0]])
            fused_row = np.concatenate(
                [alpha * patch_block, (1.0 - alpha) * image_block]
            )[None, :]
            val["fused"]["R"] += fused_row.T @ fused_row
            val["fused"]["C"] += fused_row.T @ yi
            val["fused"]["S"] += float(np.sum(yi * yi))
            val["fused"]["n"] += fused_row.shape[0]
        else:
            fit["patch"][0] += Xp.T @ Xp
            fit["patch"][1] += Xp.T @ yp
            fit["image"][0] += Xi.T @ Xi
            fit["image"][1] += Xi.T @ yi

    manifest = require_roles_nonempty(
        roles_manifest(sample_key, split_seed, val_every, role_ids, id_source),
        (ROLE_FIT, ROLE_PRECISION_VAL, ROLE_GAMMA_VAL),
        f"shrinkage diagnostic for domain {t} ({sample_key})",
    )
    stats = {
        "full": {space: tuple(full[space]) for space in full},
        "fit": {space: tuple(fit[space]) for space in fit},
        "val": val,
    }
    raw = {
        "rows": np.concatenate(prec_rows, axis=0),
        "targets": np.concatenate(prec_targets, axis=0),
        "ids": prec_ids_order,
        "raw_counts": prec_raw_counts,
        "quad": precision_quad,
    }
    return stats, raw, manifest


def estimate_precision(stats, raw, lam, epsilon):
    """Domain-local image-head fit on FIT stats, MSE on the PRECISION rows.

    Uses only the precision_val role; the gamma_val role never contributes,
    so the gamma objective is scored on data unseen by the weight family.
    """
    W = solve_ridge(stats["fit"]["image"][0], stats["fit"]["image"][1], lam)
    residual = raw["targets"] - raw["rows"] @ W
    mse = float(np.mean(residual * residual))
    quad = quadratic_mse(
        W,
        raw["quad"]["R"],
        raw["quad"]["C"],
        raw["quad"]["S"],
        raw["quad"]["n"],
    )
    if not np.isclose(mse, quad, rtol=1e-6, atol=1e-10):
        raise AssertionError(
            f"row-wise MSE {mse} disagrees with quadratic MSE {quad}"
        )
    precision = 1.0 / max(mse, epsilon)
    residuals = np.abs(residual.reshape(-1))
    return mse, precision, W, residuals


def outlier_report(raw, residuals, mse, scale, top_k):
    order = np.argsort(residuals)[::-1][: int(top_k)]
    total = float(np.sum(residuals * residuals))
    rows = []
    for index in order:
        r = float(residuals[index])
        rows.append({
            "image_id": raw["ids"][index],
            "raw_count": raw["raw_counts"][index],
            "abs_residual_normalized": r,
            "abs_residual_raw_count_equiv": r * float(scale),
            "share_of_heldout_sse": (r * r) / max(total, 1e-12),
        })
    return {
        "heldout_mse_normalized": mse,
        "n_precision_partition_images": len(raw["ids"]),
        "top_outliers": rows,
    }


def deployed_heads(record, d_in, lam):
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    patch_head.W = record["patch"]["W_deployed"]
    image_head.W = record["image"]["W_deployed"]
    return patch_head, image_head


def sequential_f1_reference(domains, patch_target, alpha, lam, scales):
    """Production-style f=1 run (begin_task/accumulate) for the equivalence check."""
    T = domains.n_domains()
    d_in = None
    for X, _, _ in domains.stream("train", 0):
        d_in = np.asarray(X).shape[1]
        break
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=lam)
    for t in range(T):
        patch_head.begin_task(1.0)
        image_head.begin_task(1.0)
        for X, Y, _ in domains.stream("train", t):
            patch_head.accumulate(X, transform_patch_target(Y, patch_target) / scales[t])
            image_head.accumulate(_image_feature(X), _image_target(Y) / scales[t])
    patch_head.solve()
    image_head.solve()
    per_domain = []
    for i in range(T):
        _, rel = eval_domain(domains, i, patch_head, image_head, alpha, scales[i])
        per_domain.append(rel)
    return float(np.mean(per_domain)), per_domain


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
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=1e-8)
    ap.add_argument("--gamma-step", type=float, default=0.05)
    ap.add_argument("--report-outliers", type=int, default=10)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="scratch runs only; paper-facing runs must be clean")
    ap.add_argument("--out", default="runs_real/shrinkage_diag")
    a = ap.parse_args()

    worktree = ensure_clean_worktree(a.allow_dirty)

    from datasets_real import RealCountingDomains, build_from_config
    with open(a.config) as handle:
        cfg = json.load(handle)
    domains = RealCountingDomains(
        build_from_config(cfg), backbone=a.backbone, img_size=a.img_size,
        max_per_domain=a.max_per_domain, sample_seed=a.sample_seed,
    )
    data_manifest = domains.data_manifest()
    T = domains.n_domains()
    names = [spec.name for spec in domains.domains]

    d_in = None
    for X, _, _ in domains.stream("train", 0):
        d_in = np.asarray(X).shape[1]
        break
    d_aug = d_in + 1

    t0 = time.time()
    scales = [domain_scale(domains, t, "mean") for t in range(T)]

    domain_stats, precisions, heldout_mses, manifests, outliers = [], [], [], [], []
    for t in range(T):
        stats, raw, manifest = collect_domain(
            domains, t, scales[t], a.patch_target, a.alpha, d_aug,
            a.val_every, a.split_seed
        )
        mse, precision, _, residuals = estimate_precision(stats, raw, a.lam, a.epsilon)
        domain_stats.append(stats)
        precisions.append(precision)
        heldout_mses.append(mse)
        manifests.append(manifest)
        outliers.append(outlier_report(raw, residuals, mse, scales[t], a.report_outliers))

    gammas = np.round(np.arange(0.0, 1.0 + 1e-9, a.gamma_step), 3)
    records = gamma_sweep(domain_stats, precisions, gammas, a.lam)

    # ---- test curves (DIAGNOSTIC ONLY; never feeds a selection rule) ----
    test_balanced, test_per_domain = [], []
    for record in records:
        patch_head, image_head = deployed_heads(record, d_in, a.lam)
        rels = []
        for i in range(T):
            _, rel = eval_domain(domains, i, patch_head, image_head, a.alpha, scales[i])
            rels.append(rel)
        test_per_domain.append(rels)
        test_balanced.append(float(np.mean(rels)))

    # ---- equivalence checks ----
    zero_index = grid_argmin(np.abs(gammas - 0.0))
    one_index = grid_argmin(np.abs(gammas - 1.0))
    f1_rel, f1_per_domain = sequential_f1_reference(
        domains, a.patch_target, a.alpha, a.lam, scales
    )
    gamma0_rel = test_balanced[zero_index]
    # Tolerance: identical model up to floating-point summation order
    # (per-image sequential adds vs per-domain block sums).
    equiv_f1 = bool(abs(gamma0_rel - f1_rel) <= 1e-6)

    # gamma=1 vs the boundary-by-boundary causal accumulation of the SAME
    # three-way precisions.  Deliberately NOT compared against the production
    # 80/20 raw-precision run: different precision-estimation protocol,
    # different method (see module docstring).
    reference_patch = causal_boundary_reference(
        [d["full"]["patch"] for d in domain_stats], precisions, a.lam
    )
    reference_image = causal_boundary_reference(
        [d["full"]["image"] for d in domain_stats], precisions, a.lam
    )
    equiv_three_way_endpoint = bool(
        np.allclose(records[one_index]["patch"]["W_deployed"], reference_patch,
                    rtol=1e-9, atol=1e-12)
        and np.allclose(records[one_index]["image"]["W_deployed"], reference_image,
                        rtol=1e-9, atol=1e-12)
    )

    # ---- argmins and representativeness (all objectives are PROXIES) ----
    val_image_curve = [r["image"]["val_balanced"] for r in records]
    val_patch_curve = [r["patch"]["val_balanced"] for r in records]
    val_fused_curve = [r["fused"]["val_balanced"] for r in records]
    argmin_val_image = grid_argmin(val_image_curve)
    argmin_val_patch = grid_argmin(val_patch_curve)
    argmin_val_fused = grid_argmin(val_fused_curve)
    argmin_test = grid_argmin(test_balanced)

    # ---- per-domain sign structure (fused proxy) at train argmin, gamma=1 ----
    def sign_table(index):
        table = []
        for j in range(T):
            val_delta = (records[index]["fused"]["val_per_domain"][j]
                         - records[zero_index]["fused"]["val_per_domain"][j])
            test_delta = test_per_domain[index][j] - test_per_domain[zero_index][j]
            table.append({
                "domain": names[j],
                "val_delta_fused_proxy": val_delta,
                "test_delta_rel_mae": test_delta,
                "sign": sign_agreement(val_delta, test_delta, tolerance=1e-10),
            })
        return table

    sign_at_train_argmin = sign_table(argmin_val_fused)
    sign_at_gamma1 = sign_table(one_index)

    # ---- console summary ----
    weights_final = causal_final_weights(precisions)
    print(f"config={a.config}  sample_seed={a.sample_seed}  split_seed={a.split_seed}")
    print(f"{'domain':<12}{'heldout_mse':>14}{'precision':>12}{'weight(g=1)':>12}")
    for name, mse, p, w in zip(names, heldout_mses, precisions, weights_final):
        print(f"{name:<12}{mse:>14.6g}{p:>12.6g}{w:>12.3f}")
    print(f"\n{'gamma':>6}{'fused*':>12}{'image*':>12}{'patch*':>12}{'test_relMAE':>13}"
          f"   (* = unclipped MSE proxy)")
    for g, vf, vi, vp, tb in zip(gammas, val_fused_curve, val_image_curve,
                                 val_patch_curve, test_balanced):
        print(f"{g:>6.2f}{vf:>12.6g}{vi:>12.6g}{vp:>12.6g}{tb:>13.5f}")
    print(f"\n  equivalence gamma=0 == f=1                     : {equiv_f1}"
          f"  (sweep {gamma0_rel:.6f} vs sequential {f1_rel:.6f})")
    print(f"  equivalence gamma=1 == 3-way precision endpoint: "
          f"{equiv_three_way_endpoint}"
          f"  (NOT the production 80/20 raw-precision method)")
    print(f"  argmin train proxies: fused={gammas[argmin_val_fused]:.2f} "
          f"image={gammas[argmin_val_image]:.2f} patch={gammas[argmin_val_patch]:.2f} "
          f"| TEST gamma = {gammas[argmin_test]:.2f}  <- diagnostic only")
    print("  sign structure (fused proxy) at train argmin:")
    for row in sign_at_train_argmin:
        print(f"    {row['domain']:<10} val {row['val_delta_fused_proxy']:+.6g} "
              f"test {row['test_delta_rel_mae']:+.5f}  -> {row['sign']}")

    payload = {
        "config": a.config,
        "diagnostic_only": True,
        "note": (
            "TEST curves are a diagnostic readout for representativeness "
            "analysis; no selection rule may consume them.  All validation "
            "objectives are UNCLIPPED quadratic MSE proxies; the deployed "
            "metric (non-negative-clipped fused relative MAE) cannot be "
            "computed exactly from O(d^2) second-order statistics."
        ),
        "normalization_note": (
            "domain_scale s_t is computed from the FULL training domain and "
            "is declared task metadata of the task-aware protocol (the paper "
            "already requires s_k at inference); it is not a fit-only "
            "quantity."
        ),
        "sample_seed": a.sample_seed,
        "split_seed": a.split_seed,
        "val_every": a.val_every,
        "holdout_split": {"split_seed": a.split_seed, "domains": manifests},
        "worktree": worktree,
        "domain_names": names,
        "scales": scales,
        "data_manifest": data_manifest,
        "heldout_domain_local_mse": heldout_mses,
        "residual_precision": precisions,
        "weights_gamma1": weights_final.tolist(),
        "gammas": gammas.tolist(),
        "val_balanced_fused_proxy": val_fused_curve,
        "val_balanced_image_proxy": val_image_curve,
        "val_balanced_patch_proxy": val_patch_curve,
        "val_per_domain_fused_proxy": [r["fused"]["val_per_domain"] for r in records],
        "val_per_domain_image_proxy": [r["image"]["val_per_domain"] for r in records],
        "val_per_domain_patch_proxy": [r["patch"]["val_per_domain"] for r in records],
        "test_balanced_rel_mae_DIAGNOSTIC": test_balanced,
        "test_per_domain_rel_mae_DIAGNOSTIC": test_per_domain,
        "argmin": {
            "train_fused_gamma": float(gammas[argmin_val_fused]),
            "train_image_gamma": float(gammas[argmin_val_image]),
            "train_patch_gamma": float(gammas[argmin_val_patch]),
            "test_gamma_DIAGNOSTIC": float(gammas[argmin_test]),
            "train_test_agree_fused": bool(argmin_val_fused == argmin_test),
            "train_test_agree_image": bool(argmin_val_image == argmin_test),
            "train_test_agree_patch": bool(argmin_val_patch == argmin_test),
        },
        "equivalence": {
            "gamma0_equals_f1": equiv_f1,
            "gamma0_rel_mae": gamma0_rel,
            "sequential_f1_rel_mae": f1_rel,
            "sequential_f1_per_domain": f1_per_domain,
            "gamma1_equals_three_way_precision_endpoint": equiv_three_way_endpoint,
            "endpoint_note": (
                "gamma=1 equals causal precision weighting with precisions "
                "from the precision_val role (three-way split).  The "
                "production raw-precision method estimates precisions on the "
                "full two-way validation set and is a SEPARATE comparator; "
                "no cross-protocol equality is claimed."
            ),
        },
        "sign_structure": {
            "objective_space": "fused_proxy",
            "at_train_argmin_fused": sign_at_train_argmin,
            "at_gamma1": sign_at_gamma1,
        },
        "outlier_audit": dict(zip(names, outliers)),
        "runtime_seconds": time.time() - t0,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_result(out / "shrinkage_diagnostic.json", with_provenance(payload, a.config, vars(a)))
    print(f"\nsaved -> {out / 'shrinkage_diagnostic.json'}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
