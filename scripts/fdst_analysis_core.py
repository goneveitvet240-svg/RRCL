#!/usr/bin/env python3
"""Shared statistical core for the frozen FDST confirmation.

Statistics are reported in relative units:
  * per-domain harm  = (E_m - E_f1) / max(E_f1, eps), gate 1%;
  * paired sequence-group block bootstrap resamples groups and computes the
    RELATIVE improvement (E_f1 - E_m) / E_f1 per replicate;
  * ``positive`` and ``safe`` are BOTH always evaluated from the final
    method vs f=1 alone.  The fixed-f oracle tests a DIFFERENT hypothesis
    (temporal forgetting) and is recorded as a diagnostic comparator only —
    it never selects which success standard applies (that would be
    outcome-dependent branching on test data; cf. the development JHU-last
    case where oracle == f=1 while reliability weighting improves +13.7%);
  * the bootstrap REFUSES per-frame fallback: every domain's test ids must
    yield decision == "detected", else protocol halt (§4/§6);
  * pairing identity is asserted (same domains, ids, order, gts) before the
    bootstrap; an atomic attempt lock is taken BEFORE any model computation
    so concurrent or repeated launches cannot race the single-shot rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import forgetting_matrix_stats
from precision_shrinkage import (
    causal_final_weights,
    ensure_clean_worktree,
    gamma_weights,
    quadratic_mse,
    solve_ridge,
    weighted_system,
)
from rls_head import ForgettingRidgeRLS
from run_real_image_aux import _image_feature, _image_target, transform_patch_target
from run_real_norm_ablation import domain_scale, eval_domain, train_eval
from run_theory_predict import predicted_factor

# ---- frozen shared constants ----
BACKBONE = "vit_base_patch14_dinov2.lvd142m"
IMG_SIZE = 518
LAM = 100.0
ALPHA = 0.25
PATCH_TARGET = "dct5"
METHOD_IDS = (
    "f1", "fixed_f_oracle", "shrinkage_gamma", "fstar_pred",
    "precision_weighting",
)
ORACLE_GRID = [1.0, 0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
BOOTSTRAP_B = 2000
MIN_IMPROVEMENT = 0.005          # 0.5% relative
MAX_DOMAIN_HARM = 0.01           # 1% relative
SAFETY_CI_FLOOR = -0.01          # -1% relative


def _aug(mat):
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat, np.ones((mat.shape[0], 1))], axis=1)


def require_detected_groups(ids, where):
    raise RuntimeError(
        "dataset-specific temporal grouping must be installed by the runner"
    )


def split_val_halves(val_ids):
    """Group-aligned front/back halves of the temporal validation block."""
    groups = require_detected_groups(val_ids, "val precision/gamma sub-split")
    total = sum(len(group) for group in groups)
    front, back, seen = [], [], 0
    for group in groups:
        (front if seen < 0.5 * total else back).extend(group)
        seen += len(group)
    if not front or not back:
        raise RuntimeError("degenerate precision/gamma sub-split of the val block")
    return front, back


def collect_domain(domains, d, scale, patch_target):
    """Full/fit stats plus disjoint precision and gamma-validation roles.

    Consistent with the development three-way protocol: the domain-local
    precision model fits on the fit block ONLY — both val halves (precision
    and the reserved gamma block) are excluded from its statistics.
    """
    precision_ids_list, gamma_ids_list = split_val_halves(domains.frozen_val_ids(d))
    precision_ids = set(precision_ids_list)
    gamma_ids = set(gamma_ids_list)
    val_ids = precision_ids | set(gamma_ids_list)
    full = {"patch": [None, None], "image": [None, None]}
    fit = {"patch": [None, None], "image": [None, None]}
    prec_rows, prec_targets = [], []
    d_aug = None
    gamma_fused = None
    full_n_images = 0

    def _add(bucket, space, rows, targets):
        if bucket[space][0] is None:
            dim = rows.shape[1]
            bucket[space][0] = np.zeros((dim, dim))
            bucket[space][1] = np.zeros((dim, 1))
        bucket[space][0] += rows.T @ rows
        bucket[space][1] += rows.T @ targets

    for X, Y, _, image_id in domains.stream("train", d, with_ids=True):
        Xp = _aug(X)
        yp = transform_patch_target(Y, patch_target) / scale
        Xi = _aug(_image_feature(X))
        yi = _image_target(Y) / scale
        if d_aug is None:
            d_aug = Xi.shape[1]
            gamma_fused = {
                "R": np.zeros((2 * d_aug, 2 * d_aug)),
                "C": np.zeros((2 * d_aug, 1)),
                "S": 0.0,
                "n": 0,
            }
        _add(full, "patch", Xp, yp)
        _add(full, "image", Xi, yi)
        full_n_images += 1
        if image_id in precision_ids:
            prec_rows.append(Xi)
            prec_targets.append(yi)
        elif image_id in gamma_ids:
            patch_block = np.concatenate([X.sum(axis=0), [float(X.shape[0])]])
            image_block = Xi[0]
            fused_row = np.concatenate(
                [ALPHA * patch_block, (1.0 - ALPHA) * image_block]
            )[None, :]
            gamma_fused["R"] += fused_row.T @ fused_row
            gamma_fused["C"] += fused_row.T @ yi
            gamma_fused["S"] += float(np.sum(yi * yi))
            gamma_fused["n"] += 1
        elif image_id not in val_ids:
            _add(fit, "patch", Xp, yp)
            _add(fit, "image", Xi, yi)
        # gamma-block images enter ONLY the full (deployed) statistics.
    if not prec_rows:
        raise RuntimeError(f"domain {d}: empty temporal precision block")
    if not gamma_fused or not gamma_fused["n"]:
        raise RuntimeError(f"domain {d}: empty temporal gamma block")
    return {
        "full": {space: tuple(full[space]) for space in full},
        "fit": {space: tuple(fit[space]) for space in fit},
        "precision_rows": np.concatenate(prec_rows),
        "precision_targets": np.concatenate(prec_targets),
        "gamma_fused": gamma_fused,
        "full_n_images": full_n_images,
    }


def per_image_errors(domains, d, patch_head, image_head, alpha, scale):
    """Per-test-image (abs_error, gt, image_id) records."""
    records = []
    for X, Y, _, image_id in domains.stream("test", d, with_ids=True):
        patch_count = float(patch_head.predict(X)[:, 0].sum())
        image_count = float(image_head.predict(_image_feature(X))[0, 0])
        pred = max((alpha * patch_count + (1.0 - alpha) * image_count) * scale, 0.0)
        gt = float(np.asarray(Y).sum())
        records.append((abs(pred - gt), gt, image_id))
    return records


def domain_metrics(records):
    errors = np.asarray([record[0] for record in records], dtype=np.float64)
    gts = np.asarray([record[1] for record in records], dtype=np.float64)
    mae = float(np.mean(errors))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "rel_mae": mae / max(float(np.mean(gts)), 1e-9),
        "n": int(errors.size),
    }


def collect_weighting_statistics(domains, epsilon=1e-8):
    T = domains.n_domains()
    scales = [domain_scale(domains, t, "mean") for t in range(T)]
    records, precisions = [], []
    for t in range(T):
        record = collect_domain(domains, t, scales[t], PATCH_TARGET)
        W_local = solve_ridge(record["fit"]["image"][0], record["fit"]["image"][1], LAM)
        residual = record["precision_targets"] - record["precision_rows"] @ W_local
        mse = float(np.mean(residual * residual))
        record["heldout_precision_mse"] = mse
        precisions.append(1.0 / max(mse, epsilon))
        records.append(record)
    return records, precisions, scales


def run_weighted_method(domains, gamma, collected=None):
    T = domains.n_domains()
    if collected is None:
        records, precisions, scales = collect_weighting_statistics(domains)
    else:
        records, precisions, scales = collected
    M_abs = np.full((T, T), np.nan)
    M_rel = np.full((T, T), np.nan)
    per_image = [None] * T
    d_in = records[0]["full"]["patch"][0].shape[0] - 1
    for t in range(T):
        gamma_t = float(gamma[t]) if isinstance(gamma, (list, tuple)) else float(gamma)
        w = gamma_weights(causal_final_weights(precisions[: t + 1]), gamma_t)
        patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
        image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
        prefix = records[: t + 1]
        R, C = weighted_system([f["full"]["patch"] for f in prefix], w)
        patch_head.W = solve_ridge(R, C, LAM)
        R, C = weighted_system([f["full"]["image"] for f in prefix], w)
        image_head.W = solve_ridge(R, C, LAM)
        for i in range(t + 1):
            M_abs[t, i], M_rel[t, i] = eval_domain(
                domains, i, patch_head, image_head, ALPHA, scales[i]
            )
        if t == T - 1:
            for i in range(T):
                per_image[i] = per_image_errors(
                    domains, i, patch_head, image_head, ALPHA, scales[i]
                )
    final_rel = float(np.nanmean(M_rel[T - 1, :T]))
    return final_rel, M_abs, M_rel, per_image, precisions, scales


def select_shrinkage_gamma(records, precisions):
    """Frozen train-only fused-MSE proxy; gamma ties prefer gamma=0."""
    selections = []
    grid = [round(value, 2) for value in np.arange(0.0, 1.0001, 0.05)]
    for boundary in range(len(records)):
        prefix = records[: boundary + 1]
        normalized = causal_final_weights(precisions[: boundary + 1])
        curve = []
        for gamma in grid:
            weights = gamma_weights(normalized, gamma)
            heads = []
            for space in ("patch", "image"):
                R, C = weighted_system([record["fit"][space] for record in prefix], weights)
                heads.append(solve_ridge(R, C, LAM))
            theta = np.concatenate(heads, axis=0)
            losses = [
                quadratic_mse(
                    theta, record["gamma_fused"]["R"],
                    record["gamma_fused"]["C"], record["gamma_fused"]["S"],
                    record["gamma_fused"]["n"],
                )
                for record in prefix
            ]
            curve.append({"gamma": gamma, "objective": float(np.mean(losses))})
        minimum = min(row["objective"] for row in curve)
        tolerance = 1e-12 * max(1.0, abs(minimum))
        selected = min(
            (row for row in curve if row["objective"] <= minimum + tolerance),
            key=lambda row: row["gamma"],
        )
        selections.append({"boundary": boundary, "selected": selected, "curve": curve})
    return float(selections[-1]["selected"]["gamma"]), selections


def predict_fixed_factor(records, precisions):
    """Apply the frozen image-level isotropic train-only approximation."""
    d_aug = records[0]["full"]["image"][0].shape[0]
    per_domain = []
    for record in records:
        full = record["full"]["image"]
        W = solve_ridge(full[0], full[1], LAM)
        mu = float(np.trace(full[0]) / d_aug)
        target_variance = max(float(np.var(record["precision_targets"])), 1e-3)
        feature_mass = mu / max(record["full_n_images"], 1)
        per_domain.append({
            "w": W, "mu": mu, "r": record["heldout_precision_mse"],
            "rho": feature_mass / target_variance,
            "n": int(record["precision_rows"].shape[0]),
        })
    factor, curve = predicted_factor(per_domain, d_aug, LAM, grid=ORACLE_GRID)
    return factor, curve


def f1_per_image(domains, scales):
    T = domains.n_domains()
    d_in = _first_dim(domains)
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
    for t in range(T):
        patch_head.begin_task(1.0)
        image_head.begin_task(1.0)
        for X, Y, _ in domains.stream("train", t):
            patch_head.accumulate(X, transform_patch_target(Y, PATCH_TARGET) / scales[t])
            image_head.accumulate(_image_feature(X), _image_target(Y) / scales[t])
    patch_head.solve()
    image_head.solve()
    return [
        per_image_errors(domains, i, patch_head, image_head, ALPHA, scales[i])
        for i in range(T)
    ]


def run_fixed_factor(domains, factor, scales=None):
    """Run one fixed temporal factor and retain paired per-image errors."""
    T = domains.n_domains()
    scales = scales or [domain_scale(domains, t, "mean") for t in range(T)]
    d_in = _first_dim(domains)
    patch_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
    image_head = ForgettingRidgeRLS(d_in=d_in, d_out=1, lam=LAM)
    M_abs = np.full((T, T), np.nan)
    M_rel = np.full((T, T), np.nan)
    for t in range(T):
        patch_head.begin_task(float(factor))
        image_head.begin_task(float(factor))
        for X, Y, _ in domains.stream("train", t):
            patch_head.accumulate(X, transform_patch_target(Y, PATCH_TARGET) / scales[t])
            image_head.accumulate(_image_feature(X), _image_target(Y) / scales[t])
        patch_head.solve()
        image_head.solve()
        for previous in range(t + 1):
            M_abs[t, previous], M_rel[t, previous] = eval_domain(
                domains, previous, patch_head, image_head, ALPHA, scales[previous]
            )
    errors = [
        per_image_errors(domains, i, patch_head, image_head, ALPHA, scales[i])
        for i in range(T)
    ]
    return {
        "factor": float(factor),
        "balanced_rel_mae": float(np.nanmean(M_rel[T - 1, :T])),
        "M_abs": M_abs,
        "M_rel": M_rel,
        "per_image": errors,
        "scales": scales,
    }


def _first_dim(domains):
    for X, _, _ in domains.stream("train", 0):
        return np.asarray(X).shape[1]
    raise RuntimeError("empty first domain")


def relative_domain_harms(per_domain_method, per_domain_f1, epsilon=1e-9):
    """(E_m - E_f1) / max(E_f1, eps) per domain — RELATIVE units."""
    harms = []
    for method_value, f1_value in zip(per_domain_method, per_domain_f1):
        harms.append(
            (float(method_value) - float(f1_value))
            / max(float(f1_value), epsilon)
        )
    return harms


def compute_verdict(improvement, bootstrap, harms, oracle_best_rel, rel_f1):
    """Protocol §6 verdict — OUTCOME-INDEPENDENT of the oracle.

    ``positive`` and ``safe`` are BOTH always evaluated from the final
    method vs f=1.  The fixed-f oracle tests the temporal-forgetting
    hypothesis, which is a different hypothesis from reliability weighting;
    it is recorded as a diagnostic only and never selects the standard
    (development counterexample: JHU-last had oracle == f=1 yet reliability
    weighting improved 13.7% — a verdict branch keyed on the oracle would
    have nulled a true positive, and would have done so AFTER seeing test
    data).
    """
    worst_harm_rel = float(max(harms))
    return {
        "units": "all thresholds in RELATIVE rel-MAE units",
        "improvement_rel": float(improvement),
        "min_required_rel": MIN_IMPROVEMENT,
        "bootstrap": bootstrap,
        "per_domain_relative_harm": harms,
        "worst_single_domain_harm_rel": worst_harm_rel,
        "max_domain_harm_rel": MAX_DOMAIN_HARM,
        "positive": bool(
            improvement >= MIN_IMPROVEMENT
            and bootstrap["ci95_lower_rel"] > 0
            and worst_harm_rel <= MAX_DOMAIN_HARM
        ),
        "safe": bool(
            improvement >= -0.005
            and bootstrap["ci95_lower_rel"] > SAFETY_CI_FLOOR
        ),
        "oracle_diagnostic_only": {
            "note": (
                "the fixed-f oracle addresses the temporal-forgetting "
                "hypothesis; it does NOT gate positive/safe"
            ),
            "oracle_equals_f1": bool(abs(oracle_best_rel - rel_f1) <= 1e-9),
        },
        "single_shot_note": "protocol §6: first and only main run; no re-runs",
    }


def assert_paired_records(per_image_method, per_image_f1):
    """Pairing identity: same domains, counts, image ids, order and gts."""
    if len(per_image_method) != len(per_image_f1):
        raise SystemExit(
            f"C7 PAIRING FAILURE: {len(per_image_method)} method domains vs "
            f"{len(per_image_f1)} f=1 domains"
        )
    for index, (method_records, f1_records) in enumerate(
        zip(per_image_method, per_image_f1)
    ):
        if len(method_records) != len(f1_records):
            raise SystemExit(
                f"C7 PAIRING FAILURE: domain {index} has "
                f"{len(method_records)} method images vs {len(f1_records)} f=1"
            )
        method_ids = [record[2] for record in method_records]
        f1_ids = [record[2] for record in f1_records]
        if method_ids != f1_ids:
            raise SystemExit(
                f"C7 PAIRING FAILURE: domain {index} image ids/order differ"
            )
        method_gts = np.asarray([record[1] for record in method_records])
        f1_gts = np.asarray([record[1] for record in f1_records])
        if not np.allclose(method_gts, f1_gts, rtol=0, atol=1e-9):
            raise SystemExit(
                f"C7 PAIRING FAILURE: domain {index} ground truths differ"
            )


def paired_block_bootstrap(per_image_method, per_image_f1, seed=0, B=BOOTSTRAP_B):
    """Paired sequence-group bootstrap of the RELATIVE balanced improvement.

    Each replicate resamples sequence groups per domain (paired for both
    methods), recomputes balanced rel-MAE for method and f=1, and records
    (E_f1 - E_m) / max(E_f1, eps).  Per-frame fallback is forbidden: every
    domain's test ids must yield a detected sequence structure.  Pairing
    identity is asserted before any resampling.
    """
    assert_paired_records(per_image_method, per_image_f1)
    rng = np.random.default_rng(seed)
    domains_blocks = []
    for method_records, f1_records in zip(per_image_method, per_image_f1):
        ids = [record[2] for record in method_records]
        groups = require_detected_groups(ids, "test-block bootstrap")
        index_of = {image_id: k for k, image_id in enumerate(ids)}
        blocks = []
        for group in groups:
            rows = [index_of[i] for i in group]
            blocks.append((
                sum(method_records[r][0] for r in rows),
                sum(f1_records[r][0] for r in rows),
                sum(method_records[r][1] for r in rows),
            ))
        domains_blocks.append(blocks)
    deltas = np.empty(B)
    for b in range(B):
        rels_m, rels_f = [], []
        for blocks in domains_blocks:
            pick = rng.integers(0, len(blocks), size=len(blocks))
            err_m = sum(blocks[p][0] for p in pick)
            err_f = sum(blocks[p][1] for p in pick)
            gt = max(sum(blocks[p][2] for p in pick), 1e-9)
            rels_m.append(err_m / gt)
            rels_f.append(err_f / gt)
        balanced_f = float(np.mean(rels_f))
        balanced_m = float(np.mean(rels_m))
        deltas[b] = (balanced_f - balanced_m) / max(balanced_f, 1e-9)
    return {
        "B": B,
        "units": "relative improvement (E_f1 - E_m) / E_f1",
        "mean_improvement_rel": float(np.mean(deltas)),
        "ci95_lower_rel": float(np.percentile(deltas, 2.5)),
        "ci95_upper_rel": float(np.percentile(deltas, 97.5)),
    }


def paired_oracle_block_bootstrap(factor_records, per_image_f1, seed=0, B=BOOTSTRAP_B):
    """Selection-adjusted fixed-factor opportunity bootstrap.

    Each replicate resamples sequence groups once, evaluates every frozen
    factor on that same draw, and reselects the minimum balanced rel-MAE.
    """
    if set(map(float, factor_records)) != set(map(float, ORACLE_GRID)):
        raise SystemExit("C7 ORACLE FAILURE: incomplete frozen factor grid")
    for records in factor_records.values():
        assert_paired_records(records, per_image_f1)
    rng = np.random.default_rng(seed)
    domain_groups = []
    for domain_index, f1_records in enumerate(per_image_f1):
        ids = [record[2] for record in f1_records]
        groups = require_detected_groups(ids, "oracle test-block bootstrap")
        index_of = {image_id: index for index, image_id in enumerate(ids)}
        domain_groups.append([[index_of[item] for item in group] for group in groups])
    deltas = np.empty(B)
    for b in range(B):
        rel_f1_domains = []
        rel_factor_domains = {float(factor): [] for factor in ORACLE_GRID}
        for domain_index, groups in enumerate(domain_groups):
            picks = rng.integers(0, len(groups), size=len(groups))
            rows = [row for pick in picks for row in groups[pick]]
            gt = max(sum(per_image_f1[domain_index][row][1] for row in rows), 1e-9)
            rel_f1_domains.append(
                sum(per_image_f1[domain_index][row][0] for row in rows) / gt
            )
            for factor in ORACLE_GRID:
                records = factor_records[float(factor)][domain_index]
                rel_factor_domains[float(factor)].append(
                    sum(records[row][0] for row in rows) / gt
                )
        baseline = float(np.mean(rel_f1_domains))
        best = min(float(np.mean(values)) for values in rel_factor_domains.values())
        deltas[b] = (baseline - best) / max(baseline, 1e-9)
    return {
        "B": B,
        "selection_adjusted": True,
        "selection_rule": "reselect minimum balanced rel-MAE over full grid per replicate",
        "mean_improvement_rel": float(np.mean(deltas)),
        "ci95_lower_rel": float(np.percentile(deltas, 2.5)),
        "ci95_upper_rel": float(np.percentile(deltas, 97.5)),
    }
