#!/usr/bin/env python3
"""Read-only diagnostic: patch-level vs fused image-level selector objective.

The deployed selector accumulates sufficient statistics on PATCH tokens and
patch-level targets, but the reported metric is an image-level relative MAE of
the FUSED prediction

    y_hat = alpha * sum_p (w_p' x_p + b_p) + (1 - alpha) * (w_i' xbar + b_i).

That prediction is linear in the stacked parameter theta = [w_p; b_p; w_i; b_i]
with the design row

    z = [ alpha * sum_p x_p' , alpha * P , (1-alpha) * xbar' , (1-alpha) ],

so the fused validation loss is exactly a quadratic form in theta and can be
accumulated with O((2d)^2) state.  This script evaluates BOTH objective curves
on the SAME held-out training split and the SAME factor grid, then reports each
curve's argmin per boundary.  It changes no selector behaviour and writes no
result artifact consumed by the release validator.

Interpretation:
  * If the fused argmin is 1.0 on the QNRF negative controls while the patch
    argmin is < 1.0, the false positives are caused by the objective/metric
    mismatch and a fused selector should fix them.
  * If both curves agree, the mismatch is NOT the cause and the failure lies in
    the held-out split itself (train-split validation not representing test).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_real import RealCountingDomains, build_from_config
from holdout_split import domain_sample_key, is_validation, iter_stream_with_ids
from run_real_image_aux import _image_target, transform_patch_target
from run_real_norm_ablation import domain_scale


def _aug(mat):
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat, np.ones((mat.shape[0], 1))], axis=1)


def _zero_stats(dim):
    return {
        "R": np.zeros((dim, dim)),
        "C": np.zeros((dim, 1)),
        "S": 0.0,
        "n": 0,
    }


def _accumulate(stats, rows, targets):
    stats["R"] += rows.T @ rows
    stats["C"] += rows.T @ targets
    stats["S"] += float(np.sum(targets * targets))
    stats["n"] += rows.shape[0]


def collect(domains, index, scale, alpha, d_aug, patch_target, val_every=5,
            split_seed=42):
    """Return (fit, val) stats for the patch head, image head and fused row."""
    fit = {
        "patch": _zero_stats(d_aug),
        "image": _zero_stats(d_aug),
        "fused": _zero_stats(2 * d_aug),
    }
    val = {
        "patch": _zero_stats(d_aug),
        "image": _zero_stats(d_aug),
        "fused": _zero_stats(2 * d_aug),
    }
    sample_key = domain_sample_key(domains, index)
    for X, Y, _, image_id, _source in iter_stream_with_ids(domains, "train", index):
        X = np.asarray(X, dtype=np.float64)
        n_patches = X.shape[0]
        target = val if is_validation(sample_key, image_id, split_seed, val_every) else fit

        patch_rows = _aug(X)
        patch_targets = transform_patch_target(Y, patch_target) / scale
        _accumulate(target["patch"], patch_rows, patch_targets)

        mean_token = X.mean(axis=0, keepdims=True)
        image_rows = _aug(mean_token)
        image_targets = _image_target(Y) / scale
        _accumulate(target["image"], image_rows, image_targets)

        patch_block = np.concatenate([X.sum(axis=0), [float(n_patches)]])
        image_block = np.concatenate([mean_token[0], [1.0]])
        fused_row = np.concatenate(
            [alpha * patch_block, (1.0 - alpha) * image_block]
        )[None, :]
        _accumulate(target["fused"], fused_row, image_targets)
    return fit, val


def solve(lam, dim, prev_R, prev_C, cur_R, cur_C, factor):
    A = lam * np.eye(dim) + factor * prev_R + cur_R
    b = factor * prev_C + cur_C
    return np.linalg.solve(A, b)


def quadratic(weights, stats, divisor):
    value = (
        float(np.sum(weights * (stats["R"] @ weights)))
        - 2.0 * float(np.sum(weights * stats["C"]))
        + stats["S"]
    )
    return value / max(divisor, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--lam", type=float, default=1e2)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--patch-target", default="dct5")
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="independent seed for the stable-hash fit/val split")
    parser.add_argument("--val-every", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = json.load(handle)
    domains = RealCountingDomains(
        build_from_config(cfg),
        backbone=args.backbone,
        img_size=args.img_size,
        max_per_domain=args.max_per_domain,
        sample_seed=args.sample_seed,
    )
    n_domains = domains.n_domains()
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 3)

    d_in = None
    for X, _, _ in domains.stream("train", 0):
        d_in = np.asarray(X).shape[1]
        break
    d_aug = d_in + 1

    fit_state = {
        "patch": _zero_stats(d_aug),
        "image": _zero_stats(d_aug),
    }
    val_balanced = {
        "patch": _zero_stats(d_aug),
        "image": _zero_stats(d_aug),
        "fused": _zero_stats(2 * d_aug),
    }
    seen = 0

    print(f"config={args.config}  seed={args.sample_seed}  alpha={args.alpha}")
    print(f"{'boundary':<12}{'patch argmin':>14}{'fused argmin':>14}"
          f"{'patch G_k':>12}{'fused G_k':>12}")

    for t in range(n_domains):
        name = domains.domains[t].name
        scale = domain_scale(domains, t, "mean")
        fit, val = collect(domains, t, scale, args.alpha, d_aug, args.patch_target,
                           args.val_every, split_seed=args.split_seed)

        if seen == 0:
            for key in ("patch", "image"):
                fit_state[key]["R"] += fit[key]["R"]
                fit_state[key]["C"] += fit[key]["C"]
            for key in ("patch", "image", "fused"):
                val_balanced[key]["R"] += val[key]["R"] / max(val[key]["n"], 1)
                val_balanced[key]["C"] += val[key]["C"] / max(val[key]["n"], 1)
                val_balanced[key]["S"] += val[key]["S"] / max(val[key]["n"], 1)
            seen = 1
            print(f"{name:<12}{'--':>14}{'--':>14}{'--':>12}{'--':>12}")
            continue

        next_val = {}
        for key in ("patch", "image", "fused"):
            next_val[key] = {
                "R": val_balanced[key]["R"] + val[key]["R"] / max(val[key]["n"], 1),
                "C": val_balanced[key]["C"] + val[key]["C"] / max(val[key]["n"], 1),
                "S": val_balanced[key]["S"] + val[key]["S"] / max(val[key]["n"], 1),
            }
        n_seen = seen + 1

        patch_curve, fused_curve = [], []
        for factor in grid:
            w_patch = solve(
                args.lam, d_aug,
                fit_state["patch"]["R"], fit_state["patch"]["C"],
                fit["patch"]["R"], fit["patch"]["C"], float(factor),
            )
            w_image = solve(
                args.lam, d_aug,
                fit_state["image"]["R"], fit_state["image"]["C"],
                fit["image"]["R"], fit["image"]["C"], float(factor),
            )
            theta = np.concatenate([w_patch, w_image], axis=0)
            patch_curve.append(quadratic(w_patch, next_val["patch"], n_seen))
            fused_curve.append(quadratic(theta, next_val["fused"], n_seen))

        patch_curve = np.asarray(patch_curve)
        fused_curve = np.asarray(fused_curve)
        one = int(np.argmin(np.abs(grid - 1.0)))
        patch_best = int(np.argmin(patch_curve))
        fused_best = int(np.argmin(fused_curve))
        patch_gain = (patch_curve[one] - patch_curve[patch_best]) / max(
            abs(patch_curve[one]), 1e-12
        )
        fused_gain = (fused_curve[one] - fused_curve[fused_best]) / max(
            abs(fused_curve[one]), 1e-12
        )
        print(f"{name:<12}{grid[patch_best]:>14.2f}{grid[fused_best]:>14.2f}"
              f"{patch_gain:>12.5f}{fused_gain:>12.5f}")

        chosen = float(grid[patch_best]) if patch_gain > 1e-4 else 1.0
        for key in ("patch", "image"):
            fit_state[key]["R"] = chosen * fit_state[key]["R"] + fit[key]["R"]
            fit_state[key]["C"] = chosen * fit_state[key]["C"] + fit[key]["C"]
        for key in ("patch", "image", "fused"):
            val_balanced[key] = {
                "R": next_val[key]["R"],
                "C": next_val[key]["C"],
                "S": next_val[key]["S"],
                "n": 0,
            }
        seen = n_seen


if __name__ == "__main__":
    main()
