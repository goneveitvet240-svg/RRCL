"""Leakage-safe grouped cross-validation primitives for KADID development.

The split unit is one pristine reference identity shared by every distortion
family. Fold assignment never uses targets, stream order, or a domain name.
"""

from __future__ import annotations

import hashlib

import numpy as np


FOLD_ALGORITHM = "sha256-rank-round-robin-v1"


def assign_group_folds(group_ids, sample_key: str, seed: int, folds: int) -> dict:
    """Assign unique group ids to balanced deterministic folds."""

    groups = sorted({str(group) for group in group_ids})
    folds = int(folds)
    if not sample_key:
        raise ValueError("sample_key must be non-empty")
    if folds < 2 or len(groups) < folds:
        raise ValueError("fold count must be >=2 and no larger than group count")

    def rank_key(group):
        token = f"{sample_key}|{group}|{int(seed)}".encode("utf-8")
        return hashlib.sha256(token).digest(), group

    ranked = sorted(groups, key=rank_key)
    return {group: index % folds for index, group in enumerate(ranked)}


def fold_manifest(assignments: dict, sample_key: str, seed: int, folds: int) -> dict:
    assignments = {str(group): int(fold) for group, fold in assignments.items()}
    rows = sorted(f"{group}\t{fold}" for group, fold in assignments.items())
    counts = [sum(value == fold for value in assignments.values()) for fold in range(folds)]
    return {
        "algorithm": FOLD_ALGORITHM,
        "sample_key": str(sample_key),
        "seed": int(seed),
        "folds": int(folds),
        "group_count": len(assignments),
        "fold_counts": counts,
        "assignments_sha256": hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(),
        "assignments": [
            {"group_id": group, "fold": assignments[group]}
            for group in sorted(assignments)
        ],
    }


def paired_group_bootstrap(
    baseline_losses: dict,
    candidate_losses: dict,
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> dict:
    """Empirical paired bootstrap over reference-group mean losses.

    Positive differences mean the candidate has lower loss. This is an
    post-selection empirical interval, not a selection-adjusted or
    distribution-free coverage guarantee.
    """

    common = sorted(set(baseline_losses) & set(candidate_losses))
    if len(common) < 2:
        raise ValueError("paired bootstrap requires at least two common groups")
    resamples = int(resamples)
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    differences = np.asarray(
        [float(baseline_losses[group]) - float(candidate_losses[group]) for group in common],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    draw_indices = rng.integers(0, len(common), size=(resamples, len(common)))
    draws = differences[draw_indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "unit": "pristine_reference_group",
        "estimand": "group_balanced_oof_normalized_mse_difference",
        "common_groups": len(common),
        "point_mean_baseline_minus_candidate": float(differences.mean()),
        "confidence": float(confidence),
        "interval": [
            float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha)),
        ],
        "candidate_better_group_fraction": float(np.mean(differences > 0.0)),
        "seed": int(seed),
        "resamples": resamples,
        "qualification": (
            "post-selection empirical paired bootstrap; not selection-adjusted "
            "and not a formal coverage guarantee"
        ),
    }
