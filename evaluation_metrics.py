"""Task-facing regression metrics used by RRCL experiment runners."""

from __future__ import annotations

import numpy as np


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def scalar_regression_metrics(prediction, target):
    """Return MAE/RMSE plus PLCC and SRCC without a SciPy dependency."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must be non-empty matching vectors")
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "plcc": _correlation(prediction, target),
        "srcc": _correlation(_average_ranks(prediction), _average_ranks(target)),
        "n": int(target.size),
    }


def balanced_metric_mean(per_domain):
    """Average task metrics with equal domain weight."""
    output = {}
    for metric in ("mae", "rmse", "plcc", "srcc"):
        values = [row[metric] for row in per_domain if row.get(metric) is not None]
        output[metric] = float(np.mean(values)) if values else None
    output["domains"] = len(per_domain)
    output["samples"] = int(sum(row.get("n", 0) for row in per_domain))
    return output
