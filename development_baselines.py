"""Pure NumPy primitives for the RRCL new-method baseline audit.

The module deliberately contains no dataset loader and no test-set model
selection.  It implements the three shared analytic objectives frozen in
``configs/new_method_development_v1.json`` and train-side ridge selection on
held-out, image-level fused quadratic statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHOD_POOLED = "pooled_f1"
METHOD_DOMAIN_BALANCED = "domain_balanced"
METHOD_DOMAIN_BALANCED_MASS_MATCHED = "domain_balanced_mass_matched"
SHARED_METHODS = (
    METHOD_POOLED,
    METHOD_DOMAIN_BALANCED,
    METHOD_DOMAIN_BALANCED_MASS_MATCHED,
)


@dataclass(frozen=True)
class RidgeSelection:
    """Selected dual-head ridge solution and its complete validation trace."""

    lam: float
    patch_weights: np.ndarray
    image_weights: np.ndarray
    validation_risk: float
    candidates: tuple[dict, ...]


@dataclass(frozen=True)
class ScalarRidgeSelection:
    """Selected scalar-head ridge solution and validation trace."""

    lam: float
    weights: np.ndarray
    validation_risk: float
    candidates: tuple[dict, ...]


def empty_stats(dim: int, out_dim: int = 1) -> dict:
    dim = int(dim)
    out_dim = int(out_dim)
    if dim <= 0 or out_dim <= 0:
        raise ValueError("dim and out_dim must be positive")
    return {
        "R": np.zeros((dim, dim), dtype=np.float64),
        "C": np.zeros((dim, out_dim), dtype=np.float64),
        "S": 0.0,
        "n": 0,
    }


def accumulate_stats(stats: dict, rows, targets) -> dict:
    rows = np.asarray(rows, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("rows must be a two-dimensional matrix")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != rows.shape[0]:
        raise ValueError("targets must have one row per design row")
    if stats["R"].shape != (rows.shape[1], rows.shape[1]):
        raise ValueError("row dimension does not match R")
    if stats["C"].shape != (rows.shape[1], targets.shape[1]):
        raise ValueError("target dimension does not match C")
    stats["R"] += rows.T @ rows
    stats["C"] += rows.T @ targets
    stats["S"] += float(np.sum(targets * targets))
    stats["n"] += int(rows.shape[0])
    return stats


def objective_weights(observation_counts, method: str) -> np.ndarray:
    """Return per-domain multipliers for one analytic objective.

    ``domain_balanced`` gives each domain total weight ``1 / T``, exactly
    matching the average-domain objective including its leading ``1 / T``.
    The mass-matched variant gives each domain total weight ``mean(n)`` so the
    sum of observation weights equals that of pooled training.  This controls
    only global mass; it does not make covariance spectra identical.
    """

    counts = np.asarray(observation_counts, dtype=np.float64).reshape(-1)
    if counts.size == 0 or not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError(f"observation counts must be finite and positive: {counts}")
    if method == METHOD_POOLED:
        return np.ones_like(counts)
    if method == METHOD_DOMAIN_BALANCED:
        return 1.0 / (counts.size * counts)
    if method == METHOD_DOMAIN_BALANCED_MASS_MATCHED:
        return counts.mean() / counts
    raise ValueError(f"unknown shared objective: {method}")


def weighted_sufficient_statistics(blocks, weights) -> tuple[np.ndarray, np.ndarray]:
    blocks = list(blocks)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not blocks or len(blocks) != weights.size:
        raise ValueError("blocks and weights must be non-empty and have equal length")
    R = np.zeros_like(np.asarray(blocks[0]["R"], dtype=np.float64))
    C = np.zeros_like(np.asarray(blocks[0]["C"], dtype=np.float64))
    for block, weight in zip(blocks, weights):
        R += float(weight) * np.asarray(block["R"], dtype=np.float64)
        C += float(weight) * np.asarray(block["C"], dtype=np.float64)
    return R, C


def solve_ridge(R, C, lam: float) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    lam = float(lam)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square")
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2 or C.shape[0] != R.shape[0]:
        raise ValueError("C must have the same leading dimension as R")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lambda must be finite and positive")
    return np.linalg.solve(R + lam * np.eye(R.shape[0]), C)


def ridge_path(R, C, lambdas) -> dict[float, np.ndarray]:
    """Solve a positive ridge grid with one symmetric eigendecomposition."""

    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    grid = sorted({float(value) for value in lambdas})
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("lambda grid must contain finite positive values")
    symmetric = 0.5 * (R + R.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-10
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("R is not positive semidefinite within numerical tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated = eigenvectors.T @ C
    return {
        lam: eigenvectors @ (rotated / (eigenvalues[:, None] + lam))
        for lam in grid
    }


def quadratic_mean_loss(weights, stats: dict) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    n = int(stats["n"])
    if n <= 0:
        raise ValueError("quadratic validation block must be non-empty")
    value = (
        float(np.sum(weights * (stats["R"] @ weights)))
        - 2.0 * float(np.sum(weights * stats["C"]))
        + float(stats["S"])
    )
    # Roundoff can create a tiny negative value for an exact fit.
    return max(value / n, 0.0)


def balanced_validation_risk(patch_weights, image_weights, validation_blocks) -> float:
    theta = np.concatenate(
        [
            np.asarray(patch_weights, dtype=np.float64).reshape(-1, 1),
            np.asarray(image_weights, dtype=np.float64).reshape(-1, 1),
        ],
        axis=0,
    )
    blocks = list(validation_blocks)
    if not blocks:
        raise ValueError("at least one validation block is required")
    return float(np.mean([quadratic_mean_loss(theta, block) for block in blocks]))


def fit_shared_heads(domain_records, method: str, lam: float, role: str):
    """Fit a dual shared head from per-domain sufficient statistics."""

    records = list(domain_records)
    if role not in {"fit", "full"}:
        raise ValueError("role must be 'fit' or 'full'")
    patch_blocks = [record[role]["patch"] for record in records]
    image_blocks = [record[role]["image"] for record in records]
    patch_weights = objective_weights([block["n"] for block in patch_blocks], method)
    image_weights = objective_weights([block["n"] for block in image_blocks], method)
    patch_R, patch_C = weighted_sufficient_statistics(patch_blocks, patch_weights)
    image_R, image_C = weighted_sufficient_statistics(image_blocks, image_weights)
    return {
        "patch_W": solve_ridge(patch_R, patch_C, lam),
        "image_W": solve_ridge(image_R, image_C, lam),
        "patch_R": patch_R,
        "image_R": image_R,
        "patch_domain_weights": patch_weights,
        "image_domain_weights": image_weights,
    }


def fit_scalar_head(domain_records, method: str, lam: float, role: str):
    """Fit one shared scalar head from per-domain sufficient statistics."""

    records = list(domain_records)
    if role not in {"fit", "full"}:
        raise ValueError("role must be 'fit' or 'full'")
    blocks = [record[role] for record in records]
    domain_weights = objective_weights([block["n"] for block in blocks], method)
    R, C = weighted_sufficient_statistics(blocks, domain_weights)
    return {
        "W": solve_ridge(R, C, lam),
        "R": R,
        "domain_weights": domain_weights,
    }


def select_scalar_lambda(domain_records, method: str, lambdas) -> ScalarRidgeSelection:
    """Select scalar-head lambda by balanced domain-mean validation MSE."""

    records = list(domain_records)
    grid = sorted({float(value) for value in lambdas})
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("lambda grid must contain finite positive values")
    blocks = [record["fit"] for record in records]
    domain_weights = objective_weights([block["n"] for block in blocks], method)
    R, C = weighted_sufficient_statistics(blocks, domain_weights)
    path = ridge_path(R, C, grid)
    trace = []
    for lam in grid:
        risk = float(
            np.mean(
                [quadratic_mean_loss(path[lam], record["val"]) for record in records]
            )
        )
        trace.append({"lambda": lam, "validation_risk": risk})
    best = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
    return ScalarRidgeSelection(
        lam=float(best["lambda"]),
        weights=path[best["lambda"]],
        validation_risk=float(best["validation_risk"]),
        candidates=tuple(trace),
    )


def select_shared_lambda(domain_records, method: str, lambdas) -> RidgeSelection:
    """Select one common dual-head lambda using validation data only.

    Candidate parameters use only each domain's ``fit`` statistics.  The
    validation objective is the balanced mean across seen domains.  Exact
    numerical ties choose the larger lambda, as frozen in protocol v1.
    """

    records = list(domain_records)
    lambdas = sorted({float(value) for value in lambdas})
    if not lambdas or any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("lambda grid must contain finite positive values")
    patch_blocks = [record["fit"]["patch"] for record in records]
    image_blocks = [record["fit"]["image"] for record in records]
    patch_domain_weights = objective_weights(
        [block["n"] for block in patch_blocks], method
    )
    image_domain_weights = objective_weights(
        [block["n"] for block in image_blocks], method
    )
    patch_R, patch_C = weighted_sufficient_statistics(
        patch_blocks, patch_domain_weights
    )
    image_R, image_C = weighted_sufficient_statistics(
        image_blocks, image_domain_weights
    )
    patch_path = ridge_path(patch_R, patch_C, lambdas)
    image_path = ridge_path(image_R, image_C, lambdas)
    trace = []
    for lam in lambdas:
        risk = balanced_validation_risk(
            patch_path[lam],
            image_path[lam],
            [record["val"]["fused"] for record in records],
        )
        trace.append({"lambda": lam, "validation_risk": risk})
    best = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
    return RidgeSelection(
        lam=float(best["lambda"]),
        patch_weights=patch_path[best["lambda"]],
        image_weights=image_path[best["lambda"]],
        validation_risk=float(best["validation_risk"]),
        candidates=tuple(trace),
    )


def condition_number(R, lam: float) -> float:
    R = np.asarray(R, dtype=np.float64)
    return float(np.linalg.cond(R + float(lam) * np.eye(R.shape[0])))


def state_cost_bytes(
    dim: int,
    *,
    heads: int = 2,
    trajectories: int = 1,
    domains: int = 1,
    bytes_per_float: int = 8,
) -> dict:
    """Separate mutable online statistics from inference-only parameters."""

    dim = int(dim)
    heads = int(heads)
    trajectories = int(trajectories)
    domains = int(domains)
    if min(dim, heads, trajectories, domains, bytes_per_float) <= 0:
        raise ValueError("all cost dimensions must be positive")
    one_stats = (dim * dim + dim) * int(bytes_per_float)  # R and C
    one_weights = dim * int(bytes_per_float)
    return {
        "online_update_state_bytes": heads * trajectories * domains * one_stats,
        "deployment_weights_bytes": heads * trajectories * domains * one_weights,
        "assumptions": {
            "dtype": f"float{8 * int(bytes_per_float)}",
            "scalar_output_per_head": True,
            "projection_matrix_excluded": True,
        },
    }


def relative_gain(reference_loss: float, candidate_loss: float) -> float:
    reference_loss = float(reference_loss)
    candidate_loss = float(candidate_loss)
    return (reference_loss - candidate_loss) / max(abs(reference_loss), 1e-12)


def baseline_gain_decomposition(final_losses: dict) -> dict:
    """Attribute batch-one gains without inventing later selector evidence."""

    required = {
        "pooled_reference_lambda",
        "pooled_tuned_lambda",
        "domain_balanced_tuned",
        "mass_matched_domain_balanced_tuned",
        "single_domain_tuned",
    }
    missing = sorted(required - set(final_losses))
    if missing:
        raise ValueError(f"missing final losses for decomposition: {missing}")
    base = float(final_losses["pooled_reference_lambda"])
    tuned = float(final_losses["pooled_tuned_lambda"])
    balanced = float(final_losses["domain_balanced_tuned"])
    mass_matched = float(final_losses["mass_matched_domain_balanced_tuned"])
    single = float(final_losses["single_domain_tuned"])
    best_simple = min(tuned, balanced, mass_matched)
    return {
        "lower_is_better": True,
        "losses": {key: float(value) for key, value in final_losses.items()},
        "regularization_gain_vs_reference": relative_gain(base, tuned),
        "canonical_domain_balance_gain_vs_tuned_pooled": relative_gain(tuned, balanced),
        "mass_matched_balance_gain_vs_tuned_pooled": relative_gain(tuned, mass_matched),
        "best_simple_shared_gain_vs_reference": relative_gain(base, best_simple),
        "shared_constraint_gap_best_simple_minus_single_domain": (
            best_simple - single
        ) / max(abs(best_simple), 1e-12),
        "candidate_family_gap": {"status": "not_measured_in_batch1"},
        "selection_gap": {"status": "not_measured_in_batch1"},
        "deployment_adoption_gap": {"status": "not_measured_in_batch1"},
    }
