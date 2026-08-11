"""Pure-numpy core for the precision-shrinkage DIAGNOSTIC (no selector).

Interpolation between absolute memory and causal precision weighting in the
*causally normalized* weight space:

    u_k          = p_k / mean(p_1..p_T)        (mean-1 normalized precisions)
    w_k(gamma)   = (1 - gamma) + gamma * u_k   (gamma in [0, 1])

Endpoints (exact, unit-tested), for WHATEVER precision vector p is supplied:
    gamma = 0  ->  all weights 1  ->  the f=1 joint model
    gamma = 1  ->  w_k = u_k      ->  causal precision weighting with THOSE
                                      precisions (boundary-by-boundary
                                      accumulation, see
                                      causal_boundary_reference)

Protocol note (Codex review 2026-07-26): the diagnostic estimates p from the
10% precision_val role of the three-way split, so its gamma=1 is the
THREE-WAY PRECISION ENDPOINT.  The production raw-precision method
(scripts/run_precision_weighted.py) estimates p on the full 20% two-way
validation set — a DIFFERENT precision vector, hence a different model.  The
two are separate comparators; only the weighted-accumulation machinery is
shared (unit-tested against production with injected identical precisions).

The interpolation preserves mean weight 1 for every gamma, so the ridge
parameter keeps its usual scale.

This module deliberately contains NO data loading and NO selection rule: it
answers whether a train-only objective over gamma can predict the test-optimal
gamma (representativeness), before any shrinkage selector is designed.
"""

from __future__ import annotations

import numpy as np


def causal_final_weights(precisions):
    """Normalized final-boundary weights u_k = p_k / mean(p).

    For a GIVEN precision vector, accumulating p_k * R_k and dividing by the
    running mean gives, at the last boundary, sum_k (p_k / mean(p)) R_k —
    the same formula run_precision_weighted.py applies to ITS OWN (two-way,
    20%) precision estimates.  Which endpoint this represents therefore
    depends entirely on where ``precisions`` came from; see the module
    docstring.  Order-invariant by construction.
    """
    p = np.asarray(precisions, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("precisions must be non-empty")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError(f"precisions must be finite and positive, got {p}")
    return p / p.mean()


def gamma_weights(normalized_weights, gamma):
    """Interpolated weights (1-gamma) + gamma*u in causal-normalized space."""
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma={gamma} outside [0, 1]")
    u = np.asarray(normalized_weights, dtype=np.float64).reshape(-1)
    return (1.0 - gamma) + gamma * u


def solve_ridge(R, C, lam):
    dim = R.shape[0]
    return np.linalg.solve(R + float(lam) * np.eye(dim), C)


def weighted_system(stats, weights):
    """Sum_k w_k (R_k, C_k) over per-domain (R, C) pairs."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(stats) != weights.size:
        raise ValueError(f"{len(stats)} stat blocks vs {weights.size} weights")
    R = np.zeros_like(stats[0][0])
    C = np.zeros_like(stats[0][1])
    for (Rk, Ck), w in zip(stats, weights):
        R += w * Rk
        C += w * Ck
    return R, C


def quadratic_mse(W, R, C, S, n):
    """Mean squared error on held-out data from its sufficient statistics."""
    value = (
        float(np.sum(W * (R @ W)))
        - 2.0 * float(np.sum(W * C))
        + float(S)
    )
    return value / max(int(n), 1)


def causal_boundary_reference(full_stats, precisions, lam):
    """Boundary-by-boundary replication of run_precision_weighted.py.

    ``full_stats``: list of per-domain (R, C).  Returns the FINAL solved W.
    Used to certify that gamma=1 of the sweep equals raw causal precision.
    """
    p = np.asarray(precisions, dtype=np.float64).reshape(-1)
    P_R = np.zeros_like(full_stats[0][0])
    P_C = np.zeros_like(full_stats[0][1])
    sum_p = 0.0
    W = None
    for t, (Rk, Ck) in enumerate(full_stats):
        P_R += p[t] * Rk
        P_C += p[t] * Ck
        sum_p += p[t]
        bar_p = sum_p / (t + 1)
        W = solve_ridge(P_R / max(bar_p, 1e-12), P_C / max(bar_p, 1e-12), lam)
    return W


def gamma_sweep(domain_stats, precisions, gammas, lam):
    """Evaluate deployed models and train-only PROXY objectives on a gamma grid.

    ``domain_stats``: list of per-domain dicts with keys
        full: {space: (R, C)}   -- full-domain stats  (deployed model)
        fit:  {space: (R, C)}   -- fit-partition stats (candidate model)
        val:  {space: {R, C, S, n}} -- held-out training quadratics
    for spaces "patch" and "image".  When a domain additionally provides
    ``val["fused"]`` (quadratics of the fused design row
    ``z = [alpha*sum_p x_p', alpha*P, (1-alpha)*xbar', 1-alpha]`` against the
    image target), the stacked candidate ``theta = [W_patch; W_image]`` is
    evaluated on it as the closest train-only proxy of the deployed fused
    prediction.

    IMPORTANT: every objective here is an UNCLIPPED quadratic (normalized MSE)
    proxy.  The deployed metric is non-negative-clipped fused relative MAE,
    which cannot be computed exactly from O(d^2) second-order statistics.
    Returns per-gamma records; test evaluation of the deployed model is the
    runner's job.
    """
    u = causal_final_weights(precisions)
    has_fused = all("fused" in d.get("val", {}) for d in domain_stats)
    records = []
    for gamma in gammas:
        w = gamma_weights(u, gamma)
        record = {"gamma": float(gamma), "weights": w.tolist()}
        for space in ("patch", "image"):
            R_dep, C_dep = weighted_system(
                [d["full"][space] for d in domain_stats], w
            )
            W_dep = solve_ridge(R_dep, C_dep, lam)
            R_cand, C_cand = weighted_system(
                [d["fit"][space] for d in domain_stats], w
            )
            W_cand = solve_ridge(R_cand, C_cand, lam)
            per_domain = [
                quadratic_mse(
                    W_cand,
                    d["val"][space]["R"],
                    d["val"][space]["C"],
                    d["val"][space]["S"],
                    d["val"][space]["n"],
                )
                for d in domain_stats
            ]
            record[space] = {
                "W_deployed": W_dep,
                "W_candidate": W_cand,
                "val_per_domain": per_domain,
                "val_balanced": float(np.mean(per_domain)),
            }
        if has_fused:
            theta = np.concatenate(
                [record["patch"]["W_candidate"], record["image"]["W_candidate"]],
                axis=0,
            )
            per_domain = [
                quadratic_mse(
                    theta,
                    d["val"]["fused"]["R"],
                    d["val"]["fused"]["C"],
                    d["val"]["fused"]["S"],
                    d["val"]["fused"]["n"],
                )
                for d in domain_stats
            ]
            record["fused"] = {
                "val_per_domain": per_domain,
                "val_balanced": float(np.mean(per_domain)),
            }
        records.append(record)
    return records


def grid_argmin(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("objective curve must be non-empty and finite")
    return int(np.argmin(values))


def sign_agreement(val_delta, test_delta, tolerance=1e-12):
    """Compare the SIGN of a validation delta against a test delta.

    Deltas are (value_at_gamma - value_at_gamma0); negative means improvement
    for both losses.  Returns one of "agree", "disagree", "val_flat",
    "test_flat", "both_flat".
    """
    def _sign(x):
        x = float(x)
        if abs(x) <= tolerance:
            return 0
        return 1 if x > 0 else -1

    sv, st = _sign(val_delta), _sign(test_delta)
    if sv == 0 and st == 0:
        return "both_flat"
    if sv == 0:
        return "val_flat"
    if st == 0:
        return "test_flat"
    return "agree" if sv == st else "disagree"


def ensure_clean_worktree(allow_dirty=False, git_status_provider=None):
    """Refuse to run a paper-facing diagnostic from a dirty tree.

    ``git_status_provider`` returns the ``git status --porcelain`` text (or
    None when git is unavailable); injectable for unit tests.
    """
    if git_status_provider is None:
        from run_provenance import _git_output

        def git_status_provider():
            return _git_output(["status", "--porcelain"])

    status = git_status_provider()
    dirty = bool(status) if status is not None else None
    if dirty and not allow_dirty:
        raise SystemExit(
            "refusing to run: git worktree is dirty "
            "(commit or stash first, or pass --allow-dirty for scratch runs)"
        )
    return {"git_dirty": dirty, "allow_dirty": bool(allow_dirty)}
