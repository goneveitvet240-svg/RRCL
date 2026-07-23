"""Bounded-memory adaptive forgetting from held-out sufficient statistics.

The selector stores only two accumulated quadratic forms: one for fitting and
one for balanced validation.  Its memory is therefore O(d^2), independent of
the number of domains.  Candidate factors are evaluated against the actual
per-boundary history rather than pretending that a single global factor had
been used at every earlier boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    factor: float
    objective: float | None
    objective_f1: float | None
    drift_score: float
    search_mode: str


class BalancedSufficientStatsSelector:
    """Select a per-boundary factor with O(d^2) total state.

    Validation losses are balanced by domain.  For domain j with held-out
    sufficient statistics (R_j, C_j, S_j, n_j), its MSE is

        (w' R_j w - 2 w' C_j + S_j) / n_j.

    Averaging these losses only requires the sums of R_j/n_j, C_j/n_j and
    S_j/n_j, so individual historical domains never need to be retained.
    """

    def __init__(
        self,
        dimension: int,
        lam: float,
        f_min: float = 0.05,
        f_max: float = 1.0,
        search_mode: str = "grid",
        grid: np.ndarray | None = None,
        abstain_relative_gain: float = 1e-4,
        continuous_iterations: int = 28,
    ):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 0.0 < f_min <= f_max <= 1.0:
            raise ValueError("forgetting-factor bounds must satisfy 0 < min <= max <= 1")
        if search_mode not in {"continuous", "grid"}:
            raise ValueError("search_mode must be 'continuous' or 'grid'")
        self.dimension = int(dimension)
        self.lam = float(lam)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.search_mode = search_mode
        self.grid = (
            np.asarray(grid, dtype=np.float64)
            if grid is not None
            else np.round(np.arange(f_min, f_max + 1e-12, 0.05), 6)
        )
        self.abstain_relative_gain = float(abstain_relative_gain)
        self.continuous_iterations = int(continuous_iterations)

        shape = (self.dimension, self.dimension)
        vector_shape = (self.dimension, 1)
        self.fit_R = np.zeros(shape, dtype=np.float64)
        self.fit_C = np.zeros(vector_shape, dtype=np.float64)
        self.val_R_balanced = np.zeros(shape, dtype=np.float64)
        self.val_C_balanced = np.zeros(vector_shape, dtype=np.float64)
        self.val_S_balanced = 0.0
        self.n_domains = 0

    @property
    def state_bytes(self) -> int:
        return int(
            self.fit_R.nbytes
            + self.fit_C.nbytes
            + self.val_R_balanced.nbytes
            + self.val_C_balanced.nbytes
        )

    def _candidate_objective(
        self,
        factor: float,
        fit_R: np.ndarray,
        fit_C: np.ndarray,
        val_R_balanced: np.ndarray,
        val_C_balanced: np.ndarray,
        val_S_balanced: float,
        n_domains: int,
    ) -> float:
        A = self.lam * np.eye(self.dimension) + factor * self.fit_R + fit_R
        b = factor * self.fit_C + fit_C
        weights = np.linalg.solve(A, b)
        quadratic = float(np.sum(weights * (val_R_balanced @ weights)))
        linear = float(np.sum(weights * val_C_balanced))
        return (quadratic - 2.0 * linear + val_S_balanced) / max(n_domains, 1)

    def _continuous_minimize(self, objective):
        """Golden-section refinement after a coarse grid pre-scan.

        J(f) is not guaranteed to be unimodal (unequal noise or ridge bias
        can create irregular loss surfaces), so pure golden-section from a
        fixed starting interval may converge to a local minimum.  We therefore
        run a coarse grid scan first to identify the best interval, then
        apply golden-section within that interval for smooth refinement.
        """
        # --- coarse grid pre-scan ---
        grid = np.round(np.arange(self.f_min, self.f_max + 1e-12, 0.05), 6)
        grid_vals = [(float(g), objective(float(g))) for g in grid]
        best_grid_f, best_grid_v = min(grid_vals, key=lambda t: t[1])

        # Find the grid interval [left, right] containing the best grid point
        # and apply golden-section refinement within it.
        idx = [i for i, (f, _) in enumerate(grid_vals) if f == best_grid_f][0]
        left = grid_vals[max(idx - 1, 0)][0]
        right = grid_vals[min(idx + 1, len(grid_vals) - 1)][0]
        if left >= right:
            # Only one grid point — return it directly.
            return best_grid_f, best_grid_v

        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        y1, y2 = objective(x1), objective(x2)
        for _ in range(self.continuous_iterations):
            if y1 <= y2:
                right, x2, y2 = x2, x1, y1
                x1 = right - ratio * (right - left)
                y1 = objective(x1)
            else:
                left, x1, y1 = x1, x2, y2
                x2 = left + ratio * (right - left)
                y2 = objective(x2)
        refined_mid = (left + right) / 2.0
        candidates = [
            (self.f_min, objective(self.f_min)),
            (self.f_max, objective(self.f_max)),
            (best_grid_f, best_grid_v),
            (x1, y1),
            (x2, y2),
            (refined_mid, objective(refined_mid)),
        ]
        return min(candidates, key=lambda item: item[1])

    def select_and_update(self, fit, val) -> SelectionResult:
        """Select f for the incoming domain and update the bounded selector state.

        ``fit`` and ``val`` are ``(R, C, S, n)`` lists/tuples.  ``S`` is unused
        for fit statistics but is accepted to keep a common representation.
        """
        fit_R, fit_C, _, _ = fit
        val_R, val_C, val_S, val_n = val
        fit_R = np.asarray(fit_R, dtype=np.float64)
        fit_C = np.asarray(fit_C, dtype=np.float64)
        val_R = np.asarray(val_R, dtype=np.float64)
        val_C = np.asarray(val_C, dtype=np.float64)
        if fit_R.shape != self.fit_R.shape or val_R.shape != self.fit_R.shape:
            raise ValueError("sufficient-statistic matrix has the wrong shape")
        if fit_C.shape != self.fit_C.shape or val_C.shape != self.fit_C.shape:
            raise ValueError("sufficient-statistic vector has the wrong shape")
        if int(val_n) <= 0:
            raise ValueError("held-out validation split must be non-empty")

        next_val_R = self.val_R_balanced + val_R / float(val_n)
        next_val_C = self.val_C_balanced + val_C / float(val_n)
        next_val_S = self.val_S_balanced + float(val_S) / float(val_n)
        next_n_domains = self.n_domains + 1

        if self.n_domains == 0:
            factor = 1.0
            best_objective = None
            objective_f1 = None
            drift_score = 0.0
        else:
            def objective(factor_value):
                return self._candidate_objective(
                    float(factor_value),
                    fit_R,
                    fit_C,
                    next_val_R,
                    next_val_C,
                    next_val_S,
                    next_n_domains,
                )

            objective_f1 = objective(1.0)
            if self.search_mode == "grid":
                candidates = [
                    (float(factor_value), objective(float(factor_value)))
                    for factor_value in self.grid
                    if self.f_min <= factor_value <= self.f_max
                ]
                candidates.append((1.0, objective_f1))
                factor, best_objective = min(candidates, key=lambda item: item[1])
            else:
                factor, best_objective = self._continuous_minimize(objective)

            relative_gain = max(objective_f1 - best_objective, 0.0) / max(
                abs(objective_f1), 1e-12
            )
            if relative_gain <= self.abstain_relative_gain:
                factor = 1.0
                best_objective = objective_f1
                relative_gain = 0.0
            drift_score = float(relative_gain)

        self.fit_R = factor * self.fit_R + fit_R
        self.fit_C = factor * self.fit_C + fit_C
        self.val_R_balanced = next_val_R
        self.val_C_balanced = next_val_C
        self.val_S_balanced = next_val_S
        self.n_domains = next_n_domains

        return SelectionResult(
            factor=float(factor),
            objective=None if best_objective is None else float(best_objective),
            objective_f1=None if objective_f1 is None else float(objective_f1),
            drift_score=float(drift_score),
            search_mode=self.search_mode,
        )
