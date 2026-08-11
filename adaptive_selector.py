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
    historical_relative_harm: float
    current_relative_harm: float
    safety_abstained: bool


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
        max_component_relative_harm: float | None = None,
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
        self.max_component_relative_harm = (
            None
            if max_component_relative_harm is None
            else float(max_component_relative_harm)
        )
        if (
            self.max_component_relative_harm is not None
            and self.max_component_relative_harm < 0.0
        ):
            raise ValueError("max_component_relative_harm must be non-negative")

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

    def _validation_objective(self, weights, R, C, S, divisor=1.0):
        quadratic = float(np.sum(weights * (R @ weights)))
        linear = float(np.sum(weights * C))
        return (quadratic - 2.0 * linear + float(S)) / max(float(divisor), 1.0)

    def _candidate_weights(self, factor, fit_R, fit_C):
        matrix = self.lam * np.eye(self.dimension) + factor * self.fit_R + fit_R
        vector = factor * self.fit_C + fit_C
        return np.linalg.solve(matrix, vector)

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
            historical_relative_harm = 0.0
            current_relative_harm = 0.0
            safety_abstained = False
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
            candidate_weights = self._candidate_weights(factor, fit_R, fit_C)
            absolute_weights = self._candidate_weights(1.0, fit_R, fit_C)

            historical_relative_harm = 0.0
            if self.n_domains:
                historical_candidate = self._validation_objective(
                    candidate_weights,
                    self.val_R_balanced,
                    self.val_C_balanced,
                    self.val_S_balanced,
                    self.n_domains,
                )
                historical_absolute = self._validation_objective(
                    absolute_weights,
                    self.val_R_balanced,
                    self.val_C_balanced,
                    self.val_S_balanced,
                    self.n_domains,
                )
                historical_relative_harm = (
                    historical_candidate - historical_absolute
                ) / max(abs(historical_absolute), 1e-12)

            current_candidate = self._validation_objective(
                candidate_weights, val_R, val_C, val_S, val_n
            )
            current_absolute = self._validation_objective(
                absolute_weights, val_R, val_C, val_S, val_n
            )
            current_relative_harm = (
                current_candidate - current_absolute
            ) / max(abs(current_absolute), 1e-12)
            safety_abstained = False

            if relative_gain <= self.abstain_relative_gain:
                factor = 1.0
                best_objective = objective_f1
                relative_gain = 0.0
            elif self.max_component_relative_harm is not None and max(
                historical_relative_harm, current_relative_harm
            ) > self.max_component_relative_harm + 1e-12:
                factor = 1.0
                best_objective = objective_f1
                relative_gain = 0.0
                safety_abstained = True
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
            historical_relative_harm=float(historical_relative_harm),
            current_relative_harm=float(current_relative_harm),
            safety_abstained=bool(safety_abstained),
        )


# ---------------------------------------------------------------------------
# Per-Domain Statistics Selector (Case C — O(T·d²) storage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerDomainSelectionResult:
    factor: float
    objective: float | None
    objective_f1: float | None
    drift_score: float
    search_mode: str
    per_domain_validation: list[float]  # J_j(f_selected) for each seen domain
    per_domain_validation_f1: list[float]  # J_j(f=1) for each seen domain
    safety_abstained: bool


class PerDomainStatsSelector:
    """Select one scalar f per boundary with O(T·d²) total state.

    This is a DIAGNOSTIC selector.  It maintains the causal accumulated
    training state (R_actual, C_actual) and stores per-domain VALIDATION
    statistics.  The selector matches the causal factor recursion of the
    deployed head.

    No per-domain fit matrices are retained.  Storage comes from per-domain
    validation matrices: O(T·d²).
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
        max_per_domain_degradation: float | None = None,
    ):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 0.0 < f_min <= f_max <= 1.0:
            raise ValueError("forgetting-factor bounds must satisfy 0 < min <= max <= 1")
        if search_mode not in {"continuous", "grid"}:
            raise ValueError("search_mode must be 'continuous' or 'grid'")
        if search_mode == "continuous":
            raise NotImplementedError(
                "PerDomainStatsSelector does not support search_mode='continuous'; "
                "use 'grid' (the paper default)."
            )
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
        self.max_per_domain_degradation = (
            None
            if max_per_domain_degradation is None
            else float(max_per_domain_degradation)
        )
        if (
            self.max_per_domain_degradation is not None
            and self.max_per_domain_degradation < 0.0
        ):
            raise ValueError("max_per_domain_degradation must be non-negative")

        shape = (self.dimension, self.dimension)
        vec_shape = (self.dimension, 1)

        # --- accumulated *actual* training state (causal recursion) ---
        self._R_actual = np.zeros(shape, dtype=np.float64)
        self._C_actual = np.zeros(vec_shape, dtype=np.float64)

        # --- per-domain validation statistics (already divided by n_j_val) ---
        # list of (R_j_val / n_j_val, C_j_val / n_j_val, S_j_val / n_j_val)
        self._val_R: list[np.ndarray] = []
        self._val_C: list[np.ndarray] = []
        self._val_S: list[float] = []

        # --- standalone counter (not derived from list length) ---
        self._n_domains: int = 0

    @property
    def n_domains(self) -> int:
        return self._n_domains

    @property
    def state_bytes(self) -> int:
        total = (
            self._R_actual.nbytes
            + self._C_actual.nbytes
        )
        for R in self._val_R:
            total += R.nbytes
        for C in self._val_C:
            total += C.nbytes
        return total

    # ------------------------------------------------------------------
    # Weight computation — CAUSAL recursion, NOT historical re-derivation
    # ------------------------------------------------------------------

    def _effective_system(self, factor: float, new_fit_R, new_fit_C):
        """Build R_eff, C_eff for a candidate scalar f using the ACTUAL
        accumulated state, matching the deployed head's causal recursion.

            R_k(f) = f · R_actual  +  R_new^{fit}
            C_k(f) = f · C_actual  +  C_new^{fit}

        This replaces the old incorrect ``Σ f^{τ_j} · R_j`` derivation
        which re-weighted historical domains by powers of f regardless
        of what had actually been applied.
        """
        A = self.lam * np.eye(self.dimension)
        A += float(factor) * self._R_actual + new_fit_R
        b = float(factor) * self._C_actual + new_fit_C
        return A, b

    def _candidate_weights(self, factor, new_fit_R, new_fit_C):
        A, b = self._effective_system(factor, new_fit_R, new_fit_C)
        return np.linalg.solve(A, b)

    # ------------------------------------------------------------------
    # Per-domain validation
    # ------------------------------------------------------------------

    def _per_domain_validation(self, weights):
        """Compute J_j = (w' R_j w - 2 w' C_j + S_j) for each old domain,
        using the stored validation statistics (already divided by n_j)."""
        losses = []
        for Rj, Cj, Sj in zip(self._val_R, self._val_C, self._val_S):
            quad = float(np.sum(weights * (Rj @ weights)))
            lin = float(np.sum(weights * Cj))
            losses.append(quad - 2.0 * lin + float(Sj))
        return losses

    def _new_domain_validation(self, weights, val_R, val_C, val_S, val_n):
        quad = float(np.sum(weights * (val_R @ weights)))
        lin = float(np.sum(weights * val_C))
        return (quad - 2.0 * lin + float(val_S)) / max(float(val_n), 1.0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_and_update(self, fit, val) -> PerDomainSelectionResult:
        """Select f for the incoming domain, update causal state, and
        append per-domain validation statistics.

        Parameters
        ----------
        fit : (R, C, _S, _n)
            Fit sufficient statistics for the NEW domain.
        val : (R, C, S, n)
            Validation sufficient statistics for the NEW domain.
        """
        import numpy as np_np

        fit_R, fit_C, _, _ = fit
        val_R, val_C, val_S, val_n = val
        fit_R = np.asarray(fit_R, dtype=np.float64)
        fit_C = np.asarray(fit_C, dtype=np.float64).reshape(-1, 1)
        val_R = np.asarray(val_R, dtype=np.float64)
        val_C = np.asarray(val_C, dtype=np.float64).reshape(-1, 1)
        val_n = int(val_n)

        if fit_R.shape != (self.dimension, self.dimension):
            raise ValueError("fit R has wrong shape")
        if val_n <= 0:
            raise ValueError("held-out validation split must be non-empty")

        # --- first domain: f=1, no selection needed ---
        if self._n_domains == 0:
            self._R_actual = fit_R.copy()
            self._C_actual = fit_C.copy()
            self._val_R.append(val_R.copy() / float(val_n))
            self._val_C.append(val_C.copy() / float(val_n))
            self._val_S.append(float(val_S) / float(val_n))
            self._n_domains = 1
            return PerDomainSelectionResult(
                factor=1.0,
                objective=None,
                objective_f1=None,
                drift_score=0.0,
                search_mode=self.search_mode,
                per_domain_validation=[],
                per_domain_validation_f1=[],
                safety_abstained=False,
            )

        # --- subsequent domains: evaluate candidates ---
        def objective(factor_value):
            fv = float(factor_value)
            w = self._candidate_weights(fv, fit_R, fit_C)
            old_losses = self._per_domain_validation(w)
            new_loss = self._new_domain_validation(w, val_R, val_C, val_S, val_n)
            all_losses = old_losses + [new_loss]
            total_domains = self._n_domains + 1
            return float(np_np.mean(all_losses))

        # Evaluate f=1 reference
        w1 = self._candidate_weights(1.0, fit_R, fit_C)
        old_losses_f1 = self._per_domain_validation(w1)
        new_loss_f1 = self._new_domain_validation(w1, val_R, val_C, val_S, val_n)

        # Grid search
        candidates = [(float(g), objective(float(g))) for g in self.grid]
        best_f, best_obj = min(candidates, key=lambda t: t[1])
        objective_f1 = objective(1.0)

        # Abstention check
        relative_gain = max(objective_f1 - best_obj, 0.0) / max(abs(objective_f1), 1e-12)
        safety_abstained = False

        if relative_gain <= self.abstain_relative_gain:
            factor = 1.0
            best_obj = objective_f1
        else:
            factor = best_f

        # Per-domain degradation check (TRUE degradation, not proxy!)
        if factor < 1.0 and self.max_per_domain_degradation is not None:
            w_f = self._candidate_weights(factor, fit_R, fit_C)
            old_losses_f = self._per_domain_validation(w_f)
            for j, (loss_f, loss_1) in enumerate(zip(old_losses_f, old_losses_f1)):
                degradation = (loss_f - loss_1) / max(abs(loss_1), 1e-12)
                if degradation > self.max_per_domain_degradation + 1e-12:
                    factor = 1.0
                    best_obj = objective_f1
                    relative_gain = 0.0
                    safety_abstained = True
                    break

        drift_score = float(relative_gain)

        # Per-domain validation values at selected f and at f=1
        w_selected = self._candidate_weights(factor, fit_R, fit_C)
        old_losses_selected = self._per_domain_validation(w_selected)
        new_loss_selected = self._new_domain_validation(
            w_selected, val_R, val_C, val_S, val_n
        )

        # --- update ACTUAL state (causal recursion, matching deployed head) ---
        self._R_actual = float(factor) * self._R_actual + fit_R
        self._C_actual = float(factor) * self._C_actual + fit_C

        # Store per-domain validation stats
        self._val_R.append(val_R.copy() / float(val_n))
        self._val_C.append(val_C.copy() / float(val_n))
        self._val_S.append(float(val_S) / float(val_n))
        self._n_domains += 1

        return PerDomainSelectionResult(
            factor=float(factor),
            objective=None if best_obj is None else float(best_obj),
            objective_f1=None if objective_f1 is None else float(objective_f1),
            drift_score=float(drift_score),
            search_mode=self.search_mode,
            per_domain_validation=old_losses_selected + [new_loss_selected],
            per_domain_validation_f1=old_losses_f1 + [new_loss_f1],
            safety_abstained=bool(safety_abstained),
        )
