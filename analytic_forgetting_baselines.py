"""Analytic forgetting baselines used by RRCL.

This module intentionally separates the algorithms' mathematical cores from
their visual-regression adapters:

* DOS-ELM [Cao et al., IEEE Access 2019] changes its forgetting factor from a
  chunk-to-chunk performance signal.  The paper's weighted design matrix uses
  ``lambda * H_old``, so old sufficient statistics receive ``lambda**2``.
* SIFt-RLS [Lai and Bernstein, 2024] forgets information only in the row space
  excited by the incoming regressor.

The DOS-ELM paper does not specify an unambiguous regression "accuracy" score.
Callers must therefore provide a score for which larger is better and must
label the resulting method as a DOS-ELM-style regression adaptation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DOSFactorUpdate:
    """One dynamic-forgetting update from the DOS-ELM rule."""

    factor: float
    error_signal: float
    statistic_factor: float


def dos_elm_factor_update(
    factor,
    previous_score,
    current_score,
    *,
    factor_min=0.0,
    factor_max=1.0,
):
    """Return the DOS-ELM forgetting factor to use for the *next* chunk.

    ``score`` must use the convention "larger is better".  The paper states
    that a positive error means degraded performance, hence the operational
    error signal is ``previous_score - current_score``.  Its Eq. (16) is

        lambda_next = lambda - atan(error) / (5 pi).

    The published equation explicitly caps values above one.  We additionally
    clip the lower end to the stated domain [0, 1], which prevents invalid
    negative factors for long or badly scaled sequences.
    """

    factor = float(factor)
    previous_score = float(previous_score)
    current_score = float(current_score)
    factor_min = float(factor_min)
    factor_max = float(factor_max)
    if not (0.0 <= factor_min <= factor_max <= 1.0):
        raise ValueError("DOS-ELM factor bounds must satisfy 0 <= min <= max <= 1")
    if not (factor_min <= factor <= factor_max):
        raise ValueError("current DOS-ELM factor is outside the configured bounds")
    if not np.isfinite([factor, previous_score, current_score]).all():
        raise ValueError("DOS-ELM inputs must be finite")

    error_signal = previous_score - current_score
    next_factor = factor - np.arctan(error_signal) / (5.0 * np.pi)
    next_factor = float(np.clip(next_factor, factor_min, factor_max))
    return DOSFactorUpdate(
        factor=next_factor,
        error_signal=float(error_signal),
        statistic_factor=next_factor * next_factor,
    )


@dataclass(frozen=True)
class SIFtStepInfo:
    """Diagnostics for one SIFt-RLS measurement update."""

    input_rows: int
    filtered_rank: int
    parameter_dim: int
    uniform_forgetting_equivalent: bool
    skipped: bool


class SIFtRLS:
    """Subspace-of-information forgetting RLS (SIFt-RLS).

    This follows Algorithm 1 of Lai and Bernstein.  Ridge regularization is
    represented as the positive-definite prior information matrix
    ``R_0 = ridge * I``.  Therefore no additional ridge term is added during
    prediction or solving.
    """

    def __init__(
        self,
        d_in,
        d_out=1,
        *,
        ridge=1e2,
        forgetting=0.5,
        epsilon=1e-8,
        bias=True,
    ):
        d_in = int(d_in)
        d_out = int(d_out)
        ridge = float(ridge)
        forgetting = float(forgetting)
        epsilon = float(epsilon)
        if d_in <= 0 or d_out <= 0:
            raise ValueError("SIFt-RLS input and output dimensions must be positive")
        if ridge <= 0.0 or not np.isfinite(ridge):
            raise ValueError("SIFt-RLS requires a finite positive ridge prior")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("SIFt-RLS forgetting must satisfy 0 < f <= 1")
        if epsilon <= 0.0 or not np.isfinite(epsilon):
            raise ValueError("SIFt-RLS epsilon must be finite and positive")

        self.bias = bool(bias)
        self.d = d_in + int(self.bias)
        self.m = d_out
        self.ridge = ridge
        self.forgetting = forgetting
        self.epsilon = epsilon
        self.R = ridge * np.eye(self.d, dtype=np.float64)
        self.W = np.zeros((self.d, self.m), dtype=np.float64)
        self.step_history = []

    def _aug(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        expected = self.d - int(self.bias)
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(f"expected X with shape (n, {expected}), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("SIFt-RLS regressors must be finite")
        if self.bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def update(self, X, Y):
        """Assimilate one vector or matrix measurement and return diagnostics."""

        phi = self._aug(X)
        Y = np.asarray(Y, dtype=np.float64).reshape(phi.shape[0], self.m)
        if not np.isfinite(Y).all():
            raise ValueError("SIFt-RLS targets must be finite")

        U, singular_values, _ = np.linalg.svd(phi, full_matrices=False)
        keep = singular_values >= np.sqrt(self.epsilon)
        filtered_rank = int(np.count_nonzero(keep))
        info = SIFtStepInfo(
            input_rows=int(phi.shape[0]),
            filtered_rank=filtered_rank,
            parameter_dim=self.d,
            uniform_forgetting_equivalent=filtered_rank == self.d,
            skipped=filtered_rank == 0,
        )
        self.step_history.append(info)
        if filtered_rank == 0:
            return info

        U_bar = U[:, keep]
        phi_bar = U_bar.T @ phi
        y_bar = U_bar.T @ Y

        L = phi_bar @ self.R
        middle = L @ phi_bar.T
        directional_information = L.T @ np.linalg.solve(middle, L)
        R_bar = self.R - (1.0 - self.forgetting) * directional_information
        R_bar = 0.5 * (R_bar + R_bar.T)

        R_next = R_bar + phi_bar.T @ phi_bar
        R_next = 0.5 * (R_next + R_next.T)
        residual = y_bar - phi_bar @ self.W
        self.W = self.W + np.linalg.solve(R_next, phi_bar.T @ residual)
        self.R = R_next
        return info

    def predict(self, X, *, non_negative=False):
        prediction = self._aug(X) @ self.W
        if non_negative:
            prediction = np.clip(prediction, 0.0, None)
        return prediction

    @property
    def state_bytes(self):
        return int(self.R.nbytes + self.W.nbytes)
