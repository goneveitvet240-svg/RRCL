"""Recursive ridge regression head with a forgetting factor, plus an OPTIONAL
fixed random feature expansion (RanPAC-style) for capacity.

Set env  RRCL_PROJ_DIM=4096  (or pass proj_dim=) to lift the frozen features
through a fixed random projection + ReLU before the ridge. The linear head
alone underfits dense crowds (lambda is inert: X^T X dominates), so this adds
capacity while staying closed-form, exemplar-free, and recursion-compatible
(the projection is fixed/data-independent).

  forget == 1.0  -> absolute memorization (increment == joint least squares)
  forget  < 1.0  -> adaptive forgetting (stability <-> plasticity)
"""
import os
import numpy as np


def f_star(rho):
    """Closed-form optimal forgetting factor for drift/noise ratio rho=q^2/r."""
    rho = max(float(rho), 0.0)
    if rho == 0.0:
        return 1.0
    return 0.5 * ((2.0 + rho) - np.sqrt(rho * rho + 4.0 * rho))


def rho_from_f(f):
    """Inverse map used for interpreting empirical best f values."""
    f = float(f)
    return (f * f + 1.0) / f - 2.0


class ForgettingRidgeRLS:
    def __init__(self, d_in, d_out=1, lam=1e2, forget=1.0, bias=True,
                 proj_dim=None, proj_seed=0, adaptive=False,
                 f_min=0.05, f_max=1.0, noise_ema=0.5):
        if proj_dim is None:
            proj_dim = int(os.environ.get("RRCL_PROJ_DIM", "0"))
        self.proj = None
        if proj_dim and proj_dim > 0:
            rng = np.random.default_rng(proj_seed)
            self.proj = (rng.standard_normal((d_in, proj_dim)) / np.sqrt(d_in)).astype(np.float64)
            d_eff = proj_dim
        else:
            d_eff = d_in
        self.bias = bias
        self.d = d_eff + (1 if bias else 0)
        self.m = d_out
        self.lam = float(lam)
        self.forget = float(forget)
        self.adaptive = bool(adaptive)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.noise_ema = float(noise_ema)
        self.r_hat = None
        self.f_history = []
        self.innovation_history = []
        self.rho_history = []
        self.R = np.zeros((self.d, self.d), dtype=np.float64)   # autocorrelation
        self.C = np.zeros((self.d, self.m), dtype=np.float64)   # cross-correlation
        self.W = None

    def _feat(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.proj is not None:
            X = np.maximum(X @ self.proj, 0.0)      # fixed random ReLU features
        return X

    def _aug(self, X):
        X = self._feat(X)
        if self.bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X

    def begin_task(self, forget=None):
        f = self.forget if forget is None else float(forget)
        self.R *= f
        self.C *= f
        self.f_history.append(float(f))
        return self

    def accumulate(self, X, Y):
        Xa = self._aug(X)
        Y = np.asarray(Y, dtype=np.float64).reshape(Xa.shape[0], self.m)
        self.R += Xa.T @ Xa
        self.C += Xa.T @ Y
        return self

    def solve(self):
        A = self.R + self.lam * np.eye(self.d)
        self.W = np.linalg.solve(A, self.C)
        return self

    def update(self, X, Y):
        return self.begin_task().accumulate(X, Y).solve()

    def predict(self, X, non_negative=True):
        Y = self._aug(X) @ self.W
        return np.clip(Y, 0.0, None) if non_negative else Y

    def residual_energy(self, X, Y):
        """Mean squared pre/post-fit residual energy per patch target."""
        if self.W is None:
            return None
        Xa = self._aug(X)
        Y = np.asarray(Y, dtype=np.float64).reshape(Xa.shape[0], self.m)
        resid = Y - Xa @ self.W
        return float(np.mean(np.sum(resid * resid, axis=1)))

    def adaptive_forget_from_innovation(self, eps):
        """Pick f_t from innovation eps using rho_hat=(eps-r_hat)+/r_hat."""
        if not self.adaptive or self.W is None or self.r_hat is None:
            return self.f_max, None
        rho_hat = max(float(eps) - self.r_hat, 0.0) / max(self.r_hat, 1e-12)
        f = float(np.clip(f_star(rho_hat), self.f_min, self.f_max))
        return f, rho_hat

    def update_noise_floor(self, eps):
        """EMA noise floor from post-fit residual energy."""
        eps = float(eps)
        if self.r_hat is None:
            self.r_hat = eps
        else:
            self.r_hat = self.noise_ema * self.r_hat + (1.0 - self.noise_ema) * eps
        return self.r_hat
