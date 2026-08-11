"""Forgetting-factor rules used as explicit RLS baselines."""

from __future__ import annotations

import numpy as np


def paleologu_vff_factor(
    error_power,
    noise_power,
    leverage_power,
    *,
    gamma=1.5,
    xi=1e-6,
    f_min=0.05,
    f_max=1.0,
):
    """Paleologu--Benesty--Ciochina robust VFF rule, batch adapted.

    The original sample-wise rule is Eq. (17)--(18) of IEEE SPL 2008,
    doi:10.1109/LSP.2008.2001559.  RRCL evaluates the same power relation once
    per known domain boundary because its analytic update is domain-batched.
    """
    sigma_error = np.sqrt(max(float(error_power), 0.0))
    sigma_noise = np.sqrt(max(float(noise_power), 0.0))
    sigma_leverage = np.sqrt(max(float(leverage_power), 0.0))
    if sigma_error <= float(gamma) * sigma_noise:
        return float(f_max)
    value = sigma_leverage * sigma_noise / (
        float(xi) + abs(sigma_error - sigma_noise)
    )
    return float(np.clip(value, f_min, f_max))
