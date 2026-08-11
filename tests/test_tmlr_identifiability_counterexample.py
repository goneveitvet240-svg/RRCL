"""Mechanical checks for Proposition: scoped train-only non-identifiability."""

from __future__ import annotations

import math


def merged_weight(factor: float) -> float:
    """Two scalar domains with w1=0, w2=1 and equal training information."""
    return 1.0 / (1.0 + factor)


def balanced_test_risk(factor: float, rho1: float, rho2: float) -> float:
    weight = merged_weight(factor)
    return 0.5 * (rho1 * weight**2 + rho2 * (weight - 1.0) ** 2)


def analytic_optimal_factor(rho1: float, rho2: float) -> float:
    # Test-risk-optimal prediction is rho2 / (rho1 + rho2).  Solving
    # 1 / (1 + f) = rho2 / (rho1 + rho2) gives f = rho1 / rho2.
    return rho1 / rho2


def test_two_worlds_have_identical_training_information_but_distinct_optima():
    training_information_world_a = {
        "w": (0.0, 1.0),
        "training_mass": (1.0, 1.0),
        "noise": (0.0, 0.0),
        "ridge": 0.0,
    }
    training_information_world_b = dict(training_information_world_a)
    assert training_information_world_a == training_information_world_b

    factor_a = analytic_optimal_factor(1.0, 1.0)
    factor_b = analytic_optimal_factor(1.0, 4.0)
    assert factor_a == 1.0
    assert factor_b == 0.25
    assert factor_a != factor_b


def test_claimed_factors_minimize_their_respective_balanced_test_risks():
    for rho1, rho2, expected in ((1.0, 1.0, 1.0), (1.0, 4.0, 0.25)):
        optimum = analytic_optimal_factor(rho1, rho2)
        assert optimum == expected
        optimum_risk = balanced_test_risk(optimum, rho1, rho2)
        for candidate in (0.05, 0.10, 0.25, 0.50, 0.75, 1.0):
            assert optimum_risk <= balanced_test_risk(candidate, rho1, rho2) + 1e-15

        expected_weight = rho2 / (rho1 + rho2)
        assert math.isclose(merged_weight(optimum), expected_weight)
