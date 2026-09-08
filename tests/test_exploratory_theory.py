import numpy as np
import pytest

from scripts.explore_candidate_family import (
    ATTRIBUTION_FAMILIES,
    candidate_families,
)
from scripts.history_dominance_phase import (
    closed_form_factor,
    expected_excess_risk,
)


@pytest.mark.parametrize(
    ("a", "delta", "expected"),
    [
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (5.0, 0.0, 1.0),
    ],
)
def test_closed_form_boundary_cases(a, delta, expected):
    assert closed_form_factor(a, delta, sigma=0.5, n2=64, dim=8) == expected


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("delta", [0.0, 0.2, 1.6])
def test_closed_form_matches_expected_risk_grid(a, delta):
    grid = np.linspace(0.001, 1.0, 10000)
    risks = np.asarray(
        [expected_excess_risk(f, a, delta, 0.5, 64, 8) for f in grid]
    )
    grid_optimum = float(grid[int(np.argmin(risks))])
    target = max(float(grid[0]), closed_form_factor(a, delta, 0.5, 64, 8))
    assert abs(grid_optimum - target) <= float(grid[1] - grid[0])


def test_candidate_attribution_track_is_nested_and_mean_normalized():
    families = candidate_families((0, 1, 2, 3), (0.0, 0.5, 1.0))

    def keys(matrix):
        return {tuple(row) for row in np.round(matrix, 10)}

    for family in ATTRIBUTION_FAMILIES:
        assert np.allclose(families[family].mean(axis=1), 1.0, atol=1e-12)
    for left, right in zip(ATTRIBUTION_FAMILIES, ATTRIBUTION_FAMILIES[1:]):
        assert keys(families[left]).issubset(keys(families[right]))
