#!/usr/bin/env python3
"""Validate the two post-freeze exploratory diagnostics and paper bindings.

Unlike the formal validators, this script does not enforce immutable hashes or
single-shot locks. It checks deterministic recomputation invariants, the
strictly nested candidate construction, and the numerical snippets retained in
the current paper source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explore_candidate_family import ATTRIBUTION_FAMILIES, candidate_families


ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _latex_integer(value):
    return f"{value:,}".replace(",", "{,}")


def validate_history(paper):
    path = ROOT / "runs_real/exploratory_history_dominance/history_dominance.json"
    payload = json.loads(path.read_text())
    records = payload["records"]
    _require(len(records) == 70, "history diagnostic must contain 70 cells")

    oracle = np.asarray([row["oracle_median"] for row in records])
    theory = np.asarray([row["closed_form"] for row in records])
    gains = np.asarray([row["oracle_opportunity_pct"] for row in records])
    correlation = float(np.corrcoef(oracle, theory)[0, 1])
    mean_abs_error = float(np.mean(np.abs(oracle - theory)))
    material = gains >= 0.5
    material_error = float(np.mean(np.abs(oracle[material] - theory[material])))

    _require(np.isclose(correlation, 0.9428, atol=5e-5), "history correlation drift")
    _require(np.isclose(mean_abs_error, payload["mean_abs_error"]), "history MAE drift")
    _require(
        np.isclose(material_error, payload["mean_abs_error_material_cells"]),
        "history material-cell MAE drift",
    )

    no_mismatch = [row for row in records if row["delta"] == 0.0]
    no_mismatch_material = sum(row["oracle_opportunity_pct"] >= 0.5 for row in no_mismatch)
    _require(no_mismatch_material == 5, "zero-mismatch conditional-Oracle count drift")
    _require(
        no_mismatch_material
        == payload["zero_mismatch_material_conditional_oracle_count"],
        "stored zero-mismatch count is inconsistent",
    )

    for row in records:
        _require(
            abs(row["expected_risk_grid_oracle"] - row["closed_form"]) <= 0.005 + 1e-12,
            "expected-risk grid optimum no longer matches the closed form",
        )

    for snippet in ("$r=0.943$", "$5/10$", "$[0.390,1.000]$", "$1.57\\%$"):
        _require(snippet in paper, f"paper is missing history binding: {snippet}")


def validate_candidate_family(paper):
    path = ROOT / "runs_real/exploratory_candidate_family/candidate_family_exploration.json"
    payload = json.loads(path.read_text())
    scenarios = payload["scenarios"]
    _require(len(scenarios) == 320, "candidate diagnostic must contain 320 scenarios")
    _require(
        tuple(payload["nested_attribution_track"]) == ATTRIBUTION_FAMILIES,
        "stored attribution family order drift",
    )
    _require(payload["all_attribution_weights_have_mean_one"], "normalization flag missing")

    config = json.loads((ROOT / payload["config"]).read_text())
    expected_counts = None
    for order in config["orders"].values():
        families = candidate_families(tuple(order), tuple(config["factor_grid"]))
        counts = {name: len(weights) for name, weights in families.items()}
        if expected_counts is None:
            expected_counts = counts
        else:
            _require(counts == expected_counts, "candidate counts vary by order")

    observed_counts = {
        name: scenarios[0][name]["candidates"] for name in payload["family_order"]
    }
    _require(observed_counts == expected_counts, "stored candidate counts drift")

    shares = payload["normalized_track_incremental_shares"]
    share_total = sum(row["share_of_normalized_track_lift_pct"] for row in shares.values())
    _require(np.isclose(share_total, 100.0), "incremental shares do not sum to 100")

    summaries = payload["summary"]
    maximum = max(row["free_normalized"]["mean_gain_pct"] for row in summaries)
    crossed = sum(row["free_normalized"]["mean_gain_pct"] >= 0.5 for row in summaries)
    _require(np.isclose(maximum, 0.4076000659103569), "free-family maximum drift")
    _require(crossed == 0, "free-family material-gate classification drift")

    final_share = shares[
        "free_current_max_normalized_to_free_normalized"
    ]["share_of_normalized_track_lift_pct"]
    for value in expected_counts.values():
        _require(_latex_integer(value) in paper, f"paper is missing candidate count {value}")
    for snippet in (f"${final_share:.1f}\\%$", "$0.408\\%$", "$0/16$"):
        _require(snippet in paper, f"paper is missing candidate binding: {snippet}")


def main():
    paper = (ROOT / "paper/main.tex").read_text()
    validate_history(paper)
    validate_candidate_family(paper)
    print("PASS exploratory diagnostics: theory boundary, nesting, outputs and paper bindings")


if __name__ == "__main__":
    main()
