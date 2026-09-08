#!/usr/bin/env python3
"""EXPLORATORY diagnostic: how much Oracle opportunity does the candidate
family itself withhold?

Status
------
This is an **exploratory** analysis, not a preregistered confirmatory study.
It writes to its own output directory and never touches the frozen
``runs_real/tmlr_synthetic_v1`` artifact or any of its hashes.  Nothing here
defines, tunes, or evaluates a deployable selector: every number below is a
*diagnostic Oracle* value computed on population risk, i.e. it is allowed to
see the test-side objective and is therefore not attainable by any train-only
rule.  Its only purpose is to attribute the small measured opportunity to
either (a) the width of the candidate family or (b) the difficulty of
selection.

Question
--------
Synthetic-v1 reported that all 16 cells fall below the preregistered 0.5%
material threshold, with a maximum fixed-factor Oracle gain of 0.200%.  That
measurement searched a **scalar** family: weights are forced to

    w_k = f^(T-1-position(k)),      f on the frozen grid.

The scalar family has three structural constraints:

  1. the last domain in the order always receives weight exactly 1;
  2. weights are monotone non-decreasing along the order;
  3. all weights are tied to a single parameter.

The frozen scalar family is first reproduced at its deployed scale.  Causal
attribution is then performed on a separate, mean-normalized track whose four
families are strictly nested and all have the same total weight. This avoids
confounding a relaxed shape constraint with a change in effective ridge scale:

    scalar-normalized
      subset-of monotone-normalized
      subset-of free-current-max-normalized
      subset-of free-normalized

The third family keeps the incoming domain at least as large as every
historical weight, which is the scale-invariant content of pinning its raw
weight to one while all other raw weights lie in [0, 1]. The fourth family
removes that current-domain-maximum constraint. Candidate unions explicitly
preserve nesting even when powers such as f^2 are not points on the base grid.

Interpretation rules (fixed before running)
-------------------------------------------
* A larger free-family opportunity does NOT imply that any train-only rule can
  reach it; it only relocates the bottleneck from selection to parameterization.
* A free-family opportunity that remains below the 0.5% material threshold is
  evidence that the setting itself, not the scalar restriction, limits the
  available headroom.
* Incremental shares are descriptive decompositions inside the nested,
  mean-normalized track. They do not apply to the deployed-scale scalar family.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmlr_synthetic import (  # noqa: E402
    cell_id,
    domain_statistics,
    factor_cells,
    generate_domains,
    load_config,
    population_risks,
)

FAMILIES = (
    "scalar",
    "scalar_normalized",
    "monotone_normalized",
    "free_current_max_normalized",
    "free_normalized",
)
ATTRIBUTION_FAMILIES = FAMILIES[1:]


# ----------------------------------------------------------------------
# candidate weight enumeration
# ----------------------------------------------------------------------
def scalar_weights(order, grid):
    """w_k = f^(T-1-position). Exactly the synthetic-v1 family."""
    count = len(order)
    out = []
    for factor in grid:
        w = np.zeros(count, dtype=np.float64)
        for position, physical in enumerate(order):
            w[physical] = float(factor) ** (count - 1 - position)
        out.append(w)
    return np.asarray(out)


def _unique_rows(*matrices):
    """Concatenate, mean-normalize and deterministically de-duplicate weights."""
    raw = np.vstack([np.asarray(m, dtype=np.float64) for m in matrices])
    raw = raw[raw.sum(axis=1) > 1e-12]
    raw = raw * (raw.shape[1] / raw.sum(axis=1, keepdims=True))
    _, keep = np.unique(np.round(raw, 10), axis=0, return_index=True)
    return raw[np.sort(keep)]


def _row_keys(matrix):
    return {tuple(row) for row in np.round(matrix, 10)}


def per_domain_weights(order, grid, monotone):
    """Free (or monotone) per-domain weights with the last-in-order pinned to 1.

    Enumerated in ORDER space, then scattered to physical domain indices, so
    that ``monotone`` means non-decreasing along the presentation order — the
    same direction the scalar family is forced into.
    """
    count = len(order)
    grid = [float(g) for g in grid]
    out = []
    for prefix in itertools.product(grid, repeat=count - 1):
        seq = list(prefix) + [1.0]
        if monotone and any(seq[i] > seq[i + 1] + 1e-12 for i in range(count - 1)):
            continue
        w = np.zeros(count, dtype=np.float64)
        for position, physical in enumerate(order):
            w[physical] = seq[position]
        out.append(w)
    return np.asarray(out)


@lru_cache(maxsize=None)
def candidate_families(order_tuple, grid_tuple):
    """Build the deployed scalar family and a strictly nested attribution track."""
    order = list(order_tuple)
    grid = list(grid_tuple)

    scalar = scalar_weights(order, grid)
    scalar_normalized = _unique_rows(scalar)

    monotone_raw = per_domain_weights(order, grid, monotone=True)
    monotone_normalized = _unique_rows(monotone_raw, scalar_normalized)

    free_current_max_raw = per_domain_weights(order, grid, monotone=False)
    free_current_max_normalized = _unique_rows(
        free_current_max_raw, monotone_normalized
    )

    count = len(order)
    all_raw = np.asarray(
        list(itertools.product(map(float, grid), repeat=count)),
        dtype=np.float64,
    )
    free_normalized = _unique_rows(all_raw, free_current_max_normalized)

    families = {
        "scalar": scalar,
        "scalar_normalized": scalar_normalized,
        "monotone_normalized": monotone_normalized,
        "free_current_max_normalized": free_current_max_normalized,
        "free_normalized": free_normalized,
    }

    for left, right in zip(ATTRIBUTION_FAMILIES, ATTRIBUTION_FAMILIES[1:]):
        if not _row_keys(families[left]).issubset(_row_keys(families[right])):
            raise AssertionError(f"candidate families are not nested: {left} -> {right}")
    for name in ATTRIBUTION_FAMILIES:
        if not np.allclose(families[name].mean(axis=1), 1.0, atol=1e-12):
            raise AssertionError(f"{name} is not mean-normalized")
    return families


# ----------------------------------------------------------------------
# batched risk evaluation
# ----------------------------------------------------------------------
def batched_population_risk(weight_matrix, stats, domains, lam):
    """Mean population balanced MSE for every candidate weight vector.

    Uses one batched linear solve instead of a Python loop.  Verified against
    the reference path in ``_selftest``.
    """
    R = np.stack([rec["full"]["R"] for rec in stats])          # (T, d, d)
    C = np.stack([rec["full"]["C"] for rec in stats])          # (T, d, 1)
    dim = R.shape[1]

    # (N, d, d) and (N, d, 1)
    R_eff = np.einsum("nt,tij->nij", weight_matrix, R)
    R_eff = R_eff + lam * np.eye(dim)[None, :, :]
    C_eff = np.einsum("nt,tij->nij", weight_matrix, C)
    W = np.linalg.solve(R_eff, C_eff)                          # (N, d, 1)

    vectors = W[:, :-1, 0]                                     # (N, d-1)
    intercepts = W[:, -1, 0]                                   # (N,)

    risks = np.zeros((W.shape[0], len(domains)), dtype=np.float64)
    for index, domain in enumerate(domains):
        delta = vectors - domain.beta[None, :]
        quad = np.einsum("ni,ij,nj->n", delta, domain.covariance, delta)
        risks[:, index] = quad + intercepts**2 + domain.sigma**2
    return risks.mean(axis=1)


def _selftest(weight_matrix, stats, domains, lam, batched):
    """Confirm the batched path reproduces the reference solve on 3 candidates."""
    from precision_shrinkage import solve_ridge, weighted_system

    pairs = [(rec["full"]["R"], rec["full"]["C"]) for rec in stats]
    for index in (0, len(weight_matrix) // 2, len(weight_matrix) - 1):
        R, C = weighted_system(pairs, weight_matrix[index])
        W = solve_ridge(R, C, lam)
        reference = float(np.mean(population_risks(W, domains)))
        if not np.isclose(reference, batched[index], rtol=0, atol=1e-10):
            raise AssertionError(
                f"batched risk mismatch at {index}: {batched[index]} vs {reference}"
            )


# ----------------------------------------------------------------------
# one scenario
# ----------------------------------------------------------------------
def evaluate_scenario(config, cell, seed, order_name, verify=False):
    order = list(config["orders"][order_name])
    grid = list(map(float, config["factor_grid"]))
    lam = float(config["ridge_lambda"])
    families = candidate_families(tuple(order), tuple(grid))

    domains = generate_domains(config, cell, seed)
    stats = domain_statistics(config, domains)

    uniform = np.ones((1, len(order)), dtype=np.float64)
    risk_f1 = float(batched_population_risk(uniform, stats, domains, lam)[0])

    record = {"f1_population_risk": risk_f1}
    for family in FAMILIES:
        candidates = families[family]
        risks = batched_population_risk(candidates, stats, domains, lam)
        if verify and family == "scalar":
            _selftest(candidates, stats, domains, lam, risks)
        best = int(np.argmin(risks))
        record[family] = {
            "candidates": int(candidates.shape[0]),
            "best_risk": float(risks[best]),
            "gain_pct": 100.0 * (risk_f1 - float(risks[best])) / abs(risk_f1),
            "best_weights": [round(float(v), 6) for v in candidates[best]],
        }
    return record


# ----------------------------------------------------------------------
# aggregation: average the two orders within a seed, then summarize seeds
# ----------------------------------------------------------------------
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci(values):
    values = [float(v) for v in values]
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None
    critical = T_CRITICAL_95.get(len(values), 1.96)
    half = critical * statistics.stdev(values) / (len(values) ** 0.5)
    return mean, mean - half, mean + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tmlr_synthetic_v1.json")
    parser.add_argument("--out", default="runs_real/exploratory_candidate_family")
    parser.add_argument("--seeds", type=int, default=0,
                        help="use only the first N seeds (0 = all)")
    parser.add_argument("--cells", default="",
                        help="comma-separated cell ids, e.g. 0000,0100")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = list(config["seeds"])
    if args.seeds:
        seeds = seeds[: args.seeds]
    wanted = {c.strip() for c in args.cells.split(",") if c.strip()}
    order_names = list(config["orders"].keys())
    threshold = 100.0 * float(config["decision_thresholds"]["minimum_relative_gain"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"config          : {args.config}")
    print(f"cells x seeds   : {'all 16' if not wanted else sorted(wanted)} x {len(seeds)}")
    print(f"orders          : {order_names} (averaged within seed)")
    print(f"material gate   : {threshold:.3f}%   [exploratory diagnostic only]\n")

    scenarios = []
    summary_rows = []
    verified = False

    for cell in factor_cells():
        cid = cell_id(cell)
        if wanted and cid not in wanted:
            continue
        per_seed = {family: [] for family in FAMILIES}
        for seed in seeds:
            per_order = {family: [] for family in FAMILIES}
            for order_name in order_names:
                rec = evaluate_scenario(config, cell, seed, order_name,
                                        verify=not verified)
                verified = True
                rec.update({"cell": cid, "seed": seed, "order": order_name})
                scenarios.append(rec)
                for family in FAMILIES:
                    per_order[family].append(rec[family]["gain_pct"])
            for family in FAMILIES:
                per_seed[family].append(float(np.mean(per_order[family])))

        row = {"cell": cid}
        for family in FAMILIES:
            mean, low, high = mean_ci(per_seed[family])
            row[family] = {"mean_gain_pct": mean, "ci95_low": low, "ci95_high": high}
        row["normalized_track_increments"] = {}
        for left, right in zip(ATTRIBUTION_FAMILIES, ATTRIBUTION_FAMILIES[1:]):
            row["normalized_track_increments"][f"{left}_to_{right}"] = (
                row[right]["mean_gain_pct"] - row[left]["mean_gain_pct"]
            )
        row["free_normalized_over_scalar_normalized_ratio"] = (
            row["free_normalized"]["mean_gain_pct"]
            / row["scalar_normalized"]["mean_gain_pct"]
            if abs(row["scalar_normalized"]["mean_gain_pct"]) > 1e-9
            else None
        )
        summary_rows.append(row)

        print(
            f"cell {cid}   deployed scalar {row['scalar']['mean_gain_pct']:7.3f}%   "
            f"normalized scalar {row['scalar_normalized']['mean_gain_pct']:7.3f}%   "
            f"free normalized {row['free_normalized']['mean_gain_pct']:7.3f}%"
            + (
                "   <-- free normalized crosses gate"
                if row["free_normalized"]["mean_gain_pct"] >= threshold
                else ""
            )
        )

    print("\n" + "=" * 92)
    print(
        f"{'cell':<6}{'deployed':>11}{'scalar-N':>11}{'monotone-N':>13}"
        f"{'current-max-N':>15}{'free-N':>10}{'free-N 95% CI':>22}"
    )
    print("-" * 92)
    for row in summary_rows:
        ci = row["free_normalized"]
        ci_text = (f"[{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}]"
                   if ci["ci95_low"] is not None else "n/a")
        print(
            f"{row['cell']:<6}{row['scalar']['mean_gain_pct']:>11.3f}"
            f"{row['scalar_normalized']['mean_gain_pct']:>11.3f}"
            f"{row['monotone_normalized']['mean_gain_pct']:>13.3f}"
            f"{row['free_current_max_normalized']['mean_gain_pct']:>15.3f}"
            f"{row['free_normalized']['mean_gain_pct']:>10.3f}"
            f"{ci_text:>22}"
        )
    print("=" * 92)

    scalar_max = max(r["scalar"]["mean_gain_pct"] for r in summary_rows)
    free_max = max(r["free_normalized"]["mean_gain_pct"] for r in summary_rows)
    crossed = [r["cell"] for r in summary_rows
               if r["free_normalized"]["mean_gain_pct"] >= threshold]

    overall_means = {
        family: statistics.mean(r[family]["mean_gain_pct"] for r in summary_rows)
        for family in FAMILIES
    }
    total_normalized_lift = (
        overall_means["free_normalized"] - overall_means["scalar_normalized"]
    )
    incremental_shares = {}
    for left, right in zip(ATTRIBUTION_FAMILIES, ATTRIBUTION_FAMILIES[1:]):
        increment = overall_means[right] - overall_means[left]
        incremental_shares[f"{left}_to_{right}"] = {
            "increment_pct_points": increment,
            "share_of_normalized_track_lift_pct": (
                100.0 * increment / total_normalized_lift
                if abs(total_normalized_lift) > 1e-12
                else None
            ),
        }

    print(f"\nmax scalar opportunity        : {scalar_max:.3f}%")
    print(f"max free-normalized opportunity: {free_max:.3f}%")
    print(f"cells crossing the {threshold:.1f}% gate under free-normalized: "
          f"{len(crossed)}/{len(summary_rows)}"
          + (f"  {crossed}" if crossed else ""))

    payload = {
        "status": "EXPLORATORY_DIAGNOSTIC_NOT_PREREGISTERED",
        "note": ("Oracle values use population risk and are not attainable by any "
                 "train-only rule. The deployed scalar family is reported separately. "
                 "The four attribution families are strictly nested and mean-normalized; "
                 "only the final family removes the current-domain-maximum constraint."),
        "family_order": list(FAMILIES),
        "nested_attribution_track": list(ATTRIBUTION_FAMILIES),
        "all_attribution_weights_have_mean_one": True,
        "overall_mean_gain_pct": overall_means,
        "normalized_track_incremental_shares": incremental_shares,
        "config": args.config,
        "seeds": seeds,
        "orders": order_names,
        "material_gate_pct": threshold,
        "summary": summary_rows,
        "scenarios": scenarios,
    }
    (out_dir / "candidate_family_exploration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nwritten -> {out_dir / 'candidate_family_exploration.json'}")


if __name__ == "__main__":
    main()
