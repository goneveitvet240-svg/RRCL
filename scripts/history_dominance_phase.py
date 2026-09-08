#!/usr/bin/env python3
"""History-dominance study: what actually determines the optimal factor?

Status
------
EXPLORATORY, not preregistered.  It writes to its own directory and does not
touch the frozen synthetic-v1 artifact.  Every reported factor is a diagnostic
Oracle quantity minimizing *population* risk; no train-only rule is defined,
tuned or evaluated here.

Question
--------
Propositions 1--2 state when f=1 stops being optimal but do not say what the
optimum is a function of.  This script isolates the two candidate drivers in
the smallest setting that admits both:

    domain 1 (history) : n1 samples, coefficient beta
    domain 2 (incoming): n2 samples, coefficient beta + delta * u,  ||u||=1

with isotropic covariance Sigma = I, shared noise sigma, ridge lambda, and
weights (f, 1) -- exactly the two-domain instance of the deployed recursion.

Closed form under this model
----------------------------
With lambda = 0 the *expected* population excess risk, where expectation is
also over the training noise, is

    L(f)  ∝  ||delta||^2 (1 + f^2 a^2) / 2  +  d sigma^2 (f^2 a + 1) / n2
             ------------------------------------------------------------
                                 (f a + 1)^2

whose stationary point is

    f* = (1 + c) / (a + c),
        a = n1 / n2                      (information ratio)
        c = 2 d sigma^2 / (||delta||^2 n2)   (noise-to-mismatch ratio)

clipped to (0, 1].  Two limits carry the interpretation:

    sigma -> 0   =>  f* = 1/a,  i.e.  f* n1 = n2
        the optimum EQUALIZES the effective information of the two domains;
    delta -> 0   =>  c -> inf,  f* = 1
        with no mapping mismatch nothing is discounted, for any n1/n2.

So the expected-risk-optimal factor is not a function of drift magnitude alone:
it is set by the *ratio* of accumulated information to mapping mismatch,
measured in units of noise.

The direct sweep below is deliberately different: for each finite training
sample it lets a diagnostic Oracle choose the factor using the true population
risk, and only then aggregates across seeds. Because minimization happens
before averaging, that conditional Oracle can exploit realization-specific
estimation noise. It is therefore a stability diagnostic, not an exact
empirical verification of the expected-risk formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------
def closed_form_factor(a, delta, sigma, n2, dim):
    """Expected-risk minimizer on ``0 < f <= 1``.

    The ratio ``(1+c)/(a+c)`` is the unconstrained stationary point when
    ``delta > 0``. It lies above one when ``a < 1``, in which case the
    constrained optimum is the boundary ``f=1``. ``delta=0`` is handled as
    the continuous limiting case because ``c`` itself is then undefined.
    """
    if a <= 0 or n2 <= 0:
        raise ValueError("a and n2 must be positive")
    if delta <= 0:
        return 1.0
    c = 2.0 * dim * sigma**2 / (delta**2 * n2)
    return float(min(1.0, (1.0 + c) / (a + c)))


def expected_excess_risk(f, a, delta, sigma, n2, dim):
    """Expected balanced excess risk for the idealized two-domain model."""
    numerator = (
        0.5 * delta**2 * (1.0 + f**2 * a**2)
        + (dim * sigma**2 / n2) * (f**2 * a + 1.0)
    )
    return float(numerator / (f * a + 1.0) ** 2)


def generate(rng, dim, n1, n2, delta, sigma, n_test):
    """Two isotropic linear domains differing by `delta` in coefficient space."""
    beta1 = rng.standard_normal(dim)
    beta1 /= np.linalg.norm(beta1)
    direction = rng.standard_normal(dim)
    direction -= direction @ beta1 * beta1
    direction /= np.linalg.norm(direction)
    beta2 = beta1 + delta * direction

    out = []
    for beta, n in ((beta1, n1), (beta2, n2)):
        X = rng.standard_normal((n, dim))
        y = X @ beta + sigma * rng.standard_normal(n)
        # Preserve the original deterministic RNG stream. These draws were
        # part of the exploratory script before the population-risk audit,
        # although the arrays themselves are not used by the Oracle.
        rng.standard_normal((n_test, dim))
        rng.standard_normal(n_test)
        out.append({"X": X, "y": y, "beta": beta})
    return out


def population_excess_risk(w, domains):
    """Balanced E[(x'w - x'beta)^2] over the two domains (Sigma = I)."""
    return float(np.mean([np.sum((w - d["beta"]) ** 2) for d in domains]))


def oracle_factor(domains, grid, lam):
    """Sweep the factor grid on population risk.

    Returns (f*, opportunity, curve) where ``opportunity`` is the relative risk
    reduction of the Oracle factor over f=1.  Reporting it alongside the argmin
    matters: when the risk curve is flat the argmin wanders while the
    achievable gain is ~0, which is exactly the regime in which a selector can
    fire without any material benefit.
    """
    R1 = domains[0]["X"].T @ domains[0]["X"]
    C1 = domains[0]["X"].T @ domains[0]["y"]
    R2 = domains[1]["X"].T @ domains[1]["X"]
    C2 = domains[1]["X"].T @ domains[1]["y"]
    dim = R1.shape[0]

    curve = []
    for f in grid:
        A = lam * np.eye(dim) + f * R1 + R2
        b = f * C1 + C2
        w = np.linalg.solve(A, b)
        curve.append(population_excess_risk(w, domains))
    curve = np.asarray(curve)
    best = int(np.argmin(curve))
    at_one = float(curve[int(np.argmin(np.abs(grid - 1.0)))])
    opportunity = 100.0 * (at_one - float(curve[best])) / abs(at_one)
    return float(grid[best]), opportunity, curve


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_real/exploratory_history_dominance")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--n2", type=int, default=64, help="incoming-domain sample count")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--n-test", type=int, default=8192)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    ratios = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    deltas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    grid = np.round(np.arange(0.005, 1.0001, 0.005), 4)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dim={args.dim}  n2={args.n2}  sigma={args.sigma}  lambda={args.lam}  "
          f"seeds={args.seeds}  grid={len(grid)} points\n")
    print("Oracle f* (median over seeds) — rows: history ratio a = n1/n2, "
          "cols: mapping mismatch ||delta||\n")

    header = "   a\\d " + "".join(f"{d:>8.2f}" for d in deltas)
    print(header)
    print("-" * len(header))

    records = []
    grid_oracle = np.zeros((len(ratios), len(deltas)))
    grid_theory = np.zeros_like(grid_oracle)
    grid_opportunity = np.zeros_like(grid_oracle)
    grid_expected_oracle = np.zeros_like(grid_oracle)
    grid_expected_opportunity = np.zeros_like(grid_oracle)

    for i, a in enumerate(ratios):
        row = []
        for j, delta in enumerate(deltas):
            n1 = a * args.n2
            factors, gains = [], []
            for s in range(args.seeds):
                rng = np.random.default_rng(20260818 + 1000 * i + 10 * j + s)
                domains = generate(rng, args.dim, n1, args.n2, delta,
                                   args.sigma, args.n_test)
                f_star, gain, _ = oracle_factor(domains, grid, args.lam)
                factors.append(f_star)
                gains.append(gain)
            median = float(np.median(factors))
            opportunity = float(np.median(gains))
            theory = closed_form_factor(a, delta, args.sigma, args.n2, args.dim)
            expected_curve = np.asarray([
                expected_excess_risk(f, a, delta, args.sigma, args.n2, args.dim)
                for f in grid
            ])
            expected_best = int(np.argmin(expected_curve))
            expected_factor = float(grid[expected_best])
            expected_at_one = float(expected_curve[-1])
            expected_opportunity = (
                100.0 * (expected_at_one - float(expected_curve[expected_best]))
                / abs(expected_at_one)
            )
            grid_target = min(1.0, max(float(grid[0]), theory))
            grid_step = float(grid[1] - grid[0])
            if abs(expected_factor - grid_target) > grid_step + 1e-12:
                raise AssertionError(
                    f"expected-risk grid optimum {expected_factor} does not match "
                    f"closed form {theory} at a={a}, delta={delta}"
                )
            grid_oracle[i, j] = median
            grid_theory[i, j] = theory
            grid_opportunity[i, j] = opportunity
            grid_expected_oracle[i, j] = expected_factor
            grid_expected_opportunity[i, j] = expected_opportunity
            row.append(median)
            records.append({
                "ratio": a, "delta": delta, "n1": n1, "n2": args.n2,
                "oracle_median": median,
                "oracle_iqr": [float(np.percentile(factors, 25)),
                               float(np.percentile(factors, 75))],
                "oracle_opportunity_pct": opportunity,
                "oracle_type": "conditional_finite_sample_population_risk",
                "closed_form": theory,
                "expected_risk_grid_oracle": expected_factor,
                "expected_risk_opportunity_pct": expected_opportunity,
                "abs_error": abs(median - theory),
            })
        print(f"{a:>6} " + "".join(f"{v:>8.3f}" for v in row))

    print("\nOracle opportunity, i.e. relative risk reduction over f=1 "
          "(median over seeds, %)\n")
    print(header)
    print("-" * len(header))
    for i, a in enumerate(ratios):
        print(f"{a:>6} " + "".join(f"{grid_opportunity[i, j]:>8.2f}"
                                   for j in range(len(deltas))))

    err = np.abs(grid_oracle - grid_theory)
    material = grid_opportunity >= 0.5
    err_material = err[material]
    print(f"\nclosed form  f* = (1+c)/(a+c),  c = 2 d sigma^2 / (||delta||^2 n2)")
    print(f"  mean |Oracle - closed form|, all {err.size} cells        : {err.mean():.4f}")
    print(f"  mean |Oracle - closed form|, {material.sum()} material cells : "
          f"{err_material.mean():.4f}")
    print(f"  Pearson r, all cells      : "
          f"{np.corrcoef(grid_oracle.ravel(), grid_theory.ravel())[0,1]:.4f}")
    print(f"  Pearson r, material cells : "
          f"{np.corrcoef(grid_oracle[material], grid_theory[material])[0,1]:.4f}")

    # The two limiting regimes. The conditional finite-sample Oracle need not
    # select f=1 on every no-mismatch realization even though expected risk does.
    no_drift = grid_oracle[:, 0]
    no_drift_material = grid_opportunity[:, 0] >= 0.5
    print(f"\n  delta = 0 column: Oracle argmin wanders over "
          f"[{no_drift.min():.3f}, {no_drift.max():.3f}] while its opportunity "
          f"stays at [{grid_opportunity[:,0].min():.3f}%, "
          f"{grid_opportunity[:,0].max():.3f}%]")
    print(f"    -> {no_drift_material.sum()}/{len(ratios)} cells exceed the 0.5% "
          "conditional-Oracle gate despite zero mismatch; this is selection on "
          "finite-sample estimation noise, not expected-risk mismatch evidence.")
    strong = grid_oracle[:, -1]
    equalizing = np.array([min(1.0, 1.0 / a) for a in ratios])
    print(f"  delta = {deltas[-1]} column vs information-equalizing 1/a: "
          f"mean |diff| = {np.abs(strong - equalizing).mean():.4f}")
    print(f"    -> in the mismatch-dominated limit the optimum equalizes "
          f"effective information, f* n1 = n2.")

    payload = {
        "status": "EXPLORATORY_DIAGNOSTIC_NOT_PREREGISTERED",
        "note": ("Conditional finite-sample Oracle factors minimize true population "
                 "risk after each training realization and are not attainable by any "
                 "train-only rule. They are not the same estimand as the closed-form "
                 "minimizer of risk averaged over training noise."),
        "setting": {"dim": args.dim, "n2": args.n2, "sigma": args.sigma,
                    "lambda": args.lam, "n_test": args.n_test,
                    "seeds": args.seeds,
                    "grid_step": 0.005},
        "closed_form": "f* = (1+c)/(a+c), c = 2 d sigma^2 / (delta^2 n2), clipped to (0,1]",
        "ratios": ratios, "deltas": deltas,
        "mean_abs_error": float(err.mean()),
        "mean_abs_error_material_cells": float(err_material.mean()),
        "material_cell_count": int(material.sum()),
        "zero_mismatch_material_conditional_oracle_count": int(no_drift_material.sum()),
        "max_abs_error": float(err.max()),
        "records": records,
    }
    (out_dir / "history_dominance.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    # ---------------- phase diagram ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(
            "rrcl", ["#C2410C", "#E8B98A", "#F5F7F8", "#7FB3C0", "#0B7285"][::-1])
        immaterial = grid_opportunity < 0.5
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
        for ax, data, title in (
            (axes[0], grid_oracle,
             r"Conditional Oracle $f^\star$ (population risk)"),
            (axes[1], grid_theory, r"Closed form $\min\{1,(1+c)/(a+c)\}$"),
        ):
            im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                           vmin=0.0, vmax=1.0)
            # mark cells whose Oracle opportunity is below the material gate:
            # there the risk curve is flat and the argmin carries no information
            for (i, j), flag in np.ndenumerate(immaterial):
                if flag:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        hatch="////", edgecolor="#55606A", linewidth=0.0))
            ax.set_xticks(range(len(deltas)))
            ax.set_xticklabels([f"{d:g}" for d in deltas], fontsize=8)
            ax.set_yticks(range(len(ratios)))
            ax.set_yticklabels([str(r) for r in ratios], fontsize=8)
            ax.set_xlabel(r"mapping mismatch $\|\Delta\|$", fontsize=9)
            ax.set_title(title, fontsize=10)
        axes[0].set_ylabel(r"history ratio $a=n_1/n_2$", fontsize=9)
        axes[0].plot([], [], marker="s", linestyle="none", markersize=9,
                     markerfacecolor="white", markeredgecolor="#55606A",
                     label=r"Oracle opportunity $<0.5\%$")
        axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9,
                       handletextpad=0.4, borderpad=0.35)
        fig.colorbar(im, ax=axes, label=r"optimal factor $f^\star$")
        fig.savefig(out_dir / "history_dominance_phase.pdf")
        fig.savefig(out_dir / "history_dominance_phase.png", dpi=160)
        print(f"\nfigure  -> {out_dir/'history_dominance_phase.pdf'}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[figure skipped: {exc}]")

    print(f"records -> {out_dir/'history_dominance.json'}")


if __name__ == "__main__":
    main()
