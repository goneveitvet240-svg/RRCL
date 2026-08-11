"""Pure-numpy core for the frozen RRCL TMLR synthetic protocol v1.

The module exposes deterministic scenario generation and method evaluation.
It does not write files, inspect git, or choose protocol parameters.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from precision_shrinkage import (
    causal_final_weights,
    gamma_weights,
    quadratic_mse,
    solve_ridge,
    weighted_system,
)
from run_theory_predict import predicted_factor
from vff_baselines import paleologu_vff_factor


PROTOCOL = "rrcl-tmlr-synthetic-v1"
CONFIG_SCHEMA = "rrcl-tmlr-synthetic-config-v1"
RAW_SCHEMA = "rrcl-tmlr-synthetic-raw-v1"
METHODS = (
    "f1",
    "fixed_f_oracle",
    "fstar_pred",
    "precision_weighting",
    "shrinkage_gamma",
    "vff_rls",
)
FACTOR_NAMES = (
    "label_scale",
    "reliability",
    "mapping_change",
    "covariance_anisotropy",
)
NUMERIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DomainData:
    X_train: np.ndarray
    y_train_normalized: np.ndarray
    y_train_raw: np.ndarray
    X_test: np.ndarray
    y_test_normalized: np.ndarray
    y_test_raw: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    sigma: float
    scale: float


def canonical_json_sha(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config: dict) -> dict:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "wrong config schema")
    _require(config.get("protocol") == PROTOCOL, "wrong synthetic protocol")
    dims = config.get("dimensions", {})
    _require(dims.get("domains") == 4, "v1 requires exactly four domains")
    _require(dims.get("features") == 8, "v1 requires exactly eight features")
    for key in ("n_train_per_domain", "n_test_per_domain"):
        _require(int(dims.get(key, 0)) >= 10, f"{key} must be at least 10")
    fractions = config.get("split_fractions", {})
    _require(set(fractions) == {"fit", "precision", "gamma"}, "wrong split roles")
    _require(
        math.isclose(sum(map(float, fractions.values())), 1.0),
        "split fractions must sum to one",
    )
    _require(all(float(value) > 0 for value in fractions.values()), "empty split role")
    for grid_name in ("factor_grid", "gamma_grid"):
        grid = list(map(float, config.get(grid_name, [])))
        _require(grid == sorted(set(grid)), f"{grid_name} must be unique and sorted")
        _require(grid and grid[0] == 0.0 and grid[-1] == 1.0, f"{grid_name} endpoints")
    seeds = config.get("seeds", [])
    _require(len(seeds) == len(set(seeds)) and len(seeds) > 0, "seeds must be unique")
    orders = config.get("orders", {})
    _require(set(orders) == {"forward", "reverse"}, "v1 requires two named orders")
    for name, order in orders.items():
        _require(sorted(order) == [0, 1, 2, 3], f"invalid {name} order")
    _require(orders["reverse"] == list(reversed(orders["forward"])), "orders not reversed")
    levels = config.get("factor_levels", {})
    _require(set(levels) == set(FACTOR_NAMES), "factor levels do not match protocol")
    _require(float(config.get("ridge_lambda", 0)) > 0, "ridge_lambda must be positive")
    _require(tuple(config.get("method_roster", ())) == METHODS, "method roster mismatch")
    _require(
        set(config.get("secondary_methods", {})) == {"SIFt", "DOS-ELM"},
        "secondary method decisions missing",
    )
    _require(
        all(
            "not computationally matched" in reason
            for reason in config["secondary_methods"].values()
        ),
        "secondary exclusions must record computational mismatch",
    )
    bootstrap = config.get("bootstrap", {})
    _require(int(bootstrap.get("replicates", 0)) > 0, "bootstrap replicates must be positive")
    _require(bootstrap.get("interval") == [0.025, 0.975], "v1 interval must be 95%")
    return config


def load_config(path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def factor_cells():
    for bits in itertools.product((0, 1), repeat=4):
        yield dict(zip(FACTOR_NAMES, bits))


def cell_id(cell: dict) -> str:
    return "".join(str(int(cell[name])) for name in FACTOR_NAMES)


def _hadamard(order: int) -> np.ndarray:
    _require(order > 0 and order & (order - 1) == 0, "Hadamard order must be power of two")
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(order)


def _base_vectors() -> tuple[np.ndarray, np.ndarray]:
    beta = np.asarray([1, -1, .75, -.75, .5, -.5, .25, -.25], dtype=np.float64)
    beta /= math.sqrt(3.75)
    direction = np.asarray([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    direction /= math.sqrt(8.0)
    if abs(float(beta @ direction)) > NUMERIC_TOLERANCE:
        raise AssertionError("mapping direction is not orthogonal to beta0")
    return beta, direction


def _covariance(config: dict, domain: int, enabled: bool) -> np.ndarray:
    d = int(config["dimensions"]["features"])
    if not enabled:
        return np.eye(d)
    spec = config["factor_levels"]["covariance_anisotropy"]
    spectrum = np.asarray(spec["spectrum"], dtype=np.float64)
    shift = int(spec["cyclic_shift_per_domain"]) * int(domain)
    spectrum = np.roll(spectrum, shift)
    Q = _hadamard(d)
    return Q @ np.diag(spectrum) @ Q.T


def _sqrt_covariance(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    if np.min(values) <= 0:
        raise ValueError("covariance must be positive definite")
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def generate_domains(config: dict, cell: dict, seed: int) -> list[DomainData]:
    validate_config(config)
    beta0, direction = _base_vectors()
    dims = config["dimensions"]
    d = int(dims["features"])
    n_train = int(dims["n_train_per_domain"])
    n_test = int(dims["n_test_per_domain"])
    levels = config["factor_levels"]
    scales = levels["label_scale"]["on" if cell["label_scale"] else "off"]
    sigmas = levels["reliability"]["on" if cell["reliability"] else "off"]
    mapping = levels["mapping_change"]
    domains = []
    for domain in range(int(dims["domains"])):
        covariance = _covariance(config, domain, bool(cell["covariance_anisotropy"]))
        root = _sqrt_covariance(covariance)
        beta = beta0.copy()
        if cell["mapping_change"]:
            beta += (
                float(mapping["amplitude"])
                * float(mapping["coefficients"][domain])
                * direction
            )
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(domain)]))
        X_train = rng.standard_normal((n_train, d)) @ root.T
        eps_train = rng.standard_normal(n_train)
        X_test = rng.standard_normal((n_test, d)) @ root.T
        eps_test = rng.standard_normal(n_test)
        sigma = float(sigmas[domain])
        scale = float(scales[domain])
        y_train_normalized = X_train @ beta + sigma * eps_train
        y_test_normalized = X_test @ beta + sigma * eps_test
        domains.append(
            DomainData(
                X_train=X_train,
                y_train_normalized=y_train_normalized,
                y_train_raw=scale * y_train_normalized,
                X_test=X_test,
                y_test_normalized=y_test_normalized,
                y_test_raw=scale * y_test_normalized,
                beta=beta,
                covariance=covariance,
                sigma=sigma,
                scale=scale,
            )
        )
    return domains


def _augmented(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def _quadratic(X: np.ndarray, y: np.ndarray) -> dict:
    Xa = _augmented(X)
    target = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return {
        "R": Xa.T @ Xa,
        "C": Xa.T @ target,
        "S": float(np.sum(target * target)),
        "n": int(Xa.shape[0]),
        "X": Xa,
        "y": target,
    }


def domain_statistics(config: dict, domains: list[DomainData], *, raw: bool = False):
    n_train = int(config["dimensions"]["n_train_per_domain"])
    fractions = config["split_fractions"]
    fit_end = int(n_train * float(fractions["fit"]))
    precision_end = fit_end + int(n_train * float(fractions["precision"]))
    if fit_end <= 0 or precision_end <= fit_end or precision_end >= n_train:
        raise ValueError("split fractions create an empty role")
    records = []
    for domain in domains:
        y = domain.y_train_raw if raw else domain.y_train_normalized
        records.append(
            {
                "full": _quadratic(domain.X_train, y),
                "fit": _quadratic(domain.X_train[:fit_end], y[:fit_end]),
                "precision": _quadratic(
                    domain.X_train[fit_end:precision_end], y[fit_end:precision_end]
                ),
                "gamma": _quadratic(
                    domain.X_train[precision_end:], y[precision_end:]
                ),
            }
        )
    return records


def _stats_pair(record: dict) -> tuple[np.ndarray, np.ndarray]:
    return record["R"], record["C"]


def _select_largest_factor(rows: list[dict], key: str) -> dict:
    minimum = min(float(row[key]) for row in rows)
    tolerance = NUMERIC_TOLERANCE * max(1.0, abs(minimum))
    return max(
        (row for row in rows if float(row[key]) <= minimum + tolerance),
        key=lambda row: float(row["factor"]),
    )


def _select_smallest_gamma(rows: list[dict], key: str) -> dict:
    minimum = min(float(row[key]) for row in rows)
    tolerance = NUMERIC_TOLERANCE * max(1.0, abs(minimum))
    return min(
        (row for row in rows if float(row[key]) <= minimum + tolerance),
        key=lambda row: float(row["gamma"]),
    )


def _temporal_weights(order: list[int], factor: float) -> np.ndarray:
    weights = np.zeros(len(order), dtype=np.float64)
    count = len(order)
    for position, physical_domain in enumerate(order):
        weights[physical_domain] = float(factor) ** (count - 1 - position)
    return weights


def _solve(stats: list[dict], weights: np.ndarray, lam: float, role: str = "full"):
    pairs = [_stats_pair(record[role]) for record in stats]
    R, C = weighted_system(pairs, weights)
    return solve_ridge(R, C, lam)


def population_risks(W: np.ndarray, domains: list[DomainData], *, raw: bool = False):
    vector = np.asarray(W[:-1, 0], dtype=np.float64)
    intercept = float(W[-1, 0])
    risks = []
    for domain in domains:
        scale = domain.scale if raw else 1.0
        delta = vector - scale * domain.beta
        noise = scale * domain.sigma
        risks.append(
            float(delta @ domain.covariance @ delta + intercept**2 + noise**2)
        )
    return risks


def empirical_risks(W: np.ndarray, domains: list[DomainData], *, raw: bool = False):
    risks = []
    for domain in domains:
        y = domain.y_test_raw if raw else domain.y_test_normalized
        prediction = _augmented(domain.X_test) @ W
        residual = prediction[:, 0] - y
        risks.append(float(np.mean(residual * residual)))
    return risks


def _method_record(
    method: str,
    W: np.ndarray,
    domains: list[DomainData],
    *,
    details: dict,
) -> dict:
    population = population_risks(W, domains)
    empirical = empirical_risks(W, domains)
    return {
        "method": method,
        "details": details,
        "population_per_domain_mse": population,
        "population_balanced_mse": float(np.mean(population)),
        "empirical_per_domain_mse": empirical,
        "empirical_balanced_mse": float(np.mean(empirical)),
    }


def _estimate_precisions(stats: list[dict], lam: float, epsilon: float = 1e-8):
    precisions, mses = [], []
    for record in stats:
        W = solve_ridge(record["fit"]["R"], record["fit"]["C"], lam)
        heldout = record["precision"]
        mse = quadratic_mse(W, heldout["R"], heldout["C"], heldout["S"], heldout["n"])
        mses.append(float(mse))
        precisions.append(1.0 / max(float(mse), epsilon))
    return np.asarray(precisions), mses


def _fstar_pred(stats: list[dict], order: list[int], lam: float, grid):
    per_domain = []
    d_aug = stats[0]["full"]["R"].shape[0]
    for physical in order:
        record = stats[physical]
        full = record["full"]
        W = solve_ridge(full["R"], full["C"], lam)
        mu = float(np.trace(full["R"]) / d_aug)
        fit_W = solve_ridge(record["fit"]["R"], record["fit"]["C"], lam)
        heldout = record["precision"]
        residual = quadratic_mse(
            fit_W, heldout["R"], heldout["C"], heldout["S"], heldout["n"]
        )
        target_variance = max(
            float(np.var(record["full"]["y"][:, 0])), 1e-3
        )
        feature_mass = mu / max(int(full["n"]), 1)
        per_domain.append(
            {
                "w": W,
                "mu": mu,
                "r": float(residual),
                "rho": feature_mass / target_variance,
                "n": int(full["n"]),
            }
        )
    return predicted_factor(per_domain, d_aug, lam, grid=grid)


def _shrinkage_gamma(
    stats: list[dict], order: list[int], precisions: np.ndarray, lam: float, grid
):
    selections = []
    for boundary in range(len(order)):
        seen = order[: boundary + 1]
        prefix_precisions = precisions[seen]
        normalized = causal_final_weights(prefix_precisions)
        rows = []
        for gamma in grid:
            prefix_weights = gamma_weights(normalized, gamma)
            R, C = weighted_system(
                [_stats_pair(stats[index]["fit"]) for index in seen], prefix_weights
            )
            W = solve_ridge(R, C, lam)
            val = [
                quadratic_mse(
                    W,
                    stats[index]["gamma"]["R"],
                    stats[index]["gamma"]["C"],
                    stats[index]["gamma"]["S"],
                    stats[index]["gamma"]["n"],
                )
                for index in seen
            ]
            rows.append({"gamma": float(gamma), "objective": float(np.mean(val))})
        selected = _select_smallest_gamma(rows, "objective")
        selections.append(
            {
                "boundary": boundary,
                "seen_physical_domains": list(seen),
                "gamma": float(selected["gamma"]),
                "objective": float(selected["objective"]),
            }
        )
    final_gamma = selections[-1]["gamma"]
    normalized = causal_final_weights(precisions)
    physical_weights = np.zeros(len(order), dtype=np.float64)
    prefix_weights = gamma_weights(normalized[order], final_gamma)
    for position, physical in enumerate(order):
        physical_weights[physical] = prefix_weights[position]
    return final_gamma, physical_weights, selections


def _vff(
    stats: list[dict], order: list[int], lam: float, parameters: dict
):
    dimension = stats[0]["full"]["R"].shape[0]
    R_state = np.zeros((dimension, dimension), dtype=np.float64)
    C_state = np.zeros((dimension, 1), dtype=np.float64)
    factors, diagnostics = [], []
    for position, physical in enumerate(order):
        record = stats[physical]
        if position == 0:
            factor = 1.0
            diagnostic = None
        else:
            W_old = solve_ridge(R_state, C_state, lam)
            W_local = solve_ridge(record["fit"]["R"], record["fit"]["C"], lam)
            heldout = record["precision"]
            error_power = quadratic_mse(
                W_old, heldout["R"], heldout["C"], heldout["S"], heldout["n"]
            )
            noise_power = quadratic_mse(
                W_local, heldout["R"], heldout["C"], heldout["S"], heldout["n"]
            )
            covariance = np.linalg.inv(R_state + lam * np.eye(dimension))
            leverage = np.einsum(
                "ij,jk,ik->i", heldout["X"], covariance, heldout["X"]
            )
            leverage_power = float(np.mean(leverage * leverage))
            factor = paleologu_vff_factor(
                error_power,
                noise_power,
                leverage_power,
                gamma=float(parameters["gamma"]),
                xi=float(parameters["xi"]),
                f_min=float(parameters["f_min"]),
                f_max=float(parameters["f_max"]),
            )
            diagnostic = {
                "error_power": float(error_power),
                "noise_power": float(noise_power),
                "leverage_power": leverage_power,
            }
        factors.append(float(factor))
        diagnostics.append(diagnostic)
        R_state = float(factor) * R_state + record["full"]["R"]
        C_state = float(factor) * C_state + record["full"]["C"]
    return solve_ridge(R_state, C_state, lam), factors, diagnostics


def data_sha(domains: list[DomainData]) -> str:
    digest = hashlib.sha256()
    for domain in domains:
        for array in (
            domain.X_train,
            domain.y_train_normalized,
            domain.X_test,
            domain.y_test_normalized,
        ):
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def evaluate_scenario(config: dict, cell: dict, seed: int, order_name: str) -> dict:
    validate_config(config)
    order = list(config["orders"][order_name])
    domains = generate_domains(config, cell, seed)
    stats = domain_statistics(config, domains)
    raw_stats = domain_statistics(config, domains, raw=True)
    lam = float(config["ridge_lambda"])
    factor_grid = list(map(float, config["factor_grid"]))
    gamma_grid = list(map(float, config["gamma_grid"]))
    results = {}

    f1_weights = np.ones(len(order), dtype=np.float64)
    W_f1 = _solve(stats, f1_weights, lam)
    results["f1"] = _method_record("f1", W_f1, domains, details={"weights": f1_weights.tolist()})

    oracle_curve = []
    raw_curve = []
    for factor in factor_grid:
        weights = _temporal_weights(order, factor)
        W = _solve(stats, weights, lam)
        risk = float(np.mean(population_risks(W, domains)))
        oracle_curve.append({"factor": factor, "population_balanced_mse": risk})
        W_raw = _solve(raw_stats, weights, lam)
        raw_curve.append(
            {
                "factor": factor,
                "population_balanced_mse": float(
                    np.mean(population_risks(W_raw, domains, raw=True))
                ),
            }
        )
    oracle = _select_largest_factor(oracle_curve, "population_balanced_mse")
    oracle_factor = float(oracle["factor"])
    oracle_weights = _temporal_weights(order, oracle_factor)
    W_oracle = _solve(stats, oracle_weights, lam)
    results["fixed_f_oracle"] = _method_record(
        "fixed_f_oracle",
        W_oracle,
        domains,
        details={
            "diagnostic_only": True,
            "factor": oracle_factor,
            "weights": oracle_weights.tolist(),
            "curve": oracle_curve,
            "raw_scale_curve": raw_curve,
        },
    )

    f_pred, prediction_curve = _fstar_pred(stats, order, lam, factor_grid)
    fstar_weights = _temporal_weights(order, f_pred)
    W_fstar = _solve(stats, fstar_weights, lam)
    results["fstar_pred"] = _method_record(
        "fstar_pred",
        W_fstar,
        domains,
        details={
            "factor": float(f_pred),
            "weights": fstar_weights.tolist(),
            "train_objective_curve": prediction_curve,
        },
    )

    precisions, precision_mses = _estimate_precisions(stats, lam)
    precision_weights = causal_final_weights(precisions)
    W_precision = _solve(stats, precision_weights, lam)
    results["precision_weighting"] = _method_record(
        "precision_weighting",
        W_precision,
        domains,
        details={
            "precisions": precisions.tolist(),
            "heldout_mse": precision_mses,
            "weights": precision_weights.tolist(),
        },
    )

    gamma, shrinkage_weights, selections = _shrinkage_gamma(
        stats, order, precisions, lam, gamma_grid
    )
    W_gamma = _solve(stats, shrinkage_weights, lam)
    results["shrinkage_gamma"] = _method_record(
        "shrinkage_gamma",
        W_gamma,
        domains,
        details={
            "gamma": gamma,
            "weights": shrinkage_weights.tolist(),
            "boundary_selections": selections,
        },
    )

    W_vff, vff_factors, vff_diagnostics = _vff(stats, order, lam, config["vff"])
    results["vff_rls"] = _method_record(
        "vff_rls",
        W_vff,
        domains,
        details={"boundary_factors": vff_factors, "diagnostics": vff_diagnostics},
    )

    f1_risk = results["f1"]["population_balanced_mse"]
    oracle_risk = results["fixed_f_oracle"]["population_balanced_mse"]
    for name, record in results.items():
        risk = record["population_balanced_mse"]
        record["relative_utility_vs_f1"] = float((f1_risk - risk) / f1_risk)
        record["relative_regret_vs_oracle"] = float((risk - oracle_risk) / f1_risk)
        record["relative_domain_harm_vs_f1"] = [
            float((value - baseline) / baseline)
            for value, baseline in zip(
                record["population_per_domain_mse"],
                results["f1"]["population_per_domain_mse"],
            )
        ]

    return {
        "schema_version": RAW_SCHEMA,
        "protocol": PROTOCOL,
        "cell": dict(cell),
        "cell_id": cell_id(cell),
        "seed": int(seed),
        "order_name": order_name,
        "order": order,
        "data_sha256": data_sha(domains),
        "domain_parameters": [
            {
                "physical_domain": index,
                "scale": domain.scale,
                "sigma": domain.sigma,
                "beta": domain.beta.tolist(),
                "covariance": domain.covariance.tolist(),
            }
            for index, domain in enumerate(domains)
        ],
        "methods": results,
    }
