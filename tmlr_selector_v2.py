"""Deterministic core for the preregistered RRCL TMLR selector-v2 study.

The module is pure NumPy and does not inspect git or write files.  It keeps the
meta-test generator behind a separate function so the formal runner can write
an exclusive hyperparameter-selection lock before any meta-test task exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analytic_forgetting_baselines import SIFtRLS, dos_elm_factor_update
from precision_shrinkage import quadratic_mse, solve_ridge
from run_theory_predict import predicted_factor


PROTOCOL = "rrcl-tmlr-selector-v2"
CONFIG_SCHEMA = "rrcl-tmlr-selector-v2-config-v1"
RESULT_SCHEMA = "rrcl-tmlr-selector-v2-result-v1"
FAMILIES = (
    "recent_domains_more_reliable",
    "recent_domains_less_reliable",
    "mapping_covariance_interaction",
    "mixed_heterogeneity",
)
METHODS = (
    "f1",
    "fixed_f_oracle",
    "fstar_pred",
    "learned_knn",
    "sift_rls",
    "dos_elm_style",
)


@dataclass(frozen=True)
class Domain:
    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    sigma: float


@dataclass(frozen=True)
class Task:
    seed: int
    family: str
    domains: tuple[Domain, ...]


def canonical_sha(payload) -> str:
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
    _require(config.get("protocol") == PROTOCOL, "wrong protocol")
    dims = config.get("dimensions", {})
    _require(dims.get("domains") == 4, "selector-v2 requires four domains")
    _require(dims.get("features") == 8, "selector-v2 requires eight features")
    _require(int(dims.get("n_train_per_domain", 0)) >= 16, "too few samples")
    _require(float(config.get("ridge_lambda", 0)) > 0, "ridge must be positive")
    fit = float(config.get("split_fraction_fit", 0))
    _require(0.5 <= fit < 1.0, "fit fraction must be in [0.5, 1)")
    factor_grid = list(map(float, config.get("factor_grid", [])))
    _require(factor_grid == sorted(set(factor_grid)), "factor grid not sorted")
    _require(factor_grid[0] == 0.0 and factor_grid[-1] == 1.0, "factor endpoints")
    sift_grid = list(map(float, config.get("sift_domain_factor_grid", [])))
    _require(sift_grid == sorted(set(sift_grid)), "SIFt grid not sorted")
    _require(sift_grid[0] > 0 and sift_grid[-1] == 1.0, "invalid SIFt grid")
    _require(
        config.get("sift", {}).get("sample_factor_conversion")
        == "domain_factor**(1/n_train_per_domain)",
        "SIFt sample-factor conversion is not frozen",
    )
    k_grid = list(map(int, config.get("knn_k_grid", [])))
    _require(k_grid == sorted(set(k_grid)) and k_grid[0] > 0, "invalid k grid")
    splits = config.get("meta_splits", {})
    _require(set(splits) == {"train", "validation", "test"}, "missing split")
    seed_sets = []
    for name in ("train", "validation", "test"):
        spec = splits[name]
        count = int(spec.get("count", 0))
        start = int(spec.get("seed_start", 0))
        _require(count > 0 and start > 0, f"invalid {name} seeds")
        _require(count % len(FAMILIES) == 0, f"{name} must balance four families")
        seed_sets.append(set(range(start, start + count)))
    _require(not (seed_sets[0] & seed_sets[1]), "train/validation seed overlap")
    _require(not (seed_sets[0] & seed_sets[2]), "train/test seed overlap")
    _require(not (seed_sets[1] & seed_sets[2]), "validation/test seed overlap")
    _require(max(k_grid) <= len(seed_sets[0]), "k exceeds meta-train count")
    generator = config.get("generator", {})
    _require(tuple(generator.get("scenario_families", ())) == FAMILIES, "family roster")
    for name in (
        "mapping_amplitude",
        "noise_base",
        "noise_ratio",
        "covariance_eigenvalue",
    ):
        bounds = list(map(float, generator.get(name, [])))
        _require(len(bounds) == 2 and 0 <= bounds[0] < bounds[1], f"bad {name}")
    _require(int(config["bootstrap"]["replicates"]) > 0, "bad bootstrap count")
    return config


def load_config(path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def split_seeds(config: dict, split: str) -> list[int]:
    spec = config["meta_splits"][split]
    start = int(spec["seed_start"])
    return list(range(start, start + int(spec["count"])))


def _log_uniform(rng, low: float, high: float, size=None):
    return np.exp(rng.uniform(math.log(low), math.log(high), size=size))


def _orthogonal(rng, dimension: int) -> np.ndarray:
    matrix = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def generate_task(config: dict, seed: int) -> Task:
    """Generate one i.i.d. train/population-risk task from its seed."""

    validate_config(config)
    rng = np.random.default_rng(int(seed))
    dims = config["dimensions"]
    d = int(dims["features"])
    n = int(dims["n_train_per_domain"])
    family = FAMILIES[int(seed) % len(FAMILIES)]
    generator = config["generator"]

    beta0 = np.asarray([1, -1, .75, -.75, .5, -.5, .25, -.25], dtype=np.float64)
    beta0 /= np.linalg.norm(beta0)
    direction = rng.standard_normal(d)
    direction -= beta0 * float(direction @ beta0)
    direction /= np.linalg.norm(direction)
    amplitude = rng.uniform(*map(float, generator["mapping_amplitude"]))
    coefficients = np.linspace(-1.0, 1.0, int(dims["domains"]))

    noise_base = float(_log_uniform(rng, *map(float, generator["noise_base"])))
    noise_ratio = float(_log_uniform(rng, *map(float, generator["noise_ratio"])))
    if family == "recent_domains_more_reliable":
        sigmas = noise_base * np.geomspace(noise_ratio, 1.0, 4)
        amplitude *= 0.35
    elif family == "recent_domains_less_reliable":
        sigmas = noise_base * np.geomspace(1.0, noise_ratio, 4)
        amplitude *= 0.35
    elif family == "mapping_covariance_interaction":
        sigmas = noise_base * _log_uniform(rng, 0.8, 1.25, size=4)
        amplitude = max(amplitude, 0.75)
    else:
        sigmas = noise_base * _log_uniform(rng, 1.0 / noise_ratio, noise_ratio, size=4)

    q = _orthogonal(rng, d)
    eig_low, eig_high = map(float, generator["covariance_eigenvalue"])
    base_spectrum = _log_uniform(rng, eig_low, eig_high, size=d)
    domains = []
    for domain_index in range(4):
        spectrum = np.roll(base_spectrum, 2 * domain_index)
        if family in {"recent_domains_more_reliable", "recent_domains_less_reliable"}:
            spectrum = np.sqrt(spectrum)
        covariance = q @ np.diag(spectrum) @ q.T
        beta = beta0 + amplitude * coefficients[domain_index] * direction
        root = q @ np.diag(np.sqrt(spectrum)) @ q.T
        X = rng.standard_normal((n, d)) @ root.T
        y = X @ beta + float(sigmas[domain_index]) * rng.standard_normal(n)
        domains.append(
            Domain(
                X=np.asarray(X, dtype=np.float64),
                y=np.asarray(y, dtype=np.float64),
                beta=np.asarray(beta, dtype=np.float64),
                covariance=np.asarray(covariance, dtype=np.float64),
                sigma=float(sigmas[domain_index]),
            )
        )
    return Task(seed=int(seed), family=family, domains=tuple(domains))


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
        "n": int(target.shape[0]),
    }


def task_stats(config: dict, task: Task) -> list[dict]:
    n = int(config["dimensions"]["n_train_per_domain"])
    fit_end = int(n * float(config["split_fraction_fit"]))
    records = []
    for domain in task.domains:
        records.append(
            {
                "full": _quadratic(domain.X, domain.y),
                "fit": _quadratic(domain.X[:fit_end], domain.y[:fit_end]),
                "validation": _quadratic(domain.X[fit_end:], domain.y[fit_end:]),
            }
        )
    return records


def _temporal_weights(factor: float, domains: int = 4) -> np.ndarray:
    return np.asarray(
        [float(factor) ** (domains - 1 - index) for index in range(domains)],
        dtype=np.float64,
    )


def _solve_weighted(stats: list[dict], weights, ridge: float) -> np.ndarray:
    R = sum(float(weight) * record["full"]["R"] for weight, record in zip(weights, stats))
    C = sum(float(weight) * record["full"]["C"] for weight, record in zip(weights, stats))
    return solve_ridge(R, C, ridge)


def population_per_domain(task: Task, W: np.ndarray) -> list[float]:
    vector = np.asarray(W[:-1, 0], dtype=np.float64)
    intercept = float(W[-1, 0])
    return [
        float(
            (vector - domain.beta) @ domain.covariance @ (vector - domain.beta)
            + intercept**2
            + domain.sigma**2
        )
        for domain in task.domains
    ]


def _risk_record(task: Task, W: np.ndarray, factor=None) -> dict:
    per_domain = population_per_domain(task, W)
    record = {
        "population_per_domain_mse": per_domain,
        "population_balanced_mse": float(np.mean(per_domain)),
    }
    if factor is not None:
        record["factor"] = float(factor)
    return record


def oracle_curve(config: dict, task: Task, stats: list[dict]) -> list[dict]:
    ridge = float(config["ridge_lambda"])
    rows = []
    for factor in map(float, config["factor_grid"]):
        W = _solve_weighted(stats, _temporal_weights(factor), ridge)
        rows.append(
            {
                "factor": factor,
                "population_balanced_mse": float(np.mean(population_per_domain(task, W))),
            }
        )
    return rows


def _best_curve_row(rows: list[dict], key="population_balanced_mse") -> dict:
    minimum = min(float(row[key]) for row in rows)
    tolerance = 1e-12 * max(1.0, abs(minimum))
    return max(
        (row for row in rows if float(row[key]) <= minimum + tolerance),
        key=lambda row: float(row["factor"]),
    )


def _fstar_factor(config: dict, stats: list[dict]) -> float:
    ridge = float(config["ridge_lambda"])
    d_aug = stats[0]["full"]["R"].shape[0]
    per_domain = []
    for record in stats:
        full = record["full"]
        W = solve_ridge(full["R"], full["C"], ridge)
        fit_W = solve_ridge(record["fit"]["R"], record["fit"]["C"], ridge)
        validation = record["validation"]
        residual = quadratic_mse(
            fit_W,
            validation["R"],
            validation["C"],
            validation["S"],
            validation["n"],
        )
        target_variance = max(
            full["S"] / full["n"] - float((full["C"][-1, 0] / full["n"]) ** 2),
            1e-3,
        )
        mu = float(np.trace(full["R"]) / d_aug)
        per_domain.append(
            {
                "w": W,
                "mu": mu,
                "r": float(residual),
                "rho": (mu / full["n"]) / target_variance,
                "n": int(full["n"]),
            }
        )
    factor, _ = predicted_factor(
        per_domain,
        d_aug,
        ridge,
        grid=list(map(float, config["factor_grid"])),
    )
    return float(factor)


def summary_features(config: dict, task: Task, stats: list[dict]) -> list[float]:
    """Order-aware invariant train-only task summary for KNN retrieval."""

    ridge = float(config["ridge_lambda"])
    local_weights = []
    residuals = []
    covariances = []
    features = []
    for domain, record in zip(task.domains, stats):
        W = solve_ridge(record["fit"]["R"], record["fit"]["C"], ridge)
        vector = W[:-1, 0]
        validation = record["validation"]
        residual = quadratic_mse(
            W,
            validation["R"],
            validation["C"],
            validation["S"],
            validation["n"],
        )
        sample_covariance = np.cov(domain.X, rowvar=False, ddof=0)
        eigenvalues = np.linalg.eigvalsh(sample_covariance)
        target_variance = max(float(np.var(domain.y)), 1e-12)
        residuals.append(max(float(residual), 1e-12))
        local_weights.append(vector)
        covariances.append(sample_covariance)
        features.extend(
            [
                math.log(max(float(residual), 1e-12)),
                math.log(target_variance),
                math.log(max(float(np.trace(sample_covariance)), 1e-12)),
                math.log(max(float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)), 1.0)),
                math.log(max(float(vector @ vector), 1e-12)),
                math.log(max(float(vector @ sample_covariance @ vector), 1e-12)),
            ]
        )
    for left in range(4):
        for right in range(left + 1, 4):
            a, b = local_weights[left], local_weights[right]
            delta = b - a
            cosine = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
            features.extend(
                [
                    math.log(max(float(delta @ delta), 1e-12)),
                    cosine,
                    math.log(max(float(delta @ covariances[left] @ delta), 1e-12)),
                    math.log(max(float(delta @ covariances[right] @ delta), 1e-12)),
                ]
            )
    for index in range(3):
        features.append(math.log(residuals[index + 1] / residuals[index]))
    array = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite learned-selector summary")
    return array.tolist()


def base_record(config: dict, task: Task) -> dict:
    stats = task_stats(config, task)
    curve = oracle_curve(config, task, stats)
    f1_risk = next(row["population_balanced_mse"] for row in curve if row["factor"] == 1.0)
    oracle = _best_curve_row(curve)
    return {
        "seed": task.seed,
        "family": task.family,
        "train_summary": summary_features(config, task, stats),
        "normalized_oracle_curve": [
            {
                "factor": row["factor"],
                "risk_over_f1": float(row["population_balanced_mse"] / f1_risk),
            }
            for row in curve
        ],
        "oracle_factor": float(oracle["factor"]),
        "oracle_opportunity": float((f1_risk - oracle["population_balanced_mse"]) / f1_risk),
    }


def _standardizer(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([record["train_summary"] for record in records], dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def knn_factor(
    train_records: list[dict],
    query_record: dict,
    mean: np.ndarray,
    scale: np.ndarray,
    k: int,
) -> tuple[float, list[dict]]:
    library = np.asarray([record["train_summary"] for record in train_records], dtype=np.float64)
    query = np.asarray(query_record["train_summary"], dtype=np.float64)
    distances = np.mean(((library - query) / scale) ** 2, axis=1)
    neighbors = np.argsort(distances, kind="stable")[: int(k)]
    factors = [row["factor"] for row in train_records[0]["normalized_oracle_curve"]]
    predicted = []
    for factor_index, factor in enumerate(factors):
        risk = float(
            np.mean(
                [
                    train_records[index]["normalized_oracle_curve"][factor_index]["risk_over_f1"]
                    for index in neighbors
                ]
            )
        )
        predicted.append({"factor": float(factor), "predicted_risk_over_f1": risk})
    selected = _best_curve_row(predicted, key="predicted_risk_over_f1")
    return float(selected["factor"]), predicted


def _sift_risk(config: dict, task: Task, domain_factor: float) -> dict:
    n = int(config["dimensions"]["n_train_per_domain"])
    sample_factor = float(domain_factor) ** (1.0 / n)
    model = SIFtRLS(
        int(config["dimensions"]["features"]),
        d_out=1,
        ridge=float(config["ridge_lambda"]),
        forgetting=sample_factor,
        epsilon=float(config["sift"]["epsilon"]),
        bias=True,
    )
    rank_counts = []
    for domain in task.domains:
        for row, target in zip(domain.X, domain.y):
            info = model.update(row.reshape(1, -1), np.asarray([[target]]))
            rank_counts.append(info.filtered_rank)
    record = _risk_record(task, model.W, factor=domain_factor)
    record.update(
        {
            "sample_factor": sample_factor,
            "state_bytes": model.state_bytes,
            "updates": len(rank_counts),
            "all_updates_rank_one": bool(all(rank == 1 for rank in rank_counts)),
        }
    )
    return record


def _dos_risk(config: dict, task: Task, stats: list[dict]) -> dict:
    dimension = int(config["dimensions"]["features"]) + 1
    ridge = float(config["ridge_lambda"])
    R = np.zeros((dimension, dimension), dtype=np.float64)
    C = np.zeros((dimension, 1), dtype=np.float64)
    factor = float(config["dos_elm"]["initial_factor"])
    previous_score = None
    records = []
    for domain_index, record in enumerate(stats):
        used = factor
        R = used**2 * R + record["full"]["R"]
        C = used**2 * C + record["full"]["C"]
        W = solve_ridge(R, C, ridge)
        validation = record["validation"]
        rmse = math.sqrt(
            max(
                quadratic_mse(
                    W,
                    validation["R"],
                    validation["C"],
                    validation["S"],
                    validation["n"],
                ),
                0.0,
            )
        )
        score = -rmse
        update = None
        if previous_score is not None:
            update = dos_elm_factor_update(
                used,
                previous_score,
                score,
                factor_min=float(config["dos_elm"]["factor_min"]),
                factor_max=1.0,
            )
            factor = update.factor
        records.append(
            {
                "domain_index": domain_index,
                "factor_used": used,
                "statistic_factor_used": used**2,
                "train_validation_score": score,
                "next_factor": None if update is None else update.factor,
            }
        )
        previous_score = score
    result = _risk_record(task, W)
    result["records"] = records
    result["state_bytes"] = int(R.nbytes + C.nbytes + W.nbytes)
    return result


def select_hyperparameters(config: dict, train_records: list[dict], validation_tasks: list[Task]):
    validation_records = [base_record(config, task) for task in validation_tasks]
    mean, scale = _standardizer(train_records)
    k_rows = []
    for k in map(int, config["knn_k_grid"]):
        regrets = []
        for record in validation_records:
            factor, _ = knn_factor(train_records, record, mean, scale, k)
            curve = {row["factor"]: row["risk_over_f1"] for row in record["normalized_oracle_curve"]}
            regrets.append(curve[factor] - min(curve.values()))
        k_rows.append({"k": k, "mean_normalized_regret": float(np.mean(regrets))})
    minimum = min(row["mean_normalized_regret"] for row in k_rows)
    selected_k = max(
        row["k"]
        for row in k_rows
        if row["mean_normalized_regret"] <= minimum + 1e-12
    )

    sift_rows = []
    for factor in map(float, config["sift_domain_factor_grid"]):
        normalized_risks = []
        for task, base in zip(validation_tasks, validation_records):
            f1_risk = next(
                row["risk_over_f1"]
                for row in base["normalized_oracle_curve"]
                if row["factor"] == 1.0
            )
            if abs(f1_risk - 1.0) > 1e-12:
                raise AssertionError("normalized f1 risk is not one")
            stats = task_stats(config, task)
            W_f1 = _solve_weighted(stats, np.ones(4), float(config["ridge_lambda"]))
            absolute_f1 = float(np.mean(population_per_domain(task, W_f1)))
            normalized_risks.append(
                _sift_risk(config, task, factor)["population_balanced_mse"] / absolute_f1
            )
        sift_rows.append(
            {"domain_factor": factor, "mean_risk_over_f1": float(np.mean(normalized_risks))}
        )
    minimum = min(row["mean_risk_over_f1"] for row in sift_rows)
    selected_sift = max(
        row["domain_factor"]
        for row in sift_rows
        if row["mean_risk_over_f1"] <= minimum + 1e-12
    )
    selection = {
        "knn_k": int(selected_k),
        "sift_domain_factor": float(selected_sift),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "knn_validation_curve": k_rows,
        "sift_validation_curve": sift_rows,
        "meta_train_records_sha256": canonical_sha(train_records),
        "meta_validation_records_sha256": canonical_sha(validation_records),
    }
    return validation_records, selection


def prepare_selection(config: dict):
    train_records = [
        base_record(config, generate_task(config, seed))
        for seed in split_seeds(config, "train")
    ]
    validation_tasks = [
        generate_task(config, seed) for seed in split_seeds(config, "validation")
    ]
    validation_records, selection = select_hyperparameters(
        config, train_records, validation_tasks
    )
    return train_records, validation_records, selection


def _attach_relative_metrics(methods: dict):
    f1 = methods["f1"]["population_balanced_mse"]
    oracle = methods["fixed_f_oracle"]["population_balanced_mse"]
    f1_domains = methods["f1"]["population_per_domain_mse"]
    for record in methods.values():
        risk = record["population_balanced_mse"]
        record["relative_utility_vs_f1"] = float((f1 - risk) / f1)
        record["relative_regret_vs_oracle"] = float((risk - oracle) / f1)
        record["relative_domain_harm_vs_f1"] = [
            float((value - baseline) / baseline)
            for value, baseline in zip(record["population_per_domain_mse"], f1_domains)
        ]


def evaluate_test_task(
    config: dict,
    task: Task,
    train_records: list[dict],
    selection: dict,
) -> dict:
    stats = task_stats(config, task)
    base = base_record(config, task)
    ridge = float(config["ridge_lambda"])
    methods = {}
    W_f1 = _solve_weighted(stats, np.ones(4), ridge)
    methods["f1"] = _risk_record(task, W_f1, factor=1.0)
    oracle_factor = float(base["oracle_factor"])
    W_oracle = _solve_weighted(stats, _temporal_weights(oracle_factor), ridge)
    methods["fixed_f_oracle"] = _risk_record(task, W_oracle, factor=oracle_factor)
    fstar = _fstar_factor(config, stats)
    methods["fstar_pred"] = _risk_record(
        task, _solve_weighted(stats, _temporal_weights(fstar), ridge), factor=fstar
    )
    mean = np.asarray(selection["feature_mean"], dtype=np.float64)
    scale = np.asarray(selection["feature_scale"], dtype=np.float64)
    learned_factor, predicted_curve = knn_factor(
        train_records, base, mean, scale, int(selection["knn_k"])
    )
    methods["learned_knn"] = _risk_record(
        task,
        _solve_weighted(stats, _temporal_weights(learned_factor), ridge),
        factor=learned_factor,
    )
    methods["learned_knn"]["predicted_curve"] = predicted_curve
    methods["sift_rls"] = _sift_risk(
        config, task, float(selection["sift_domain_factor"])
    )
    methods["dos_elm_style"] = _dos_risk(config, task, stats)
    _attach_relative_metrics(methods)
    return {
        "seed": task.seed,
        "family": task.family,
        "oracle_opportunity": methods["fixed_f_oracle"]["relative_utility_vs_f1"],
        "methods": methods,
    }


def evaluate_meta_test(config: dict, train_records: list[dict], selection: dict):
    return [
        evaluate_test_task(config, generate_task(config, seed), train_records, selection)
        for seed in split_seeds(config, "test")
    ]


def _bootstrap(values, config: dict, salt: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95_lower": None, "ci95_upper": None}
    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(bootstrap["seed"]), int(salt)])
    )
    indices = rng.integers(
        0, values.size, size=(int(bootstrap["replicates"]), values.size)
    )
    means = np.mean(values[indices], axis=1)
    lower, upper = map(float, bootstrap["interval"])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(means, lower)),
        "ci95_upper": float(np.quantile(means, upper)),
    }


def summarize(config: dict, test_records: list[dict]) -> dict:
    threshold = float(config["decision_thresholds"]["opportunity_relative_gain"])
    eligible = [row for row in test_records if row["oracle_opportunity"] >= threshold]
    subsets = {"all": test_records, "opportunity_present": eligible}
    summaries = {}
    salt = 0
    for subset_name, rows in subsets.items():
        method_rows = {}
        for method in METHODS:
            utility = _bootstrap(
                [row["methods"][method]["relative_utility_vs_f1"] for row in rows],
                config,
                salt,
            )
            salt += 1
            regret = _bootstrap(
                [row["methods"][method]["relative_regret_vs_oracle"] for row in rows],
                config,
                salt,
            )
            salt += 1
            mean_harm = (
                np.mean(
                    [row["methods"][method]["relative_domain_harm_vs_f1"] for row in rows],
                    axis=0,
                ).tolist()
                if rows
                else []
            )
            method_rows[method] = {
                "utility_vs_f1": utility,
                "regret_vs_oracle": regret,
                "mean_relative_domain_harm": mean_harm,
                "worst_mean_domain_harm": max(mean_harm) if mean_harm else None,
            }
            if method in {"f1", "fixed_f_oracle", "fstar_pred", "learned_knn"}:
                method_rows[method]["factor_mae"] = (
                    float(
                        np.mean(
                            [
                                abs(
                                    float(row["methods"][method].get("factor", 1.0))
                                    - float(row["methods"]["fixed_f_oracle"]["factor"])
                                )
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else None
                )
        summaries[subset_name] = {
            "task_count": len(rows),
            "methods": method_rows,
        }

    opportunity = summaries["opportunity_present"]["methods"]
    learned = opportunity["learned_knn"]
    fstar = opportunity["fstar_pred"]
    all_methods = summaries["all"]["methods"]
    maximum_harm = float(config["decision_thresholds"]["maximum_mean_domain_harm"])
    gates = {
        "factor_mae_better_than_f1": bool(
            all_methods["learned_knn"]["factor_mae"] < all_methods["f1"]["factor_mae"]
        ),
        "opportunity_regret_better_than_fstar": bool(
            eligible
            and learned["regret_vs_oracle"]["mean"] < fstar["regret_vs_oracle"]["mean"]
        ),
        "opportunity_utility_ci_lower_positive": bool(
            eligible and learned["utility_vs_f1"]["ci95_lower"] > 0.0
        ),
        "mean_domain_harm_within_limit": bool(
            learned["worst_mean_domain_harm"] is not None
            and learned["worst_mean_domain_harm"] <= maximum_harm
        ),
    }
    gates["structured_learnability_supported"] = bool(all(gates.values()))
    return {
        "meta_test_task_count": len(test_records),
        "opportunity_threshold": threshold,
        "opportunity_present_count": len(eligible),
        "subsets": summaries,
        "learned_selector_gates": gates,
    }


def assemble_result(
    config: dict,
    train_records: list[dict],
    validation_records: list[dict],
    selection: dict,
    test_records: list[dict],
) -> dict:
    return {
        "schema_version": RESULT_SCHEMA,
        "protocol": PROTOCOL,
        "information_policy": {
            "test_generated_after_selection_lock": True,
            "meta_test_not_used_for_hyperparameters": True,
            "deployment_selector_uses_train_summary_only": True,
            "fdst_reused": False,
            "model_merging_in_scope": False,
        },
        "meta_train_records": train_records,
        "meta_validation_records": validation_records,
        "selection": selection,
        "meta_test_records": test_records,
        "summary": summarize(config, test_records),
    }
