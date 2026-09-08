#!/usr/bin/env python3
"""Run leakage-safe five-fold KADID baseline restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data_manifest import identifiers_manifest  # noqa: E402
from development_baselines import (  # noqa: E402
    METHOD_DOMAIN_BALANCED,
    METHOD_SAMPLE_MEAN_POOLED,
    condition_number,
    objective_weights,
    ridge_path,
    solve_ridge,
    state_cost_bytes,
    weighted_sufficient_statistics,
)
from features import resolve_device  # noqa: E402
from kadid_group_cv import assign_group_folds, fold_manifest, paired_group_bootstrap  # noqa: E402
from result_io import dump_result  # noqa: E402
from run_new_method_kadid_baselines import (  # noqa: E402
    _ToyScalarDomains,
    _config_hash,
    _matrix_summary,
    assert_kadid_config,
    require_output,
)
from run_provenance import with_provenance  # noqa: E402


METHOD_LINEAR_POOLED = "linear_sample_mean_pooled"
METHOD_LINEAR_BALANCED = "linear_domain_balanced"
METHOD_INDEPENDENT = "independent_linear_domain_heads"
METHOD_PROJECTION = "image_mean_gaussian_relu_projection_sample_mean_pooled"
METHOD_ROSTER = (
    METHOD_LINEAR_POOLED,
    METHOD_LINEAR_BALANCED,
    METHOD_INDEPENDENT,
    METHOD_PROJECTION,
)


class _ToyGroupCVDomains(_ToyScalarDomains):
    """Small deterministic fixture with disjoint train/test content ids."""

    def __init__(self, seed=9):
        super().__init__(seed=seed)
        self._data["test"] = [
            [
                (features, target, f"test-{group}")
                for features, target, group in rows
            ]
            for rows in self._data["test"]
        ]


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_lambda_grid(protocol):
    spec = protocol["lambda_grid"]
    if spec.get("kind") != "log10":
        raise RuntimeError("v4 requires a log10 lambda grid")
    return np.logspace(
        float(spec["minimum_exponent"]),
        float(spec["maximum_exponent"]),
        int(spec["points"]),
        dtype=np.float64,
    ).tolist()


def require_frozen_arguments(arguments, protocol):
    expected = {
        "backbone": protocol["backbone"],
        "img_size": int(protocol["image_size"]),
        "max_per_domain": int(protocol["max_train_per_domain"]),
        "sample_seed": int(protocol["sample_seed"]),
        "lambdas": protocol_lambda_grid(protocol),
        "reference_lambda": float(protocol["reference_normalized_lambda"]),
        "projection_dim": int(protocol["projection"]["dimension"]),
        "projection_seed": int(protocol["projection"]["seed"]),
    }
    differences = {
        key: {"expected": value, "actual": getattr(arguments, key)}
        for key, value in expected.items()
        if getattr(arguments, key) != value
    }
    if arguments.skip_projection:
        differences["skip_projection"] = {"expected": False, "actual": True}
    if differences:
        raise RuntimeError(
            "natural run arguments differ from frozen KADID group-CV-v4 protocol: "
            + json.dumps(differences, sort_keys=True)
        )
    return True


def _aug(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return np.concatenate([rows, np.ones((rows.shape[0], 1))], axis=1)


def _stats(rows, targets):
    rows = np.asarray(rows, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    return {
        "R": rows.T @ rows,
        "C": rows.T @ targets,
        "S": float(np.sum(targets * targets)),
        "n": int(rows.shape[0]),
    }


def collect_arrays(domains):
    records = []
    dimension = None
    for index in range(domains.n_domains()):
        name = str(domains.specs[index].name)
        record = {"name": name}
        for split in ("train", "test"):
            features, targets, groups = [], [], []
            for values, target, group in domains.stream(split, index):
                values = np.asarray(values, dtype=np.float64)
                if values.ndim == 1:
                    values = values[None, :]
                if values.shape[0] != 1:
                    raise RuntimeError(
                        "group-CV v4 requires one mean-pooled descriptor per image"
                    )
                if dimension is None:
                    dimension = int(values.shape[1])
                elif values.shape[1] != dimension:
                    raise RuntimeError("feature dimension changed across KADID domains")
                features.append(values[0])
                targets.append(float(np.asarray(target).reshape(-1)[0]))
                groups.append(str(group))
            if not features:
                raise RuntimeError(f"{name}: empty {split} split")
            record[split] = {
                "X": np.asarray(features, dtype=np.float64),
                "y": np.asarray(targets, dtype=np.float64),
                "groups": np.asarray(groups, dtype=str),
            }
        records.append(record)
    return records, int(dimension)


class LinearView:
    name = "mean_pooled_dinov2_linear"

    def __init__(self, input_dim):
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)
        self.projection_bytes = 0

    def fold_parameters(self, records, assignments, fold):
        return None

    def final_parameters(self, records):
        return None

    def transform(self, rows, parameters):
        return np.asarray(rows, dtype=np.float64)


class ImageProjectionView:
    name = "image_mean_gaussian_relu"

    def __init__(self, input_dim, output_dim, seed):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        rng = np.random.default_rng(int(seed))
        self.projection = (
            rng.standard_normal((self.input_dim, self.output_dim))
            / math.sqrt(self.input_dim)
        )
        self.projection_bytes = int(self.projection.nbytes)

    @staticmethod
    def _standardizer(rows):
        rows = np.asarray(rows, dtype=np.float64)
        mean = rows.mean(axis=0)
        std = np.sqrt(np.maximum(np.mean(rows * rows, axis=0) - mean * mean, 1e-6))
        return {"mean": mean, "std": std}

    def fold_parameters(self, records, assignments, fold):
        base = records[0]["train"]
        mask = np.asarray([assignments[group] != fold for group in base["groups"]])
        if not np.any(mask):
            raise RuntimeError("projection standardizer fold has no fit samples")
        return self._standardizer(base["X"][mask])

    def final_parameters(self, records):
        return self._standardizer(records[0]["train"]["X"])

    def transform(self, rows, parameters):
        normalized = (
            np.asarray(rows, dtype=np.float64) - parameters["mean"]
        ) / parameters["std"]
        return np.maximum(normalized @ self.projection, 0.0)


def build_cv_contexts(records, assignments, folds, view):
    contexts = []
    for fold in range(folds):
        parameters = view.fold_parameters(records, assignments, fold)
        domains = []
        for record in records:
            train = record["train"]
            validation_mask = np.asarray(
                [assignments[group] == fold for group in train["groups"]],
                dtype=bool,
            )
            fit_mask = ~validation_mask
            if not np.any(fit_mask) or not np.any(validation_mask):
                raise RuntimeError(
                    f"{record['name']}: empty fold role at fold {fold}"
                )
            scale = max(float(np.mean(train["y"][fit_mask])), 1e-6)
            transformed = _aug(view.transform(train["X"], parameters))
            domains.append(
                {
                    "X": transformed,
                    "y_normalized": train["y"] / scale,
                    "y_raw": train["y"],
                    "groups": train["groups"],
                    "fit_mask": fit_mask,
                    "validation_mask": validation_mask,
                    "scale": scale,
                }
            )
        contexts.append({"fold": fold, "parameters": parameters, "domains": domains})
    return contexts


def _objective_for_method(method):
    if method == METHOD_LINEAR_POOLED or method == METHOD_PROJECTION:
        return METHOD_SAMPLE_MEAN_POOLED
    if method == METHOD_LINEAR_BALANCED:
        return METHOD_DOMAIN_BALANCED
    raise ValueError(f"unsupported shared method: {method}")


def _fit_shared(blocks, objective, lam):
    weights = objective_weights([block["n"] for block in blocks], objective)
    R, C = weighted_sufficient_statistics(blocks, weights)
    return solve_ridge(R, C, lam), R, weights


def _oof_summary(contexts, paths, boundary, lambda_index):
    domain_sse = np.zeros(boundary + 1, dtype=np.float64)
    domain_n = np.zeros(boundary + 1, dtype=np.int64)
    domain_abs = np.zeros(boundary + 1, dtype=np.float64)
    domain_target_abs = np.zeros(boundary + 1, dtype=np.float64)
    group_domain = {}
    for context in contexts:
        weights = paths[boundary][context["fold"]][lambda_index]
        for domain_index in range(boundary + 1):
            domain = context["domains"][domain_index]
            mask = domain["validation_mask"]
            predictions = domain["X"][mask] @ weights
            targets = domain["y_normalized"][mask]
            squared = (predictions - targets) ** 2
            raw_predictions = np.maximum(predictions, 0.0) * domain["scale"]
            raw_targets = domain["y_raw"][mask]
            domain_sse[domain_index] += float(np.sum(squared))
            domain_n[domain_index] += int(np.sum(mask))
            domain_abs[domain_index] += float(
                np.sum(np.abs(raw_predictions - raw_targets))
            )
            domain_target_abs[domain_index] += float(np.sum(np.abs(raw_targets)))
            for group, loss in zip(domain["groups"][mask], squared):
                per_domain = group_domain.setdefault(str(group), {})
                bucket = per_domain.setdefault(domain_index, [0.0, 0])
                bucket[0] += float(loss)
                bucket[1] += 1
    normalized_mse = float(np.mean(domain_sse / domain_n))
    relative_mae = float(np.mean(domain_abs / np.maximum(domain_target_abs, 1e-12)))
    group_losses = {
        group: float(np.mean([total / count for total, count in domains.values()]))
        for group, domains in group_domain.items()
    }
    return {
        "balanced_normalized_mse": normalized_mse,
        "balanced_relative_mae": relative_mae,
        "per_domain_normalized_mse": (domain_sse / domain_n).tolist(),
        "per_domain_relative_mae": (
            domain_abs / np.maximum(domain_target_abs, 1e-12)
        ).tolist(),
        "group_losses": [
            {"group_id": group, "balanced_normalized_mse": group_losses[group]}
            for group in sorted(group_losses)
        ],
    }


def _test_trajectory(records, view, objective, lambdas_by_boundary):
    parameters = view.final_parameters(records)
    train_features = [
        _aug(view.transform(record["train"]["X"], parameters)) for record in records
    ]
    test_features = [
        _aug(view.transform(record["test"]["X"], parameters)) for record in records
    ]
    scales = [max(float(np.mean(record["train"]["y"])), 1e-6) for record in records]
    blocks = [
        _stats(features, record["train"]["y"] / scale)
        for features, record, scale in zip(train_features, records, scales)
    ]
    count = len(records)
    relative = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    final_R = None
    for boundary, lam in enumerate(lambdas_by_boundary):
        weights, R, _ = _fit_shared(blocks[: boundary + 1], objective, lam)
        final_R = R
        for domain_index in range(boundary + 1):
            prediction_normalized = (
                test_features[domain_index] @ weights
            ).reshape(-1)
            target_normalized = records[domain_index]["test"]["y"] / scales[domain_index]
            mse[boundary, domain_index] = float(
                np.mean((prediction_normalized - target_normalized) ** 2)
            )
            prediction_raw = np.maximum(prediction_normalized, 0.0) * scales[domain_index]
            target_raw = records[domain_index]["test"]["y"]
            relative[boundary, domain_index] = float(
                np.mean(np.abs(prediction_raw - target_raw))
                / max(float(np.mean(np.abs(target_raw))), 1e-12)
            )
    return {
        "relative_mae": _matrix_summary(relative),
        "unclipped_normalized_mse": _matrix_summary(mse),
        "final_condition_number": condition_number(final_R, lambdas_by_boundary[-1]),
        "final_target_scales": scales,
    }


def run_shared_cv(records, contexts, view, method, lambdas, reference_lambda):
    objective = _objective_for_method(method)
    boundaries = len(records)
    folds = len(contexts)
    grid = [float(value) for value in lambdas]
    grid_index = {value: index for index, value in enumerate(grid)}
    if float(reference_lambda) not in grid_index:
        raise RuntimeError("reference lambda must be a member of the frozen grid")
    risk_sse = np.zeros((boundaries, len(grid), boundaries), dtype=np.float64)
    risk_n = np.zeros((boundaries, boundaries), dtype=np.int64)
    paths = [[None for _ in range(folds)] for _ in range(boundaries)]

    for context in contexts:
        blocks = []
        for domain in context["domains"]:
            mask = domain["fit_mask"]
            blocks.append(_stats(domain["X"][mask], domain["y_normalized"][mask]))
        for boundary in range(boundaries):
            seen = blocks[: boundary + 1]
            domain_weights = objective_weights(
                [block["n"] for block in seen], objective
            )
            R, C = weighted_sufficient_statistics(seen, domain_weights)
            path = ridge_path(R, C, grid)
            stacked = np.stack([path[lam].reshape(-1) for lam in grid], axis=0)
            paths[boundary][context["fold"]] = stacked
            for domain_index in range(boundary + 1):
                domain = context["domains"][domain_index]
                mask = domain["validation_mask"]
                predictions = domain["X"][mask] @ stacked.T
                targets = domain["y_normalized"][mask, None]
                risk_sse[boundary, :, domain_index] += np.sum(
                    (predictions - targets) ** 2, axis=0
                )
                risk_n[boundary, domain_index] += int(np.sum(mask))

    selections = []
    selected_lambdas = []
    reference_oof = []
    for boundary in range(boundaries):
        risks = np.mean(
            risk_sse[boundary, :, : boundary + 1]
            / risk_n[boundary, None, : boundary + 1],
            axis=1,
        )
        trace = [
            {"lambda": lam, "oof_validation_risk": float(risk)}
            for lam, risk in zip(grid, risks)
        ]
        best = min(trace, key=lambda item: (item["oof_validation_risk"], -item["lambda"]))
        selected_index = grid_index[best["lambda"]]
        selected_oof = _oof_summary(contexts, paths, boundary, selected_index)
        if not math.isclose(
            selected_oof["balanced_normalized_mse"],
            best["oof_validation_risk"],
            rel_tol=1e-11,
            abs_tol=1e-12,
        ):
            raise AssertionError("OOF prediction recomputation disagrees with risk trace")
        selections.append(
            {
                "boundary": boundary,
                "selected_lambda": best["lambda"],
                "candidate_trace": trace,
                "selected_oof": selected_oof,
            }
        )
        selected_lambdas.append(best["lambda"])
        reference_oof.append(
            _oof_summary(
                contexts, paths, boundary, grid_index[float(reference_lambda)]
            )
        )

    return {
        "method": method,
        "objective": objective,
        "feature_view": view.name,
        "projection_matrix_bytes": view.projection_bytes,
        "cv_selected": {
            "lambda_selection": selections,
            "development_test": _test_trajectory(
                records, view, objective, selected_lambdas
            ),
        },
        "fixed_reference_lambda": {
            "lambda": float(reference_lambda),
            "oof_by_boundary": reference_oof,
            "development_test": _test_trajectory(
                records,
                view,
                objective,
                [float(reference_lambda)] * boundaries,
            ),
        },
        "state_cost": state_cost_bytes(view.output_dim + 1, heads=1),
    }


def _single_domain_oof(contexts, domain_index, lambdas, reference_lambda):
    grid = [float(value) for value in lambdas]
    risk_sse = np.zeros(len(grid), dtype=np.float64)
    risk_n = 0
    paths = {}
    for context in contexts:
        domain = context["domains"][domain_index]
        fit = domain["fit_mask"]
        block = _stats(domain["X"][fit], domain["y_normalized"][fit])
        normalized_R = block["R"] / block["n"]
        normalized_C = block["C"] / block["n"]
        path = ridge_path(normalized_R, normalized_C, grid)
        stacked = np.stack([path[lam].reshape(-1) for lam in grid], axis=0)
        paths[context["fold"]] = stacked
        validation = domain["validation_mask"]
        predictions = domain["X"][validation] @ stacked.T
        targets = domain["y_normalized"][validation, None]
        risk_sse += np.sum((predictions - targets) ** 2, axis=0)
        risk_n += int(np.sum(validation))
    risks = risk_sse / risk_n
    trace = [
        {"lambda": lam, "oof_validation_risk": float(risk)}
        for lam, risk in zip(grid, risks)
    ]
    best = min(trace, key=lambda item: (item["oof_validation_risk"], -item["lambda"]))

    def summarize(lam):
        index = grid.index(float(lam))
        sse = count = absolute = target_abs = 0.0
        groups = {}
        for context in contexts:
            domain = context["domains"][domain_index]
            mask = domain["validation_mask"]
            predictions = domain["X"][mask] @ paths[context["fold"]][index]
            targets = domain["y_normalized"][mask]
            squared = (predictions - targets) ** 2
            raw_predictions = np.maximum(predictions, 0.0) * domain["scale"]
            raw_targets = domain["y_raw"][mask]
            sse += float(np.sum(squared))
            count += int(np.sum(mask))
            absolute += float(np.sum(np.abs(raw_predictions - raw_targets)))
            target_abs += float(np.sum(np.abs(raw_targets)))
            for group, loss in zip(domain["groups"][mask], squared):
                bucket = groups.setdefault(str(group), [0.0, 0])
                bucket[0] += float(loss)
                bucket[1] += 1
        return {
            "normalized_mse": sse / count,
            "relative_mae": absolute / max(target_abs, 1e-12),
            "group_losses": {
                group: total / observations
                for group, (total, observations) in groups.items()
            },
        }

    return best, trace, summarize(best["lambda"]), summarize(reference_lambda)


def run_independent_cv(records, contexts, view, lambdas, reference_lambda):
    details = []
    selected_test, reference_test = [], []
    selected_group_domains = {}
    reference_group_domains = {}
    parameters = view.final_parameters(records)
    for domain_index, record in enumerate(records):
        best, trace, selected_oof, reference_oof = _single_domain_oof(
            contexts, domain_index, lambdas, reference_lambda
        )
        train_X = _aug(view.transform(record["train"]["X"], parameters))
        test_X = _aug(view.transform(record["test"]["X"], parameters))
        scale = max(float(np.mean(record["train"]["y"])), 1e-6)
        block = _stats(train_X, record["train"]["y"] / scale)

        def test_metric(lam):
            weights = solve_ridge(
                block["R"] / block["n"], block["C"] / block["n"], lam
            )
            normalized = (test_X @ weights).reshape(-1)
            raw = np.maximum(normalized, 0.0) * scale
            target = record["test"]["y"]
            return {
                "relative_mae": float(
                    np.mean(np.abs(raw - target))
                    / max(float(np.mean(np.abs(target))), 1e-12)
                ),
                "unclipped_normalized_mse": float(
                    np.mean((normalized - target / scale) ** 2)
                ),
            }

        selected_metric = test_metric(best["lambda"])
        reference_metric = test_metric(reference_lambda)
        selected_test.append(selected_metric)
        reference_test.append(reference_metric)
        for group, loss in selected_oof["group_losses"].items():
            selected_group_domains.setdefault(group, []).append(loss)
        for group, loss in reference_oof["group_losses"].items():
            reference_group_domains.setdefault(group, []).append(loss)
        details.append(
            {
                "domain": record["name"],
                "selected_lambda": best["lambda"],
                "candidate_trace": trace,
                "selected_oof": {
                    key: value
                    for key, value in selected_oof.items()
                    if key != "group_losses"
                },
                "reference_oof": {
                    key: value
                    for key, value in reference_oof.items()
                    if key != "group_losses"
                },
                "development_test_selected": selected_metric,
                "development_test_reference": reference_metric,
            }
        )

    def combined(groups):
        return {
            group: float(np.mean(losses)) for group, losses in groups.items()
        }

    selected_groups = combined(selected_group_domains)
    reference_groups = combined(reference_group_domains)
    return {
        "method": METHOD_INDEPENDENT,
        "task_aware_domain_id_required": True,
        "feature_view": view.name,
        "domains": details,
        "cv_selected": {
            "oof_balanced_normalized_mse": float(
                np.mean([item["selected_oof"]["normalized_mse"] for item in details])
            ),
            "development_test_balanced_relative_mae": float(
                np.mean([item["relative_mae"] for item in selected_test])
            ),
            "development_test_balanced_normalized_mse": float(
                np.mean([item["unclipped_normalized_mse"] for item in selected_test])
            ),
            "final_group_losses": [
                {"group_id": group, "balanced_normalized_mse": selected_groups[group]}
                for group in sorted(selected_groups)
            ],
        },
        "fixed_reference_lambda": {
            "lambda": float(reference_lambda),
            "oof_balanced_normalized_mse": float(
                np.mean([item["reference_oof"]["normalized_mse"] for item in details])
            ),
            "development_test_balanced_relative_mae": float(
                np.mean([item["relative_mae"] for item in reference_test])
            ),
            "development_test_balanced_normalized_mse": float(
                np.mean([item["unclipped_normalized_mse"] for item in reference_test])
            ),
            "final_group_losses": [
                {"group_id": group, "balanced_normalized_mse": reference_groups[group]}
                for group in sorted(reference_groups)
            ],
        },
        "state_cost": state_cost_bytes(
            view.output_dim + 1, heads=1, domains=len(records)
        ),
    }


def _group_loss_dict(result):
    return {
        item["group_id"]: float(item["balanced_normalized_mse"])
        for item in result
    }


def _final_selected_group_losses(method_result):
    if method_result["method"] == METHOD_INDEPENDENT:
        rows = method_result["cv_selected"]["final_group_losses"]
    else:
        rows = method_result["cv_selected"]["lambda_selection"][-1][
            "selected_oof"
        ]["group_losses"]
    return _group_loss_dict(rows)


def _records_manifest(records):
    manifests = []
    for record in records:
        train_groups = sorted(set(record["train"]["groups"]))
        test_groups = sorted(set(record["test"]["groups"]))
        manifests.append(
            {
            "name": record["name"],
            "train_samples": int(record["train"]["X"].shape[0]),
            "test_samples": int(record["test"]["X"].shape[0]),
            "train_groups": identifiers_manifest(train_groups),
            "test_groups": identifiers_manifest(test_groups),
            "train_group_ids": train_groups,
            "test_group_ids": test_groups,
            }
        )
    return manifests


def run_audit(domains, protocol, arguments, data_manifest, runtime):
    records, input_dim = collect_arrays(domains)
    all_train_groups = sorted(
        set().union(*(set(record["train"]["groups"]) for record in records))
    )
    cv = protocol["group_cv"]
    assignments = assign_group_folds(
        all_train_groups, cv["sample_key"], cv["seed"], cv["folds"]
    )
    folds = fold_manifest(assignments, cv["sample_key"], cv["seed"], cv["folds"])
    linear = LinearView(input_dim)
    linear_contexts = build_cv_contexts(records, assignments, cv["folds"], linear)
    methods = {
        METHOD_LINEAR_POOLED: run_shared_cv(
            records,
            linear_contexts,
            linear,
            METHOD_LINEAR_POOLED,
            arguments.lambdas,
            arguments.reference_lambda,
        ),
        METHOD_LINEAR_BALANCED: run_shared_cv(
            records,
            linear_contexts,
            linear,
            METHOD_LINEAR_BALANCED,
            arguments.lambdas,
            arguments.reference_lambda,
        ),
        METHOD_INDEPENDENT: run_independent_cv(
            records,
            linear_contexts,
            linear,
            arguments.lambdas,
            arguments.reference_lambda,
        ),
    }
    if arguments.skip_projection:
        methods[METHOD_PROJECTION] = {
            "method": METHOD_PROJECTION,
            "status": "not_run",
            "reason": "--skip-projection",
        }
    else:
        projection = ImageProjectionView(
            input_dim, arguments.projection_dim, arguments.projection_seed
        )
        projection_contexts = build_cv_contexts(
            records, assignments, cv["folds"], projection
        )
        methods[METHOD_PROJECTION] = run_shared_cv(
            records,
            projection_contexts,
            projection,
            METHOD_PROJECTION,
            arguments.lambdas,
            arguments.reference_lambda,
        )
        methods[METHOD_PROJECTION]["qualification"] = protocol["projection"][
            "qualification"
        ]

    bootstrap = protocol["bootstrap"]
    baseline = _final_selected_group_losses(methods[METHOD_LINEAR_POOLED])
    comparisons = {}
    for offset, method in enumerate(
        (METHOD_LINEAR_BALANCED, METHOD_INDEPENDENT, METHOD_PROJECTION)
    ):
        if methods[method].get("status") == "not_run":
            comparisons[method] = {"status": "not_run"}
            continue
        comparisons[method] = paired_group_bootstrap(
            baseline,
            _final_selected_group_losses(methods[method]),
            seed=int(bootstrap["seed"]) + offset,
            resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence"]),
        )

    return {
        "protocol_id": protocol["protocol_id"],
        "evidence_role": "development-only",
        "confirmation_data_used": False,
        "task": "KADID-10k image quality assessment",
        "runtime": runtime,
        "domain_config": {
            "path": str(Path(arguments.domain_config).resolve())
            if arguments.domain_config
            else None,
            "sha256": _config_hash(arguments.domain_config)
            if arguments.domain_config
            else None,
        },
        "data_manifest": data_manifest,
        "domain_records": _records_manifest(records),
        "feature_dimension": input_dim,
        "fold_manifest": folds,
        "lambda_grid": arguments.lambdas,
        "reference_normalized_lambda": arguments.reference_lambda,
        "methods": methods,
        "paired_oof_bootstrap_vs_linear_sample_mean_pooled": comparisons,
        "scope_status": protocol["scope_status"],
        "test_use_warning": (
            "KADID test is repeatedly inspected development data; it did not select "
            "a method or hyperparameter and is not independent confirmation."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "kadid_group_cv_v4.json"),
    )
    parser.add_argument(
        "--domain-config", default=str(ROOT / "configs" / "domains_iqa_kadid.json")
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--out", default=str(ROOT / "runs_real" / "kadid_group_cv_v4" / "main")
    )
    parser.add_argument("--backbone")
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--max-per-domain", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--lambdas")
    parser.add_argument("--reference-lambda", type=float)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--projection-seed", type=int)
    parser.add_argument("--skip-projection", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = _load_json(args.protocol_config)
    args.backbone = args.backbone or protocol["backbone"]
    args.img_size = args.img_size or int(protocol["image_size"])
    args.max_per_domain = args.max_per_domain or int(protocol["max_train_per_domain"])
    args.sample_seed = int(protocol["sample_seed"]) if args.sample_seed is None else args.sample_seed
    args.lambdas = (
        [float(value) for value in args.lambdas.split(",")]
        if args.lambdas
        else protocol_lambda_grid(protocol)
    )
    args.reference_lambda = (
        float(protocol["reference_normalized_lambda"])
        if args.reference_lambda is None
        else args.reference_lambda
    )
    args.projection_dim = (
        int(protocol["projection"]["dimension"])
        if args.projection_dim is None
        else args.projection_dim
    )
    args.projection_seed = (
        int(protocol["projection"]["seed"])
        if args.projection_seed is None
        else args.projection_seed
    )
    output = require_output(args.out, protocol)

    if args.selftest:
        domains = _ToyGroupCVDomains()
        args.domain_config = None
        args.projection_dim = min(args.projection_dim, 12)
        data_manifest = domains.data_manifest()
        runtime = {"device": "none-toy", "cache_dir": None}
    else:
        require_frozen_arguments(args, protocol)
        domain_payload = _load_json(args.domain_config)
        assert_kadid_config(args.domain_config, domain_payload, protocol)
        from datasets_iqa import IQADomains, build_iqa_config

        device = resolve_device(args.device)
        domains = IQADomains(
            build_iqa_config(domain_payload),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
            device=device,
            cache_dir=args.cache_dir,
        )
        data_manifest = domains.data_manifest()
        runtime = {"device": device, "cache_dir": domains.cache_dir}

    started = time.time()
    payload = run_audit(domains, protocol, args, data_manifest, runtime)
    payload["elapsed_seconds"] = time.time() - started
    payload["selftest"] = bool(args.selftest)
    result = with_provenance(payload, args.protocol_config, vars(args))
    destination = output / protocol["result_filename"]
    dump_result(destination, result)
    summary = {
        method: (
            value.get("cv_selected", {})
            .get("development_test", {})
            .get("relative_mae", {})
            .get("final_balanced_loss")
            if method != METHOD_INDEPENDENT
            else value.get("cv_selected", {}).get(
                "development_test_balanced_relative_mae"
            )
        )
        for method, value in payload["methods"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
