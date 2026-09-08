#!/usr/bin/env python3
"""Run the first-batch analytic baseline audit on KADID-10k."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_manifest import identifiers_manifest  # noqa: E402
from development_baselines import (  # noqa: E402
    METHOD_DOMAIN_BALANCED,
    METHOD_DOMAIN_BALANCED_MASS_MATCHED,
    METHOD_POOLED,
    SHARED_METHODS,
    accumulate_stats,
    baseline_gain_decomposition,
    condition_number,
    empty_stats,
    fit_scalar_head,
    objective_weights,
    quadratic_mean_loss,
    ridge_path,
    select_scalar_lambda,
    solve_ridge,
    state_cost_bytes,
    weighted_sufficient_statistics,
)
from features import resolve_device  # noqa: E402
from holdout_split import is_validation  # noqa: E402
from metrics import forgetting_matrix_stats  # noqa: E402
from result_io import dump_result  # noqa: E402
from run_provenance import with_provenance  # noqa: E402


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aug(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    return np.concatenate([rows, np.ones((rows.shape[0], 1))], axis=1)


def _config_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_kadid_config(path, payload, protocol):
    path = Path(path)
    if path.name != protocol["domain_config_basename"]:
        raise RuntimeError(
            f"expected domain config {protocol['domain_config_basename']!r}; got {path.name!r}"
        )
    serialized = f"{path}\n{json.dumps(payload, sort_keys=True)}".lower()
    hits = [
        token
        for token in protocol.get("forbidden_dataset_tokens", [])
        if str(token).lower() in serialized
    ]
    if hits:
        raise RuntimeError(f"forbidden dataset token(s): {hits}")
    names = [domain.get("name") for domain in payload.get("domains", [])]
    if names != protocol["expected_domains"]:
        raise RuntimeError(f"KADID domain roster/order mismatch: {names}")
    for domain in payload["domains"]:
        if domain.get("kind") != "csv":
            raise RuntimeError(f"unsupported KADID domain kind: {domain.get('kind')}")
        root = Path(os.path.expandvars(os.path.expanduser(domain["root"])))
        csv_path = Path(os.path.expandvars(os.path.expanduser(domain["csv"])))
        if not root.is_dir():
            raise RuntimeError(f"missing KADID image root: {root}")
        if not csv_path.is_file():
            raise RuntimeError(f"missing KADID CSV: {csv_path}")
    return True


def require_output(path, protocol):
    expected_root = (ROOT / protocol["output_root"]).resolve()
    actual = Path(path).expanduser().resolve()
    if actual != expected_root and expected_root not in actual.parents:
        raise RuntimeError(f"output must stay below {expected_root}; got {actual}")
    return actual


def require_frozen_arguments(arguments, protocol):
    expected = {
        "backbone": protocol["backbone"],
        "img_size": int(protocol["image_size"]),
        "max_per_domain": int(protocol["max_train_per_domain"]),
        "sample_seed": int(protocol["sample_seed"]),
        "split_seed": int(protocol["selector_split_seed"]),
        "val_every": int(protocol["selector_val_every"]),
        "lambdas": [float(value) for value in protocol["lambda_grid"]],
        "reference_lambda": float(protocol["reference_lambda"]),
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
            "natural run arguments differ from frozen KADID development-v1 protocol: "
            + json.dumps(differences, sort_keys=True)
        )
    return True


def _domain_name(domains, index):
    specs = getattr(domains, "specs", None)
    if specs is None:
        specs = getattr(domains, "domains", None)
    if specs is not None:
        return str(specs[index].name)
    return f"domain-{index}"


def collect_domain_records(domains, split_seed, val_every):
    records = []
    dimension = None
    for index in range(domains.n_domains()):
        name = _domain_name(domains, index)
        fit_groups, validation_groups = set(), set()
        fit_targets = []
        for _, target, group in domains.stream("train", index):
            group = str(group)
            if is_validation(name, group, split_seed, val_every):
                validation_groups.add(group)
            else:
                fit_groups.add(group)
                fit_targets.append(float(np.asarray(target).reshape(-1)[0]))
        if not fit_groups or not validation_groups:
            raise RuntimeError(
                f"{name}: empty fit/validation group role "
                f"({len(fit_groups)}/{len(validation_groups)})"
            )
        if fit_groups & validation_groups:
            raise RuntimeError(f"{name}: reference group crosses fit/validation roles")
        scale = max(float(np.mean(fit_targets)), 1e-6)
        fit = None
        full = None
        validation = None
        fit_samples = validation_samples = 0
        for features, target, group in domains.stream("train", index):
            rows = _aug(features)
            if dimension is None:
                dimension = rows.shape[1]
            elif rows.shape[1] != dimension:
                raise RuntimeError("feature dimension changed across KADID domains")
            if fit is None:
                fit = empty_stats(dimension)
                full = empty_stats(dimension)
                validation = empty_stats(dimension)
            normalized = np.asarray(target, dtype=np.float64).reshape(-1, 1) / scale
            accumulate_stats(full, rows, normalized)
            if str(group) in validation_groups:
                accumulate_stats(validation, rows, normalized)
                validation_samples += rows.shape[0]
            else:
                accumulate_stats(fit, rows, normalized)
                fit_samples += rows.shape[0]
        records.append(
            {
                "name": name,
                "scale": scale,
                "fit": fit,
                "full": full,
                "val": validation,
                "split_manifest": {
                    "unit": "reference_content_group",
                    "split_seed": int(split_seed),
                    "val_every": int(val_every),
                    "fit_groups": identifiers_manifest(fit_groups),
                    "validation_groups": identifiers_manifest(validation_groups),
                    "fit_samples": fit_samples,
                    "validation_samples": validation_samples,
                },
            }
        )
    return records, int(dimension)


def evaluate(domains, index, weights, scale, transform=None):
    errors = []
    normalized_squared_errors = []
    targets = []
    for features, target, _ in domains.stream("test", index):
        rows = _aug(transform(features) if transform is not None else features)
        normalized_prediction = float((rows @ weights)[0, 0])
        raw_target = float(np.asarray(target).reshape(-1)[0])
        normalized_target = raw_target / scale
        normalized_squared_errors.append((normalized_prediction - normalized_target) ** 2)
        prediction = max(normalized_prediction, 0.0) * scale
        errors.append(abs(prediction - raw_target))
        targets.append(raw_target)
    if not targets:
        raise RuntimeError(f"empty KADID test domain {index}")
    return {
        "relative_mae": float(np.mean(errors)) / max(float(np.mean(np.abs(targets))), 1e-6),
        "unclipped_normalized_mse": float(np.mean(normalized_squared_errors)),
        "test_samples": len(targets),
    }


def _matrix_summary(matrix):
    summary = forgetting_matrix_stats(matrix)
    return {
        "final_balanced_loss": summary.pop("final_avg_mae"),
        **summary,
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
    }


def run_shared_method(domains, records, method, lambdas):
    count = len(records)
    fit_rel = np.full((count, count), np.nan)
    fit_mse = np.full((count, count), np.nan)
    refit_rel = np.full((count, count), np.nan)
    refit_mse = np.full((count, count), np.nan)
    selections = []
    final_fit = None
    for boundary in range(count):
        seen = records[: boundary + 1]
        selected = select_scalar_lambda(seen, method, lambdas)
        fit_model = fit_scalar_head(seen, method, selected.lam, "fit")
        refit_model = fit_scalar_head(seen, method, selected.lam, "full")
        for index in range(boundary + 1):
            fit_eval = evaluate(domains, index, fit_model["W"], records[index]["scale"])
            refit_eval = evaluate(domains, index, refit_model["W"], records[index]["scale"])
            fit_rel[boundary, index] = fit_eval["relative_mae"]
            fit_mse[boundary, index] = fit_eval["unclipped_normalized_mse"]
            refit_rel[boundary, index] = refit_eval["relative_mae"]
            refit_mse[boundary, index] = refit_eval["unclipped_normalized_mse"]
        selections.append(
            {
                "boundary": boundary,
                "lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
            }
        )
        final_fit = fit_model
    return {
        "method": method,
        "lambda_selection": selections,
        "aligned_fit_only": {
            "relative_mae": _matrix_summary(fit_rel),
            "unclipped_normalized_mse": _matrix_summary(fit_mse),
        },
        "full_refit_ablation": {
            "warning": "Fit-only validation risk does not certify refitted parameters.",
            "relative_mae": _matrix_summary(refit_rel),
            "unclipped_normalized_mse": _matrix_summary(refit_mse),
        },
        "final_system": {
            "selected_lambda": selections[-1]["lambda"],
            "condition_number": condition_number(
                final_fit["R"], selections[-1]["lambda"]
            ),
            "domain_weights": final_fit["domain_weights"].tolist(),
        },
    }


def run_fixed_reference(domains, records, lam):
    count = len(records)
    rel = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    for boundary in range(count):
        model = fit_scalar_head(records[: boundary + 1], METHOD_POOLED, lam, "fit")
        for index in range(boundary + 1):
            result = evaluate(domains, index, model["W"], records[index]["scale"])
            rel[boundary, index] = result["relative_mae"]
            mse[boundary, index] = result["unclipped_normalized_mse"]
    return {
        "lambda": float(lam),
        "relative_mae": _matrix_summary(rel),
        "unclipped_normalized_mse": _matrix_summary(mse),
    }


def run_single_domain(domains, records, lambdas):
    fit_losses, refit_losses, details = [], [], []
    for index, record in enumerate(records):
        selected = select_scalar_lambda([record], METHOD_POOLED, lambdas)
        fit_model = fit_scalar_head([record], METHOD_POOLED, selected.lam, "fit")
        refit_model = fit_scalar_head([record], METHOD_POOLED, selected.lam, "full")
        fit_eval = evaluate(domains, index, fit_model["W"], record["scale"])
        refit_eval = evaluate(domains, index, refit_model["W"], record["scale"])
        fit_losses.append(fit_eval["relative_mae"])
        refit_losses.append(refit_eval["relative_mae"])
        details.append(
            {
                "domain": record["name"],
                "selected_lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
                "aligned_fit_only": fit_eval,
                "full_refit_ablation": refit_eval,
            }
        )
    return {
        "task_aware_domain_id_required": True,
        "per_domain_lambda": True,
        "aligned_fit_only_final_balanced_loss": float(np.mean(fit_losses)),
        "full_refit_ablation_final_balanced_loss": float(np.mean(refit_losses)),
        "domains": details,
    }


def collect_projection_records(domains, records, projection_dim, projection_seed, split_seed, val_every):
    total = total_sq = None
    count = 0
    first_name = records[0]["name"]
    for features, _, group in domains.stream("train", 0):
        if is_validation(first_name, str(group), split_seed, val_every):
            continue
        values = np.asarray(features, dtype=np.float64)
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        total_sq = (values * values).sum(axis=0) if total_sq is None else total_sq + (values * values).sum(axis=0)
        count += values.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-6))
    rng = np.random.default_rng(int(projection_seed))
    projection = rng.standard_normal((mean.size, int(projection_dim))) / math.sqrt(mean.size)

    def transform(features):
        return np.maximum(((np.asarray(features) - mean) / std) @ projection, 0.0)

    projected = []
    for index, base in enumerate(records):
        fit = empty_stats(int(projection_dim) + 1)
        full = empty_stats(int(projection_dim) + 1)
        validation = empty_stats(int(projection_dim) + 1)
        for features, target, group in domains.stream("train", index):
            rows = _aug(transform(features))
            normalized = np.asarray(target, dtype=np.float64).reshape(-1, 1) / base["scale"]
            accumulate_stats(full, rows, normalized)
            if is_validation(base["name"], str(group), split_seed, val_every):
                accumulate_stats(validation, rows, normalized)
            else:
                accumulate_stats(fit, rows, normalized)
        projected.append({"fit": fit, "full": full, "val": validation})
    return projected, transform, projection


def run_projection(domains, base_records, lambdas, projection_dim, projection_seed, split_seed, val_every):
    records, transform, projection = collect_projection_records(
        domains,
        base_records,
        projection_dim,
        projection_seed,
        split_seed,
        val_every,
    )
    count = len(records)
    fit_rel = np.full((count, count), np.nan)
    fit_mse = np.full((count, count), np.nan)
    refit_rel = np.full((count, count), np.nan)
    refit_mse = np.full((count, count), np.nan)
    selections = []
    final_R = None
    for boundary in range(count):
        seen = records[: boundary + 1]
        selected = select_scalar_lambda(seen, METHOD_POOLED, lambdas)
        fit_model = fit_scalar_head(seen, METHOD_POOLED, selected.lam, "fit")
        refit_model = fit_scalar_head(seen, METHOD_POOLED, selected.lam, "full")
        for index in range(boundary + 1):
            fit_eval = evaluate(
                domains, index, fit_model["W"], base_records[index]["scale"], transform
            )
            refit_eval = evaluate(
                domains, index, refit_model["W"], base_records[index]["scale"], transform
            )
            fit_rel[boundary, index] = fit_eval["relative_mae"]
            fit_mse[boundary, index] = fit_eval["unclipped_normalized_mse"]
            refit_rel[boundary, index] = refit_eval["relative_mae"]
            refit_mse[boundary, index] = refit_eval["unclipped_normalized_mse"]
        selections.append(
            {
                "boundary": boundary,
                "lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
            }
        )
        final_R = fit_model["R"]
    return {
        "label": "RanPAC-style Gaussian projection + ReLU + ridge; not full RanPAC",
        "projection_dim": int(projection_dim),
        "projection_seed": int(projection_seed),
        "projection_matrix_bytes": int(projection.nbytes),
        "lambda_selection": selections,
        "aligned_fit_only": {
            "relative_mae": _matrix_summary(fit_rel),
            "unclipped_normalized_mse": _matrix_summary(fit_mse),
        },
        "full_refit_ablation": {
            "warning": "Fit-only validation risk does not certify refitted parameters.",
            "relative_mae": _matrix_summary(refit_rel),
            "unclipped_normalized_mse": _matrix_summary(refit_mse),
        },
        "final_condition_number": condition_number(final_R, selections[-1]["lambda"]),
    }


def _serializable_records(records):
    return [
        {
            "name": record["name"],
            "fit_only_target_scale": record["scale"],
            "fit_samples": record["fit"]["n"],
            "validation_samples": record["val"]["n"],
            "full_samples": record["full"]["n"],
            "split_manifest": record["split_manifest"],
        }
        for record in records
    ]


def run_audit(domains, protocol, arguments, data_manifest, runtime):
    records, dimension = collect_domain_records(
        domains, arguments.split_seed, arguments.val_every
    )
    reference = run_fixed_reference(domains, records, arguments.reference_lambda)
    shared = {
        method: run_shared_method(domains, records, method, arguments.lambdas)
        for method in SHARED_METHODS
    }
    single = run_single_domain(domains, records, arguments.lambdas)
    projection = (
        {"status": "not_run", "reason": "--skip-projection"}
        if arguments.skip_projection
        else run_projection(
            domains,
            records,
            arguments.lambdas,
            arguments.projection_dim,
            arguments.projection_seed,
            arguments.split_seed,
            arguments.val_every,
        )
    )
    losses = {
        "pooled_reference_lambda": reference["relative_mae"]["final_balanced_loss"],
        "pooled_tuned_lambda": shared[METHOD_POOLED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "domain_balanced_tuned": shared[METHOD_DOMAIN_BALANCED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "mass_matched_domain_balanced_tuned": shared[METHOD_DOMAIN_BALANCED_MASS_MATCHED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "single_domain_tuned": single["aligned_fit_only_final_balanced_loss"],
    }
    single_one_pass = state_cost_bytes(dimension, heads=1)
    single_one_pass["deployment_weights_bytes"] *= len(records)
    return {
        "protocol_id": protocol["protocol_id"],
        "evidence_role": "development-only",
        "confirmation_data_used": False,
        "task": "KADID-10k image quality assessment",
        "runtime": runtime,
        "domain_config": {
            "path": str(Path(arguments.domain_config).resolve()) if arguments.domain_config else None,
            "sha256": _config_hash(arguments.domain_config) if arguments.domain_config else None,
        },
        "data_manifest": data_manifest,
        "domain_records": _serializable_records(records),
        "pooled_reference_lambda": reference,
        "shared_methods": shared,
        "single_domain": single,
        "ranpac_style_projection": projection,
        "cost_report": {
            "shared_scalar_head": state_cost_bytes(dimension, heads=1),
            "single_domain_one_pass_peak_update_state": single_one_pass,
            "single_domain_revisitable_update_state": state_cost_bytes(
                dimension, heads=1, trajectories=1, domains=len(records)
            ),
            "projection_scalar_head": state_cost_bytes(
                arguments.projection_dim + 1, heads=1
            ),
        },
        "gain_decomposition": baseline_gain_decomposition(losses),
    }


class _ToySpec:
    def __init__(self, name):
        self.name = name


class _ToyScalarDomains:
    def __init__(self, seed=9):
        self.specs = [_ToySpec(f"toy-iqa-{index}") for index in range(3)]
        self._data = {"train": [], "test": []}
        rng = np.random.default_rng(seed)
        for domain in range(3):
            weight = np.asarray([0.7, -0.2, 0.3]) + domain * 0.05
            for split, groups in (("train", 12), ("test", 5)):
                rows = []
                for group in range(groups):
                    for variant in range(3):
                        features = rng.normal(size=(1, 3))
                        target = max(
                            float((features @ weight)[0] + 2.0 + rng.normal(scale=0.05)),
                            0.1,
                        )
                        rows.append((features, np.asarray([[target]]), f"g{group:02d}"))
                self._data[split].append(rows)

    def n_domains(self):
        return len(self.specs)

    def stream(self, split, index):
        for features, target, group in self._data[split][index]:
            yield features.copy(), target.copy(), group

    def data_manifest(self):
        return {"kind": "toy-grouped-scalar", "domains": 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "new_method_kadid_development_v1.json"),
    )
    parser.add_argument(
        "--domain-config",
        default=str(ROOT / "configs" / "domains_iqa_kadid.json"),
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "runs_real" / "new_method_kadid_development_v1" / "main"),
    )
    parser.add_argument("--backbone")
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--max-per-domain", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--val-every", type=int)
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
    args.sample_seed = (
        int(protocol["sample_seed"]) if args.sample_seed is None else args.sample_seed
    )
    args.split_seed = (
        int(protocol["selector_split_seed"]) if args.split_seed is None else args.split_seed
    )
    args.val_every = args.val_every or int(protocol["selector_val_every"])
    args.lambdas = (
        [float(value) for value in args.lambdas.split(",")]
        if args.lambdas
        else [float(value) for value in protocol["lambda_grid"]]
    )
    args.reference_lambda = (
        float(protocol["reference_lambda"])
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
        domains = _ToyScalarDomains()
        data_manifest = domains.data_manifest()
        args.domain_config = None
        args.projection_dim = min(args.projection_dim, 12)
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
    destination = output / "kadid_baseline_audit.json"
    dump_result(destination, result)
    print(json.dumps(payload["gain_decomposition"], indent=2, ensure_ascii=False))
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
