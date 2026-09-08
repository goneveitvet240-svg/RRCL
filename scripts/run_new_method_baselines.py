#!/usr/bin/env python3
"""Development-only RRCL baseline audit from protocol v1.

This runner never imports the FDST loader and rejects any non-allow-listed
natural-domain configuration.  Lambda is selected on deterministic held-out
training images.  The primary evaluated parameters are fit-only, so the model
being validated and the model being deployed are identical; full-data refits
are emitted only as a labelled ablation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from development_baselines import (  # noqa: E402
    METHOD_DOMAIN_BALANCED,
    METHOD_DOMAIN_BALANCED_MASS_MATCHED,
    METHOD_POOLED,
    SHARED_METHODS,
    accumulate_stats,
    baseline_gain_decomposition,
    condition_number,
    empty_stats,
    fit_shared_heads,
    objective_weights,
    quadratic_mean_loss,
    ridge_path,
    select_shared_lambda,
    solve_ridge,
    state_cost_bytes,
    weighted_sufficient_statistics,
)
from holdout_split import (  # noqa: E402
    domain_sample_key,
    is_validation,
    iter_stream_with_ids,
    require_both_partitions,
    split_manifest,
)
from metrics import forgetting_matrix_stats  # noqa: E402
from result_io import dump_result  # noqa: E402
from run_provenance import with_provenance  # noqa: E402
from run_real_image_aux import _image_feature, _image_target, transform_patch_target  # noqa: E402


def _aug(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return np.concatenate([rows, np.ones((rows.shape[0], 1))], axis=1)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_development_domain_config(path, payload, protocol):
    """Fail closed before any dataset or feature loader is instantiated."""

    path = Path(path)
    allowed = set(protocol["allowed_domain_config_basenames"])
    if path.name not in allowed:
        raise RuntimeError(
            f"domain config {path.name!r} is not allow-listed by "
            f"{protocol['protocol_id']}"
        )
    serialized = f"{path}\n{json.dumps(payload, sort_keys=True)}".lower()
    hits = [
        token
        for token in protocol.get("forbidden_dataset_tokens", [])
        if str(token).lower() in serialized
    ]
    if hits:
        raise RuntimeError(f"forbidden confirmation dataset token(s): {hits}")
    domains = payload.get("domains", [])
    if not domains:
        raise RuntimeError("domain config has no domains")
    allowed_keys = set(protocol["allowed_sample_keys"])
    allowed_kinds = set(protocol["allowed_domain_kinds"])
    unexpected = [
        {
            "name": domain.get("name"),
            "sample_key": domain.get("sample_key"),
            "kind": domain.get("kind"),
        }
        for domain in domains
        if domain.get("sample_key") not in allowed_keys
        or domain.get("kind", "jhu") not in allowed_kinds
    ]
    if unexpected:
        raise RuntimeError(f"domain identities are not allow-listed: {unexpected}")
    return True


def require_development_output(path):
    required = (ROOT / "runs_real" / "new_method_development_v1").resolve()
    actual = Path(path).expanduser().resolve()
    if actual != required and required not in actual.parents:
        raise RuntimeError(f"output must stay below {required}; got {actual}")
    return actual


def require_frozen_method_arguments(arguments, protocol):
    """Prevent a natural run from silently drifting while retaining the v1 id."""

    expected = {
        "val_every": int(protocol["val_every"]),
        "patch_target": protocol["patch_target"],
        "alpha": float(protocol["alpha_patch"]),
        "lambdas": [float(value) for value in protocol["lambda_grid"]],
        "reference_lambda": float(protocol["reference_lambda"]),
        "projection_dim": int(protocol["projection"]["dimension"]),
        "projection_seed": int(protocol["projection"]["seed"]),
        "backbone": protocol["backbone"],
        "img_size": int(protocol["image_size"]),
        "max_per_domain": int(protocol["max_per_domain"]),
    }
    actual = {key: getattr(arguments, key) for key in expected}
    differences = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if arguments.skip_ranpac:
        differences["skip_ranpac"] = {"expected": False, "actual": True}
    if differences:
        raise RuntimeError(
            "natural run arguments differ from frozen development-v1 protocol: "
            + json.dumps(differences, sort_keys=True)
        )
    return True


def _first_feature_dim(domains):
    for X, _, _ in domains.stream("train", 0):
        return int(np.asarray(X).shape[1])
    raise RuntimeError("empty first training domain")


def _fit_scale_and_split(domains, domain_index, split_seed, val_every):
    key = domain_sample_key(domains, domain_index)
    fit_ids, val_ids = [], []
    fit_totals = []
    id_source = None
    for _, Y, _, image_id, source in iter_stream_with_ids(
        domains, "train", domain_index
    ):
        id_source = source if id_source is None else id_source
        if source != id_source:
            raise RuntimeError("mixed image-id sources inside one domain")
        if is_validation(key, image_id, split_seed, val_every):
            val_ids.append(image_id)
        else:
            fit_ids.append(image_id)
            fit_totals.append(float(np.asarray(Y, dtype=np.float64).sum()))
    manifest = require_both_partitions(
        split_manifest(key, split_seed, val_every, fit_ids, val_ids, id_source),
        where=f"development baseline domain {domain_index}",
    )
    scale = max(float(np.mean(fit_totals)), 1e-6)
    return scale, manifest


def collect_domain_records(domains, patch_target, alpha, split_seed, val_every):
    """Collect fit/full dual-head stats and image-level fused validation stats."""

    dim = _first_feature_dim(domains)
    d_aug = dim + 1
    records = []
    for index in range(domains.n_domains()):
        scale, manifest = _fit_scale_and_split(
            domains, index, split_seed, val_every
        )
        record = {
            "name": getattr(domains.domains[index], "name", f"domain-{index}")
            if hasattr(domains, "domains")
            else f"domain-{index}",
            "sample_key": domain_sample_key(domains, index),
            "scale": scale,
            "split_manifest": manifest,
            "fit": {
                "patch": empty_stats(d_aug),
                "image": empty_stats(d_aug),
            },
            "full": {
                "patch": empty_stats(d_aug),
                "image": empty_stats(d_aug),
            },
            "val": {"fused": empty_stats(2 * d_aug)},
        }
        for X, Y, _, image_id, _ in iter_stream_with_ids(domains, "train", index):
            X = np.asarray(X, dtype=np.float64)
            patch_rows = _aug(X)
            patch_targets = transform_patch_target(Y, patch_target) / scale
            image_rows = _aug(_image_feature(X))
            image_targets = _image_target(Y) / scale
            accumulate_stats(record["full"]["patch"], patch_rows, patch_targets)
            accumulate_stats(record["full"]["image"], image_rows, image_targets)
            if is_validation(record["sample_key"], image_id, split_seed, val_every):
                fused_row = np.concatenate(
                    [
                        alpha * patch_rows.sum(axis=0),
                        (1.0 - alpha) * image_rows[0],
                    ]
                )[None, :]
                accumulate_stats(record["val"]["fused"], fused_row, image_targets)
            else:
                accumulate_stats(record["fit"]["patch"], patch_rows, patch_targets)
                accumulate_stats(record["fit"]["image"], image_rows, image_targets)
        records.append(record)
    return records, d_aug


def evaluate_dual(domains, domain_index, patch_W, image_W, alpha, scale):
    absolute_errors = []
    normalized_squared_errors = []
    ground_truths = []
    for X, Y, _ in domains.stream("test", domain_index):
        patch_rows = _aug(X)
        image_rows = _aug(_image_feature(X))
        patch_unclipped = float((patch_rows @ patch_W).sum())
        image_unclipped = float((image_rows @ image_W)[0, 0])
        normalized_unclipped = alpha * patch_unclipped + (1.0 - alpha) * image_unclipped
        normalized_target = float(np.asarray(Y, dtype=np.float64).sum()) / scale
        normalized_squared_errors.append((normalized_unclipped - normalized_target) ** 2)

        patch_clipped = float(np.clip(patch_rows @ patch_W, 0.0, None).sum())
        image_clipped = float(np.clip(image_rows @ image_W, 0.0, None)[0, 0])
        prediction = max(alpha * patch_clipped + (1.0 - alpha) * image_clipped, 0.0) * scale
        target = float(np.asarray(Y, dtype=np.float64).sum())
        absolute_errors.append(abs(prediction - target))
        ground_truths.append(target)
    if not ground_truths:
        raise RuntimeError(f"empty test domain {domain_index}")
    rel_mae = float(np.mean(absolute_errors)) / max(float(np.mean(ground_truths)), 1e-6)
    return {
        "relative_mae": rel_mae,
        "unclipped_normalized_mse": float(np.mean(normalized_squared_errors)),
        "n_test_images": len(ground_truths),
    }


def _matrix_summary(matrix):
    summary = forgetting_matrix_stats(matrix)
    return {
        "final_balanced_loss": summary.pop("final_avg_mae"),
        **summary,
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
    }


def run_shared_method(domains, records, method, lambdas, alpha):
    count = len(records)
    aligned_rel = np.full((count, count), np.nan)
    aligned_mse = np.full((count, count), np.nan)
    refit_rel = np.full((count, count), np.nan)
    refit_mse = np.full((count, count), np.nan)
    selections = []
    final_systems = None
    for boundary in range(count):
        seen = records[: boundary + 1]
        selected = select_shared_lambda(seen, method, lambdas)
        aligned = fit_shared_heads(seen, method, selected.lam, role="fit")
        refit = fit_shared_heads(seen, method, selected.lam, role="full")
        for domain_index in range(boundary + 1):
            aligned_eval = evaluate_dual(
                domains,
                domain_index,
                aligned["patch_W"],
                aligned["image_W"],
                alpha,
                records[domain_index]["scale"],
            )
            refit_eval = evaluate_dual(
                domains,
                domain_index,
                refit["patch_W"],
                refit["image_W"],
                alpha,
                records[domain_index]["scale"],
            )
            aligned_rel[boundary, domain_index] = aligned_eval["relative_mae"]
            aligned_mse[boundary, domain_index] = aligned_eval["unclipped_normalized_mse"]
            refit_rel[boundary, domain_index] = refit_eval["relative_mae"]
            refit_mse[boundary, domain_index] = refit_eval["unclipped_normalized_mse"]
        selections.append(
            {
                "boundary": boundary,
                "lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
            }
        )
        final_systems = aligned
    return {
        "method": method,
        "lambda_selection": selections,
        "aligned_fit_only": {
            "relative_mae": _matrix_summary(aligned_rel),
            "unclipped_normalized_mse": _matrix_summary(aligned_mse),
        },
        "full_refit_ablation": {
            "warning": "Validation risk selected fit-only parameters; it does not certify refitted parameters.",
            "relative_mae": _matrix_summary(refit_rel),
            "unclipped_normalized_mse": _matrix_summary(refit_mse),
        },
        "final_system": {
            "selected_lambda": selections[-1]["lambda"],
            "patch_condition_number": condition_number(
                final_systems["patch_R"], selections[-1]["lambda"]
            ),
            "image_condition_number": condition_number(
                final_systems["image_R"], selections[-1]["lambda"]
            ),
            "patch_domain_weights": final_systems["patch_domain_weights"].tolist(),
            "image_domain_weights": final_systems["image_domain_weights"].tolist(),
        },
    }


def run_fixed_reference(domains, records, lam, alpha):
    count = len(records)
    rel = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    for boundary in range(count):
        fitted = fit_shared_heads(records[: boundary + 1], METHOD_POOLED, lam, role="fit")
        for domain_index in range(boundary + 1):
            evaluated = evaluate_dual(
                domains,
                domain_index,
                fitted["patch_W"],
                fitted["image_W"],
                alpha,
                records[domain_index]["scale"],
            )
            rel[boundary, domain_index] = evaluated["relative_mae"]
            mse[boundary, domain_index] = evaluated["unclipped_normalized_mse"]
    return {
        "lambda": float(lam),
        "training_protocol": "aligned_fit_only",
        "relative_mae": _matrix_summary(rel),
        "unclipped_normalized_mse": _matrix_summary(mse),
    }


def run_single_domain(domains, records, lambdas, alpha):
    aligned, refit, details = [], [], []
    for index, record in enumerate(records):
        selected = select_shared_lambda([record], METHOD_POOLED, lambdas)
        fit_model = fit_shared_heads([record], METHOD_POOLED, selected.lam, role="fit")
        full_model = fit_shared_heads([record], METHOD_POOLED, selected.lam, role="full")
        fit_eval = evaluate_dual(
            domains, index, fit_model["patch_W"], fit_model["image_W"], alpha, record["scale"]
        )
        full_eval = evaluate_dual(
            domains, index, full_model["patch_W"], full_model["image_W"], alpha, record["scale"]
        )
        aligned.append(fit_eval["relative_mae"])
        refit.append(full_eval["relative_mae"])
        details.append(
            {
                "domain": record["name"],
                "selected_lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
                "aligned_fit_only": fit_eval,
                "full_refit_ablation": full_eval,
            }
        )
    return {
        "task_aware_domain_id_required": True,
        "per_domain_lambda": True,
        "aligned_fit_only_final_balanced_loss": float(np.mean(aligned)),
        "full_refit_ablation_final_balanced_loss": float(np.mean(refit)),
        "domains": details,
    }


def _fit_standardizer(domains, records, split_seed, val_every):
    total = None
    total_sq = None
    count = 0
    key = records[0]["sample_key"]
    for X, _, _, image_id, _ in iter_stream_with_ids(domains, "train", 0):
        if is_validation(key, image_id, split_seed, val_every):
            continue
        X = np.asarray(X, dtype=np.float64)
        total = X.sum(axis=0) if total is None else total + X.sum(axis=0)
        total_sq = (X * X).sum(axis=0) if total_sq is None else total_sq + (X * X).sum(axis=0)
        count += X.shape[0]
    if count <= 0:
        raise RuntimeError("empty fit partition for projection standardizer")
    mean = total / count
    variance = total_sq / count - mean * mean
    return mean, np.sqrt(np.maximum(variance, 1e-6))


def collect_projected_image_records(
    domains, records, projection_dim, projection_seed, split_seed, val_every
):
    mean, std = _fit_standardizer(domains, records, split_seed, val_every)
    rng = np.random.default_rng(int(projection_seed))
    projection = rng.standard_normal((mean.size, int(projection_dim))) / math.sqrt(mean.size)
    projected = []
    for index, base in enumerate(records):
        record = {
            "fit": empty_stats(int(projection_dim) + 1),
            "full": empty_stats(int(projection_dim) + 1),
            "val": empty_stats(int(projection_dim) + 1),
        }
        for X, Y, _, image_id, _ in iter_stream_with_ids(domains, "train", index):
            image = _image_feature(X)
            rows = _aug(np.maximum(((image - mean) / std) @ projection, 0.0))
            targets = _image_target(Y) / base["scale"]
            accumulate_stats(record["full"], rows, targets)
            if is_validation(base["sample_key"], image_id, split_seed, val_every):
                accumulate_stats(record["val"], rows, targets)
            else:
                accumulate_stats(record["fit"], rows, targets)
        projected.append(record)
    return projected, mean, std, projection


def _select_projected_lambda(records, lambdas):
    grid = sorted({float(value) for value in lambdas})
    weights = objective_weights(
        [record["fit"]["n"] for record in records], METHOD_POOLED
    )
    R, C = weighted_sufficient_statistics(
        [record["fit"] for record in records], weights
    )
    path = ridge_path(R, C, grid)
    trace = []
    for lam in grid:
        W = path[lam]
        risk = float(np.mean([quadratic_mean_loss(W, record["val"]) for record in records]))
        trace.append({"lambda": lam, "validation_risk": risk})
    best = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
    return best, trace, (path[best["lambda"]], R)


def _evaluate_projected(domains, index, W, mean, std, projection, scale):
    errors, squared, targets = [], [], []
    for X, Y, _ in domains.stream("test", index):
        row = _aug(np.maximum(((_image_feature(X) - mean) / std) @ projection, 0.0))
        raw_normalized = float((row @ W)[0, 0])
        target = float(np.asarray(Y, dtype=np.float64).sum())
        errors.append(abs(max(raw_normalized, 0.0) * scale - target))
        squared.append((raw_normalized - target / scale) ** 2)
        targets.append(target)
    return {
        "relative_mae": float(np.mean(errors)) / max(float(np.mean(targets)), 1e-6),
        "unclipped_normalized_mse": float(np.mean(squared)),
    }


def run_ranpac_style(domains, base_records, lambdas, projection_dim, projection_seed, split_seed, val_every):
    records, mean, std, projection = collect_projected_image_records(
        domains,
        base_records,
        projection_dim,
        projection_seed,
        split_seed,
        val_every,
    )
    count = len(records)
    rel = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    refit_rel = np.full((count, count), np.nan)
    refit_mse = np.full((count, count), np.nan)
    selections = []
    final_R = None
    for boundary in range(count):
        seen = records[: boundary + 1]
        best, trace, (W, R) = _select_projected_lambda(seen, lambdas)
        full_weights = objective_weights(
            [record["full"]["n"] for record in seen], METHOD_POOLED
        )
        full_R, full_C = weighted_sufficient_statistics(
            [record["full"] for record in seen], full_weights
        )
        full_W = solve_ridge(full_R, full_C, best["lambda"])
        for index in range(boundary + 1):
            evaluated = _evaluate_projected(
                domains, index, W, mean, std, projection, base_records[index]["scale"]
            )
            refit_evaluated = _evaluate_projected(
                domains,
                index,
                full_W,
                mean,
                std,
                projection,
                base_records[index]["scale"],
            )
            rel[boundary, index] = evaluated["relative_mae"]
            mse[boundary, index] = evaluated["unclipped_normalized_mse"]
            refit_rel[boundary, index] = refit_evaluated["relative_mae"]
            refit_mse[boundary, index] = refit_evaluated["unclipped_normalized_mse"]
        selections.append({"boundary": boundary, **best, "candidate_trace": trace})
        final_R = R
    return {
        "label": "RanPAC-style image-only Gaussian projection + ReLU + ridge; not full RanPAC",
        "projection_dim": int(projection_dim),
        "projection_seed": int(projection_seed),
        "projection_matrix_bytes": int(projection.nbytes),
        "lambda_selection": selections,
        "aligned_fit_only": {
            "relative_mae": _matrix_summary(rel),
            "unclipped_normalized_mse": _matrix_summary(mse),
        },
        "full_refit_ablation": {
            "warning": "Validation risk selected fit-only parameters; it does not certify refitted parameters.",
            "relative_mae": _matrix_summary(refit_rel),
            "unclipped_normalized_mse": _matrix_summary(refit_mse),
        },
        "final_condition_number": condition_number(final_R, selections[-1]["lambda"]),
    }


def _serializable_records(records):
    return [
        {
            "name": record["name"],
            "sample_key": record["sample_key"],
            "fit_only_target_scale": record["scale"],
            "split_manifest": record["split_manifest"],
            "observation_counts": {
                role: {
                    "patch": record[role]["patch"]["n"],
                    "image": record[role]["image"]["n"],
                }
                for role in ("fit", "full")
            },
            "validation_images": record["val"]["fused"]["n"],
        }
        for record in records
    ]


def build_cost_report(d_aug, domain_count, projection_dim):
    shared = state_cost_bytes(d_aug, heads=2)
    single_revisitable = state_cost_bytes(d_aug, heads=2, domains=domain_count)
    single_one_pass = state_cost_bytes(d_aug, heads=2)
    single_one_pass["deployment_weights_bytes"] *= domain_count
    ranpac = state_cost_bytes(int(projection_dim) + 1, heads=1)
    ranpac["projection_matrix_bytes"] = int((d_aug - 1) * int(projection_dim) * 8)
    return {
        "shared_dual_head": shared,
        "single_domain_one_pass_peak_update_state": single_one_pass,
        "single_domain_revisitable_update_state": single_revisitable,
        "ranpac_style_image_only": ranpac,
        "warning": "R/C update state and W-only deployment state are intentionally separated.",
    }


def run_audit(domains, protocol, arguments, data_manifest):
    records, d_aug = collect_domain_records(
        domains,
        arguments.patch_target,
        arguments.alpha,
        arguments.split_seed,
        arguments.val_every,
    )
    reference = run_fixed_reference(
        domains, records, arguments.reference_lambda, arguments.alpha
    )
    shared = {
        method: run_shared_method(domains, records, method, arguments.lambdas, arguments.alpha)
        for method in SHARED_METHODS
    }
    single = run_single_domain(domains, records, arguments.lambdas, arguments.alpha)
    if arguments.skip_ranpac:
        ranpac = {"status": "not_run", "reason": "--skip-ranpac"}
    else:
        ranpac = run_ranpac_style(
            domains,
            records,
            arguments.lambdas,
            arguments.projection_dim,
            arguments.projection_seed,
            arguments.split_seed,
            arguments.val_every,
        )

    final_losses = {
        "pooled_reference_lambda": reference["relative_mae"]["final_balanced_loss"],
        "pooled_tuned_lambda": shared[METHOD_POOLED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "domain_balanced_tuned": shared[METHOD_DOMAIN_BALANCED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "mass_matched_domain_balanced_tuned": shared[METHOD_DOMAIN_BALANCED_MASS_MATCHED]["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
        "single_domain_tuned": single["aligned_fit_only_final_balanced_loss"],
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "evidence_role": "development-only",
        "confirmation_data_used": False,
        "primary_training_protocol": "aligned_fit_only",
        "secondary_training_protocol": "full_refit_ablation",
        "selection_objective": protocol["lambda_selection_objective"],
        "downstream_metric": protocol["downstream_metric"],
        "data_manifest": data_manifest,
        "domain_records": _serializable_records(records),
        "pooled_reference_lambda": reference,
        "shared_methods": shared,
        "single_domain": single,
        "ranpac_style_image": ranpac,
        "cost_report": build_cost_report(d_aug, len(records), arguments.projection_dim),
        "gain_decomposition": baseline_gain_decomposition(final_losses),
    }


class _ToySpec:
    def __init__(self, name):
        self.name = name
        self.sample_key = name


class _ToyDomains:
    """Small deterministic stream for smoke tests; no natural evidence."""

    def __init__(self, seed=7):
        self.domains = [_ToySpec(f"toy-{index}") for index in range(3)]
        rng = np.random.default_rng(seed)
        self._data = {"train": [], "test": []}
        base = np.asarray([0.8, -0.4, 0.3])
        for domain in range(3):
            direction = base + domain * np.asarray([0.12, 0.05, -0.08])
            for split, size in (("train", 30), ("test", 12)):
                images = []
                for image in range(size):
                    X = rng.normal(loc=0.15 * domain, scale=1.0, size=(4, 3))
                    raw = np.maximum(X @ direction + 1.0 + 0.1 * rng.normal(size=4), 0.02)
                    Y = raw[:, None]
                    images.append((X, Y, X.shape[0], f"d{domain}-{split}-{image:03d}"))
                self._data[split].append(images)

    def n_domains(self):
        return len(self.domains)

    def stream(self, split, index, with_ids=False):
        for X, Y, count, image_id in self._data[split][index]:
            if with_ids:
                yield X.copy(), Y.copy(), count, image_id
            else:
                yield X.copy(), Y.copy(), count

    def data_manifest(self):
        return {"kind": "deterministic-toy", "domains": 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "new_method_development_v1.json"),
    )
    parser.add_argument("--domain-config")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "runs_real" / "new_method_development_v1" / "smoke"))
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--patch-target")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lambdas")
    parser.add_argument("--reference-lambda", type=float)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--projection-seed", type=int)
    parser.add_argument("--skip-ranpac", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = _load_json(args.protocol_config)
    args.val_every = args.val_every or int(protocol["val_every"])
    args.patch_target = args.patch_target or protocol["patch_target"]
    args.alpha = float(protocol["alpha_patch"] if args.alpha is None else args.alpha)
    args.lambdas = (
        [float(value) for value in args.lambdas.split(",")]
        if args.lambdas
        else [float(value) for value in protocol["lambda_grid"]]
    )
    args.reference_lambda = float(
        protocol["reference_lambda"]
        if args.reference_lambda is None
        else args.reference_lambda
    )
    args.projection_dim = int(
        protocol["projection"]["dimension"]
        if args.projection_dim is None
        else args.projection_dim
    )
    args.projection_seed = int(
        protocol["projection"]["seed"]
        if args.projection_seed is None
        else args.projection_seed
    )
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if args.projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    output = require_development_output(args.out)

    if args.selftest:
        if args.domain_config:
            raise RuntimeError("--selftest and --domain-config are mutually exclusive")
        domains = _ToyDomains()
        data_manifest = domains.data_manifest()
        args.projection_dim = min(args.projection_dim, 16)
    else:
        if not args.domain_config:
            raise RuntimeError("pass --domain-config or --selftest")
        require_frozen_method_arguments(args, protocol)
        domain_payload = _load_json(args.domain_config)
        assert_development_domain_config(args.domain_config, domain_payload, protocol)
        from datasets_real import RealCountingDomains, build_from_config

        domains = RealCountingDomains(
            build_from_config(domain_payload),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
        )
        data_manifest = domains.data_manifest()

    started = time.time()
    payload = run_audit(domains, protocol, args, data_manifest)
    payload["elapsed_seconds"] = time.time() - started
    payload["selftest"] = bool(args.selftest)
    destination = output / "baseline_audit.json"
    result = with_provenance(payload, args.protocol_config, vars(args))
    dump_result(destination, result)
    print(json.dumps(payload["gain_decomposition"], indent=2, ensure_ascii=False))
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
