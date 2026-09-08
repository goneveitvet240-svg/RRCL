#!/usr/bin/env python3
"""Validate one leakage-safe KADID group-CV-v4 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_manifest import identifiers_manifest  # noqa: E402
from kadid_group_cv import (  # noqa: E402
    FOLD_ALGORITHM,
    assign_group_folds,
    fold_manifest,
    paired_group_bootstrap,
)


DEFAULT_PROTOCOL = ROOT / "configs" / "kadid_group_cv_v4.json"
METHOD_LINEAR_POOLED = "linear_sample_mean_pooled"
METHOD_LINEAR_BALANCED = "linear_domain_balanced"
METHOD_INDEPENDENT = "independent_linear_domain_heads"
METHOD_PROJECTION = "image_mean_gaussian_relu_projection_sample_mean_pooled"
EXPECTED_METHODS = {
    METHOD_LINEAR_POOLED,
    METHOD_LINEAR_BALANCED,
    METHOD_INDEPENDENT,
    METHOD_PROJECTION,
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _finite(value, where):
    _require(
        isinstance(value, (int, float)) and math.isfinite(value),
        f"{where} is not finite",
    )


def _close(actual, expected, where, *, rel_tol=1e-11, abs_tol=1e-12):
    _finite(actual, where)
    _finite(expected, f"{where}/expected")
    _require(
        math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{where} mismatch: {actual} != {expected}",
    )


def _lambda_grid(protocol):
    spec = protocol["lambda_grid"]
    _require(spec.get("kind") == "log10", "wrong lambda-grid kind")
    return np.logspace(
        float(spec["minimum_exponent"]),
        float(spec["maximum_exponent"]),
        int(spec["points"]),
        dtype=np.float64,
    ).tolist()


def _validate_group_loss_rows(rows, expected_groups, where):
    _require(isinstance(rows, list), f"{where}: group losses missing")
    ids = [str(row.get("group_id")) for row in rows]
    _require(ids == sorted(set(ids)), f"{where}: duplicate or unsorted group ids")
    _require(set(ids) == set(expected_groups), f"{where}: group coverage mismatch")
    for index, row in enumerate(rows):
        _finite(row.get("balanced_normalized_mse"), f"{where}/{index}/loss")
        _require(row["balanced_normalized_mse"] >= 0.0, f"{where}/{index}: negative loss")


def _validate_matrix_summary(summary, domains, where):
    matrix = summary.get("matrix")
    _require(isinstance(matrix, list) and len(matrix) == domains, f"{where}: matrix rows")
    for row_index, row in enumerate(matrix):
        _require(isinstance(row, list) and len(row) == domains, f"{where}: matrix columns")
        for column, value in enumerate(row):
            if column <= row_index:
                _finite(value, f"{where}/{row_index}/{column}")
            else:
                _require(
                    value is None or (isinstance(value, float) and math.isnan(value)),
                    f"{where}/{row_index}/{column}: unseen entry must be null",
                )
    final = [float(value) for value in matrix[-1]]
    _close(summary.get("final_balanced_loss"), float(np.mean(final)), f"{where}/final")
    for key in ("nBwT", "final_oldest_mae", "final_newest_mae"):
        _finite(summary.get(key), f"{where}/{key}")


def _validate_trace(trace, selected, grid, where):
    _require(len(trace) == len(grid), f"{where}: incomplete lambda trace")
    _require([row.get("lambda") for row in trace] == grid, f"{where}: grid mismatch")
    for index, row in enumerate(trace):
        _finite(row.get("oof_validation_risk"), f"{where}/{index}/risk")
        _require(row["oof_validation_risk"] >= 0.0, f"{where}/{index}: negative risk")
    expected = min(
        trace, key=lambda item: (item["oof_validation_risk"], -item["lambda"])
    )
    _require(selected == expected["lambda"], f"{where}: wrong selected lambda")
    return expected


def _validate_shared(result, method, grid, reference_lambda, domains, groups):
    _require(result.get("method") == method, f"{method}: method label mismatch")
    expected_objective = (
        "domain_balanced" if method == METHOD_LINEAR_BALANCED else "sample_mean_pooled"
    )
    _require(result.get("objective") == expected_objective, f"{method}: objective mismatch")
    selections = result.get("cv_selected", {}).get("lambda_selection", [])
    _require(len(selections) == domains, f"{method}: boundary count mismatch")
    for boundary, selection in enumerate(selections):
        _require(selection.get("boundary") == boundary, f"{method}/{boundary}: boundary mismatch")
        expected = _validate_trace(
            selection.get("candidate_trace", []),
            selection.get("selected_lambda"),
            grid,
            f"{method}/{boundary}",
        )
        oof = selection.get("selected_oof", {})
        _close(
            oof.get("balanced_normalized_mse"),
            expected["oof_validation_risk"],
            f"{method}/{boundary}/selected OOF",
        )
        _finite(oof.get("balanced_relative_mae"), f"{method}/{boundary}/relative MAE")
        _require(
            len(oof.get("per_domain_normalized_mse", [])) == boundary + 1,
            f"{method}/{boundary}: per-domain MSE length",
        )
        _require(
            len(oof.get("per_domain_relative_mae", [])) == boundary + 1,
            f"{method}/{boundary}: per-domain relative-MAE length",
        )
        if boundary == domains - 1:
            _validate_group_loss_rows(oof.get("group_losses"), groups, f"{method}/final OOF")
    fixed = result.get("fixed_reference_lambda", {})
    _require(fixed.get("lambda") == reference_lambda, f"{method}: reference lambda mismatch")
    fixed_oof = fixed.get("oof_by_boundary", [])
    _require(len(fixed_oof) == domains, f"{method}: fixed OOF boundary count")
    _validate_group_loss_rows(
        fixed_oof[-1].get("group_losses"), groups, f"{method}/fixed final OOF"
    )
    for branch in (result["cv_selected"], fixed):
        trajectory = branch.get("development_test", {})
        _validate_matrix_summary(trajectory.get("relative_mae", {}), domains, f"{method}/test relative")
        _validate_matrix_summary(
            trajectory.get("unclipped_normalized_mse", {}),
            domains,
            f"{method}/test normalized MSE",
        )
        _finite(trajectory.get("final_condition_number"), f"{method}/condition number")
        _require(
            len(trajectory.get("final_target_scales", [])) == domains,
            f"{method}: target-scale count",
        )


def _validate_independent(result, grid, reference_lambda, records, groups):
    _require(result.get("method") == METHOD_INDEPENDENT, "independent method label")
    _require(result.get("task_aware_domain_id_required") is True, "independent task-id warning")
    details = result.get("domains", [])
    _require(len(details) == len(records), "independent domain count")
    for index, (detail, record) in enumerate(zip(details, records)):
        _require(detail.get("domain") == record.get("name"), f"independent/{index}: domain order")
        expected = _validate_trace(
            detail.get("candidate_trace", []),
            detail.get("selected_lambda"),
            grid,
            f"independent/{index}",
        )
        _close(
            detail.get("selected_oof", {}).get("normalized_mse"),
            expected["oof_validation_risk"],
            f"independent/{index}/selected OOF",
        )
        for branch in (
            "selected_oof",
            "reference_oof",
            "development_test_selected",
            "development_test_reference",
        ):
            for value in detail.get(branch, {}).values():
                _finite(value, f"independent/{index}/{branch}")
    for branch in ("cv_selected", "fixed_reference_lambda"):
        summary = result.get(branch, {})
        if branch == "fixed_reference_lambda":
            _require(summary.get("lambda") == reference_lambda, "independent reference lambda")
        for key, value in summary.items():
            if key not in {"lambda", "final_group_losses"}:
                _finite(value, f"independent/{branch}/{key}")
        _validate_group_loss_rows(
            summary.get("final_group_losses"), groups, f"independent/{branch}"
        )


def _group_loss_dict(rows):
    return {
        str(row["group_id"]): float(row["balanced_normalized_mse"])
        for row in rows
    }


def _selected_final_group_losses(result):
    if result["method"] == METHOD_INDEPENDENT:
        rows = result["cv_selected"]["final_group_losses"]
    else:
        rows = result["cv_selected"]["lambda_selection"][-1]["selected_oof"]["group_losses"]
    return _group_loss_dict(rows)


def validate_payload(payload, protocol=None):
    protocol = protocol or json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    _require(payload.get("protocol_id") == protocol["protocol_id"], "wrong protocol id")
    _require(payload.get("evidence_role") == "development-only", "wrong evidence role")
    _require(payload.get("confirmation_data_used") is False, "confirmation flag")
    _require("fdst" not in json.dumps(payload.get("data_manifest", {})).lower(), "FDST contamination")

    cv = protocol["group_cv"]
    manifest = payload.get("fold_manifest", {})
    rows = manifest.get("assignments", [])
    group_ids = [str(row.get("group_id")) for row in rows]
    _require(group_ids == sorted(set(group_ids)), "fold assignments duplicate or unsorted")
    assignments = {str(row["group_id"]): int(row["fold"]) for row in rows}
    recomputed_assignments = assign_group_folds(
        group_ids, cv["sample_key"], cv["seed"], cv["folds"]
    )
    _require(assignments == recomputed_assignments, "forged fold assignment")
    expected_manifest = fold_manifest(
        recomputed_assignments, cv["sample_key"], cv["seed"], cv["folds"]
    )
    _require(manifest == expected_manifest, "fold manifest mismatch")
    _require(manifest.get("algorithm") == FOLD_ALGORITHM == cv["algorithm"], "fold algorithm")
    _require(max(manifest["fold_counts"]) - min(manifest["fold_counts"]) <= 1, "unbalanced folds")
    if not payload.get("selftest"):
        _require(
            manifest["group_count"] == int(protocol["expected_train_reference_groups"]),
            "natural train reference-group count mismatch",
        )

    records = payload.get("domain_records", [])
    _require(records, "domain records missing")
    if not payload.get("selftest"):
        _require(
            [record.get("name") for record in records] == protocol["expected_domains"],
            "natural domain roster/order mismatch",
        )
    global_train = set()
    for index, record in enumerate(records):
        train_ids = [str(value) for value in record.get("train_group_ids", [])]
        test_ids = [str(value) for value in record.get("test_group_ids", [])]
        _require(train_ids == sorted(set(train_ids)) and train_ids, f"domain {index}: train ids")
        _require(test_ids == sorted(set(test_ids)) and test_ids, f"domain {index}: test ids")
        _require(not (set(train_ids) & set(test_ids)), f"domain {index}: train/test leakage")
        _require(identifiers_manifest(train_ids) == record.get("train_groups"), f"domain {index}: train manifest")
        _require(identifiers_manifest(test_ids) == record.get("test_groups"), f"domain {index}: test manifest")
        _require(record.get("train_samples", 0) >= len(train_ids), f"domain {index}: train sample count")
        _require(record.get("test_samples", 0) >= len(test_ids), f"domain {index}: test sample count")
        global_train.update(train_ids)
    _require(global_train == set(group_ids), "fold assignments do not cover domain train groups")

    grid = payload.get("lambda_grid", [])
    _require(grid == _lambda_grid(protocol), "frozen lambda grid mismatch")
    reference_lambda = float(protocol["reference_normalized_lambda"])
    _require(payload.get("reference_normalized_lambda") == reference_lambda, "reference lambda")
    methods = payload.get("methods", {})
    _require(set(methods) == EXPECTED_METHODS, "method roster mismatch")
    for method in (METHOD_LINEAR_POOLED, METHOD_LINEAR_BALANCED, METHOD_PROJECTION):
        _require(methods[method].get("status") != "not_run", f"{method}: required method not run")
        _validate_shared(methods[method], method, grid, reference_lambda, len(records), group_ids)
    _require(
        methods[METHOD_PROJECTION].get("qualification") == protocol["projection"]["qualification"],
        "projection qualification mismatch",
    )
    _validate_independent(methods[METHOD_INDEPENDENT], grid, reference_lambda, records, group_ids)

    comparisons = payload.get("paired_oof_bootstrap_vs_linear_sample_mean_pooled", {})
    expected_candidates = [METHOD_LINEAR_BALANCED, METHOD_INDEPENDENT, METHOD_PROJECTION]
    _require(set(comparisons) == set(expected_candidates), "bootstrap roster mismatch")
    baseline = _selected_final_group_losses(methods[METHOD_LINEAR_POOLED])
    spec = protocol["bootstrap"]
    for offset, method in enumerate(expected_candidates):
        expected = paired_group_bootstrap(
            baseline,
            _selected_final_group_losses(methods[method]),
            seed=int(spec["seed"]) + offset,
            resamples=int(spec["resamples"]),
            confidence=float(spec["confidence"]),
        )
        _require(comparisons[method] == expected, f"{method}: forged bootstrap summary")

    expected_scope = {
        "fixed_trajectory_bank": "not_measured_in_batch4",
        "risk_controlled_hard_selection": "not_measured_in_batch4",
        "analytic_shrinkage": "not_measured_in_batch4",
        "patch_level_nonlinear_projection": "not_measured_in_batch4",
        "automatic_f": "not_measured_in_batch4",
    }
    _require(payload.get("scope_status") == expected_scope, "later method scope changed or omitted")
    _require("not independent confirmation" in payload.get("test_use_warning", ""), "test-use warning")
    return True


def _resolve_domain_config(recorded_path, protocol, explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
    elif recorded_path and Path(recorded_path).expanduser().is_file():
        path = Path(recorded_path).expanduser()
    else:
        path = ROOT / "configs" / protocol["domain_config_basename"]
    _require(path.is_file(), f"domain config missing: {path}")
    return path


def validate_file(path, protocol_path=DEFAULT_PROTOCOL, domain_config_path=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol_path = Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_payload(payload, protocol)
    provenance = payload.get("_provenance", {})
    _require(
        provenance.get("config_sha256") == hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol hash mismatch",
    )
    _require(provenance.get("config_snapshot") == protocol, "protocol snapshot mismatch")
    if not payload.get("selftest"):
        domain_record = payload.get("domain_config", {})
        domain_path = _resolve_domain_config(domain_record.get("path"), protocol, domain_config_path)
        _require(domain_path.name == protocol["domain_config_basename"], "domain config basename")
        _require(
            domain_record.get("sha256") == hashlib.sha256(domain_path.read_bytes()).hexdigest(),
            "domain config hash mismatch",
        )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--domain-config")
    args = parser.parse_args()
    validate_file(args.path, args.protocol_config, args.domain_config)
    print(f"VALID: {args.path}")


if __name__ == "__main__":
    main()
