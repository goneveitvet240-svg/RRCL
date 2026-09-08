#!/usr/bin/env python3
"""Validate one KADID scale-audit-v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "new_method_kadid_scale_audit_v2.json"
EXPECTED_METHODS = {"sample_mean_pooled", "domain_balanced"}
EXPECTED_SCOPE = {
    "projection_and_independent_heads": "retained_from_batch1_not_rerun",
    "fixed_trajectory_bank": "not_measured_in_batch2",
    "risk_controlled_hard_selection": "not_measured_in_batch2",
    "analytic_shrinkage": "not_measured_in_batch2",
    "automatic_f": "not_measured_in_batch2",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _finite(value, where):
    _require(
        isinstance(value, (int, float)) and math.isfinite(value),
        f"{where} is not finite",
    )


def _resolve_domain_config(recorded_path, protocol, explicit_path=None):
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        _require(candidate.is_file(), f"explicit domain config missing: {candidate}")
        return candidate
    recorded = Path(recorded_path).expanduser() if recorded_path else None
    if recorded is not None and recorded.is_file():
        return recorded
    fallback = ROOT / "configs" / protocol["domain_config_basename"]
    _require(fallback.is_file(), f"domain config missing: {fallback}")
    return fallback


def validate_payload(payload, protocol=None):
    protocol = protocol or json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    _require(payload.get("protocol_id") == protocol["protocol_id"], "wrong protocol id")
    _require(payload.get("evidence_role") == "development-only", "wrong evidence role")
    _require(payload.get("confirmation_data_used") is False, "confirmation flag must be false")
    _require("fdst" not in json.dumps(payload.get("data_manifest", {})).lower(), "FDST contamination")

    records = payload.get("domain_records", [])
    _require(len(records) > 0, "domain records missing")
    for index, record in enumerate(records):
        _require(record.get("fit_samples", 0) > 0, f"domain {index}: empty fit")
        _require(record.get("validation_samples", 0) > 0, f"domain {index}: empty validation")
        manifest = record.get("split_manifest", {})
        _require(manifest.get("unit") == "reference_content_group", f"domain {index}: wrong split unit")
        _require(manifest.get("fit_groups", {}).get("count", 0) > 0, f"domain {index}: empty fit groups")
        _require(manifest.get("validation_groups", {}).get("count", 0) > 0, f"domain {index}: empty validation groups")

    grid = payload.get("lambda_grid", [])
    _require(len(grid) == int(protocol["lambda_grid"]["points"]), "lambda grid length mismatch")
    _require(all(isinstance(value, (int, float)) and value > 0 for value in grid), "invalid lambda grid")
    _require(grid == sorted(grid) and len(set(grid)) == len(grid), "lambda grid must be unique and sorted")

    methods = payload.get("normalized_methods", {})
    _require(set(methods) == EXPECTED_METHODS, "normalized method roster mismatch")
    tolerance = float(protocol["equivalence_relative_tolerance"])
    for method, result in methods.items():
        _require(result.get("objective_total_observation_weight") == 1.0, f"{method}: objective mass mismatch")
        selections = result.get("lambda_selection", [])
        _require(len(selections) == len(records), f"{method}: boundary count mismatch")
        for boundary, selection in enumerate(selections):
            trace = selection.get("candidate_trace", [])
            _require(len(trace) == len(grid), f"{method}/{boundary}: incomplete lambda trace")
            expected = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
            _require(selection.get("lambda") == expected["lambda"], f"{method}/{boundary}: wrong selected lambda")
            audit = selection.get("equivalence_audit", {})
            _require(audit.get("normalized_method") == method, f"{method}/{boundary}: audit method mismatch")
            _require(audit.get("relation") == "lambda_raw = fit_observation_count * lambda_normalized", f"{method}/{boundary}: relation missing")
            _require(audit.get("fit_observation_count", 0) > 0, f"{method}/{boundary}: count missing")
            expected_raw = audit["fit_observation_count"] * selection["lambda"]
            _require(math.isclose(audit.get("raw_equivalent_lambda"), expected_raw, rel_tol=1e-12), f"{method}/{boundary}: raw lambda mismatch")
            _require(audit.get("weights_relative_l2", math.inf) <= tolerance, f"{method}/{boundary}: weight equivalence failed")
            _require(audit.get("weights_max_abs", math.inf) <= tolerance, f"{method}/{boundary}: absolute weight equivalence failed")
            _require(audit.get("validation_risk_absolute_difference", math.inf) <= tolerance, f"{method}/{boundary}: risk equivalence failed")
            statistics = audit.get("statistics", {})
            expected_statistics = {
                "R_max_abs_after_dividing_raw_by_N",
                "C_max_abs_after_dividing_raw_by_N",
                "domain_weight_max_abs_after_dividing_raw_by_N",
            }
            _require(set(statistics) == expected_statistics, f"{method}/{boundary}: statistic audit incomplete")
            for key, value in statistics.items():
                _finite(value, f"{method}/{boundary}/{key}")
                _require(value <= tolerance, f"{method}/{boundary}: statistic equivalence failed")
        final = result.get("final_system", {})
        _require(math.isclose(final.get("total_observation_weight", 0.0), 1.0, rel_tol=1e-12, abs_tol=1e-12), f"{method}: normalized mass is not one")
        for metric in ("relative_mae", "unclipped_normalized_mse"):
            _finite(result["aligned_fit_only"][metric]["final_balanced_loss"], f"{method}/{metric}")

    pairs = payload.get("same_lambda_pairs", {})
    _require(set(pairs) == EXPECTED_METHODS, "paired anchor roster mismatch")
    for anchor, result in pairs.items():
        _require(result.get("anchor_selected_on_validation_only") is True, f"{anchor}: selection warning missing")
        _require(len(result.get("lambda_by_boundary", [])) == len(records), f"{anchor}: lambda path length mismatch")
        _require(set(result.get("evaluations", {})) == EXPECTED_METHODS, f"{anchor}: paired method mismatch")
        _finite(result.get("final_domain_balance_gain_vs_sample_mean_pooled"), f"{anchor}: paired gain")

    practical = payload.get("practical_comparison", {})
    for key in (
        "sample_mean_pooled_final_relative_mae",
        "domain_balanced_final_relative_mae",
        "domain_balance_gain_vs_sample_mean_pooled",
    ):
        _finite(practical.get(key), f"practical/{key}")
    _require(payload.get("scope_status") == EXPECTED_SCOPE, "later-stage scope was changed or omitted")
    return True


def validate_file(path, protocol_path=DEFAULT_PROTOCOL, domain_config_path=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol_path = Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_payload(payload, protocol)
    provenance = payload.get("_provenance", {})
    _require(provenance.get("config_sha256") == hashlib.sha256(protocol_bytes).hexdigest(), "protocol hash mismatch")
    _require(provenance.get("config_snapshot") == protocol, "protocol snapshot mismatch")
    if not payload.get("selftest"):
        domain_record = payload.get("domain_config", {})
        domain_path = _resolve_domain_config(
            domain_record.get("path"), protocol, domain_config_path
        )
        _require(domain_path.name == protocol["domain_config_basename"], "domain config basename mismatch")
        _require(domain_record.get("sha256") == hashlib.sha256(domain_path.read_bytes()).hexdigest(), "domain config hash mismatch")
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
