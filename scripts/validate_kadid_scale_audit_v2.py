#!/usr/bin/env python3
"""Validate one KADID scale-audit-v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from holdout_split import is_validation  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "configs" / "new_method_kadid_scale_audit_v2.json"
EXPECTED_METHODS = {"sample_mean_pooled", "domain_balanced"}
INVALIDATED_PROTOCOLS = {
    "rrcl-new-method-kadid-scale-audit-v2": (
        "cross-domain reference-content leakage between fit and validation roles"
    )
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _finite(value, where):
    _require(
        isinstance(value, (int, float)) and math.isfinite(value),
        f"{where} is not finite",
    )


def _ids_manifest(values):
    values = sorted(str(value) for value in values)
    return {
        "count": len(values),
        "ids_sha256": hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest(),
    }


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


def validate_payload(payload, protocol=None, *, allow_invalidated=False):
    protocol = protocol or json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    _require(payload.get("protocol_id") == protocol["protocol_id"], "wrong protocol id")
    invalidation = INVALIDATED_PROTOCOLS.get(payload.get("protocol_id"))
    _require(
        allow_invalidated or invalidation is None,
        f"invalidated protocol: {invalidation}",
    )
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

    selector_sample_key = protocol.get("selector_sample_key")
    if selector_sample_key is not None:
        audit = payload.get("selector_isolation_audit", {})
        _require(audit.get("scope") == "global_across_domains", "selector split is not global")
        _require(audit.get("selector_sample_key") == selector_sample_key, "selector sample key mismatch")
        _require(audit.get("cross_role_overlap_count") == 0, "cross-domain fit/validation leakage")
        domain_roles = audit.get("domains", [])
        _require(len(domain_roles) == len(records), "selector domain-role audit missing")
        global_fit, global_validation = set(), set()
        for index, (record, roles) in enumerate(zip(records, domain_roles)):
            _require(roles.get("name") == record.get("name"), f"domain {index}: role name mismatch")
            fit = set(roles.get("fit_groups", []))
            validation = set(roles.get("validation_groups", []))
            _require(fit and validation and not (fit & validation), f"domain {index}: invalid role groups")
            split = record["split_manifest"]
            _require(split.get("selector_sample_key") == selector_sample_key, f"domain {index}: wrong selector key")
            _require(split.get("scope") == "global_across_domains", f"domain {index}: wrong selector scope")
            _require(_ids_manifest(fit) == split["fit_groups"], f"domain {index}: fit manifest mismatch")
            _require(_ids_manifest(validation) == split["validation_groups"], f"domain {index}: validation manifest mismatch")
            for group in fit:
                _require(not is_validation(selector_sample_key, group, protocol["selector_split_seed"], protocol["selector_val_every"]), f"domain {index}: forged fit role")
            for group in validation:
                _require(is_validation(selector_sample_key, group, protocol["selector_split_seed"], protocol["selector_val_every"]), f"domain {index}: forged validation role")
            global_fit.update(fit)
            global_validation.update(validation)
        _require(not (global_fit & global_validation), "recomputed cross-domain role overlap")
        _require(_ids_manifest(global_fit) == audit["global_fit_groups"], "global fit manifest mismatch")
        _require(_ids_manifest(global_validation) == audit["global_validation_groups"], "global validation manifest mismatch")
        _require(
            audit.get("cross_role_overlap_ids_sha256")
            == _ids_manifest(global_fit & global_validation)["ids_sha256"],
            "cross-role overlap hash mismatch",
        )

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
    batch_label = protocol.get("scope_batch_label", "batch2")
    expected_scope = {
        "projection_and_independent_heads": protocol.get(
            "projection_and_independent_heads_status",
            "retained_from_batch1_not_rerun",
        ),
        "fixed_trajectory_bank": f"not_measured_in_{batch_label}",
        "risk_controlled_hard_selection": f"not_measured_in_{batch_label}",
        "analytic_shrinkage": f"not_measured_in_{batch_label}",
        "automatic_f": f"not_measured_in_{batch_label}",
    }
    _require(payload.get("scope_status") == expected_scope, "later-stage scope was changed or omitted")
    return True


def validate_file(
    path,
    protocol_path=DEFAULT_PROTOCOL,
    domain_config_path=None,
    *,
    allow_invalidated=False,
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol_path = Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_payload(payload, protocol, allow_invalidated=allow_invalidated)
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
    parser.add_argument(
        "--allow-invalidated-audit-trail",
        action="store_true",
        help="check historical file integrity even though its evidence protocol is invalid",
    )
    args = parser.parse_args()
    validate_file(
        args.path,
        args.protocol_config,
        args.domain_config,
        allow_invalidated=args.allow_invalidated_audit_trail,
    )
    print(f"VALID: {args.path}")


if __name__ == "__main__":
    main()
