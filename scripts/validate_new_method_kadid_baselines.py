#!/usr/bin/env python3
"""Validate one KADID development-v1 first-batch artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "new_method_kadid_development_v1.json"
EXPECTED_SHARED = {
    "pooled_f1",
    "domain_balanced",
    "domain_balanced_mass_matched",
}
INVALIDATION_REASON = (
    "cross-domain reference-content leakage between fit and validation roles"
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _finite(value, where):
    _require(isinstance(value, (int, float)) and math.isfinite(value), f"{where} is not finite")


def validate_payload(payload, *, allow_invalidated=False):
    _require(
        payload.get("protocol_id") == "rrcl-new-method-kadid-development-v1",
        "wrong protocol id",
    )
    _require(
        allow_invalidated,
        f"invalidated protocol: {INVALIDATION_REASON}",
    )
    _require(payload.get("evidence_role") == "development-only", "wrong evidence role")
    _require(payload.get("confirmation_data_used") is False, "confirmation flag must be false")
    _require("fdst" not in json.dumps(payload.get("data_manifest", {})).lower(), "FDST contamination")
    records = payload.get("domain_records", [])
    _require(records, "domain records missing")
    for index, record in enumerate(records):
        _require(record.get("fit_samples", 0) > 0, f"domain {index}: empty fit")
        _require(record.get("validation_samples", 0) > 0, f"domain {index}: empty validation")
        manifest = record.get("split_manifest", {})
        _require(manifest.get("unit") == "reference_content_group", f"domain {index}: wrong split unit")
        _require(manifest.get("fit_groups", {}).get("count", 0) > 0, f"domain {index}: empty fit groups")
        _require(
            manifest.get("validation_groups", {}).get("count", 0) > 0,
            f"domain {index}: empty validation groups",
        )

    shared = payload.get("shared_methods", {})
    _require(set(shared) == EXPECTED_SHARED, "shared method roster mismatch")
    for method, result in shared.items():
        selections = result.get("lambda_selection", [])
        _require(len(selections) == len(records), f"{method}: boundary count mismatch")
        for selection in selections:
            trace = selection.get("candidate_trace", [])
            _require(trace, f"{method}: empty lambda trace")
            expected = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
            _require(selection.get("lambda") == expected["lambda"], f"{method}: wrong lambda")
        for role in ("aligned_fit_only", "full_refit_ablation"):
            for metric in ("relative_mae", "unclipped_normalized_mse"):
                _finite(
                    result[role][metric]["final_balanced_loss"],
                    f"{method} {role} {metric}",
                )
        _require("warning" in result["full_refit_ablation"], f"{method}: refit warning missing")

    single = payload.get("single_domain", {})
    _require(single.get("task_aware_domain_id_required") is True, "single-head domain-id warning missing")
    _require(len(single.get("domains", [])) == len(records), "single-head domain count mismatch")

    projection = payload.get("ranpac_style_projection", {})
    if projection.get("status") != "not_run":
        _require("not full RanPAC" in projection.get("label", ""), "projection qualification missing")
        for role in ("aligned_fit_only", "full_refit_ablation"):
            for metric in ("relative_mae", "unclipped_normalized_mse"):
                _finite(
                    projection[role][metric]["final_balanced_loss"],
                    f"projection {role} {metric}",
                )

    decomposition = payload.get("gain_decomposition", {})
    for key in ("candidate_family_gap", "selection_gap", "deployment_adoption_gap"):
        _require(
            decomposition.get(key, {}).get("status") == "not_measured_in_batch1",
            f"{key} must remain unmeasured",
        )
    for key in (
        "regularization_gain_vs_reference",
        "canonical_domain_balance_gain_vs_tuned_pooled",
        "mass_matched_balance_gain_vs_tuned_pooled",
        "best_simple_shared_gain_vs_reference",
        "shared_constraint_gap_best_simple_minus_single_domain",
    ):
        _finite(decomposition.get(key), f"decomposition {key}")
    return True


def _resolve_domain_config(recorded_path, protocol, explicit_path=None):
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        _require(candidate.is_file(), f"explicit domain config missing: {candidate}")
        return candidate

    recorded = Path(recorded_path).expanduser() if recorded_path else None
    if recorded is not None and recorded.is_file():
        return recorded

    basename = protocol.get("domain_config_basename")
    _require(bool(basename), "protocol domain config basename missing")
    fallback = ROOT / "configs" / basename
    _require(
        fallback.is_file(),
        f"domain config missing at recorded path and repository fallback: {fallback}",
    )
    return fallback


def validate_file(
    path,
    protocol_path=DEFAULT_PROTOCOL,
    domain_config_path=None,
    *,
    allow_invalidated=False,
):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload(payload, allow_invalidated=allow_invalidated)
    protocol_path = Path(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    provenance = payload.get("_provenance", {})
    _require(
        provenance.get("config_sha256") == hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol hash mismatch",
    )
    _require(
        provenance.get("config_snapshot") == json.loads(protocol_bytes),
        "protocol snapshot mismatch",
    )
    if not payload.get("selftest"):
        domain_config = payload.get("domain_config", {})
        domain_path = _resolve_domain_config(
            domain_config.get("path"),
            json.loads(protocol_bytes),
            domain_config_path,
        )
        _require(
            domain_path.name == json.loads(protocol_bytes)["domain_config_basename"],
            "domain config basename mismatch",
        )
        _require(
            domain_config.get("sha256")
            == hashlib.sha256(domain_path.read_bytes()).hexdigest(),
            "domain config hash mismatch",
        )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--domain-config",
        help="relocated domain config; its basename and SHA-256 must match the artifact",
    )
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
