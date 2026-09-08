#!/usr/bin/env python3
"""Validate one RRCL new-method development-v1 baseline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "new_method_development_v1.json"

EXPECTED_SHARED = {
    "pooled_f1",
    "domain_balanced",
    "domain_balanced_mass_matched",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _finite(value, where):
    _require(isinstance(value, (int, float)) and math.isfinite(value), f"{where} is not finite")


def validate_payload(payload):
    _require(payload.get("protocol_id") == "rrcl-new-method-development-v1", "wrong protocol id")
    _require(payload.get("evidence_role") == "development-only", "wrong evidence role")
    _require(payload.get("confirmation_data_used") is False, "confirmation data flag must be false")
    identity_text = json.dumps(
        {
            "data_manifest": payload.get("data_manifest", {}),
            "domain_records": payload.get("domain_records", []),
        },
        sort_keys=True,
    ).lower()
    _require("fdst" not in identity_text, "FDST token found in dataset identity")

    records = payload.get("domain_records", [])
    _require(records, "domain_records must be non-empty")
    for index, record in enumerate(records):
        split = record.get("split_manifest", {})
        _require(split.get("fit_count", 0) > 0, f"domain {index} has empty fit split")
        _require(split.get("val_count", 0) > 0, f"domain {index} has empty validation split")
        _require(record.get("validation_images") == split.get("val_count"), f"domain {index} split count mismatch")

    shared = payload.get("shared_methods", {})
    _require(set(shared) == EXPECTED_SHARED, "shared method roster mismatch")
    for method, result in shared.items():
        selections = result.get("lambda_selection", [])
        _require(len(selections) == len(records), f"{method}: one selection required per boundary")
        for selection in selections:
            trace = selection.get("candidate_trace", [])
            _require(trace, f"{method}: empty lambda trace")
            for item in trace:
                _finite(item.get("lambda"), f"{method} lambda")
                _finite(item.get("validation_risk"), f"{method} validation risk")
            expected = min(trace, key=lambda item: (item["validation_risk"], -item["lambda"]))
            _require(selection.get("lambda") == expected["lambda"], f"{method}: wrong selected lambda")
        _require("warning" in result.get("full_refit_ablation", {}), f"{method}: refit warning missing")
        for protocol_role in ("aligned_fit_only", "full_refit_ablation"):
            for metric in ("relative_mae", "unclipped_normalized_mse"):
                _finite(
                    result[protocol_role][metric]["final_balanced_loss"],
                    f"{method} {protocol_role} {metric}",
                )

    single = payload.get("single_domain", {})
    _require(single.get("task_aware_domain_id_required") is True, "single-domain task-awareness missing")
    _require(len(single.get("domains", [])) == len(records), "single-domain roster mismatch")

    ranpac = payload.get("ranpac_style_image", {})
    if ranpac.get("status") != "not_run":
        _require("not full RanPAC" in ranpac.get("label", ""), "RanPAC-style qualification missing")
        _finite(
            ranpac["aligned_fit_only"]["relative_mae"]["final_balanced_loss"],
            "RanPAC-style relative MAE",
        )
        _require("warning" in ranpac.get("full_refit_ablation", {}), "RanPAC-style refit warning missing")
        _finite(
            ranpac["full_refit_ablation"]["relative_mae"]["final_balanced_loss"],
            "RanPAC-style refit relative MAE",
        )

    decomposition = payload.get("gain_decomposition", {})
    for gap in ("candidate_family_gap", "selection_gap", "deployment_adoption_gap"):
        _require(
            decomposition.get(gap, {}).get("status") == "not_measured_in_batch1",
            f"{gap} must remain unmeasured in batch 1",
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


def validate_file(path, protocol_path=DEFAULT_PROTOCOL):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload(payload)
    protocol_path = Path(protocol_path)
    expected_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    provenance = payload.get("_provenance", {})
    _require(provenance.get("config_sha256") == expected_hash, "protocol config hash mismatch")
    current_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(provenance.get("config_snapshot") == current_protocol, "protocol snapshot mismatch")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL))
    args = parser.parse_args()
    validate_file(args.path, args.protocol_config)
    print(f"VALID: {args.path}")


if __name__ == "__main__":
    main()
