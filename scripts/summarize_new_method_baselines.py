#!/usr/bin/env python3
"""Aggregate validated RRCL development-v1 baseline audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from result_io import dump_result  # noqa: E402
from validate_new_method_baselines import validate_file  # noqa: E402


FIELDS = (
    "pooled_reference_lambda",
    "pooled_tuned_lambda",
    "domain_balanced_tuned",
    "mass_matched_domain_balanced_tuned",
    "single_domain_tuned",
    "regularization_gain_vs_reference",
    "canonical_domain_balance_gain_vs_tuned_pooled",
    "mass_matched_balance_gain_vs_tuned_pooled",
    "best_simple_shared_gain_vs_reference",
    "shared_constraint_gap_best_simple_minus_single_domain",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean_sem(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "sem": None}
    return {
        "n": len(values),
        "mean": mean,
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
    }


def summarize(paths, include_selftest=False):
    rows = []
    sources = []
    for path in sorted(Path(item) for item in paths):
        validate_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("selftest") and not include_selftest:
            continue
        decomposition = payload["gain_decomposition"]
        losses = decomposition["losses"]
        arguments = payload.get("_provenance", {}).get("arguments", {})
        row = {
            "source": str(path),
            "domain_config": arguments.get("domain_config"),
            "sample_seed": arguments.get("sample_seed"),
            "split_seed": arguments.get("split_seed"),
            **losses,
            **{
                key: decomposition[key]
                for key in FIELDS
                if key not in losses
            },
            "pooled_selected_lambda": payload["shared_methods"]["pooled_f1"]["final_system"]["selected_lambda"],
            "domain_balanced_selected_lambda": payload["shared_methods"]["domain_balanced"]["final_system"]["selected_lambda"],
            "mass_matched_selected_lambda": payload["shared_methods"]["domain_balanced_mass_matched"]["final_system"]["selected_lambda"],
        }
        ranpac = payload.get("ranpac_style_image", {})
        row["ranpac_style_image"] = (
            None
            if ranpac.get("status") == "not_run"
            else ranpac["aligned_fit_only"]["relative_mae"]["final_balanced_loss"]
        )
        rows.append(row)
        sources.append({"path": str(path), "sha256": _sha256(path)})
    if not rows:
        raise RuntimeError("no non-selftest development artifacts found")
    aggregates = {
        field: _mean_sem([row[field] for row in rows])
        for field in FIELDS
    }
    ranpac_values = [row["ranpac_style_image"] for row in rows if row["ranpac_style_image"] is not None]
    if ranpac_values:
        aggregates["ranpac_style_image"] = _mean_sem(ranpac_values)
    return {
        "protocol_id": "rrcl-new-method-development-v1",
        "evidence_role": "development-only",
        "run_count": len(rows),
        "rows": rows,
        "aggregates": aggregates,
        "sources": sources,
        "decision_status": {
            "simple_baseline_oracle_recovery": "not_evaluable_until_oracle_opportunity_is_joined",
            "candidate_family_gap": "not_measured_in_batch1",
            "selection_gap": "not_measured_in_batch1",
            "external_confirmation": "not_run_and_fdst_remains_forbidden",
        },
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(ROOT / "runs_real" / "new_method_development_v1"),
    )
    parser.add_argument("--include-selftest", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.out) if args.out else root / "summary"
    payload = summarize(
        root.glob("**/baseline_audit.json"),
        include_selftest=args.include_selftest,
    )
    dump_result(output / "baseline_summary.json", payload)
    write_csv(output / "baseline_runs.csv", payload["rows"])
    print(f"validated and summarized {payload['run_count']} run(s) -> {output}")


if __name__ == "__main__":
    main()
