#!/usr/bin/env python3
"""Validate that RRCL paper-facing results are complete and traceable."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REORDER_GROUPS = [
    ["jhu_sha_shb", "jhu_shb_sha", "sha_shb_jhu"],
    ["qnrf_sha_shb", "qnrf_shb_sha", "sha_shb_qnrf"],
]
REORDER_CONFIGS = sorted({name for group in REORDER_GROUPS for name in group})


def expected_json_files():
    files = []
    core = ["jhu_sha_shb", "jhu_shb_sha", "sha_shb", "sha_shb_jhu"]
    for name in core:
        files.extend(
            [
                f"normabl_{name}/norm_ablation.json",
                f"adaptf2_{name}/adaptive_f.json",
                f"base_{name}/baselines.json",
                f"vff_{name}/vff_baselines.json",
            ]
        )
    qnrf = ["qnrf_sha_shb", "qnrf_shb_sha", "sha_shb_qnrf"]
    for name in qnrf:
        files.extend(
            [
                f"normabl_{name}/norm_ablation.json",
                f"adaptf2_{name}/adaptive_f.json",
            ]
        )
    for name in ["qnrf_sha_shb", "qnrf_shb_sha", "jhu_sha_shb"]:
        files.append(f"normabl_proj_{name}/norm_ablation.json")
    for name in ["age_utk_young_old", "age_agedb_utk", "age_utk_agedb"]:
        files.append(f"{name}/age_result.json")
    files.extend(
        [
            "kadid/domains_iqa_kadid_result.json",
            "ava/ava_result.json",
            "nyu/nyu_result.json",
        ]
    )
    return files


def validate_json(path, relative):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid JSON: {exc}"]
    errors = []
    provenance = payload.get("_provenance")
    if not isinstance(provenance, dict):
        errors.append("missing _provenance")
    else:
        if provenance.get("schema_version") != "rrcl-result-v2":
            errors.append("wrong provenance schema")
        if not provenance.get("git_commit"):
            errors.append("missing git commit")
        if not provenance.get("config_sha256"):
            errors.append("missing config hash")
        if provenance.get("git_dirty") is not False:
            errors.append("paper-facing result was not generated from a clean git tree")

    if relative.startswith(("normabl_", "adaptf2_", "base_", "vff_")):
        manifest = payload.get("data_manifest")
        if not isinstance(manifest, dict) or not manifest.get("domains"):
            errors.append("missing data_manifest with selected-file hashes")
    if relative.startswith("adaptf2_"):
        verdict = payload.get("verdict")
        if not isinstance(verdict, dict) or "status" not in verdict:
            errors.append("missing adaptive verdict")
        evidence = payload.get("adaptive_evidence")
        if not isinstance(evidence, dict) or "M_rel" not in evidence:
            errors.append("missing adaptive evaluation matrix")
    if relative.startswith("base_"):
        baseline = payload.get("cf_f1_full")
        if not isinstance(baseline, dict) or "M_rel" not in baseline:
            errors.append("missing full f=1 evaluation matrix")
        if "ranpac_image" in payload:
            errors.append("legacy baseline key ranpac_image must be named ranpac_style_image")
    if relative.startswith("vff_"):
        if "paleologu_batch_vff" not in payload:
            errors.append("missing explicit Paleologu batch-VFF result")
    if relative in {
        "kadid/domains_iqa_kadid_result.json",
        "ava/ava_result.json",
    }:
        evaluation = payload.get("evaluation", {})
        for method in ("f1", "adaptive"):
            metrics = evaluation.get(method, {}).get("balanced", {})
            if not all(key in metrics for key in ("mae", "rmse", "plcc", "srcc")):
                errors.append(f"missing task-standard metrics for {method}")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    if relative == "nyu/nyu_result.json":
        metrics = payload.get("adaptive_standard_metrics")
        required = {"mae_m", "rmse_m", "abs_rel", "delta1", "delta2", "delta3", "silog"}
        if not isinstance(metrics, list) or not metrics:
            errors.append("missing adaptive depth standard metrics")
        elif any(not required.issubset(row) for row in metrics):
            errors.append("incomplete adaptive depth standard metrics")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    if relative.endswith("/age_result.json"):
        evaluation = payload.get("evaluation", {})
        for method in ("f1", "adaptive"):
            metrics = evaluation.get(method, {}).get("balanced", {})
            if not all(key in metrics for key in ("mae", "rmse")):
                errors.append(f"missing age MAE/RMSE in years for {method}")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    return errors


def config_identity_errors():
    errors = []
    signatures = {}
    for name in REORDER_CONFIGS:
        path = ROOT / "configs" / f"domains_{name}.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"CONFIG {path.name}: {exc}")
            continue
        for domain in payload.get("domains", []):
            key = domain.get("sample_key")
            if not key:
                errors.append(
                    f"CONFIG {path.name}: domain {domain.get('name')} has no sample_key"
                )
                continue
            signature = (
                domain.get("kind"),
                domain.get("root"),
                domain.get("train_split", "train"),
                domain.get("test_split", "test"),
            )
            previous = signatures.get(key)
            if previous is not None and previous != signature:
                errors.append(
                    f"CONFIG identity mismatch for {key}: {previous} != {signature}"
                )
            else:
                signatures[key] = signature
    return errors


def _manifest_by_key(payload):
    manifest = payload.get("data_manifest", {})
    return {
        domain["sample_key"]: {
            "train": domain.get("train"),
            "test": domain.get("test"),
        }
        for domain in manifest.get("domains", [])
        if domain.get("sample_key")
    }


def reorder_invariance_errors(runs, tolerance=1e-6):
    """Check the f=1 identity that must hold for matched reordered datasets."""
    errors = []
    for group in REORDER_GROUPS:
        loaded = {}
        for name in group:
            path = runs / f"normabl_{name}" / "norm_ablation.json"
            if not path.exists():
                continue
            try:
                loaded[name] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
        if len(loaded) != len(group):
            continue

        reference_name = group[0]
        reference = loaded[reference_name]
        reference_manifest = _manifest_by_key(reference)
        reference_score = reference["results"]["norm=mean,f=1"]["final_avg_rel"]
        for name in group[1:]:
            candidate = loaded[name]
            candidate_manifest = _manifest_by_key(candidate)
            if candidate_manifest != reference_manifest:
                errors.append(
                    f"ORDER INVARIANCE {name}: selected-file manifest differs from "
                    f"{reference_name}"
                )
                continue
            candidate_score = candidate["results"]["norm=mean,f=1"]["final_avg_rel"]
            if abs(float(candidate_score) - float(reference_score)) > tolerance:
                errors.append(
                    f"ORDER INVARIANCE {name}: f=1 final_avg_rel={candidate_score} "
                    f"differs from {reference_name}={reference_score}"
                )
    return errors


def missing_bibliography_keys():
    tex = (ROOT / "paper" / "main.tex").read_text()
    bib = (ROOT / "paper" / "refs.bib").read_text()
    cited = set()
    for match in re.finditer(r"\\cite\{([^}]*)\}", tex):
        cited.update(key.strip() for key in match.group(1).split(","))
    present = set(re.findall(r"@\w+\{\s*([^,\s]+)", bib))
    return sorted(cited - present)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_real")
    args = parser.parse_args()
    runs = args.runs.resolve()
    failures = config_identity_errors()
    for relative in expected_json_files():
        path = runs / relative
        if not path.exists():
            failures.append(f"MISSING {relative}")
            continue
        for error in validate_json(path, relative):
            failures.append(f"INVALID {relative}: {error}")
    failures.extend(reorder_invariance_errors(runs))

    theory = runs / "theory_points.csv"
    if not theory.exists():
        failures.append("MISSING theory_points.csv")
    else:
        with theory.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 10:
            failures.append(f"INVALID theory_points.csv: expected 10 rows, found {len(rows)}")

    missing_bib = missing_bibliography_keys()
    if missing_bib:
        failures.append("MISSING bibliography keys: " + ", ".join(missing_bib))

    print(f"RRCL release validation: {runs}")
    if failures:
        for failure in failures:
            print(f"  {failure}")
        print(f"STATUS: NOT READY ({len(failures)} issue(s))")
        return 1
    print(f"STATUS: READY ({len(expected_json_files())} JSON artifacts + theory CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
