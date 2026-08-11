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
FSTAR_CALIBRATION_CONFIGS = [
    "jhu_sha_shb",
    "jhu_shb_sha",
    "sha_jhu_shb",
    "sha_shb_jhu",
    "qnrf_sha_shb",
    "qnrf_shb_sha",
    "sha_shb_qnrf",
    "sha_shb",
]


def _reject_nonstandard_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


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
        files.append(f"analytic_{name}/analytic_baselines.json")
    for name in core:
        files.append(f"analytic_crowd_{name}/analytic_baselines.json")
    files.extend(
        [
            "kadid/domains_iqa_kadid_result.json",
            "ava/ava_result.json",
            "nyu/nyu_result.json",
            "analytic_kadid/analytic_baselines.json",
            "analytic_ava/analytic_baselines.json",
        ]
    )
    return files


def validate_json(path, relative):
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
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
        errors.extend(_holdout_split_errors(payload, evidence))
    if relative.startswith("base_"):
        baseline = payload.get("cf_f1_full")
        if not isinstance(baseline, dict) or "M_rel" not in baseline:
            errors.append("missing full f=1 evaluation matrix")
        if "ranpac_image" in payload:
            errors.append("legacy baseline key ranpac_image must be named ranpac_style_image")
    if relative.startswith("vff_"):
        vff = payload.get("paleologu_batch_vff")
        if not isinstance(vff, dict):
            errors.append("missing explicit Paleologu batch-VFF result")
        else:
            # VFF's OWN boundary manifests (t >= 1), not just the RRCL
            # comparator's evidence.  A T-domain sequence must contain
            # exactly T-1 boundary records (the first entry is null).
            diagnostics = vff.get("diagnostics")
            boundary_records = [
                record for record in (diagnostics or []) if isinstance(record, dict)
            ]
            expected_boundaries = max(len(_data_manifest_keys(payload)) - 1, 0)
            if not boundary_records:
                errors.append("missing VFF boundary diagnostics")
            elif expected_boundaries and len(boundary_records) != expected_boundaries:
                errors.append(
                    f"VFF has {len(boundary_records)} boundary records, "
                    f"expected {expected_boundaries}"
                )
            top_seed = payload.get("split_seed")
            for record in boundary_records:
                split = record.get("holdout_split")
                if not isinstance(split, dict) or not (
                    split.get("fit_ids_sha256") and split.get("val_ids_sha256")
                ):
                    errors.append("missing VFF boundary holdout manifest")
                    break
                nested = split.get("split_seed")
                if (
                    isinstance(top_seed, int)
                    and nested is not None
                    and int(nested) != top_seed
                ):
                    errors.append("VFF boundary split_seed differs from top-level")
                    break
        rrcl = payload.get("rrcl_bounded", {})
        evidence = rrcl.get("evidence") if isinstance(rrcl, dict) else None
        errors.extend(_holdout_split_errors(payload, evidence))
    if relative in {
        "kadid/domains_iqa_kadid_result.json",
        "ava/ava_result.json",
    }:
        manifest = payload.get("data_manifest")
        if not isinstance(manifest, dict) or not manifest.get("domains"):
            errors.append("missing data_manifest with selected-file hashes")
        evaluation = payload.get("evaluation", {})
        for method in ("f1", "adaptive"):
            metrics = evaluation.get(method, {}).get("balanced", {})
            if not all(key in metrics for key in ("mae", "rmse", "plcc", "srcc")):
                errors.append(f"missing task-standard metrics for {method}")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    if relative == "nyu/nyu_result.json":
        manifest = payload.get("data_manifest")
        if not isinstance(manifest, dict) or not manifest.get("domains"):
            errors.append("missing data_manifest with selected-file hashes")
        metrics = payload.get("adaptive_standard_metrics")
        required = {"mae_m", "rmse_m", "abs_rel", "delta1", "delta2", "delta3", "silog"}
        if not isinstance(metrics, list) or not metrics:
            errors.append("missing adaptive depth standard metrics")
        elif any(not required.issubset(row) for row in metrics):
            errors.append("incomplete adaptive depth standard metrics")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    if relative.endswith("/age_result.json"):
        manifest = payload.get("data_manifest")
        if not isinstance(manifest, dict) or not manifest.get("domains"):
            errors.append("missing data_manifest with selected-file hashes")
        evaluation = payload.get("evaluation", {})
        for method in ("f1", "adaptive"):
            metrics = evaluation.get(method, {}).get("balanced", {})
            if not all(key in metrics for key in ("mae", "rmse")):
                errors.append(f"missing age MAE/RMSE in years for {method}")
        if not isinstance(payload.get("verdict"), dict):
            errors.append("missing structured adaptive verdict")
    if relative.endswith("/analytic_baselines.json"):
        for method in ("dos_elm_style", "sift_rls"):
            result = payload.get(method)
            if not isinstance(result, dict):
                errors.append(f"missing analytic baseline {method}")
                continue
            if not isinstance(result.get("matrix"), list):
                errors.append(f"missing evaluation matrix for {method}")
            metrics = result.get("evaluation", {}).get("balanced", {})
            if not all(
                key in metrics for key in ("mae", "rmse", "plcc", "srcc", "relative_mae")
            ):
                errors.append(f"missing task metrics for {method}")
        sift = payload.get("sift_rls", {})
        if not isinstance(sift.get("train_only_selection_curve"), list):
            errors.append("missing train-only SIFt factor selection evidence")
        if relative.startswith("analytic_crowd_"):
            if "image-level" not in payload.get("comparison_scope", ""):
                errors.append("crowd analytic baseline scope is not marked image-level")
            manifest = payload.get("data_manifest")
            if not isinstance(manifest, dict) or not manifest.get("domains"):
                errors.append("missing crowd data_manifest")
    return errors


def _provenance_errors(payload):
    errors = []
    provenance = payload.get("_provenance")
    if not isinstance(provenance, dict):
        return ["missing _provenance"]
    if provenance.get("schema_version") != "rrcl-result-v2":
        errors.append("wrong provenance schema")
    if not provenance.get("git_commit"):
        errors.append("missing git commit")
    if not provenance.get("config_sha256"):
        errors.append("missing config hash")
    if provenance.get("git_dirty") is not False:
        errors.append("paper-facing result was not generated from a clean git tree")
    return errors


def _data_manifest_keys(payload):
    return [
        domain.get("sample_key")
        for domain in payload.get("data_manifest", {}).get("domains", [])
        if isinstance(domain, dict)
    ]


def _holdout_coverage_errors(entries, payload, top_seed):
    """Shared strictness for holdout manifests vs the data manifest."""
    errors = []
    keys = [
        entry.get("sample_key") for entry in entries if isinstance(entry, dict)
    ]
    if len(set(keys)) != len(keys) or not all(keys):
        errors.append("holdout manifest sample_keys are missing or duplicated")
    data_keys = _data_manifest_keys(payload)
    if data_keys and (
        len(entries) != len(data_keys) or set(keys) != set(data_keys)
    ):
        errors.append("holdout manifest does not cover data_manifest domains")
    if isinstance(top_seed, int):
        for entry in entries:
            nested = entry.get("split_seed") if isinstance(entry, dict) else None
            if nested is not None and int(nested) != top_seed:
                errors.append("nested holdout split_seed differs from top-level")
                break
    return errors


def _holdout_split_errors(payload, evidence):
    """Selector runs must record the per-seed stable-hash holdout partition."""
    errors = []
    top_seed = payload.get("split_seed")
    if not isinstance(top_seed, int):
        errors.append("missing selector split_seed")
    block = (evidence or {}).get("holdout_split") if isinstance(evidence, dict) else None
    domains = block.get("domains") if isinstance(block, dict) else None
    if not domains:
        errors.append("missing holdout split manifest")
        return errors
    for domain in domains:
        if not isinstance(domain, dict) or not (
            domain.get("fit_ids_sha256") and domain.get("val_ids_sha256")
        ):
            errors.append("incomplete holdout split manifest entry")
            break
    if isinstance(block, dict):
        nested = block.get("split_seed")
        if (
            isinstance(top_seed, int)
            and nested is not None
            and int(nested) != top_seed
        ):
            errors.append("holdout block split_seed differs from top-level")
    errors.extend(_holdout_coverage_errors(domains, payload, top_seed))
    return errors


def optional_artifact_errors(runs):
    """Validate NEW-METHOD artifacts wherever present.

    precision_*/precision_weighted.json and shrinkage_diag*/
    shrinkage_diagnostic.json are not yet in the frozen expected-file list
    (the paper route is undecided), but any that exist must already satisfy
    provenance, split and equivalence requirements.
    """
    failures = []
    for path in sorted(runs.rglob("precision_weighted.json")):
        relative = str(path.relative_to(runs))
        try:
            payload = json.loads(
                path.read_text(), parse_constant=_reject_nonstandard_json_constant
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"INVALID {relative}: invalid JSON: {exc}")
            continue
        for error in _provenance_errors(payload):
            failures.append(f"INVALID {relative}: {error}")
        if not isinstance(payload.get("split_seed"), int):
            failures.append(f"INVALID {relative}: missing split_seed")
        block = payload.get("holdout_split", {})
        if not isinstance(block, dict) or not block.get("domains"):
            failures.append(f"INVALID {relative}: missing holdout split manifest")
        else:
            for error in _holdout_coverage_errors(
                block["domains"], payload, payload.get("split_seed")
            ):
                failures.append(f"INVALID {relative}: {error}")
        if not isinstance(payload.get("data_manifest"), dict):
            failures.append(f"INVALID {relative}: missing data_manifest")
    for path in sorted(runs.rglob("shrinkage_diagnostic.json")):
        relative = str(path.relative_to(runs))
        try:
            payload = json.loads(
                path.read_text(), parse_constant=_reject_nonstandard_json_constant
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"INVALID {relative}: invalid JSON: {exc}")
            continue
        for error in _provenance_errors(payload):
            failures.append(f"INVALID {relative}: {error}")
        if payload.get("diagnostic_only") is not True:
            failures.append(f"INVALID {relative}: diagnostic_only flag missing")
        if not isinstance(payload.get("split_seed"), int):
            failures.append(f"INVALID {relative}: missing split_seed")
        equivalence = payload.get("equivalence", {})
        if equivalence.get("gamma0_equals_f1") is not True:
            failures.append(f"INVALID {relative}: gamma0 != f=1 equivalence")
        if equivalence.get("gamma1_equals_three_way_precision_endpoint") is not True:
            failures.append(
                f"INVALID {relative}: gamma1 != three-way precision endpoint"
            )
        block = payload.get("holdout_split", {})
        domains = block.get("domains") if isinstance(block, dict) else None
        if not domains:
            failures.append(f"INVALID {relative}: missing three-way role manifest")
        else:
            for domain in domains:
                roles = domain.get("roles", {}) if isinstance(domain, dict) else {}
                for role in ("fit", "precision_val", "gamma_val"):
                    if roles.get(role, {}).get("count", 0) <= 0:
                        failures.append(
                            f"INVALID {relative}: empty role {role} in manifest"
                        )
                        break
            for error in _holdout_coverage_errors(
                domains, payload, payload.get("split_seed")
            ):
                failures.append(f"INVALID {relative}: {error}")
    return failures


def config_identity_errors():
    errors = []
    signatures = {}
    for name in sorted(set(REORDER_CONFIGS) | set(FSTAR_CALIBRATION_CONFIGS)):
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
    failures.extend(optional_artifact_errors(runs))

    theory_root = runs / "fstar_calibration"
    theory = theory_root / "fstar_calibration_summary.json"
    theory_csv = theory_root / "fstar_points.csv"
    if not theory.exists():
        failures.append("MISSING fstar_calibration/fstar_calibration_summary.json")
    else:
        try:
            payload = json.loads(
                theory.read_text(),
                parse_constant=_reject_nonstandard_json_constant,
            )
            if payload.get("protocol") != "theory-f-calibration-v1":
                failures.append("INVALID fstar calibration protocol")
            if payload.get("diagnostic_only") is not True:
                failures.append("INVALID fstar calibration diagnostic_only flag")
            if len(payload.get("rows", [])) != 40:
                failures.append(
                    "INVALID fstar calibration: expected 40 scenario rows"
                )
            if not payload.get("source_commit"):
                failures.append("INVALID fstar calibration: missing source commit")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"INVALID fstar calibration summary: {error}")
    if not theory_csv.exists():
        failures.append("MISSING fstar_calibration/fstar_points.csv")
    else:
        with theory_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 40:
            failures.append(
                f"INVALID fstar_points.csv: expected 40 rows, found {len(rows)}"
            )

    missing_bib = missing_bibliography_keys()
    if missing_bib:
        failures.append("MISSING bibliography keys: " + ", ".join(missing_bib))

    print(f"RRCL release validation: {runs}")
    if failures:
        for failure in failures:
            print(f"  {failure}")
        print(f"STATUS: NOT READY ({len(failures)} issue(s))")
        return 1
    print(
        f"STATUS: READY ({len(expected_json_files())} JSON artifacts "
        "+ f* calibration)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
