#!/usr/bin/env python3
"""Validate and independently reproduce a synthetic-v1 result package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_tmlr_synthetic_v1 import summarize
from tmlr_synthetic import METHODS, PROTOCOL, file_sha, load_config


ROOT = Path(__file__).resolve().parents[1]


def validate_paper_table(runs: Path):
    failures = []
    summary = json.loads((runs / "summary.json").read_text(encoding="utf-8"))
    table_path = ROOT / "paper" / "figs" / "table_tmlr_synthetic.csv"
    if not table_path.is_file():
        return [f"missing paper table {table_path}"]
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111"]
    if [row.get("cell_id") for row in rows] != expected_ids:
        return ["paper synthetic table must contain the eight scale-off cells in order"]
    columns = {
        "fixed_f_oracle_gain_pct": "fixed_f_oracle",
        "fstar_pred_gain_pct": "fstar_pred",
        "precision_gain_pct": "precision_weighting",
        "shrinkage_gamma_gain_pct": "shrinkage_gamma",
        "vff_rls_gain_pct": "vff_rls",
    }
    for row in rows:
        identifier = row["cell_id"]
        cell = summary["cells"][identifier]
        for column, method in columns.items():
            expected = 100.0 * cell["methods"][method]["utility_vs_f1"]["mean"]
            if abs(float(row[column]) - expected) > 0.0005:
                failures.append(
                    f"paper table {identifier}/{column} differs: "
                    f"{row[column]} vs {expected:.6f}"
                )
        if (row["opportunity_present"].lower() == "true") != bool(
            cell["opportunity_present"]
        ):
            failures.append(f"paper table {identifier} opportunity verdict differs")
    return failures


def validate(runs: Path, config_path: Path, allow_testing: bool = False):
    config = load_config(config_path)
    failures = []
    manifest_path = runs / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        failures.append("manifest protocol mismatch")
    if manifest.get("config_sha256") != file_sha(config_path):
        failures.append("manifest config hash mismatch")
    if tuple(manifest.get("method_roster", ())) != METHODS:
        failures.append("manifest method roster mismatch")
    if manifest.get("secondary_methods") != config.get("secondary_methods"):
        failures.append("manifest secondary-method decision mismatch")
    if manifest.get("testing") and not allow_testing:
        failures.append("testing artifacts cannot validate as formal evidence")
    if manifest.get("git_dirty") and not allow_testing:
        failures.append("formal artifact provenance is dirty")
    entries = manifest.get("entries", [])
    expected = 16 * len(config["seeds"]) * len(config["orders"])
    if len(entries) != expected or manifest.get("expected_scenarios") != expected:
        failures.append(f"scenario count mismatch: expected {expected}")
    for entry in entries:
        path = runs / entry.get("path", "")
        if not path.is_file():
            failures.append(f"missing raw artifact {path}")
            continue
        if file_sha(path) != entry.get("sha256"):
            failures.append(f"raw artifact hash mismatch {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("_provenance", {})
        if provenance.get("config_sha256") != file_sha(config_path):
            failures.append(f"raw config hash mismatch {path}")
        if provenance.get("git_dirty") and not allow_testing:
            failures.append(f"dirty raw provenance {path}")
    if failures:
        return failures
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        regenerated_json = temporary / "summary.json"
        regenerated_csv = temporary / "summary.csv"
        summarize(runs, config_path, regenerated_json, regenerated_csv)
        for regenerated, existing in (
            (regenerated_json, runs / "summary.json"),
            (regenerated_csv, runs / "summary.csv"),
        ):
            if not existing.is_file():
                failures.append(f"missing derived artifact {existing}")
            elif regenerated.read_bytes() != existing.read_bytes():
                failures.append(f"derived artifact does not reproduce: {existing}")
    if not allow_testing:
        failures.extend(validate_paper_table(runs))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_real/tmlr_synthetic_v1")
    parser.add_argument("--config", default="configs/tmlr_synthetic_v1.json")
    parser.add_argument("--allow-testing", action="store_true")
    args = parser.parse_args()
    failures = validate(Path(args.runs), Path(args.config), args.allow_testing)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print("PASS TMLR synthetic-v1: raw hashes and derived summary/CSV reproduced")


if __name__ == "__main__":
    main()
