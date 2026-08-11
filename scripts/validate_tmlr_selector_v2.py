#!/usr/bin/env python3
"""Independently regenerate and validate a TMLR selector-v2 result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_io import dump_result
from tmlr_selector_v2 import (
    METHODS,
    PROTOCOL,
    assemble_result,
    canonical_sha,
    evaluate_meta_test,
    file_sha,
    load_config,
    prepare_selection,
)


ROOT = Path(__file__).resolve().parents[1]


def validate_paper_table(result: dict):
    failures = []
    table_path = ROOT / "paper" / "figs" / "table_tmlr_selector_v2.csv"
    if not table_path.is_file():
        return [f"missing {table_path}"]
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        return ["selector-v2 paper table must contain 12 rows"]
    summary = result["summary"]["subsets"]
    for row in rows:
        subset = summary[row["subset"]]
        metrics = subset["methods"][row["method"]]
        expected = {
            "n": subset["task_count"],
            "utility_pct": 100 * metrics["utility_vs_f1"]["mean"],
            "utility_ci95_lower_pct": 100 * metrics["utility_vs_f1"]["ci95_lower"],
            "utility_ci95_upper_pct": 100 * metrics["utility_vs_f1"]["ci95_upper"],
            "regret_pct": 100 * metrics["regret_vs_oracle"]["mean"],
            "worst_mean_domain_harm_pct": 100 * metrics["worst_mean_domain_harm"],
        }
        if "factor_mae" in metrics:
            expected["factor_mae"] = metrics["factor_mae"]
        for key, value in expected.items():
            if key == "n":
                matches = int(row[key]) == int(value)
            else:
                matches = abs(float(row[key]) - float(value)) <= 5e-7
            if not matches:
                failures.append(
                    f"paper table {row['subset']}/{row['method']}/{key} differs"
                )
    return failures


def validate(runs: Path, config_path: Path, allow_testing: bool = False):
    failures = []
    config = load_config(config_path)
    required = (
        "attempt.lock",
        "selection.lock.json",
        "meta_train_records.json",
        "meta_validation_records.json",
        "result.json",
        "manifest.json",
    )
    for name in required:
        if not (runs / name).is_file():
            failures.append(f"missing {runs / name}")
    if failures:
        return failures
    manifest = json.loads((runs / "manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((runs / "selection.lock.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or lock.get("protocol") != PROTOCOL:
        failures.append("protocol mismatch")
    if manifest.get("config_sha256") != file_sha(config_path):
        failures.append("config hash mismatch")
    if manifest.get("testing") and not allow_testing:
        failures.append("testing result is not formal evidence")
    if manifest.get("git_dirty") and not allow_testing:
        failures.append("dirty result is not formal evidence")
    if not lock.get("created_before_meta_test"):
        failures.append("selection lock does not assert pre-test creation")
    if manifest.get("result_sha256") != file_sha(runs / "result.json"):
        failures.append("result hash mismatch")
    if manifest.get("selection_lock_sha256") != file_sha(runs / "selection.lock.json"):
        failures.append("selection-lock hash mismatch")
    if tuple(manifest.get("method_roster", ())) != METHODS:
        failures.append("method roster mismatch")
    stored_train = json.loads((runs / "meta_train_records.json").read_text(encoding="utf-8"))
    stored_validation = json.loads(
        (runs / "meta_validation_records.json").read_text(encoding="utf-8")
    )
    if lock.get("meta_train_records_sha256") != canonical_sha(stored_train):
        failures.append("meta-train records hash mismatch")
    if lock.get("meta_validation_records_sha256") != canonical_sha(stored_validation):
        failures.append("meta-validation records hash mismatch")
    if failures:
        return failures

    recomputed_train, recomputed_validation, recomputed_selection = prepare_selection(config)
    if recomputed_train != stored_train:
        failures.append("meta-train records do not regenerate")
    if recomputed_validation != stored_validation:
        failures.append("meta-validation records do not regenerate")
    if recomputed_selection != lock.get("selection"):
        failures.append("hyperparameter selection does not regenerate")
    if failures:
        return failures
    test_records = evaluate_meta_test(config, recomputed_train, recomputed_selection)
    result = assemble_result(
        config,
        recomputed_train,
        recomputed_validation,
        recomputed_selection,
        test_records,
    )
    with tempfile.TemporaryDirectory() as temporary:
        regenerated = Path(temporary) / "result.json"
        dump_result(regenerated, result)
        if regenerated.read_bytes() != (runs / "result.json").read_bytes():
            failures.append("result.json does not reproduce byte-for-byte")
    if not allow_testing:
        failures.extend(validate_paper_table(result))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_real/tmlr_selector_v2")
    parser.add_argument("--config", default="configs/tmlr_selector_v2.json")
    parser.add_argument("--allow-testing", action="store_true")
    args = parser.parse_args()
    failures = validate(Path(args.runs), Path(args.config), args.allow_testing)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print("PASS TMLR selector-v2: locks, hashes, selection and full result reproduced")


if __name__ == "__main__":
    main()
