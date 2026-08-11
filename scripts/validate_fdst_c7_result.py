#!/usr/bin/env python3
"""Fail-closed validation for the single-shot C7-fdst-v3 result.

The formal result lives below ignored ``runs_real/``.  This validator binds it
to the frozen commit, protocol files, split manifest and publication table.  It
does not run a model, regenerate features or mutate any artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RESULT_SHA256 = (
    "0dfa51f7bf1008be170b43daf0969bb4c037f35c1cf50936d314770384bccdc3"
)
EXPECTED_FREEZE_COMMIT = "a398bc908c32670ac68f8e38b6fbb9ba404ccf3e"
EXPECTED_PROTOCOL = "C7-fdst-v3"
EXPECTED_MANIFEST_SHA256 = (
    "e0eb6692b62b979579da1af97fb4e802c6127cd91fcdb40d9b914b6c2cb87c8c"
)
EXPECTED_APPENDIX_SHA256 = (
    "3043263c071d24ee351f2a252c24147421c9554eb34830261de5c0747c8007a1"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "runs_real" / "fdst_c7" / "fdst_c7.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper" / "figs" / "table_fdst_c7.csv",
    )
    args = parser.parse_args()

    result_path = args.result.resolve()
    table_path = args.table.resolve()
    failures: list[str] = []
    if not result_path.is_file():
        print(f"FAIL missing formal result: {result_path}")
        return 1
    if not table_path.is_file():
        print(f"FAIL missing publication table: {table_path}")
        return 1

    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / "configs" / "fdst_splits" / "manifest.json"
    appendix_path = ROOT / "docs" / "C7_FDST_APPENDIX_A.json"
    lock_path = result_path.parent / "attempt.lock"

    require(digest(result_path) == EXPECTED_RESULT_SHA256,
            "formal result SHA256 differs from the first-run artifact", failures)
    require(result.get("protocol") == EXPECTED_PROTOCOL,
            "protocol is not C7-fdst-v3", failures)
    provenance = result.get("_provenance", {})
    require(provenance.get("git_commit") == EXPECTED_FREEZE_COMMIT,
            "result is not bound to the split-freeze commit", failures)
    require(provenance.get("git_dirty") is False,
            "formal run did not start from a clean worktree", failures)
    require(result.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            "result manifest hash differs from the frozen value", failures)
    require(digest(manifest_path) == EXPECTED_MANIFEST_SHA256,
            "tracked manifest no longer matches the formal result", failures)
    require(result.get("appendix_a_sha256") == EXPECTED_APPENDIX_SHA256,
            "result Appendix A hash differs from the frozen value", failures)
    require(digest(appendix_path) == EXPECTED_APPENDIX_SHA256,
            "tracked Appendix A no longer matches the formal result", failures)

    require(lock_path.is_file(), "single-shot attempt.lock is missing", failures)
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        require(lock.get("commit") == EXPECTED_FREEZE_COMMIT,
                "attempt.lock commit differs from the result", failures)
        require(isinstance(lock.get("started_at_utc"), str),
                "attempt.lock has no start time", failures)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest.get("label_audit", {})
    require(manifest.get("protocol") == EXPECTED_PROTOCOL,
            "manifest protocol mismatch", failures)
    require(audit.get("frames_checked") == 9000,
            "formal label audit did not cover all 9,000 train frames", failures)
    require(audit.get("clipped_points") == 327,
            "formal clipped-point count differs from the frozen manifest", failures)
    require(audit.get("frames_with_clipped_points") == 319,
            "formal clipped-frame count differs from the frozen manifest", failures)
    require(audit.get("hard_out_of_bounds_points") == 0,
            "formal label audit contains a hard out-of-bounds point", failures)

    domains = result.get("data_manifest", {}).get("domains", [])
    require(len(domains) == 6, "formal result does not contain six domains", failures)
    require(len({row.get("scene_id") for row in domains}) == 6,
            "formal result domains are not six distinct scenes", failures)
    require(all(row.get("test", {}).get("count") == 30 for row in domains),
            "a formal FDST domain does not have 30 test frames", failures)

    baseline = float(result.get("rel_f1", float("nan")))
    oracle = result.get("oracle_best", {})
    require(close(baseline, 0.09446164637323735),
            "f=1 balanced rel-MAE changed", failures)
    require(oracle.get("factor") == 1.0 and close(float(oracle.get("rel_MAE", -1)), baseline),
            "fixed-factor Oracle does not equal f=1", failures)
    require(close(float(result.get("oracle_improvement_rel", float("nan"))), 0.0),
            "Oracle improvement is not zero", failures)
    bootstrap = result.get("oracle_selection_adjusted_bootstrap", {})
    require(bootstrap.get("B") == 2000,
            "Oracle bootstrap replicate count is not 2,000", failures)
    require(close(float(bootstrap.get("ci95_lower_rel", float("nan"))), 0.0)
            and close(float(bootstrap.get("ci95_upper_rel", float("nan"))), 0.0),
            "Oracle selection-adjusted interval is not [0,0]", failures)

    expected_classification = {
        "construction_candidate": False,
        "gap_confirmed": False,
        "opportunity_absent": True,
        "opportunity_present": False,
        "useful_train_only_methods": [],
    }
    require(result.get("classification") == expected_classification,
            "formal classification differs from the frozen NO-GO outcome", failures)

    expected_rows = {
        "f1": (baseline, 0.0, "", ""),
        "fixed_f_oracle": (baseline, 0.0, "false", ""),
        "fstar_pred": (
            float(result["methods"]["fstar_pred"]["balanced_rel_mae"]),
            float(result["methods"]["fstar_pred"]["verdict"]["improvement_rel"]),
            "false", "true"),
        "shrinkage_gamma": (
            float(result["methods"]["shrinkage_gamma"]["balanced_rel_mae"]),
            float(result["methods"]["shrinkage_gamma"]["verdict"]["improvement_rel"]),
            "false", "true"),
        "precision_weighting": (
            float(result["methods"]["precision_weighting"]["balanced_rel_mae"]),
            float(result["methods"]["precision_weighting"]["verdict"]["improvement_rel"]),
            "false", "false"),
    }
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["rule"]: row for row in csv.DictReader(handle)}
    require(set(rows) == set(expected_rows),
            "publication table method roster differs from the formal result", failures)
    for name, expected in expected_rows.items():
        if name not in rows:
            continue
        row = rows[name]
        require(close(float(row["balanced_rel_mae"]), expected[0]),
                f"{name} rel-MAE differs from formal result", failures)
        require(close(float(row["improvement_rel"]), expected[1]),
                f"{name} improvement differs from formal result", failures)
        require(row["positive"] == expected[2],
                f"{name} positive verdict differs from formal result", failures)
        require(row["safe"] == expected[3],
                f"{name} safety verdict differs from formal result", failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(
        "PASS FDST C7-v3 single-shot result: hash/provenance/manifest/"
        "classification/publication table verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
