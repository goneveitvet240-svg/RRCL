#!/usr/bin/env python3
"""Validate only the artifacts that support the current RRCL TMLR claims.

The historical full-release validator intentionally remains stricter and
currently fails on experiments removed from the manuscript.  This validator
does not weaken that audit.  It verifies the two retained evidence packages:
C4 reliability reweighting and fixed-factor theory calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C4_ROOT = ROOT / "runs_real" / "c4_gamma"
FSTAR_ROOT = ROOT / "runs_real" / "fstar_calibration"

C4_COMMIT = "bb2e3afe969670f74eee69a4952dbaab8618a8e5"
FSTAR_COMMIT = "609399ac2039c9f280b61d54206da439ff5a53bc"

EXPECTED_HASHES = {
    C4_ROOT / "c4_gamma_summary.json":
        "5636f037631a3b0eee9cd79036a504eb2ba034206496f375908b52262a434ec5",
    FSTAR_ROOT / "fstar_calibration_summary.json":
        "d880313e8d76c45fe324d4229ccb5a1fdc64da50db4fd4ab201daedcc4e97bc7",
    FSTAR_ROOT / "fstar_points.csv":
        "8bd15865c63a38b7d7e04de536c64ddfb6b2aacafe93577fc4860b4ecd7d3d9d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_records(
    root: Path,
    filename: str,
    expected_count: int,
    expected_commit: str,
) -> list[str]:
    failures: list[str] = []
    records = sorted(root.rglob(filename))
    if len(records) != expected_count:
        failures.append(
            f"{root.relative_to(ROOT)}: expected {expected_count} {filename}, "
            f"found {len(records)}"
        )
    for path in records:
        relative = path.relative_to(ROOT)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{relative}: invalid JSON: {error}")
            continue
        provenance = payload.get("_provenance", {})
        if provenance.get("git_commit") != expected_commit:
            failures.append(f"{relative}: wrong source commit")
        if provenance.get("git_dirty") is not False:
            failures.append(f"{relative}: source worktree was not clean")
        if not payload.get("data_manifest"):
            failures.append(f"{relative}: missing data_manifest")
        if not payload.get("holdout_split"):
            failures.append(f"{relative}: missing holdout_split")
        if payload.get("diagnostic_only") is not True:
            failures.append(f"{relative}: diagnostic_only is not true")
    return failures


def run_checked(
    command: list[str],
    environment: dict[str, str],
    allowed_returncodes: tuple[int, ...] = (0,),
) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed.stdout


def validate_paper_tables() -> list[str]:
    failures: list[str] = []
    c4 = json.loads((C4_ROOT / "c4_gamma_summary.json").read_text())
    with (ROOT / "paper" / "figs" / "table_c4_material_decision.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        table_rows = list(csv.DictReader(handle))
    expected_rows = sorted(
        c4["rows"], key=lambda row: (row["family"], int(row["seed"]))
    )
    actual_rows = sorted(
        table_rows, key=lambda row: (row["family"].lower(), int(row["seed"]))
    )
    if len(actual_rows) != len(expected_rows):
        failures.append("C4 paper table row count differs from summary")
    else:
        for actual, expected in zip(actual_rows, expected_rows):
            checks = {
                "family": actual["family"].lower() == expected["family"],
                "seed": int(actual["seed"]) == int(expected["seed"]),
                "gamma_pred": abs(float(actual["gamma_pred"]) - expected["train_fused_gamma"]) < 1e-12,
                "gamma_oracle": abs(float(actual["gamma_oracle"]) - expected["test_gamma_diagnostic"]) < 1e-12,
                "oracle_gain_pct": abs(float(actual["oracle_gain_pct"]) - 100 * expected["oracle_gain_relative"]) < 5e-4,
                "selected_gain_pct": abs(float(actual["selected_gain_pct"]) - 100 * expected["selected_gain_relative"]) < 5e-4,
                "correct": (actual["correct"].lower() == "true") is bool(expected["correct_material_decision"]),
            }
            for field, passed in checks.items():
                if not passed:
                    failures.append(
                        f"C4 paper table mismatch: {expected['family']} "
                        f"seed {expected['seed']} field {field}"
                    )

    fstar = json.loads(
        (FSTAR_ROOT / "fstar_calibration_summary.json").read_text()
    )
    with (ROOT / "paper" / "figs" / "table_fstar_calibration.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        aggregate_rows = {row["family"]: row for row in csv.DictReader(handle)}
    sources = {"All": fstar["overall"], **fstar["by_family"]}
    raw_rows = fstar["rows"]
    for family, metrics in sources.items():
        actual = aggregate_rows.get(family)
        if actual is None:
            failures.append(f"fixed-factor paper table missing family {family}")
            continue
        family_rows = raw_rows if family == "All" else [
            row for row in raw_rows if row["family"] == family
        ]
        bias = sum(row["f_pred"] - row["f_oracle"] for row in family_rows) / len(family_rows)
        checks = {
            "n": int(actual["n"]) == int(metrics["n_points"]),
            "factor_mae": abs(float(actual["factor_mae"]) - metrics["mae_factor"]) < 5e-4,
            "bias": abs(float(actual["bias"]) - bias) < 5e-4,
            "oracle_mse_gain_pct": abs(float(actual["oracle_mse_gain_pct"]) - 100 * metrics["mean_oracle_mse_gain_relative"]) < 5e-4,
            "selected_mse_gain_pct": abs(float(actual["selected_mse_gain_pct"]) - 100 * metrics["mean_selected_mse_gain_relative"]) < 5e-4,
            "mse_regret_pct": abs(float(actual["mse_regret_pct"]) - 100 * metrics["mean_oracle_mse_regret_at_f_pred_relative"]) < 5e-4,
            "selected_rel_gain_pct": abs(float(actual["selected_rel_gain_pct"]) - 100 * metrics["mean_selected_rel_gain_relative"]) < 5e-4,
        }
        for field, passed in checks.items():
            if not passed:
                failures.append(
                    f"fixed-factor paper table mismatch: {family} field {field}"
                )
    return failures


def main() -> int:
    failures: list[str] = []
    failures.extend(
        validate_records(C4_ROOT, "shrinkage_diagnostic.json", 20, C4_COMMIT)
    )
    failures.extend(
        validate_records(FSTAR_ROOT, "fstar_calibration.json", 40, FSTAR_COMMIT)
    )
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            failures.append(f"missing retained artifact: {path.relative_to(ROOT)}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {path.relative_to(ROOT)}")

    if not failures:
        with tempfile.TemporaryDirectory(prefix="rrcl-tmlr-audit-") as directory:
            temporary = Path(directory)
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(temporary / "matplotlib")
            try:
                run_checked(
                    [
                        sys.executable,
                        "scripts/summarize_gamma_c4.py",
                        "--runs",
                        str(C4_ROOT),
                        "--out",
                        str(temporary / "c4.json"),
                    ],
                    environment,
                    allowed_returncodes=(0, 1),  # 1 is the frozen C4 scientific FAIL.
                )
                run_checked(
                    [
                        sys.executable,
                        "scripts/summarize_fstar_calibration.py",
                        "--runs",
                        str(FSTAR_ROOT),
                        "--out",
                        str(temporary / "fstar.json"),
                        "--csv",
                        str(temporary / "fstar.csv"),
                    ],
                    environment,
                )
                run_checked(
                    [
                        sys.executable,
                        "scripts/plot_fstar_calibration.py",
                        "--summary",
                        str(temporary / "fstar.json"),
                        "--out",
                        str(temporary / "scatter.pdf"),
                    ],
                    environment,
                )
            except subprocess.CalledProcessError as error:
                failures.append(
                    "recomputation command failed: "
                    + (error.stderr.strip() or str(error))
                )
            else:
                byte_pairs = (
                    (temporary / "c4.json", C4_ROOT / "c4_gamma_summary.json"),
                    (temporary / "fstar.json", FSTAR_ROOT / "fstar_calibration_summary.json"),
                    (temporary / "fstar.csv", FSTAR_ROOT / "fstar_points.csv"),
                    (temporary / "scatter.png", ROOT / "paper" / "figs" / "fstar_calibration_scatter.png"),
                )
                for generated, reference in byte_pairs:
                    if generated.read_bytes() != reference.read_bytes():
                        failures.append(
                            f"recomputed {generated.name} differs from "
                            f"{reference.relative_to(ROOT)}"
                        )

    if not failures:
        failures.extend(validate_paper_tables())

    print("RRCL TMLR claim-set validation")
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        print(f"STATUS: NOT READY ({len(failures)} issue(s))")
        return 1
    print("  C4: 20/20 clean raw artifacts; summary and paper table reproduced")
    print("  fixed-factor: 40/40 clean raw artifacts; summary, CSV, plot and paper table reproduced")
    print("STATUS: CORE CLAIM SET READY (archive validation is a separate build step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
