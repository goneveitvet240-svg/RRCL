#!/usr/bin/env python3
"""Frozen single-command runner for RRCL TMLR controlled synthetic study v1."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_shrinkage import ensure_clean_worktree
from result_io import dump_result
from scripts.summarize_tmlr_synthetic_v1 import summarize
from tmlr_synthetic import (
    PROTOCOL,
    evaluate_scenario,
    factor_cells,
    file_sha,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "tmlr_synthetic_v1.json"


def _git(arguments):
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _expected(config):
    return 16 * len(config["seeds"]) * len(config["orders"])


def preflight(config_path: Path, output: Path, testing: bool) -> dict:
    config = load_config(config_path)
    issues = []
    formal_output = (ROOT / config["formal_output"]).resolve()
    if testing and output.resolve() == formal_output:
        issues.append("testing mode may not use the formal output path")
    if not testing:
        if config_path.resolve() != FORMAL_CONFIG.resolve():
            issues.append("formal run must use the frozen config path")
        if output.resolve() != formal_output:
            issues.append("formal run must use the frozen output path")
        status = _git(["status", "--porcelain"])
        if status:
            issues.append("git worktree is dirty")
    if output.exists():
        issues.append(f"output path already exists: {output}")
    return {
        "protocol": PROTOCOL,
        "config": str(config_path),
        "config_sha256": file_sha(config_path),
        "expected_scenarios": _expected(config),
        "output": str(output),
        "testing": testing,
        "issues": issues,
        "ready": not issues,
    }


def _acquire_lock(output: Path, commit: str | None):
    output.mkdir(parents=True, exist_ok=False)
    lock = output / "attempt.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": PROTOCOL,
                "git_commit": commit,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pid": os.getpid(),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return lock


def run(config_path: Path, output: Path, testing: bool):
    report = preflight(config_path, output, testing)
    if not report["ready"]:
        raise SystemExit("synthetic preflight blocked:\n  " + "\n  ".join(report["issues"]))
    config = load_config(config_path)
    commit = _git(["rev-parse", "HEAD"])
    if not testing:
        ensure_clean_worktree(False)
    _acquire_lock(output, commit)
    raw_root = output / "raw"
    entries = []
    common_provenance = {
        "schema_version": "rrcl-tmlr-synthetic-provenance-v1",
        "git_commit": commit,
        "git_dirty": bool(_git(["status", "--porcelain"])) if testing else False,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha(config_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "testing": testing,
    }
    for cell in factor_cells():
        identifier = "".join(str(cell[name]) for name in cell)
        for seed in config["seeds"]:
            for order_name in config["orders"]:
                payload = evaluate_scenario(config, cell, int(seed), order_name)
                payload["_provenance"] = dict(common_provenance)
                destination = (
                    raw_root / f"cell_{identifier}" / f"seed_{seed}" / order_name / "scenario.json"
                )
                dump_result(destination, payload)
                entries.append(
                    {
                        "path": destination.relative_to(output).as_posix(),
                        "sha256": file_sha(destination),
                        "cell_id": identifier,
                        "seed": int(seed),
                        "order_name": order_name,
                    }
                )
    manifest = {
        "schema_version": "rrcl-tmlr-synthetic-manifest-v1",
        "protocol": PROTOCOL,
        "config_sha256": file_sha(config_path),
        "git_commit": commit,
        "git_dirty": common_provenance["git_dirty"],
        "testing": testing,
        "expected_scenarios": _expected(config),
        "method_roster": config["method_roster"],
        "secondary_methods": config["secondary_methods"],
        "entries": entries,
    }
    dump_result(output / "manifest.json", manifest)
    summarize(
        output,
        config_path,
        output / "summary.json",
        output / "summary.csv",
    )
    print(f"completed {len(entries)} scenarios -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description="RRCL TMLR synthetic-v1 formal runner; no formal parameters"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.config or args.out) and not args.testing:
        raise SystemExit("--config/--out are testing-only; the formal command has no parameters")
    config_path = args.config.resolve() if args.config else FORMAL_CONFIG
    config = load_config(config_path)
    output = (
        args.out.resolve()
        if args.out
        else (ROOT / config["formal_output"]).resolve()
    )
    if args.testing and not args.out:
        raise SystemExit("--testing requires an explicit non-formal --out")
    if args.preflight:
        report = preflight(config_path, output, args.testing)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["ready"] else 1)
    run(config_path, output, args.testing)


if __name__ == "__main__":
    main()
