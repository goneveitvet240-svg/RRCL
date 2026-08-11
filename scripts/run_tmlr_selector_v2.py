#!/usr/bin/env python3
"""Single-command formal runner for the preregistered TMLR selector-v2 study."""

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

from result_io import dump_result
from tmlr_selector_v2 import (
    PROTOCOL,
    assemble_result,
    canonical_sha,
    evaluate_meta_test,
    file_sha,
    load_config,
    prepare_selection,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "tmlr_selector_v2.json"


def _git(arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def preflight(config_path: Path, output: Path, testing: bool) -> dict:
    config = load_config(config_path)
    formal_output = (ROOT / config["formal_output"]).resolve()
    issues = []
    if testing and output.resolve() == formal_output:
        issues.append("testing mode may not use the formal output path")
    if not testing:
        if config_path.resolve() != FORMAL_CONFIG.resolve():
            issues.append("formal run must use the frozen config path")
        if output.resolve() != formal_output:
            issues.append("formal run must use the frozen output path")
        if _git(["status", "--porcelain"]):
            issues.append("git worktree is dirty")
    if output.exists():
        issues.append(f"output path already exists: {output}")
    counts = {
        name: int(config["meta_splits"][name]["count"])
        for name in ("train", "validation", "test")
    }
    return {
        "protocol": PROTOCOL,
        "config": str(config_path),
        "config_sha256": file_sha(config_path),
        "counts": counts,
        "output": str(output),
        "testing": testing,
        "issues": issues,
        "ready": not issues,
    }


def _exclusive_json(path: Path, payload: dict):
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(config_path: Path, output: Path, testing: bool):
    report = preflight(config_path, output, testing)
    if not report["ready"]:
        raise SystemExit("selector-v2 preflight blocked:\n  " + "\n  ".join(report["issues"]))
    config = load_config(config_path)
    commit = _git(["rev-parse", "HEAD"])
    output.mkdir(parents=True, exist_ok=False)
    _exclusive_json(
        output / "attempt.lock",
        {
            "protocol": PROTOCOL,
            "git_commit": commit,
            "config_sha256": file_sha(config_path),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "testing": testing,
            "pid": os.getpid(),
        },
    )

    train_records, validation_records, selection = prepare_selection(config)
    dump_result(output / "meta_train_records.json", train_records)
    dump_result(output / "meta_validation_records.json", validation_records)
    lock = {
        "schema_version": "rrcl-tmlr-selector-v2-selection-lock-v1",
        "protocol": PROTOCOL,
        "git_commit": commit,
        "config_sha256": file_sha(config_path),
        "created_before_meta_test": True,
        "selection": selection,
        "meta_train_records_sha256": canonical_sha(train_records),
        "meta_validation_records_sha256": canonical_sha(validation_records),
    }
    _exclusive_json(output / "selection.lock.json", lock)

    # This is the first line in the formal execution path that is allowed to
    # instantiate any meta-test task.
    test_records = evaluate_meta_test(config, train_records, selection)
    result = assemble_result(
        config, train_records, validation_records, selection, test_records
    )
    dump_result(output / "result.json", result)
    manifest = {
        "schema_version": "rrcl-tmlr-selector-v2-manifest-v1",
        "protocol": PROTOCOL,
        "git_commit": commit,
        "git_dirty": bool(_git(["status", "--porcelain"])) if testing else False,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha(config_path),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "platform": platform.platform(),
        "testing": testing,
        "result_sha256": file_sha(output / "result.json"),
        "selection_lock_sha256": file_sha(output / "selection.lock.json"),
        "method_roster": list(result["summary"]["subsets"]["all"]["methods"]),
    }
    dump_result(output / "manifest.json", manifest)
    summary = result["summary"]
    print(
        "selector-v2 complete: "
        f"test={summary['meta_test_task_count']} "
        f"opportunity={summary['opportunity_present_count']} "
        f"learnability={summary['learned_selector_gates']['structured_learnability_supported']}"
    )
    print(f"saved -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description="RRCL TMLR selector-v2 formal runner; no formal parameters"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.config or args.out) and not args.testing:
        raise SystemExit("--config/--out are testing-only")
    config_path = args.config.resolve() if args.config else FORMAL_CONFIG
    config = load_config(config_path)
    output = args.out.resolve() if args.out else (ROOT / config["formal_output"]).resolve()
    if args.testing and not args.out:
        raise SystemExit("--testing requires an explicit non-formal --out")
    if args.preflight:
        report = preflight(config_path, output, args.testing)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["ready"] else 1)
    run(config_path, output, args.testing)


if __name__ == "__main__":
    main()
