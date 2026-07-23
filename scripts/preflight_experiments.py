#!/usr/bin/env python3
"""Fail fast before launching the expensive RRCL experiment closure."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validate_release import REORDER_CONFIGS, config_identity_errors


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNNERS = [
    "run_real_norm_ablation.py",
    "run_adaptive_f.py",
    "run_baselines.py",
    "run_vff_baselines.py",
    "run_age_cl.py",
    "run_iqa_cl.py",
    "run_ava_cl.py",
    "run_depth_cl.py",
    "run_theory_predict.py",
]


def git_is_clean():
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="also require every configured dataset root to exist",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="debug only; paper-facing result validation still rejects dirty runs",
    )
    args = parser.parse_args()

    failures = []
    for relative in REQUIRED_RUNNERS:
        if not (ROOT / relative).is_file():
            failures.append(f"missing runner: {relative}")
    failures.extend(config_identity_errors())

    if not args.allow_dirty and not git_is_clean():
        failures.append(
            "git worktree is dirty; commit the exact experiment code before running"
        )

    if args.check_data:
        for name in REORDER_CONFIGS:
            config_path = ROOT / "configs" / f"domains_{name}.json"
            payload = json.loads(config_path.read_text())
            for domain in payload["domains"]:
                root = Path(domain["root"])
                if not root.is_dir():
                    failures.append(
                        f"missing dataset root for {domain['sample_key']}: {root}"
                    )

    if failures:
        print("RRCL experiment preflight: FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("RRCL experiment preflight: PASSED")
    print("  required runners present")
    print("  reordered configs use consistent dataset identities")
    if args.allow_dirty:
        print("  git cleanliness check skipped (debug mode)")
    else:
        print("  git worktree clean")
    if args.check_data:
        print("  configured dataset roots present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
