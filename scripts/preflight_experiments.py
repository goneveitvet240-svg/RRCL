#!/usr/bin/env python3
"""Fail fast before launching the expensive RRCL experiment closure."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validate_release import (
    FSTAR_CALIBRATION_CONFIGS,
    REORDER_CONFIGS,
    config_identity_errors,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUNNERS = [
    "run_real_norm_ablation.py",
    "run_adaptive_f.py",
    "run_baselines.py",
    "run_vff_baselines.py",
    "run_analytic_scalar_baselines.py",
    "run_age_cl.py",
    "run_iqa_cl.py",
    "run_ava_cl.py",
    "run_depth_cl.py",
    "run_theory_predict.py",
    "scripts/run_precision_shrinkage_diagnostic.py",
    "scripts/run_c4_gamma_multiseed.sh",
    "scripts/summarize_gamma_c4.py",
    "scripts/run_fstar_calibration_multiseed.sh",
    "scripts/summarize_fstar_calibration.py",
    "scripts/plot_fstar_calibration.py",
]

PAPER_CONFIGS = [
    *(f"domains_{name}.json" for name in REORDER_CONFIGS),
    *(f"domains_{name}.json" for name in FSTAR_CALIBRATION_CONFIGS),
    "domains_qnrf_sha_shb.json",
    "domains_qnrf_shb_sha.json",
    "domains_sha_shb_qnrf.json",
    "domains_age_utk_young_old.json",
    "domains_age_agedb_utk.json",
    "domains_age_utk_agedb.json",
    "domains_iqa_kadid.json",
    "domains_ava.json",
    "domains_depth_nyu.json",
]

DOMAIN_FILE_FIELDS = ("csv", "split_file", "test_split_file")


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
        checked_paths = set()
        for config_name in PAPER_CONFIGS:
            config_path = ROOT / "configs" / config_name
            if not config_path.is_file():
                failures.append(f"missing paper config: {config_path}")
                continue
            payload = json.loads(config_path.read_text())
            for domain in payload["domains"]:
                root = Path(domain["root"])
                root_key = ("directory", str(root))
                if root_key not in checked_paths and not root.is_dir():
                    failures.append(
                        f"missing dataset root for {domain['name']}: {root}"
                    )
                checked_paths.add(root_key)
                for field in DOMAIN_FILE_FIELDS:
                    if field not in domain:
                        continue
                    path = Path(domain[field])
                    file_key = ("file", str(path))
                    if file_key not in checked_paths and not path.is_file():
                        failures.append(
                            f"missing {field} for {domain['name']}: {path}"
                        )
                    checked_paths.add(file_key)

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
        print("  all paper-closure dataset roots and metadata files present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
