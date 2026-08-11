#!/usr/bin/env python3
"""Validate and summarize confirmatory RRCL runs.

The script fails on missing, dirty, non-standard, or identity-inconsistent
artifacts by default.  ``--allow-missing`` is intended only for monitoring an
unfinished server run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


T_CRITICAL_95_BY_N = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}

CROWD_ORDERS = ["jhu_sha_shb", "jhu_shb_sha", "sha_shb_jhu"]
QNRF_ORDERS = ["qnrf_sha_shb", "qnrf_shb_sha", "sha_shb_qnrf"]
METHODS = ("original", "safe")


def mean_ci(values):
    values = [float(value) for value in values]
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None
    critical = T_CRITICAL_95_BY_N.get(len(values), 1.96)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def _reject_nonstandard_json_constant(value):
    raise ValueError(f"non-standard JSON constant {value}")


def load(path):
    return json.loads(
        path.read_text(),
        parse_constant=_reject_nonstandard_json_constant,
    )


def _manifest_by_key(payload):
    return {
        domain.get("sample_key", domain.get("name")): {
            key: domain.get(key)
            for key in ("train", "test", "train_groups", "test_groups")
            if key in domain
        }
        for domain in payload.get("data_manifest", {}).get("domains", [])
    }


def _selection_records(payload):
    if isinstance(payload.get("selection_records"), list):
        return payload["selection_records"]
    evidence = payload.get("adaptive_evidence", {})
    return evidence.get("selections", []) if isinstance(evidence, dict) else []


def _holdout_by_key(payload):
    """Per-domain fit/val id hashes of the recorded holdout split."""
    evidence = payload.get("adaptive_evidence", {})
    block = evidence.get("holdout_split", {}) if isinstance(evidence, dict) else {}
    return {
        domain.get("sample_key"): (
            domain.get("fit_ids_sha256"),
            domain.get("val_ids_sha256"),
        )
        for domain in block.get("domains", [])
        if isinstance(domain, dict)
    }


def _record(payload, task):
    if task == "kadid":
        f1 = float(payload["f1"])
        adaptive = float(payload["adaptive"])
        oracle = float(payload["opt_rel_mean"])
        factors = payload["f_used"]
    else:
        f1 = float(payload["rel_f1"])
        adaptive = float(payload["adaptive_rel"])
        oracle = float(payload["oracle_rel"])
        factors = payload["f_used"]
    verdict = payload["verdict"]
    gain = (f1 - adaptive) / max(abs(f1), 1e-12)
    oracle_gain = (f1 - oracle) / max(abs(f1), 1e-12)
    minimum_oracle_gain = float(
        verdict.get("minimum_oracle_relative_gain", 0.005)
    )
    negative_control = oracle_gain < minimum_oracle_gain
    selections = _selection_records(payload)
    return {
        "gain_pct": 100.0 * gain,
        "oracle_gain_pct": 100.0 * oracle_gain,
        "recovery_pct": (
            None
            if verdict.get("oracle_gain_recovery") is None
            else 100.0 * float(verdict["oracle_gain_recovery"])
        ),
        "negative_control": negative_control,
        "harmful_false_positive": negative_control and gain < -1e-6,
        "selected_forgetting": any(float(factor) < 1.0 - 1e-9 for factor in factors),
        "safety_abstained": any(
            bool(selection.get("safety_abstained", False))
            for selection in selections
        ),
    }


def _format(value):
    return "" if value is None else f"{value:.4f}"


def _result_path(runs, seed, method, task):
    if task == "kadid":
        prefix = "kadid_safe" if method == "safe" else "kadid"
        return runs / f"{prefix}_seed_{seed}" / "domains_iqa_kadid_result.json"
    return (
        runs
        / f"seed_{seed}"
        / f"{method}_{task}"
        / "adaptive_f.json"
    )


def _validate_provenance(payload, path, commits, failures):
    provenance = payload.get("_provenance", {})
    commit = provenance.get("git_commit")
    if not commit:
        failures.append(f"{path}: missing git commit")
    else:
        commits.add(commit)
    if provenance.get("git_dirty") is not False:
        failures.append(f"{path}: result was generated from a dirty worktree")
    manifest = payload.get("data_manifest")
    if not isinstance(manifest, dict) or not manifest.get("domains"):
        failures.append(f"{path}: missing data manifest")


def _validate_seed(payload, path, seed, task, failures):
    manifest = payload.get("data_manifest", {})
    if task == "kadid":
        actual = manifest.get("split_seed")
        if not isinstance(actual, list) or any(
            int(value) != seed for value in actual
        ):
            failures.append(f"{path}: KADID split seed does not equal {seed}")
        return
    if int(manifest.get("sample_seed", -1)) != seed:
        failures.append(f"{path}: sample seed does not equal {seed}")
    # Crowd runs must ALSO re-partition the selector holdout per seed
    # (pilot audit 2026-07-26: stream-order split left SHA/SHB fixed).
    split_seed = payload.get("split_seed")
    if split_seed is None or int(split_seed) != seed:
        failures.append(f"{path}: selector split seed does not equal {seed}")
    holdout = _holdout_by_key(payload)
    if not holdout:
        failures.append(f"{path}: missing holdout split manifest")
    else:
        for key, (fit_hash, val_hash) in holdout.items():
            if not fit_hash or not val_hash:
                failures.append(f"{path}: incomplete holdout manifest for {key}")


def _validate_order_identity(loaded, seeds, methods, orders, failures):
    for seed in seeds:
        for method in methods:
            rows = [
                loaded.get((seed, method, order))
                for order in orders
            ]
            if any(row is None for row in rows):
                continue
            reference_score = float(rows[0]["rel_f1"])
            reference_manifest = _manifest_by_key(rows[0])
            reference_holdout = _holdout_by_key(rows[0])
            for order, row in zip(orders[1:], rows[1:]):
                score = float(row["rel_f1"])
                if abs(score - reference_score) > 1e-6:
                    failures.append(
                        f"seed {seed} {method} {order}: f=1 order invariance failed"
                    )
                if _manifest_by_key(row) != reference_manifest:
                    failures.append(
                        f"seed {seed} {method} {order}: sample manifest differs"
                    )
                if _holdout_by_key(row) != reference_holdout:
                    failures.append(
                        f"seed {seed} {method} {order}: holdout split manifest differs"
                    )


def _validate_method_identity(loaded, seeds, tasks, failures):
    for seed in seeds:
        for task in tasks:
            original = loaded.get((seed, "original", task))
            safe = loaded.get((seed, "safe", task))
            if original is None or safe is None:
                continue
            score_key = "f1" if task == "kadid" else "rel_f1"
            if abs(float(original[score_key]) - float(safe[score_key])) > 1e-6:
                failures.append(
                    f"seed {seed} {task}: original/safe f=1 references differ"
                )
            if _manifest_by_key(original) != _manifest_by_key(safe):
                failures.append(
                    f"seed {seed} {task}: original/safe manifests differ"
                )
            if task != "kadid" and _holdout_by_key(original) != _holdout_by_key(safe):
                failures.append(
                    f"seed {seed} {task}: original/safe holdout splits differ"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs_real/development"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="summarize partial output while a run is still in progress",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    tasks = [*CROWD_ORDERS, *QNRF_ORDERS, "kadid"]
    loaded = {}
    commits = set()
    failures = []
    missing = []

    for seed in seeds:
        for method in METHODS:
            for task in tasks:
                path = _result_path(args.runs, seed, method, task)
                if not path.exists():
                    missing.append(path)
                    continue
                try:
                    payload = load(path)
                except (OSError, json.JSONDecodeError, ValueError) as error:
                    failures.append(f"{path}: invalid JSON: {error}")
                    continue
                _validate_provenance(payload, path, commits, failures)
                _validate_seed(payload, path, seed, task, failures)
                loaded[(seed, method, task)] = payload

    _validate_order_identity(
        loaded, seeds, METHODS, CROWD_ORDERS, failures
    )
    _validate_order_identity(
        loaded, seeds, METHODS, QNRF_ORDERS, failures
    )
    _validate_method_identity(loaded, seeds, tasks, failures)
    if len(commits) > 1:
        failures.append(
            "confirmatory artifacts were generated by multiple commits: "
            + ", ".join(sorted(commits))
        )
    if missing and not args.allow_missing:
        failures.extend(f"missing result: {path}" for path in missing)

    if failures:
        print("CONFIRMATORY VALIDATION: FAILED")
        for failure in failures:
            print(f"  {failure}")
        print("AVAILABLE-RESULT SUMMARY FOLLOWS")
    elif missing:
        print(
            "CONFIRMATORY VALIDATION: PARTIAL "
            f"({len(missing)} result(s) still missing)"
        )
    else:
        print("CONFIRMATORY VALIDATION: PASSED")
        if commits:
            print(f"  source commit: {next(iter(commits))}")

    header = [
        "method",
        "task",
        "n",
        "mean_gain_pct",
        "ci95_low",
        "ci95_high",
        "mean_oracle_gain_pct",
        "mean_recovery_pct",
        "forgetting_rate",
        "negative_n",
        "harmful_fp_rate",
        "safety_abstention_rate",
    ]
    print("\t".join(header))
    for method in METHODS:
        for task in tasks:
            records = [
                _record(loaded[(seed, method, task)], task)
                for seed in seeds
                if (seed, method, task) in loaded
            ]
            if not records:
                continue
            mean, low, high = mean_ci(
                [record["gain_pct"] for record in records]
            )
            recoveries = [
                record["recovery_pct"]
                for record in records
                if record["recovery_pct"] is not None
            ]
            negative = [
                record for record in records if record["negative_control"]
            ]
            values = [
                method,
                task,
                str(len(records)),
                _format(mean),
                _format(low),
                _format(high),
                _format(statistics.mean(
                    record["oracle_gain_pct"] for record in records
                )),
                _format(statistics.mean(recoveries) if recoveries else None),
                _format(statistics.mean(
                    record["selected_forgetting"] for record in records
                )),
                str(len(negative)),
                _format(
                    statistics.mean(
                        record["harmful_false_positive"] for record in negative
                    )
                    if negative
                    else None
                ),
                _format(statistics.mean(
                    record["safety_abstained"] for record in records
                )),
            ]
            print("\t".join(values))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
