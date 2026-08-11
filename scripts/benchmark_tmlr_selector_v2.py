#!/usr/bin/env python3
"""Post-result compute audit for frozen selector-v2 methods.

This audit never changes a factor, selector, task result, or decision gate. It
times the already-frozen methods on a fixed prefix of meta-test seeds and is
reported separately from the formal evidence manifest.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_io import dump_result
from tmlr_selector_v2 import (
    _dos_risk,
    _sift_risk,
    _solve_weighted,
    base_record,
    file_sha,
    generate_task,
    knn_factor,
    load_config,
    population_per_domain,
    split_seeds,
    task_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def _milliseconds(samples):
    ordered = sorted(samples)
    return {
        "measurements": len(samples),
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(np.quantile(ordered, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs_real" / "tmlr_selector_v2_compute_audit.json",
    )
    args = parser.parse_args()
    if args.tasks <= 0 or args.repeats <= 0:
        raise SystemExit("tasks and repeats must be positive")

    config_path = ROOT / "configs" / "tmlr_selector_v2.json"
    result_path = ROOT / "runs_real" / "tmlr_selector_v2" / "result.json"
    config = load_config(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = result["selection"]
    train_records = result["meta_train_records"]
    seeds = split_seeds(config, "test")[: args.tasks]
    timings = {name: [] for name in ("f1_batch", "learned_knn", "sift_rls", "dos_elm_style")}

    for seed in seeds:
        task = generate_task(config, seed)
        stats = task_stats(config, task)
        query = base_record(config, task)
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            W = _solve_weighted(stats, np.ones(4), float(config["ridge_lambda"]))
            population_per_domain(task, W)
            timings["f1_batch"].append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            factor, _ = knn_factor(
                train_records,
                query,
                np.asarray(selection["feature_mean"]),
                np.asarray(selection["feature_scale"]),
                int(selection["knn_k"]),
            )
            W = _solve_weighted(
                stats,
                [factor**3, factor**2, factor, 1.0],
                float(config["ridge_lambda"]),
            )
            population_per_domain(task, W)
            timings["learned_knn"].append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            sift = _sift_risk(config, task, float(selection["sift_domain_factor"]))
            timings["sift_rls"].append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            dos = _dos_risk(config, task, stats)
            timings["dos_elm_style"].append((time.perf_counter_ns() - started) / 1e6)

    payload = {
        "schema_version": "rrcl-tmlr-selector-v2-compute-audit-v1",
        "post_result_diagnostic_only": True,
        "changes_formal_result_or_selection": False,
        "config_sha256": file_sha(config_path),
        "formal_result_sha256": file_sha(result_path),
        "task_seed_prefix": seeds,
        "repeats_per_task": args.repeats,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "timings": {name: _milliseconds(values) for name, values in timings.items()},
        "state_and_updates": {
            "f1_batch": {"domain_updates": 4, "state_bytes": 792},
            "learned_knn": {
                "domain_updates": 4,
                "deployed_head_state_bytes": 792,
                "meta_library_tasks": len(train_records),
            },
            "sift_rls": {
                "rank_one_updates": sift["updates"],
                "state_bytes": sift["state_bytes"],
            },
            "dos_elm_style": {"domain_updates": 4, "state_bytes": dos["state_bytes"]},
        },
    }
    dump_result(args.out, payload)
    print(json.dumps(payload["timings"], indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
