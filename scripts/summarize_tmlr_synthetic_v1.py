#!/usr/bin/env python3
"""Mechanically summarize RRCL TMLR synthetic-v1 raw scenario artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_io import dump_result
from tmlr_synthetic import (
    FACTOR_NAMES,
    METHODS,
    PROTOCOL,
    RAW_SCHEMA,
    canonical_json_sha,
    cell_id,
    factor_cells,
    file_sha,
    load_config,
)


SUMMARY_SCHEMA = "rrcl-tmlr-synthetic-summary-v1"


def _interval(values, bootstrap, salt):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be non-empty and finite")
    B = int(bootstrap["replicates"])
    rng = np.random.default_rng(
        np.random.SeedSequence([int(bootstrap["seed"]), int(salt)])
    )
    samples = values[rng.integers(0, values.size, size=(B, values.size))]
    means = np.mean(samples, axis=1)
    lower, upper = map(float, bootstrap["interval"])
    return {
        "n_seeds": int(values.size),
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(means, lower)),
        "ci95_upper": float(np.quantile(means, upper)),
        "bootstrap_replicates": B,
    }


def load_raw_records(runs: Path, config: dict):
    records = []
    for path in sorted((runs / "raw").rglob("scenario.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RAW_SCHEMA:
            raise ValueError(f"{path}: wrong raw schema")
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"{path}: wrong protocol")
        if set(payload.get("methods", {})) != set(METHODS):
            raise ValueError(f"{path}: incomplete method roster")
        if payload.get("cell_id") != cell_id(payload.get("cell", {})):
            raise ValueError(f"{path}: cell id mismatch")
        records.append((path, payload))
    expected = 16 * len(config["seeds"]) * len(config["orders"])
    if len(records) != expected:
        raise ValueError(f"expected {expected} raw scenarios, found {len(records)}")
    identities = {
        (payload["cell_id"], int(payload["seed"]), payload["order_name"])
        for _, payload in records
    }
    if len(identities) != expected:
        raise ValueError("duplicate cell/seed/order identities")
    return records


def _seed_averages(cell_records, method):
    grouped = defaultdict(list)
    for record in cell_records:
        grouped[int(record["seed"])].append(record["methods"][method])
    output = []
    for seed in sorted(grouped):
        rows = grouped[seed]
        output.append(
            {
                "seed": seed,
                "utility": float(np.mean([row["relative_utility_vs_f1"] for row in rows])),
                "regret": float(np.mean([row["relative_regret_vs_oracle"] for row in rows])),
                "domain_harm": np.mean(
                    [row["relative_domain_harm_vs_f1"] for row in rows], axis=0
                ).tolist(),
            }
        )
    return output


def _paired_factor_error(cell_records):
    grouped = defaultdict(list)
    for record in cell_records:
        grouped[int(record["seed"])].append(
            abs(
                float(record["methods"]["fstar_pred"]["details"]["factor"])
                - float(record["methods"]["fixed_f_oracle"]["details"]["factor"])
            )
        )
    return [float(np.mean(grouped[seed])) for seed in sorted(grouped)]


def build_summary(records, config: dict, config_sha256: str) -> dict:
    by_cell = defaultdict(list)
    for _, payload in records:
        by_cell[payload["cell_id"]].append(payload)
    expected_cells = {cell_id(cell) for cell in factor_cells()}
    if set(by_cell) != expected_cells:
        raise ValueError("raw artifacts do not cover the complete 2^4 design")

    thresholds = config["decision_thresholds"]
    minimum_gain = float(thresholds["minimum_relative_gain"])
    maximum_harm = float(thresholds["maximum_single_domain_relative_harm"])
    cells = {}
    for cell_index, identifier in enumerate(sorted(by_cell)):
        cell_records = by_cell[identifier]
        method_summaries = {}
        for method_index, method in enumerate(METHODS):
            seeds = _seed_averages(cell_records, method)
            utility = _interval(
                [row["utility"] for row in seeds],
                config["bootstrap"],
                1000 * cell_index + method_index,
            )
            regret = _interval(
                [row["regret"] for row in seeds],
                config["bootstrap"],
                100000 + 1000 * cell_index + method_index,
            )
            mean_domain_harm = np.mean([row["domain_harm"] for row in seeds], axis=0)
            method_summaries[method] = {
                "utility_vs_f1": utility,
                "regret_vs_oracle": regret,
                "mean_relative_domain_harm": mean_domain_harm.tolist(),
                "worst_mean_domain_harm": float(np.max(mean_domain_harm)),
                "useful": bool(
                    method not in {"f1", "fixed_f_oracle"}
                    and utility["mean"] >= minimum_gain
                    and utility["ci95_lower"] > 0
                    and float(np.max(mean_domain_harm)) <= maximum_harm
                ),
            }
        factor_errors = _paired_factor_error(cell_records)
        method_summaries["fstar_pred"]["factor_absolute_error"] = _interval(
            factor_errors, config["bootstrap"], 200000 + cell_index
        )
        opportunity = method_summaries["fixed_f_oracle"]["utility_vs_f1"]
        opportunity_present = bool(
            opportunity["mean"] >= minimum_gain and opportunity["ci95_lower"] > 0
        )
        train_only_useful = {
            method: method_summaries[method]["useful"]
            for method in METHODS
            if method not in {"f1", "fixed_f_oracle"}
        }
        cells[identifier] = {
            "factors": by_cell[identifier][0]["cell"],
            "opportunity_present": opportunity_present,
            "gap_confirmed": bool(opportunity_present and not any(train_only_useful.values())),
            "construction_candidate": bool(opportunity_present and any(train_only_useful.values())),
            "methods": method_summaries,
        }

    paired = defaultdict(dict)
    for _, payload in records:
        paired[(int(payload["seed"]), payload["order_name"])][payload["cell_id"]] = payload
    scale_differences = []
    for scenarios in paired.values():
        base = scenarios["0000"]["methods"]["fixed_f_oracle"]["details"]["curve"]
        scale = scenarios["1000"]["methods"]["fixed_f_oracle"]["details"]["curve"]
        scale_differences.extend(
            abs(float(left["population_balanced_mse"]) - float(right["population_balanced_mse"]))
            for left, right in zip(base, scale)
        )

    reliability_rows = by_cell["0100"]
    monotone = []
    for record in reliability_rows:
        weights = record["methods"]["precision_weighting"]["details"]["weights"]
        sigmas = [item["sigma"] for item in record["domain_parameters"]]
        order = np.argsort(sigmas)
        ordered_weights = np.asarray(weights)[order]
        monotone.append(bool(np.all(np.diff(ordered_weights) <= 1e-12)))

    mechanism_checks = {
        "null_oracle_opportunity": {
            "value": cells["0000"]["methods"]["fixed_f_oracle"]["utility_vs_f1"]["mean"],
            "maximum": float(thresholds["null_maximum_oracle_opportunity"]),
            "pass": bool(
                cells["0000"]["methods"]["fixed_f_oracle"]["utility_vs_f1"]["mean"]
                <= float(thresholds["null_maximum_oracle_opportunity"])
            ),
        },
        "normalized_scale_isolation": {
            "maximum_absolute_curve_difference": float(max(scale_differences)),
            "tolerance": float(thresholds["scale_isolation_absolute_tolerance"]),
            "pass": bool(
                max(scale_differences)
                <= float(thresholds["scale_isolation_absolute_tolerance"])
            ),
        },
        "reliability_precision_direction": {
            "monotone_scenarios": int(sum(monotone)),
            "total_scenarios": len(monotone),
            "pass": bool(all(monotone)),
        },
        "mapping_only_opportunity_present": cells["0010"]["opportunity_present"],
        "mapping_anisotropy_opportunity_present": cells["0011"]["opportunity_present"],
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "config_sha256": config_sha256,
        "raw_scenario_count": len(records),
        "seed_count": len(config["seeds"]),
        "orders_are_sensitivity_not_independent": True,
        "cells": cells,
        "mechanism_checks": mechanism_checks,
    }


def write_csv(path: Path, summary: dict):
    fields = [
        "cell_id", "method", "opportunity_present", "gap_confirmed",
        "construction_candidate", "utility_mean", "utility_ci95_lower",
        "utility_ci95_upper", "regret_mean", "worst_mean_domain_harm", "useful",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identifier, cell in sorted(summary["cells"].items()):
            for method, metrics in cell["methods"].items():
                writer.writerow(
                    {
                        "cell_id": identifier,
                        "method": method,
                        "opportunity_present": cell["opportunity_present"],
                        "gap_confirmed": cell["gap_confirmed"],
                        "construction_candidate": cell["construction_candidate"],
                        "utility_mean": metrics["utility_vs_f1"]["mean"],
                        "utility_ci95_lower": metrics["utility_vs_f1"]["ci95_lower"],
                        "utility_ci95_upper": metrics["utility_vs_f1"]["ci95_upper"],
                        "regret_mean": metrics["regret_vs_oracle"]["mean"],
                        "worst_mean_domain_harm": metrics["worst_mean_domain_harm"],
                        "useful": metrics["useful"],
                    }
                )


def summarize(runs: Path, config_path: Path, out: Path, csv_out: Path):
    config = load_config(config_path)
    records = load_raw_records(runs, config)
    summary = build_summary(records, config, file_sha(config_path))
    dump_result(out, summary)
    write_csv(csv_out, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_real/tmlr_synthetic_v1")
    parser.add_argument("--config", default="configs/tmlr_synthetic_v1.json")
    parser.add_argument("--out")
    parser.add_argument("--csv")
    args = parser.parse_args()
    runs = Path(args.runs)
    out = Path(args.out) if args.out else runs / "summary.json"
    csv_out = Path(args.csv) if args.csv else runs / "summary.csv"
    summary = summarize(runs, Path(args.config), out, csv_out)
    print(
        f"summarized {summary['raw_scenario_count']} scenarios -> {out} and {csv_out}"
    )


if __name__ == "__main__":
    main()
