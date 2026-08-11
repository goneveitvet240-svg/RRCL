#!/usr/bin/env python3
"""Aggregate the frozen multi-seed f* prediction/oracle diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_io import dump_result
from run_theory_predict import FACTOR_GRID, NUMERIC_TOLERANCE, PROTOCOL


EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CONFIG_FAMILIES = {
    "jhu_sha_shb": "JHU/SHA/SHB",
    "jhu_shb_sha": "JHU/SHA/SHB",
    "sha_jhu_shb": "JHU/SHA/SHB",
    "sha_shb_jhu": "JHU/SHA/SHB",
    "qnrf_sha_shb": "QNRF/SHA/SHB",
    "qnrf_shb_sha": "QNRF/SHA/SHB",
    "sha_shb_qnrf": "QNRF/SHA/SHB",
    "sha_shb": "SHA/SHB",
}
MIN_MATERIAL_GAIN = 0.005


def _strict_load(path):
    def reject(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def _close(left, right, tolerance=1e-10):
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right))
    )


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left)
        * sum((b - mean_right) ** 2 for b in right)
    )
    return None if denominator <= 0 else numerator / denominator


def _manifest_signature(record):
    return (
        int(record["train"]["count"]),
        record["train"]["ids_sha256"],
        int(record["test"]["count"]),
        record["test"]["ids_sha256"],
    )


def _holdout_signature(record):
    return (
        int(record["split_seed"]),
        int(record["fit_count"]),
        record["fit_ids_sha256"],
        int(record["val_count"]),
        record["val_ids_sha256"],
        record["algorithm_version"],
        record["id_source"],
    )


def _validate_artifact(payload, config, seed, path):
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"{path}: wrong protocol")
    if payload.get("diagnostic_only") is not True:
        raise ValueError(f"{path}: diagnostic_only must be true")
    if payload.get("test_data_used_only_for_oracle_diagnostic") is not True:
        raise ValueError(f"{path}: oracle leakage warning is missing")
    if payload.get("task") != "crowd":
        raise ValueError(f"{path}: formal v1 calibration is crowd-only")
    if payload.get("config_name") != f"domains_{config}.json":
        raise ValueError(f"{path}: config name mismatch")
    if payload.get("sample_seed") != seed or payload.get("split_seed") != seed:
        raise ValueError(f"{path}: sample/split seed mismatch")

    grid = [float(value) for value in payload.get("factor_grid", [])]
    if grid != list(FACTOR_GRID):
        raise ValueError(f"{path}: factor grid differs from frozen grid")
    prediction = payload.get("prediction_curve_train_only", [])
    oracle = payload.get("oracle_curve_DIAGNOSTIC", [])
    if [float(row["factor"]) for row in prediction] != grid:
        raise ValueError(f"{path}: prediction curve grid mismatch")
    if [float(row["factor"]) for row in oracle] != grid:
        raise ValueError(f"{path}: oracle curve grid mismatch")

    pred_best = min(float(row["objective"]) for row in prediction)
    pred_candidates = [
        float(row["factor"])
        for row in prediction
        if float(row["objective"])
        <= pred_best + NUMERIC_TOLERANCE * max(1.0, abs(pred_best))
    ]
    oracle_best = min(float(row["balanced_normalized_mse"]) for row in oracle)
    oracle_candidates = [
        float(row["factor"])
        for row in oracle
        if float(row["balanced_normalized_mse"])
        <= oracle_best + NUMERIC_TOLERANCE * max(1.0, abs(oracle_best))
    ]
    f_pred = max(pred_candidates)
    f_oracle = max(oracle_candidates)
    if not _close(payload["f_pred"], f_pred):
        raise ValueError(f"{path}: stored f_pred disagrees with curve")
    if not _close(payload["f_oracle_DIAGNOSTIC"], f_oracle):
        raise ValueError(f"{path}: stored f_oracle disagrees with curve")

    provenance = payload.get("_provenance", {})
    if provenance.get("git_dirty") is not False:
        raise ValueError(f"{path}: git_dirty must be false")
    if not provenance.get("git_commit"):
        raise ValueError(f"{path}: missing source commit")
    arguments = provenance.get("arguments", {})
    expected = {
        "task": "crowd",
        "lam": 100.0,
        "img_size": 518,
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "max_per_domain": 400,
        "sample_seed": seed,
        "split_seed": seed,
        "val_every": 5,
        "allow_dirty": False,
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise ValueError(
                f"{path}: provenance argument {key}={arguments.get(key)!r}, "
                f"expected {value!r}"
            )
    return provenance["git_commit"]


def _point(payload, config, family, seed):
    f_pred = float(payload["f_pred"])
    f_oracle = float(payload["f_oracle_DIAGNOSTIC"])
    f1 = float(payload["test_mse_f1_DIAGNOSTIC"])
    selected = float(payload["test_mse_at_f_pred_DIAGNOSTIC"])
    oracle = float(payload["test_mse_oracle_DIAGNOSTIC"])
    oracle_gain = (f1 - oracle) / max(abs(f1), 1e-12)
    selected_gain = (f1 - selected) / max(abs(f1), 1e-12)
    material = oracle_gain >= MIN_MATERIAL_GAIN
    predicts_reweight = f_pred < 1.0 - NUMERIC_TOLERANCE
    return {
        "seed": seed,
        "config": config,
        "family": family,
        "f_pred": f_pred,
        "f_oracle": f_oracle,
        "absolute_error": abs(f_pred - f_oracle),
        "squared_error": (f_pred - f_oracle) ** 2,
        "exact_match": bool(_close(f_pred, f_oracle)),
        "within_one_grid_step": abs(f_pred - f_oracle) <= 0.05 + 1e-12,
        "test_normalized_mse_f1": f1,
        "test_normalized_mse_at_f_pred": selected,
        "test_normalized_mse_oracle": oracle,
        "oracle_mse_gain_relative": oracle_gain,
        "selected_mse_gain_relative": selected_gain,
        "oracle_mse_regret_at_f_pred_relative": (
            (selected - oracle) / max(abs(oracle), 1e-12)
        ),
        "f_oracle_relmae": float(payload["f_oracle_relmae_DIAGNOSTIC"]),
        "test_rel_f1": float(payload["test_rel_f1_DIAGNOSTIC"]),
        "test_rel_at_f_pred": float(
            payload["test_rel_at_f_pred_DIAGNOSTIC"]
        ),
        "test_rel_oracle": float(payload["test_rel_oracle_DIAGNOSTIC"]),
        "selected_rel_gain_relative": float(
            payload["selected_rel_gain_over_f1_DIAGNOSTIC"]
        ),
        "oracle_rel_regret_at_f_pred_relative": float(
            payload["oracle_rel_regret_at_f_pred_DIAGNOSTIC"]
        ),
        "oracle_material_reweight": material,
        "theory_predicts_reweight": predicts_reweight,
        "correct_material_decision": material == predicts_reweight,
    }


def _metrics(rows):
    predicted = [row["f_pred"] for row in rows]
    oracle = [row["f_oracle"] for row in rows]
    constant_one_error = [abs(1.0 - value) for value in oracle]
    return {
        "n_points": len(rows),
        "mae_factor": statistics.mean(row["absolute_error"] for row in rows),
        "rmse_factor": math.sqrt(
            statistics.mean(row["squared_error"] for row in rows)
        ),
        "exact_match_rate": statistics.mean(row["exact_match"] for row in rows),
        "within_one_grid_step_rate": statistics.mean(
            row["within_one_grid_step"] for row in rows
        ),
        "pearson_r": _pearson(predicted, oracle),
        "spearman_rho": _pearson(
            _average_ranks(predicted), _average_ranks(oracle)
        ),
        "material_decision_accuracy": statistics.mean(
            row["correct_material_decision"] for row in rows
        ),
        "mean_oracle_mse_gain_relative": statistics.mean(
            row["oracle_mse_gain_relative"] for row in rows
        ),
        "mean_selected_mse_gain_relative": statistics.mean(
            row["selected_mse_gain_relative"] for row in rows
        ),
        "mean_oracle_mse_regret_at_f_pred_relative": statistics.mean(
            row["oracle_mse_regret_at_f_pred_relative"] for row in rows
        ),
        "mean_selected_rel_gain_relative": statistics.mean(
            row["selected_rel_gain_relative"] for row in rows
        ),
        "mean_oracle_rel_regret_at_f_pred_relative": statistics.mean(
            row["oracle_rel_regret_at_f_pred_relative"] for row in rows
        ),
        "constant_f1_mae_factor": statistics.mean(constant_one_error),
        "mae_improvement_over_constant_f1": (
            statistics.mean(constant_one_error)
            - statistics.mean(row["absolute_error"] for row in rows)
        ),
    }


def summarize(runs):
    root = Path(runs)
    rows = []
    commits = set()
    data_signatures = defaultdict(set)
    holdout_signatures = defaultdict(set)
    config_hashes = defaultdict(set)

    for seed in EXPECTED_SEEDS:
        for config, family in CONFIG_FAMILIES.items():
            path = (
                root
                / f"seed_{seed}"
                / f"theory_{config}"
                / "fstar_calibration.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing calibration artifact: {path}")
            payload = _strict_load(path)
            commits.add(_validate_artifact(payload, config, seed, path))
            config_hashes[config].add(payload["_provenance"]["config_sha256"])

            for record in payload.get("data_manifest", {}).get("domains", []):
                key = record.get("sample_key", record.get("name"))
                data_signatures[(seed, key)].add(_manifest_signature(record))
            for record in payload.get("holdout_split", {}).get("domains", []):
                holdout_signatures[(seed, record["sample_key"])].add(
                    _holdout_signature(record)
                )
            rows.append(_point(payload, config, family, seed))

    if len(commits) != 1:
        raise ValueError(
            f"all calibration artifacts must use one commit; got {sorted(commits)}"
        )
    if any(len(values) != 1 for values in config_hashes.values()):
        raise ValueError("a config changed across calibration seeds")
    if any(len(values) != 1 for values in data_signatures.values()):
        raise ValueError("same dataset/seed uses inconsistent sample manifests")
    if any(len(values) != 1 for values in holdout_signatures.values()):
        raise ValueError("same dataset/seed uses inconsistent holdout manifests")

    by_family = {
        family: _metrics([row for row in rows if row["family"] == family])
        for family in sorted(set(CONFIG_FAMILIES.values()))
    }
    return {
        "protocol": PROTOCOL,
        "development_data_only": True,
        "diagnostic_only": True,
        "source_commit": next(iter(commits)),
        "seeds": list(EXPECTED_SEEDS),
        "configs": list(CONFIG_FAMILIES),
        "factor_grid": list(FACTOR_GRID),
        "minimum_material_gain": MIN_MATERIAL_GAIN,
        "primary_oracle_metric": "unclipped domain-balanced normalized MSE",
        "secondary_metric": "clipped domain-balanced relative MAE",
        "independence_warning": (
            "Orderings and seeds reuse underlying datasets; the 40 rows are "
            "calibration scenarios, not 40 independent datasets. Metrics are "
            "descriptive and carry no iid confidence interval."
        ),
        "rows": rows,
        "overall": _metrics(rows),
        "by_family": by_family,
    }


def _write_csv(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(
            descriptor, "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_real/fstar_calibration")
    parser.add_argument("--out", default="")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    summary = summarize(args.runs)
    metrics = summary["overall"]
    print("F* THEORY CALIBRATION (development diagnostic)")
    print("  source commit:", summary["source_commit"])
    print("  points:", metrics["n_points"])
    print(
        f"  MAE={metrics['mae_factor']:.4f} "
        f"RMSE={metrics['rmse_factor']:.4f} "
        f"exact={metrics['exact_match_rate']:.3f} "
        f"within-0.05={metrics['within_one_grid_step_rate']:.3f}"
    )
    print(
        f"  Pearson={metrics['pearson_r']} "
        f"Spearman={metrics['spearman_rho']} "
        f"material-accuracy={metrics['material_decision_accuracy']:.3f}"
    )
    print(
        f"  constant-f=1 MAE={metrics['constant_f1_mae_factor']:.4f} "
        f"theory improvement={metrics['mae_improvement_over_constant_f1']:+.4f}"
    )
    print(
        f"  matched-MSE selected gain="
        f"{100*metrics['mean_selected_mse_gain_relative']:+.3f}% "
        f"oracle gain={100*metrics['mean_oracle_mse_gain_relative']:+.3f}% "
        f"regret="
        f"{100*metrics['mean_oracle_mse_regret_at_f_pred_relative']:+.3f}%"
    )
    print(
        f"  downstream relMAE selected gain="
        f"{100*metrics['mean_selected_rel_gain_relative']:+.3f}% "
        f"regret="
        f"{100*metrics['mean_oracle_rel_regret_at_f_pred_relative']:+.3f}%"
    )

    root = Path(args.runs)
    output = Path(args.out) if args.out else root / "fstar_calibration_summary.json"
    csv_path = Path(args.csv) if args.csv else root / "fstar_points.csv"
    dump_result(output, summary)
    _write_csv(csv_path, summary["rows"])
    print("  saved ->", output)
    print("  saved ->", csv_path)


if __name__ == "__main__":
    main()
