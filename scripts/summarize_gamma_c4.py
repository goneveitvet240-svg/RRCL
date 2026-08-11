#!/usr/bin/env python3
"""Frozen C4 material-decision summary for the gamma diagnostic.

This is a development-data gate, not a deployed selector.  Test curves are
read only to assess whether the train-only fused proxy predicts the presence
of a *material* reweighting opportunity:

  oracle_material_reweight = best test gain over gamma=0 >= 0.5%
  train_predicts_reweight   = fused train argmin gamma > 0

The two orderings of one dataset family are consistency checks, not
independent repetitions.  Each family receives one vote per seed and must be
correct on at least four of seeds 42..46.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from result_io import dump_result


EXPECTED_SEEDS = (42, 43, 44, 45, 46)
FAMILIES = {
    "jhu": ("jhu_sha_shb", "sha_shb_jhu"),
    "qnrf": ("qnrf_shb_sha", "qnrf_sha_shb"),
}
MIN_MATERIAL_GAIN = 0.005
REQUIRED_CORRECT_SEEDS = 4
NUMERIC_TOLERANCE = 1e-10


def _strict_load(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value!r} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _finite_vector(payload, key):
    values = [float(value) for value in payload[key]]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{key} must be a non-empty finite vector")
    return values


def material_decision(payload, minimum_gain=MIN_MATERIAL_GAIN):
    """Compute one artifact's frozen C4 decision and descriptive quantities."""
    gammas = _finite_vector(payload, "gammas")
    train_curve = _finite_vector(payload, "val_balanced_fused_proxy")
    test_curve = _finite_vector(payload, "test_balanced_rel_mae_DIAGNOSTIC")
    if not (len(gammas) == len(train_curve) == len(test_curve)):
        raise ValueError("gamma/train/test curve length mismatch")
    if 0.0 not in gammas:
        raise ValueError("gamma grid does not contain the absolute-memory endpoint")

    train_index = min(range(len(gammas)), key=lambda index: train_curve[index])
    test_index = min(range(len(gammas)), key=lambda index: test_curve[index])
    zero_index = gammas.index(0.0)
    train_gamma = gammas[train_index]
    test_gamma = gammas[test_index]

    stored = payload.get("argmin", {})
    if abs(float(stored.get("train_fused_gamma")) - train_gamma) > NUMERIC_TOLERANCE:
        raise ValueError("stored fused train argmin disagrees with the curve")
    if abs(float(stored.get("test_gamma_DIAGNOSTIC")) - test_gamma) > NUMERIC_TOLERANCE:
        raise ValueError("stored test argmin disagrees with the curve")

    baseline = test_curve[zero_index]
    best = test_curve[test_index]
    selected = test_curve[train_index]
    denominator = max(abs(baseline), 1e-12)
    oracle_gain = (baseline - best) / denominator
    selected_gain = (baseline - selected) / denominator
    oracle_material = oracle_gain >= float(minimum_gain)
    train_reweights = train_gamma > NUMERIC_TOLERANCE

    return {
        "train_fused_gamma": train_gamma,
        "test_gamma_diagnostic": test_gamma,
        "gamma_absolute_error": abs(train_gamma - test_gamma),
        "test_gamma0_rel_mae": baseline,
        "test_best_rel_mae": best,
        "test_at_train_gamma_rel_mae": selected,
        "oracle_gain_relative": oracle_gain,
        "selected_gain_relative": selected_gain,
        "oracle_gain_recovery": (
            selected_gain / oracle_gain if oracle_material and oracle_gain > 0 else None
        ),
        "minimum_material_gain": float(minimum_gain),
        "oracle_material_reweight": oracle_material,
        "train_predicts_reweight": train_reweights,
        "correct_material_decision": train_reweights == oracle_material,
    }


def _manifest_by_key(payload):
    records = payload.get("data_manifest", {}).get("domains", [])
    return {
        record["sample_key"]: {
            key: value for key, value in record.items() if key != "name"
        }
        for record in records
    }


def _holdout_by_key(payload):
    records = payload.get("holdout_split", {}).get("domains", [])
    return {
        record["sample_key"]: {
            key: value for key, value in record.items() if key != "sample_key"
        }
        for record in records
    }


def _vectors_close(left, right, tolerance=NUMERIC_TOLERANCE):
    if len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def validate_order_pair(left, right):
    """Require the two orderings to be the same experiment up to domain order."""
    left_manifest = _manifest_by_key(left)
    right_manifest = _manifest_by_key(right)
    if not left_manifest or not right_manifest:
        raise ValueError("order pair is missing sampled data manifests")
    if left_manifest != right_manifest:
        raise ValueError("order pair uses different sampled data manifests")
    left_holdout = _holdout_by_key(left)
    right_holdout = _holdout_by_key(right)
    if not left_holdout or not right_holdout:
        raise ValueError("order pair is missing three-way holdout manifests")
    if left_holdout != right_holdout:
        raise ValueError("order pair uses different three-way holdout manifests")
    for key in (
        "gammas",
        "val_balanced_fused_proxy",
        "test_balanced_rel_mae_DIAGNOSTIC",
    ):
        if not _vectors_close(left[key], right[key]):
            raise ValueError(f"order-invariant C4 curve mismatch in {key}")


def summarize(runs):
    runs = Path(runs)
    rows = []
    source_commits = set()
    for seed in EXPECTED_SEEDS:
        for family, tasks in FAMILIES.items():
            payloads = []
            for task in tasks:
                path = (
                    runs
                    / f"seed_{seed}"
                    / f"shrinkage_diag_{task}"
                    / "shrinkage_diagnostic.json"
                )
                if not path.exists():
                    raise FileNotFoundError(f"missing required C4 artifact: {path}")
                payload = _strict_load(path)
                if payload.get("diagnostic_only") is not True:
                    raise ValueError(f"{path}: diagnostic_only must be true")
                provenance = payload.get("_provenance", {})
                if provenance.get("git_dirty") is not False:
                    raise ValueError(f"{path}: git_dirty must be false")
                commit = provenance.get("git_commit")
                if not commit:
                    raise ValueError(f"{path}: missing source commit")
                source_commits.add(commit)
                if payload.get("sample_seed") != seed or payload.get("split_seed") != seed:
                    raise ValueError(f"{path}: sample/split seed does not equal {seed}")
                equivalence = payload.get("equivalence", {})
                if not equivalence.get("gamma0_equals_f1"):
                    raise ValueError(f"{path}: gamma=0 endpoint equivalence failed")
                if not equivalence.get("gamma1_equals_three_way_precision_endpoint"):
                    raise ValueError(f"{path}: gamma=1 endpoint equivalence failed")
                payloads.append(payload)

            validate_order_pair(payloads[0], payloads[1])
            left = material_decision(payloads[0])
            right = material_decision(payloads[1])
            for key in (
                "train_predicts_reweight",
                "oracle_material_reweight",
                "correct_material_decision",
            ):
                if left[key] != right[key]:
                    raise ValueError(
                        f"seed {seed} family {family}: order pair disagrees on {key}"
                    )
            rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "tasks_consistency_only": list(tasks),
                    **left,
                }
            )

    if len(source_commits) != 1:
        raise ValueError(
            "all final C4 artifacts must come from one exact commit; got "
            f"{sorted(source_commits)}"
        )

    family_summary = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        correct = sum(row["correct_material_decision"] for row in family_rows)
        family_summary[family] = {
            "correct_seeds": int(correct),
            "total_seeds": len(family_rows),
            "required_correct_seeds": REQUIRED_CORRECT_SEEDS,
            "passed": correct >= REQUIRED_CORRECT_SEEDS,
        }
    passed = all(record["passed"] for record in family_summary.values())
    return {
        "protocol": "C4-material-decision-v1",
        "development_data_only": True,
        "source_commit": next(iter(source_commits)),
        "seeds": list(EXPECTED_SEEDS),
        "minimum_material_gain": MIN_MATERIAL_GAIN,
        "vote_unit": "one dataset family per seed; orderings are consistency checks",
        "rows": rows,
        "families": family_summary,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_real/c4_gamma")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    summary = summarize(args.runs)
    print("C4 MATERIAL-DECISION VALIDATION:", "PASSED" if summary["passed"] else "FAILED")
    print("  source commit:", summary["source_commit"])
    for family, record in summary["families"].items():
        print(
            f"  {family}: {record['correct_seeds']}/{record['total_seeds']} "
            f"(required {record['required_correct_seeds']}) -> "
            f"{'PASS' if record['passed'] else 'FAIL'}"
        )
    for row in summary["rows"]:
        print(
            f"    seed={row['seed']} family={row['family']} "
            f"train_gamma={row['train_fused_gamma']:.2f} "
            f"test_gamma={row['test_gamma_diagnostic']:.2f} "
            f"oracle_gain={100*row['oracle_gain_relative']:+.3f}% "
            f"correct={row['correct_material_decision']}"
        )

    output = Path(args.out) if args.out else Path(args.runs) / "c4_gamma_summary.json"
    dump_result(output, summary)
    print("  saved ->", output)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
