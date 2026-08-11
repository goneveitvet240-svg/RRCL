#!/usr/bin/env python3
"""Train-only prediction of the oracle fixed forgetting factor.

This runner is a falsification diagnostic for the isotropic finite-horizon
approximation in the paper.  It does *not* deploy a selector:

* ``f_pred`` minimizes the analytic approximation using training data only;
* ``f_oracle`` minimizes balanced test relative MAE and is diagnostic only;
* both use exactly the same predeclared factor grid;
* a stable hash split estimates residual energy;
* every run records data/split manifests and clean-code provenance.

Crowd counting is deliberately reduced to image-level scalar regression
(mean-pooled DINOv2 feature -> total count).  It therefore tests the theory's
ordering/calibration, not the paper's fused patch/image prediction head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from holdout_split import (
    domain_sample_key,
    is_validation,
    iter_stream_with_ids,
    require_both_partitions,
    split_manifest,
)
from result_io import dump_result
from rls_head import ForgettingRidgeRLS
from run_provenance import with_provenance


PROTOCOL = "theory-f-calibration-v1"
FACTOR_GRID = tuple(round(value, 2) for value in np.arange(0.05, 1.0001, 0.05))
NUMERIC_TOLERANCE = 1e-12


def _git_clean():
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _task_domains(config, task, backbone, img_size, max_per_domain, sample_seed):
    if task == "age":
        from datasets_age import AgeDomains, build_age_config

        return AgeDomains(
            build_age_config(config),
            backbone=backbone,
            img_size=img_size,
            max_per_domain=max_per_domain,
        )
    from datasets_real import RealCountingDomains, build_from_config

    return RealCountingDomains(
        build_from_config(config),
        backbone=backbone,
        img_size=img_size,
        max_per_domain=max_per_domain,
        sample_seed=sample_seed,
    )


def load_task(
    config,
    task,
    backbone,
    img_size,
    max_per_domain,
    sample_seed=42,
):
    """Load image-level scalar arrays plus stable image identities."""
    domains = _task_domains(
        config, task, backbone, img_size, max_per_domain, sample_seed
    )

    def arrays(split, domain_index):
        features, targets, identifiers = [], [], []
        id_sources = set()
        for feature, target, _, image_id, id_source in iter_stream_with_ids(
            domains, split, domain_index
        ):
            features.append(np.asarray(feature, dtype=np.float64).mean(axis=0))
            targets.append(float(np.asarray(target, dtype=np.float64).sum()))
            identifiers.append(str(image_id))
            id_sources.add(str(id_source))
        if not features:
            raise RuntimeError(
                f"empty {split} stream for domain index {domain_index}"
            )
        if len(id_sources) != 1:
            raise RuntimeError(
                f"mixed identity sources in {split} domain {domain_index}: "
                f"{sorted(id_sources)}"
            )
        return {
            "X": np.asarray(features, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.float64),
            "ids": identifiers,
            "id_source": next(iter(id_sources)),
            "sample_key": domain_sample_key(domains, domain_index),
        }

    count = domains.n_domains()
    train = {index: arrays("train", index) for index in range(count)}
    test = {index: arrays("test", index) for index in range(count)}
    names = [getattr(spec, "name", f"domain-{index}") for index, spec in enumerate(domains.domains)]
    return train, test, names, domains.data_manifest()


def relative_mae(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(
        np.mean(np.abs(prediction - target)) / max(np.mean(np.abs(target)), 1e-6)
    )


def estimate_training_quantities(train, lam, split_seed, val_every=5):
    """Estimate the isotropic approximation using training data only."""
    per_domain = []
    manifests = []
    d_aug = None
    for index in range(len(train)):
        record = train[index]
        X = record["X"]
        y = record["y"]
        scale = max(float(np.mean(y)), 1e-6)
        normalized = y / scale

        val_mask = np.asarray(
            [
                is_validation(
                    record["sample_key"], image_id, split_seed, val_every
                )
                for image_id in record["ids"]
            ],
            dtype=bool,
        )
        fit_mask = ~val_mask
        manifest = split_manifest(
            record["sample_key"],
            split_seed,
            val_every,
            [item for item, keep in zip(record["ids"], fit_mask) if keep],
            [item for item, keep in zip(record["ids"], val_mask) if keep],
            record["id_source"],
        )
        require_both_partitions(manifest, f"theory calibration domain {index}")
        manifests.append(manifest)

        full_head = ForgettingRidgeRLS(
            d_in=X.shape[1], d_out=1, lam=lam
        )
        full_head.begin_task()
        full_head.accumulate(X, normalized.reshape(-1, 1))
        full_head.solve()

        fit_head = ForgettingRidgeRLS(
            d_in=X.shape[1], d_out=1, lam=lam
        )
        fit_head.begin_task()
        fit_head.accumulate(
            X[fit_mask], normalized[fit_mask].reshape(-1, 1)
        )
        fit_head.solve()

        d_aug = full_head.R.shape[0]
        mu = float(np.trace(full_head.R) / d_aug)
        residual = float(
            fit_head.residual_energy(
                X[val_mask], normalized[val_mask].reshape(-1, 1)
            )
        )
        target_variance = max(float(np.var(normalized)), 1e-3)
        feature_mass = mu / max(X.shape[0], 1)
        per_domain.append(
            {
                "w": full_head.W.copy(),
                "mu": mu,
                "r": residual,
                "rho": feature_mass / target_variance,
                "n": int(X.shape[0]),
                "scale": scale,
                "target_variance": target_variance,
            }
        )
    return per_domain, int(d_aug), manifests


def _select_minimum(rows, value_key):
    """Choose the least-forgetting factor among numerical ties."""
    if not rows:
        raise ValueError("cannot select a minimum from an empty curve")
    values = [float(row[value_key]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"curve {value_key} contains a non-finite value")
    best_value = min(values)
    tolerance = NUMERIC_TOLERANCE * max(1.0, abs(best_value))
    candidates = [
        row for row in rows if float(row[value_key]) <= best_value + tolerance
    ]
    return max(candidates, key=lambda row: float(row["factor"]))


def predicted_factor(per_domain, d_aug, lam, grid=FACTOR_GRID, gain=1.0):
    """Return the train-only predicted optimum and its objective curve."""
    count = len(per_domain)
    mus = np.asarray([item["mu"] for item in per_domain], dtype=np.float64)
    residuals = np.asarray([item["r"] for item in per_domain], dtype=np.float64)
    rhos = np.asarray([item["rho"] for item in per_domain], dtype=np.float64)
    weights = [item["w"] for item in per_domain]
    ages = np.asarray(
        [count - 1 - index for index in range(count)], dtype=np.float64
    )

    curve = []
    for factor in grid:
        factor = float(factor)
        aged_mass = (factor**ages) * mus
        denominator = float(np.sum(aged_mass) + lam)
        coefficients = aged_mass / denominator
        expected_weight = sum(
            coefficients[index] * weights[index] for index in range(count)
        )
        bias = float(
            np.mean(
                [
                    rhos[index]
                    * np.sum((expected_weight - weights[index]) ** 2)
                    for index in range(count)
                ]
            )
        )
        variance_base = float(
            np.sum((factor ** (2 * ages)) * residuals * mus)
            / denominator**2
        )
        variance = (
            float(gain)
            * int(d_aug)
            * float(np.mean(rhos))
            * variance_base
        )
        curve.append(
            {
                "factor": factor,
                "bias_term": bias,
                "variance_term": variance,
                "objective": bias + variance,
            }
        )
    best = _select_minimum(curve, "objective")
    return float(best["factor"]), curve


def oracle_factor(train, test, grid=FACTOR_GRID, lam=100.0):
    """Diagnostic test sweep on the same factor grid as the prediction.

    The primary oracle minimizes balanced normalized MSE with *unclipped*
    predictions, matching the squared-risk object approximated by J(f).
    Clipped relative MAE is retained as a secondary downstream task metric.
    """
    count = len(train)
    dimension = train[0]["X"].shape[1]
    scales = {
        index: max(float(np.mean(train[index]["y"])), 1e-6)
        for index in range(count)
    }
    curve = []
    for factor in grid:
        head = ForgettingRidgeRLS(
            d_in=dimension, d_out=1, lam=lam, forget=float(factor)
        )
        for index in range(count):
            head.begin_task()
            head.accumulate(
                train[index]["X"],
                (train[index]["y"] / scales[index]).reshape(-1, 1),
            )
            head.solve()
        per_domain = []
        per_domain_mse = []
        for index in range(count):
            normalized_prediction = head.predict(
                test[index]["X"], non_negative=False
            )[:, 0]
            normalized_target = test[index]["y"] / scales[index]
            per_domain_mse.append(
                float(
                    np.mean(
                        (normalized_prediction - normalized_target) ** 2
                    )
                )
            )
            prediction = (
                np.clip(normalized_prediction, 0.0, None) * scales[index]
            )
            per_domain.append(relative_mae(prediction, test[index]["y"]))
        curve.append(
            {
                "factor": float(factor),
                "balanced_normalized_mse": float(np.mean(per_domain_mse)),
                "per_domain_normalized_mse": per_domain_mse,
                "balanced_rel_mae": float(np.mean(per_domain)),
                "per_domain_rel_mae": per_domain,
            }
        )
    best = _select_minimum(curve, "balanced_normalized_mse")
    return (
        float(best["factor"]),
        float(best["balanced_normalized_mse"]),
        curve,
    )


def _curve_row(curve, factor):
    for row in curve:
        if abs(float(row["factor"]) - float(factor)) <= NUMERIC_TOLERANCE:
            return row
    raise KeyError(f"factor {factor} not present in curve")


def _append_legacy_csv(path, payload):
    """Optional compatibility output; formal aggregation uses JSON artifacts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    new = not destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                [
                    "config",
                    "task",
                    "T",
                    "sample_seed",
                    "split_seed",
                    "f_pred",
                    "f_oracle",
                    "rel_oracle",
                ]
            )
        writer.writerow(
            [
                payload["config_name"],
                payload["task"],
                payload["domain_count"],
                payload["sample_seed"],
                payload["split_seed"],
                payload["f_pred"],
                payload["f_oracle_DIAGNOSTIC"],
                payload["test_rel_oracle_DIAGNOSTIC"],
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["crowd", "age"], required=True)
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument(
        "--backbone", default="vit_base_patch14_dinov2.lvd142m"
    )
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty and not _git_clean():
        raise SystemExit(
            "theory calibration refused: git worktree is dirty"
        )

    started = time.time()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    train, test, names, data_manifest = load_task(
        config,
        args.task,
        args.backbone,
        args.img_size,
        args.max_per_domain,
        sample_seed=args.sample_seed,
    )
    quantities, d_aug, holdout = estimate_training_quantities(
        train,
        args.lam,
        split_seed=args.split_seed,
        val_every=args.val_every,
    )
    f_pred, prediction_curve = predicted_factor(
        quantities, d_aug, args.lam, FACTOR_GRID, gain=1.0
    )
    f_oracle, mse_oracle, oracle_curve = oracle_factor(
        train, test, FACTOR_GRID, args.lam
    )
    relmae_best = _select_minimum(oracle_curve, "balanced_rel_mae")
    f_oracle_relmae = float(relmae_best["factor"])
    prediction_test_row = _curve_row(oracle_curve, f_pred)
    f1_test_row = _curve_row(oracle_curve, 1.0)
    mse_pred = float(prediction_test_row["balanced_normalized_mse"])
    mse_f1 = float(f1_test_row["balanced_normalized_mse"])
    rel_pred = float(prediction_test_row["balanced_rel_mae"])
    rel_f1 = float(f1_test_row["balanced_rel_mae"])
    rel_best = float(relmae_best["balanced_rel_mae"])

    payload = {
        "protocol": PROTOCOL,
        "diagnostic_only": True,
        "test_data_used_only_for_oracle_diagnostic": True,
        "protocol_note": (
            "Image-level scalar diagnostic. f_pred uses training data only; "
            "the primary f_oracle minimizes unclipped balanced normalized "
            "test MSE to match the theory's squared-risk target. The clipped "
            "relative-MAE oracle is secondary. All test fields are diagnostic "
            "only and must never select a deployed method."
        ),
        "config_name": config_path.name,
        "task": args.task,
        "domain_count": len(train),
        "domain_names": names,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "factor_grid": list(FACTOR_GRID),
        "variance_gain_fixed": 1.0,
        "data_manifest": data_manifest,
        "holdout_split": {
            "split_seed": args.split_seed,
            "domains": holdout,
        },
        "training_quantities": [
            {
                key: value
                for key, value in record.items()
                if key != "w"
            }
            for record in quantities
        ],
        "prediction_curve_train_only": prediction_curve,
        "oracle_curve_DIAGNOSTIC": oracle_curve,
        "f_pred": f_pred,
        "f_oracle_DIAGNOSTIC": f_oracle,
        "f_oracle_relmae_DIAGNOSTIC": f_oracle_relmae,
        "factor_absolute_error_DIAGNOSTIC": abs(f_pred - f_oracle),
        "test_mse_f1_DIAGNOSTIC": mse_f1,
        "test_mse_at_f_pred_DIAGNOSTIC": mse_pred,
        "test_mse_oracle_DIAGNOSTIC": mse_oracle,
        "oracle_mse_gain_over_f1_DIAGNOSTIC": (
            (mse_f1 - mse_oracle) / max(abs(mse_f1), 1e-12)
        ),
        "selected_mse_gain_over_f1_DIAGNOSTIC": (
            (mse_f1 - mse_pred) / max(abs(mse_f1), 1e-12)
        ),
        "oracle_mse_regret_at_f_pred_DIAGNOSTIC": (
            (mse_pred - mse_oracle) / max(abs(mse_oracle), 1e-12)
        ),
        "test_rel_f1_DIAGNOSTIC": rel_f1,
        "test_rel_at_f_pred_DIAGNOSTIC": rel_pred,
        "test_rel_oracle_DIAGNOSTIC": rel_best,
        "oracle_rel_gain_over_f1_DIAGNOSTIC": (
            (rel_f1 - rel_best) / max(abs(rel_f1), 1e-12)
        ),
        "selected_rel_gain_over_f1_DIAGNOSTIC": (
            (rel_f1 - rel_pred) / max(abs(rel_f1), 1e-12)
        ),
        "oracle_rel_regret_at_f_pred_DIAGNOSTIC": (
            (rel_pred - rel_best) / max(abs(rel_best), 1e-12)
        ),
        "runtime_seconds": time.time() - started,
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "fstar_calibration.json"
    dump_result(
        artifact,
        with_provenance(payload, config_path, vars(args)),
    )
    if args.csv:
        _append_legacy_csv(args.csv, payload)

    print(
        f"[{config_path.name}] task={args.task} seed={args.sample_seed} "
        f"domains={names}"
    )
    print(
        f"  f*_pred={f_pred:.2f} | f*_oracle={f_oracle:.2f} "
        f"(matched MSE) | |error|={abs(f_pred-f_oracle):.2f}"
    )
    print(
        f"  test MSE f=1={mse_f1:.6f} pred={mse_pred:.6f} "
        f"oracle={mse_oracle:.6f}"
    )
    print(
        f"  downstream relMAE f=1={rel_f1:.6f} pred={rel_pred:.6f} "
        f"oracle={rel_best:.6f} at f={f_oracle_relmae:.2f}"
    )
    print(f"  saved -> {artifact} ({time.time()-started:.1f}s)")


if __name__ == "__main__":
    main()
