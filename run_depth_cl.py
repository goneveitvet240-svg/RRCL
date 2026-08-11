"""Continual patch-level depth regression with a frozen DINOv2 backbone.

This runner is deliberately separate from the crowd-counting pipeline: depth
targets are per-patch metric depths and must never be summed as counts.  It
reports balanced normalized MAE and records the bounded-memory selector state.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from adaptive_selector import BalancedSufficientStatsSelector
from rls_head import ForgettingRidgeRLS
from run_adaptive_f import classify_adaptive_result
from run_provenance import with_provenance
from result_io import dump_result


def _first_dim(domains):
    for X, _, _ in domains.stream("train", 0):
        return int(np.asarray(X).shape[1])
    raise RuntimeError("empty first depth domain")


def domain_scale(domains, domain):
    total, count = 0.0, 0
    for _, Y, _ in domains.stream("train", domain):
        values = np.asarray(Y, dtype=np.float64)
        total += float(values.sum())
        count += int(values.size)
    return max(total / max(count, 1), 1e-6)


def evaluate_domain(domains, domain, head, scale):
    absolute_errors, squared_errors, predictions, targets = [], [], [], []
    for X, Y, _ in domains.stream("test", domain):
        truth = np.asarray(Y, dtype=np.float64).reshape(-1)
        prediction = head.predict(X)[:, 0] * scale
        error = prediction - truth
        absolute_errors.append(np.abs(error))
        squared_errors.append(error * error)
        predictions.append(prediction)
        targets.append(truth)
    absolute = np.concatenate(absolute_errors)
    squared = np.concatenate(squared_errors)
    truth = np.concatenate(targets)
    prediction_all = np.concatenate(predictions)
    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction_all)
        & (truth > 1e-3)
    )
    if not np.any(valid):
        raise RuntimeError(f"depth domain {domain} has no valid evaluation patches")
    safe_truth = truth[valid]
    safe_prediction = np.maximum(prediction_all[valid], 1e-3)
    valid_error = safe_prediction - safe_truth
    ratio = np.maximum(safe_prediction / safe_truth, safe_truth / safe_prediction)
    log_difference = np.log(safe_prediction) - np.log(safe_truth)
    mae = float(np.mean(np.abs(valid_error)))
    return {
        "mae_m": mae,
        "normalized_mae": mae / max(float(np.mean(np.abs(safe_truth))), 1e-6),
        "rmse_m": float(np.sqrt(np.mean(valid_error * valid_error))),
        "abs_rel": float(np.mean(np.abs(valid_error) / safe_truth)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "silog": float(
            100.0
            * np.sqrt(
                max(
                    float(np.mean(log_difference ** 2) - np.mean(log_difference) ** 2),
                    0.0,
                )
            )
        ),
        "valid_patches": int(np.sum(valid)),
    }


def fixed_factor(domains, factor, lam):
    T = domains.n_domains()
    d = _first_dim(domains)
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam, forget=factor)
    scales = [domain_scale(domains, t) for t in range(T)]
    matrix = np.full((T, T), np.nan)
    for t in range(T):
        head.begin_task()
        for X, Y, _ in domains.stream("train", t):
            head.accumulate(X, np.asarray(Y, dtype=np.float64) / scales[t])
        head.solve()
        for previous in range(t + 1):
            matrix[t, previous] = evaluate_domain(
                domains, previous, head, scales[previous]
            )["normalized_mae"]
    final_metrics = [
        evaluate_domain(domains, domain, head, scales[domain])
        for domain in range(T)
    ]
    return float(np.nanmean(matrix[T - 1, :T])), matrix.tolist(), final_metrics


def _split_stats(domains, domain, scale, dimension, val_every=5):
    def empty():
        return [np.zeros((dimension, dimension)), np.zeros((dimension, 1)), 0.0, 0]

    fit, val = empty(), empty()
    for index, (X, Y, _) in enumerate(domains.stream("train", domain)):
        Xa = np.concatenate(
            [np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1))], axis=1
        )
        target = np.asarray(Y, dtype=np.float64).reshape(-1, 1) / scale
        destination = val if index % val_every == val_every - 1 else fit
        destination[0] += Xa.T @ Xa
        destination[1] += Xa.T @ target
        destination[2] += float(np.sum(target * target))
        destination[3] += int(Xa.shape[0])
    if val[3] == 0:
        raise RuntimeError(f"depth domain {domain} has no held-out validation images")
    return fit, val


def adaptive_factor(
    domains,
    lam,
    f_min,
    search_mode,
    abstain_relative_gain,
    grid,
    max_component_relative_harm=None,
):
    T = domains.n_domains()
    d = _first_dim(domains)
    scales = [domain_scale(domains, t) for t in range(T)]
    head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    selector = BalancedSufficientStatsSelector(
        d + 1,
        lam,
        f_min=f_min,
        search_mode=search_mode,
        grid=grid,
        abstain_relative_gain=abstain_relative_gain,
        max_component_relative_harm=max_component_relative_harm,
    )
    factors, drift_scores = [], []
    matrix = np.full((T, T), np.nan)
    for t in range(T):
        fit, val = _split_stats(domains, t, scales[t], d + 1)
        selection = selector.select_and_update(fit, val)
        factors.append(selection.factor)
        drift_scores.append(selection.drift_score)
        head.begin_task(selection.factor)
        for X, Y, _ in domains.stream("train", t):
            head.accumulate(X, np.asarray(Y, dtype=np.float64) / scales[t])
        head.solve()
        for previous in range(t + 1):
            matrix[t, previous] = evaluate_domain(
                domains, previous, head, scales[previous]
            )["normalized_mae"]
    final_metrics = [
        evaluate_domain(domains, domain, head, scales[domain])
        for domain in range(T)
    ]
    return (
        float(np.nanmean(matrix[T - 1, :T])),
        matrix.tolist(),
        factors,
        drift_scores,
        selector.state_bytes,
        final_metrics,
    )


def single_domain(domains, lam):
    scores = []
    for t in range(domains.n_domains()):
        d = _first_dim(domains)
        scale = domain_scale(domains, t)
        head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
        head.begin_task()
        for X, Y, _ in domains.stream("train", t):
            head.accumulate(X, np.asarray(Y, dtype=np.float64) / scale)
        head.solve()
        scores.append(evaluate_domain(domains, t, head, scale)["normalized_mae"])
    return float(np.mean(scores)), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--forgets", default="1.0,0.8,0.6,0.4,0.3,0.25,0.2,0.1,0.05")
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--max-per-domain", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--f-min", type=float, default=0.05)
    parser.add_argument(
        "--selector-search",
        choices=["continuous", "grid"],
        default="grid",
        help="grid is the paper default because the objective is not proven unimodal",
    )
    parser.add_argument("--abstain-relative-gain", type=float, default=1e-4)
    parser.add_argument("--max-component-relative-harm", type=float, default=None)
    parser.add_argument("--out", default="runs_real/nyu")
    args = parser.parse_args()

    from datasets_depth import DepthDomains, build_depth_config

    with open(args.config) as handle:
        config = json.load(handle)
    domains = DepthDomains(
        build_depth_config(config),
        backbone=args.backbone,
        img_size=args.img_size,
        max_per_domain=args.max_per_domain,
        sample_seed=args.sample_seed,
    )
    factors = [float(value) for value in args.forgets.split(",")]
    started = time.time()

    sweep = []
    for factor in factors:
        score, matrix_fixed, standard_metrics = fixed_factor(
            domains, factor, args.lam
        )
        sweep.append(
            {
                "factor": factor,
                "normalized_mae": score,
                "matrix": matrix_fixed,
                "final_standard_metrics": standard_metrics,
            }
        )
        print(f"[depth fixed f={factor:g}] balanced normalized MAE={score:.5f}")
    oracle = min(sweep, key=lambda row: row["normalized_mae"])
    f1 = next(row["normalized_mae"] for row in sweep if row["factor"] == 1.0)
    (
        adaptive,
        matrix,
        used,
        drift_scores,
        selector_bytes,
        adaptive_standard_metrics,
    ) = adaptive_factor(
        domains,
        args.lam,
        args.f_min,
        args.selector_search,
        args.abstain_relative_gain,
        np.round(np.arange(args.f_min, 1.0001, 0.05), 3),
        max_component_relative_harm=args.max_component_relative_harm,
    )
    independent, independent_per_domain = single_domain(domains, args.lam)
    verdict = classify_adaptive_result(
        f1,
        oracle["normalized_mae"],
        adaptive,
        used,
    )
    payload = {
        "config": args.config,
        "domain_note": config.get("_note", ""),
        "metric": "balanced normalized patch-depth MAE",
        "sweep": sweep,
        "oracle_f": oracle["factor"],
        "oracle_rel": oracle["normalized_mae"],
        "f1": f1,
        "adaptive": adaptive,
        "adaptive_matrix": matrix,
        "adaptive_standard_metrics": adaptive_standard_metrics,
        "f_used": used,
        "drift_scores": drift_scores,
        "selector_search": args.selector_search,
        "selector_state_bytes": selector_bytes,
        "data_manifest": domains.data_manifest(),
        "verdict": verdict,
        "single": independent,
        "single_per_domain": independent_per_domain,
        "elapsed_seconds": time.time() - started,
    }
    os.makedirs(args.out, exist_ok=True)
    output = os.path.join(args.out, "nyu_result.json")
    dump_result(output, with_provenance(payload, args.config, vars(args)))
    print(
        f"adaptive={adaptive:.5f}, f_used={[round(value, 4) for value in used]}, "
        f"oracle={oracle['normalized_mae']:.5f} at f={oracle['factor']:g}"
    )
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
