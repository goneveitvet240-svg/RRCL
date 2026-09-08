#!/usr/bin/env python3
"""Run the KADID normalized-objective and ridge-scale audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from development_baselines import (  # noqa: E402
    METHOD_DOMAIN_BALANCED,
    METHOD_DOMAIN_BALANCED_MASS_MATCHED,
    METHOD_POOLED,
    METHOD_SAMPLE_MEAN_POOLED,
    condition_number,
    fit_scalar_head,
    objective_weights,
    quadratic_mean_loss,
    relative_gain,
    select_scalar_lambda,
    weighted_sufficient_statistics,
)
from features import resolve_device  # noqa: E402
from result_io import dump_result  # noqa: E402
from run_new_method_kadid_baselines import (  # noqa: E402
    _ToyScalarDomains,
    _config_hash,
    _matrix_summary,
    _serializable_records,
    assert_kadid_config,
    collect_domain_records,
    evaluate,
    require_output,
)
from run_provenance import with_provenance  # noqa: E402


NORMALIZED_METHODS = (METHOD_SAMPLE_MEAN_POOLED, METHOD_DOMAIN_BALANCED)
RAW_EQUIVALENTS = {
    METHOD_SAMPLE_MEAN_POOLED: METHOD_POOLED,
    METHOD_DOMAIN_BALANCED: METHOD_DOMAIN_BALANCED_MASS_MATCHED,
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_lambda_grid(protocol):
    spec = protocol["lambda_grid"]
    if spec.get("kind") != "log10":
        raise RuntimeError("v2 requires a log10 lambda grid")
    points = int(spec["points"])
    if points < 2:
        raise RuntimeError("lambda grid requires at least two points")
    return np.logspace(
        float(spec["minimum_exponent"]),
        float(spec["maximum_exponent"]),
        points,
        dtype=np.float64,
    ).tolist()


def require_frozen_arguments(arguments, protocol):
    expected = {
        "backbone": protocol["backbone"],
        "img_size": int(protocol["image_size"]),
        "max_per_domain": int(protocol["max_train_per_domain"]),
        "sample_seed": int(protocol["sample_seed"]),
        "split_seed": int(protocol["selector_split_seed"]),
        "val_every": int(protocol["selector_val_every"]),
        "lambdas": protocol_lambda_grid(protocol),
    }
    if "selector_sample_key" in protocol:
        expected["selector_sample_key"] = protocol["selector_sample_key"]
    differences = {
        key: {"expected": value, "actual": getattr(arguments, key)}
        for key, value in expected.items()
        if getattr(arguments, key) != value
    }
    if differences:
        raise RuntimeError(
            "natural run arguments differ from frozen KADID scale-audit-v2 protocol: "
            + json.dumps(differences, sort_keys=True)
        )
    return True


def _relative_l2(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-15))


def _maximum_absolute(left, right):
    return float(
        np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))
    )


def _validation_risk(weights, records):
    return float(np.mean([quadratic_mean_loss(weights, record["val"]) for record in records]))


def equivalence_audit(records, method, selected_lambda, selected_weights):
    """Check the exact raw/normalized ridge reparameterization."""

    blocks = [record["fit"] for record in records]
    counts = np.asarray([block["n"] for block in blocks], dtype=np.float64)
    total_count = float(counts.sum())
    normalized_domain_weights = objective_weights(counts, method)
    normalized_R, normalized_C = weighted_sufficient_statistics(
        blocks, normalized_domain_weights
    )
    raw_method = RAW_EQUIVALENTS[method]
    raw_lambda = total_count * float(selected_lambda)
    raw_model = fit_scalar_head(records, raw_method, raw_lambda, "fit")
    normalized_validation_risk = _validation_risk(selected_weights, records)
    raw_validation_risk = _validation_risk(raw_model["W"], records)
    return {
        "normalized_method": method,
        "raw_equivalent_method": raw_method,
        "fit_observation_count": int(total_count),
        "normalized_lambda": float(selected_lambda),
        "raw_equivalent_lambda": raw_lambda,
        "relation": "lambda_raw = fit_observation_count * lambda_normalized",
        "statistics": {
            "R_max_abs_after_dividing_raw_by_N": _maximum_absolute(
                normalized_R, raw_model["R"] / total_count
            ),
            "C_max_abs_after_dividing_raw_by_N": _maximum_absolute(
                normalized_C, weighted_sufficient_statistics(
                    blocks, objective_weights(counts, raw_method)
                )[1] / total_count
            ),
            "domain_weight_max_abs_after_dividing_raw_by_N": _maximum_absolute(
                normalized_domain_weights,
                raw_model["domain_weights"] / total_count,
            ),
        },
        "weights_max_abs": _maximum_absolute(selected_weights, raw_model["W"]),
        "weights_relative_l2": _relative_l2(selected_weights, raw_model["W"]),
        "normalized_validation_risk": normalized_validation_risk,
        "raw_equivalent_validation_risk": raw_validation_risk,
        "validation_risk_absolute_difference": abs(
            normalized_validation_risk - raw_validation_risk
        ),
    }


def _evaluate_trajectory(domains, records, method, lambdas_by_boundary):
    count = len(records)
    relative = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    for boundary, lam in enumerate(lambdas_by_boundary):
        model = fit_scalar_head(records[: boundary + 1], method, lam, "fit")
        for index in range(boundary + 1):
            result = evaluate(domains, index, model["W"], records[index]["scale"])
            relative[boundary, index] = result["relative_mae"]
            mse[boundary, index] = result["unclipped_normalized_mse"]
    return {
        "relative_mae": _matrix_summary(relative),
        "unclipped_normalized_mse": _matrix_summary(mse),
    }


def run_normalized_method(domains, records, method, lambdas):
    count = len(records)
    relative = np.full((count, count), np.nan)
    mse = np.full((count, count), np.nan)
    selections = []
    final_model = None
    for boundary in range(count):
        seen = records[: boundary + 1]
        selected = select_scalar_lambda(seen, method, lambdas)
        model = fit_scalar_head(seen, method, selected.lam, "fit")
        for index in range(boundary + 1):
            result = evaluate(domains, index, model["W"], records[index]["scale"])
            relative[boundary, index] = result["relative_mae"]
            mse[boundary, index] = result["unclipped_normalized_mse"]
        selections.append(
            {
                "boundary": boundary,
                "lambda": selected.lam,
                "validation_risk": selected.validation_risk,
                "candidate_trace": list(selected.candidates),
                "equivalence_audit": equivalence_audit(
                    seen, method, selected.lam, selected.weights
                ),
            }
        )
        final_model = model
    return {
        "method": method,
        "objective_total_observation_weight": 1.0,
        "lambda_selection": selections,
        "aligned_fit_only": {
            "relative_mae": _matrix_summary(relative),
            "unclipped_normalized_mse": _matrix_summary(mse),
        },
        "final_system": {
            "selected_lambda": selections[-1]["lambda"],
            "condition_number": condition_number(
                final_model["R"], selections[-1]["lambda"]
            ),
            "domain_weights": final_model["domain_weights"].tolist(),
            "total_observation_weight": float(
                np.dot(
                    final_model["domain_weights"],
                    [record["fit"]["n"] for record in records],
                )
            ),
        },
    }


def run_same_lambda_pairs(domains, records, methods):
    paired = {}
    for anchor_method, anchor_result in methods.items():
        lambdas = [item["lambda"] for item in anchor_result["lambda_selection"]]
        evaluations = {
            method: (
                anchor_result["aligned_fit_only"]
                if method == anchor_method
                else _evaluate_trajectory(domains, records, method, lambdas)
            )
            for method in NORMALIZED_METHODS
        }
        pooled_loss = evaluations[METHOD_SAMPLE_MEAN_POOLED]["relative_mae"][
            "final_balanced_loss"
        ]
        balanced_loss = evaluations[METHOD_DOMAIN_BALANCED]["relative_mae"][
            "final_balanced_loss"
        ]
        paired[anchor_method] = {
            "anchor_selected_on_validation_only": True,
            "lambda_by_boundary": lambdas,
            "evaluations": evaluations,
            "final_domain_balance_gain_vs_sample_mean_pooled": relative_gain(
                pooled_loss, balanced_loss
            ),
        }
    return paired


def run_audit(domains, protocol, arguments, data_manifest, runtime):
    records, dimension = collect_domain_records(
        domains,
        arguments.split_seed,
        arguments.val_every,
        getattr(arguments, "selector_sample_key", None),
    )
    methods = {
        method: run_normalized_method(domains, records, method, arguments.lambdas)
        for method in NORMALIZED_METHODS
    }
    pooled_loss = methods[METHOD_SAMPLE_MEAN_POOLED]["aligned_fit_only"][
        "relative_mae"
    ]["final_balanced_loss"]
    balanced_loss = methods[METHOD_DOMAIN_BALANCED]["aligned_fit_only"][
        "relative_mae"
    ]["final_balanced_loss"]
    fit_groups = set().union(*(record["_fit_groups"] for record in records))
    validation_groups = set().union(
        *(record["_validation_groups"] for record in records)
    )
    cross_role_overlap = fit_groups & validation_groups
    batch_label = protocol.get("scope_batch_label", "batch2")
    return {
        "protocol_id": protocol["protocol_id"],
        "evidence_role": "development-only",
        "confirmation_data_used": False,
        "task": "KADID-10k image quality assessment",
        "audit_question": "domain weighting separated from global ridge scale",
        "runtime": runtime,
        "feature_dimension_with_bias": dimension,
        "domain_config": {
            "path": str(Path(arguments.domain_config).resolve())
            if arguments.domain_config
            else None,
            "sha256": _config_hash(arguments.domain_config)
            if arguments.domain_config
            else None,
        },
        "data_manifest": data_manifest,
        "domain_records": _serializable_records(records),
        "selector_isolation_audit": {
            "selector_sample_key": getattr(arguments, "selector_sample_key", None),
            "scope": (
                "global_across_domains"
                if getattr(arguments, "selector_sample_key", None) is not None
                else "domain_specific"
            ),
            "global_fit_groups": {
                "count": len(fit_groups),
                "ids_sha256": hashlib.sha256(
                    "\n".join(sorted(fit_groups)).encode("utf-8")
                ).hexdigest(),
            },
            "global_validation_groups": {
                "count": len(validation_groups),
                "ids_sha256": hashlib.sha256(
                    "\n".join(sorted(validation_groups)).encode("utf-8")
                ).hexdigest(),
            },
            "cross_role_overlap_count": len(cross_role_overlap),
            "cross_role_overlap_ids_sha256": hashlib.sha256(
                "\n".join(sorted(cross_role_overlap)).encode("utf-8")
            ).hexdigest(),
            "domains": [
                {
                    "name": record["name"],
                    "fit_groups": sorted(record["_fit_groups"]),
                    "validation_groups": sorted(record["_validation_groups"]),
                }
                for record in records
            ],
        },
        "lambda_grid": arguments.lambdas,
        "normalized_methods": methods,
        "same_lambda_pairs": run_same_lambda_pairs(domains, records, methods),
        "practical_comparison": {
            "sample_mean_pooled_final_relative_mae": pooled_loss,
            "domain_balanced_final_relative_mae": balanced_loss,
            "domain_balance_gain_vs_sample_mean_pooled": relative_gain(
                pooled_loss, balanced_loss
            ),
            "selection_source": "fit models selected by validation only",
        },
        "scope_status": {
            "projection_and_independent_heads": protocol.get(
                "projection_and_independent_heads_status",
                "retained_from_batch1_not_rerun",
            ),
            "fixed_trajectory_bank": f"not_measured_in_{batch_label}",
            "risk_controlled_hard_selection": f"not_measured_in_{batch_label}",
            "analytic_shrinkage": f"not_measured_in_{batch_label}",
            "automatic_f": f"not_measured_in_{batch_label}",
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "new_method_kadid_scale_audit_v2.json"),
    )
    parser.add_argument(
        "--domain-config",
        default=str(ROOT / "configs" / "domains_iqa_kadid.json"),
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "runs_real" / "new_method_kadid_scale_audit_v2" / "main"),
    )
    parser.add_argument("--backbone")
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--max-per-domain", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--lambdas")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = _load_json(args.protocol_config)
    args.backbone = args.backbone or protocol["backbone"]
    args.img_size = args.img_size or int(protocol["image_size"])
    args.max_per_domain = args.max_per_domain or int(protocol["max_train_per_domain"])
    args.sample_seed = (
        int(protocol["sample_seed"]) if args.sample_seed is None else args.sample_seed
    )
    args.split_seed = (
        int(protocol["selector_split_seed"])
        if args.split_seed is None
        else args.split_seed
    )
    args.val_every = args.val_every or int(protocol["selector_val_every"])
    args.selector_sample_key = protocol.get("selector_sample_key")
    args.lambdas = (
        [float(value) for value in args.lambdas.split(",")]
        if args.lambdas
        else protocol_lambda_grid(protocol)
    )
    output = require_output(args.out, protocol)

    if args.selftest:
        domains = _ToyScalarDomains()
        data_manifest = domains.data_manifest()
        args.domain_config = None
        runtime = {"device": "none-toy", "cache_dir": None}
    else:
        require_frozen_arguments(args, protocol)
        domain_payload = _load_json(args.domain_config)
        assert_kadid_config(args.domain_config, domain_payload, protocol)
        from datasets_iqa import IQADomains, build_iqa_config

        device = resolve_device(args.device)
        domains = IQADomains(
            build_iqa_config(domain_payload),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
            device=device,
            cache_dir=args.cache_dir,
        )
        data_manifest = domains.data_manifest()
        runtime = {"device": device, "cache_dir": domains.cache_dir}

    started = time.time()
    payload = run_audit(domains, protocol, args, data_manifest, runtime)
    payload["elapsed_seconds"] = time.time() - started
    payload["selftest"] = bool(args.selftest)
    result = with_provenance(payload, args.protocol_config, vars(args))
    destination = output / protocol.get(
        "result_filename", "kadid_scale_audit_v2.json"
    )
    dump_result(destination, result)
    print(json.dumps(payload["practical_comparison"], indent=2))
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
