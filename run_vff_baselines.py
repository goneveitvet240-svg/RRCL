"""Compare RRCL against an explicit classic VFF-RLS rule.

The Paleologu et al. (IEEE SPL 2008) rule is sample-wise in its original
system-identification setting.  Here it is transparently adapted to RRCL's
task-aware, domain-batched update: powers are estimated on a held-out part of
the incoming domain and one factor is applied at that domain boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from holdout_split import (
    domain_sample_key,
    is_validation,
    iter_stream_with_ids,
    require_both_partitions,
    split_manifest,
)
from rls_head import ForgettingRidgeRLS
from run_adaptive_f import _aug_raw, _domain_suffstats_split, adaptive_train_eval_suff
from run_provenance import with_provenance
from result_io import dump_result
from run_real import _first_dim
from run_real_image_aux import _image_feature, _image_target, transform_patch_target
from run_real_norm_ablation import domain_scale, eval_domain, train_eval
from vff_baselines import paleologu_vff_factor


def _boundary_powers(domains, domain, head, scale, lam, val_every=5, split_seed=42):
    """Estimate pre-fit error, post-fit noise and leverage powers on held-out train.

    Powers are measured at the IMAGE level (mean-pooled features, total-count
    targets) to align the VFF factor selection with the image-level evaluation
    metric (rel-MAE), mirroring the same fix applied to the sufficient-statistics
    selector in run_adaptive_f.py.  The fit/validation partition uses the shared
    stable hash split (holdout_split.py)."""
    d = head.d - 1
    domain_head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    domain_head.begin_task()
    held_out = []
    sample_key = domain_sample_key(domains, domain)
    fit_ids, val_ids = [], []
    id_source = None
    for X, Y, _, image_id, source in iter_stream_with_ids(domains, "train", domain):
        id_source = source
        target = _image_target(Y) / scale
        if is_validation(sample_key, image_id, split_seed, val_every):
            held_out.append((_image_feature(np.asarray(X, dtype=np.float64)), target))
            val_ids.append(image_id)
        else:
            domain_head.accumulate(_image_feature(X), target)
            fit_ids.append(image_id)
    manifest = require_both_partitions(
        split_manifest(sample_key, split_seed, val_every, fit_ids, val_ids, id_source),
        f"VFF boundary powers for domain {domain} ({sample_key})",
    )
    domain_head.solve()
    covariance = np.linalg.inv(head.R + lam * np.eye(head.d))
    old_residuals, floor_residuals, leverages = [], [], []
    for X, target in held_out:
        augmented = _aug_raw(X)
        old_residuals.append(target - augmented @ head.W)
        floor_residuals.append(target - augmented @ domain_head.W)
        leverages.append(np.einsum("ij,jk,ik->i", augmented, covariance, augmented))
    old = np.concatenate(old_residuals).reshape(-1)
    floor = np.concatenate(floor_residuals).reshape(-1)
    leverage = np.concatenate(leverages).reshape(-1)
    return {
        "error_power": float(np.mean(old * old)),
        "noise_power": float(np.mean(floor * floor)),
        "leverage_power": float(np.mean(leverage * leverage)),
        "holdout_split": manifest,
    }


def run_paleologu(domains, patch_target, alpha, lam, gamma, xi, f_min, split_seed=42):
    T = domains.n_domains()
    d = _first_dim(domains)
    patch_head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    image_head = ForgettingRidgeRLS(d_in=d, d_out=1, lam=lam)
    scales, factors, diagnostics = [], [], []
    matrix = np.full((T, T), np.nan)
    for t in range(T):
        scale = domain_scale(domains, t, "mean")
        scales.append(scale)
        if t == 0:
            factor = 1.0
            powers = None
        else:
            powers = _boundary_powers(
                domains, t, image_head, scale, lam, split_seed=split_seed
            )
            factor = paleologu_vff_factor(
                error_power=powers["error_power"],
                noise_power=powers["noise_power"],
                leverage_power=powers["leverage_power"],
                gamma=gamma, xi=xi, f_min=f_min, f_max=1.0,
            )
            # diagnostic field added AFTER factor computation
            powers["power_space"] = "image_head"
        factors.append(factor)
        diagnostics.append(powers)
        patch_head.begin_task(factor)
        image_head.begin_task(factor)
        for X, Y, _ in domains.stream("train", t):
            patch_head.accumulate(X, transform_patch_target(Y, patch_target) / scale)
            image_head.accumulate(_image_feature(X), _image_target(Y) / scale)
        patch_head.solve()
        image_head.solve()
        for previous in range(t + 1):
            _, matrix[t, previous] = eval_domain(
                domains,
                previous,
                patch_head,
                image_head,
                alpha,
                scales[previous],
            )
    return float(np.nanmean(matrix[T - 1, :T])), factors, diagnostics, matrix.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--patch-target", default="dct5")
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="independent seed for the stable-hash fit/val split")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--xi", type=float, default=1e-6)
    parser.add_argument("--f-min", type=float, default=0.05)
    parser.add_argument(
        "--selector-search",
        choices=["continuous", "grid"],
        default="grid",
        help="RRCL comparator search; grid is the defensible paper default",
    )
    parser.add_argument("--out", default="runs_real/vff")
    args = parser.parse_args()

    from datasets_real import RealCountingDomains, build_from_config

    with open(args.config) as handle:
        config = json.load(handle)
    domains = RealCountingDomains(
        build_from_config(config),
        backbone=args.backbone,
        img_size=args.img_size,
        max_per_domain=args.max_per_domain,
        sample_seed=args.sample_seed,
    )
    data_manifest = domains.data_manifest()
    started = time.time()
    _, f1, _, f1_matrix, f1_scales = train_eval(
        domains, args.patch_target, 1.0, args.alpha, args.lam, "mean"
    )
    vff, vff_factors, vff_diagnostics, vff_matrix = run_paleologu(
        domains,
        args.patch_target,
        args.alpha,
        args.lam,
        args.gamma,
        args.xi,
        args.f_min,
        split_seed=args.split_seed,
    )
    grid = np.round(np.arange(args.f_min, 1.0001, 0.05), 3)
    (
        ours,
        ours_factors,
        objectives,
        drift_scores,
        state_bytes,
        ours_evidence,
    ) = adaptive_train_eval_suff(
        domains,
        args.patch_target,
        args.alpha,
        args.lam,
        args.f_min,
        1.0,
        grid,
        search_mode=args.selector_search,
        split_seed=args.split_seed,
    )
    payload = {
        "config": args.config,
        "split_seed": args.split_seed,
        "f1": f1,
        "f1_evidence": {
            "domain_names": [spec.name for spec in domains.domains],
            "scales": f1_scales,
            "M_rel": f1_matrix,
        },
        "data_manifest": data_manifest,
        "paleologu_batch_vff": {
            "score": vff,
            "f_used": vff_factors,
            "diagnostics": vff_diagnostics,
            "matrix": vff_matrix,
            "gamma": args.gamma,
            "xi": args.xi,
            "power_space": "image_head",
            "adaptation_note": "Eq.17-18 power rule evaluated once per known domain boundary; "
                              "powers estimated from image-head residuals (image-head proxy, "
                              "not exact fused-objective).",
        },
        "rrcl_bounded": {
            "score": ours,
            "f_used": ours_factors,
            "selector_objectives": objectives,
            "drift_scores": drift_scores,
            "selector_state_bytes": state_bytes,
            "evidence": ours_evidence,
        },
        "elapsed_seconds": time.time() - started,
    }
    os.makedirs(args.out, exist_ok=True)
    output = os.path.join(args.out, "vff_baselines.json")
    dump_result(output, with_provenance(payload, args.config, vars(args)))
    print(f"f=1={f1:.5f} | Paleologu batch VFF={vff:.5f} | RRCL={ours:.5f}")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
