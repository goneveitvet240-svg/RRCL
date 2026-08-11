"""Run DOS-ELM-style and SIFt-RLS baselines on matched image-level heads.

Supported tasks are age estimation, AVA aesthetic prediction, KADID IQA, and
an explicitly diagnostic image-level crowd-counting head.  The crowd pipeline's
main method also has a dense patch head, so crowd-image results must not be put
in the full-model headline table.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from analytic_scalar_protocol import (
    default_validation_masks,
    grouped_validation_masks,
    mean_scales,
    run_dos_elm_style,
    run_sift,
)
from run_provenance import with_provenance
from result_io import dump_result


def _load_arrays(domains):
    train, test = {}, {}
    for domain in range(domains.n_domains()):
        for split, destination in (("train", train), ("test", test)):
            features, targets = [], []
            for feature, target, _ in domains.stream(split, domain):
                features.append(np.asarray(feature, dtype=np.float64).reshape(-1))
                targets.append(float(np.asarray(target).reshape(-1)[0]))
            if not features:
                raise RuntimeError(f"domain {domain} has no {split} samples")
            destination[domain] = (
                np.asarray(features, dtype=np.float64),
                np.asarray(targets, dtype=np.float64),
            )
    return train, test


def _load_crowd_image_arrays(domains):
    from run_real_image_aux import _image_feature, _image_target

    train, test = {}, {}
    for domain in range(domains.n_domains()):
        for split, destination in (("train", train), ("test", test)):
            features, targets = [], []
            for patch_features, patch_targets, _ in domains.stream(split, domain):
                features.append(
                    np.asarray(
                        _image_feature(patch_features), dtype=np.float64
                    ).reshape(-1)
                )
                targets.append(float(np.asarray(_image_target(patch_targets)).reshape(-1)[0]))
            if not features:
                raise RuntimeError(f"crowd domain {domain} has no {split} samples")
            destination[domain] = (
                np.asarray(features, dtype=np.float64),
                np.asarray(targets, dtype=np.float64),
            )
    return train, test


def _build_domains(args, config):
    if args.task == "age":
        from datasets_age import AgeDomains, build_age_config

        return AgeDomains(
            build_age_config(config),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
        )
    if args.task == "ava":
        from datasets_ava import AVADomains, build_ava_config

        return AVADomains(
            build_ava_config(config),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
        )
    if args.task == "iqa":
        from datasets_iqa import IQADomains, build_iqa_config

        return IQADomains(
            build_iqa_config(config),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
        )
    if args.task == "crowd_image":
        from datasets_real import RealCountingDomains, build_from_config

        return RealCountingDomains(
            build_from_config(config),
            backbone=args.backbone,
            img_size=args.img_size,
            max_per_domain=args.max_per_domain,
            sample_seed=args.sample_seed,
        )
    raise ValueError(f"unsupported task: {args.task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["age", "ava", "iqa", "crowd_image"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--max-per-domain", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--dos-initial-factor", type=float, default=1.0)
    parser.add_argument("--dos-factor-min", type=float, default=0.0)
    parser.add_argument("--sift-factors", default="1.0,0.8,0.6,0.4,0.2")
    parser.add_argument("--sift-epsilon", type=float, default=1e-8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = json.load(handle)
    domains = _build_domains(args, config)
    if args.task == "crowd_image":
        train, test = _load_crowd_image_arrays(domains)
        groups = None
    else:
        train, test = _load_arrays(domains)
        groups = (
            {
                domain: np.asarray(domains.groups("train", domain))
                for domain in range(domains.n_domains())
            }
            if args.task == "iqa"
            else None
        )
    validation_masks = (
        grouped_validation_masks(groups, args.val_every)
        if groups is not None
        else default_validation_masks(train, args.val_every)
    )
    scales = mean_scales(train)
    sift_factors = [float(value) for value in args.sift_factors.split(",")]

    started = time.time()
    dos = run_dos_elm_style(
        train,
        test,
        ridge=args.lam,
        scales=scales,
        initial_factor=args.dos_initial_factor,
        factor_min=args.dos_factor_min,
    )
    sift = run_sift(
        train,
        test,
        validation_masks,
        ridge=args.lam,
        factors=sift_factors,
        epsilon=args.sift_epsilon,
        scales=scales,
    )
    names = [domain["name"] for domain in config["domains"]]
    payload = {
        "task": args.task,
        "config": args.config,
        "domain_names": names,
        "sample_counts": {
            "train": [int(train[index][0].shape[0]) for index in train],
            "test": [int(test[index][0].shape[0]) for index in test],
        },
        "normalization": {
            "kind": "per-domain mean absolute target",
            "scales": scales,
        },
        "comparison_scope": (
            "image-level forgetting-rule diagnostic; not comparable to the "
            "full dual-head crowd model"
            if args.task == "crowd_image"
            else "matched frozen-backbone image-level linear head"
        ),
        "dos_elm_style": dos,
        "sift_rls": sift,
        "elapsed_seconds": time.time() - started,
    }
    if hasattr(domains, "data_manifest"):
        payload["data_manifest"] = domains.data_manifest()

    os.makedirs(args.out, exist_ok=True)
    output = os.path.join(args.out, "analytic_baselines.json")
    dump_result(output, with_provenance(payload, args.config, vars(args)))
    print(
        f"DOS-ELM-style={dos['score']:.5f} | "
        f"SIFt-RLS={sift['score']:.5f} (train-only f={sift['selected_factor']:g})"
    )
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
