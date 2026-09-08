#!/usr/bin/env python3
"""Resumable KADID DINOv2 feature precomputation with atomic caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets_iqa import IQADomains, build_iqa_config  # noqa: E402
from features import resolve_device  # noqa: E402
from result_io import dump_result  # noqa: E402
from run_new_method_kadid_baselines import assert_kadid_config  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def precompute(domains, *, limit=0, progress_every=50):
    planned = sum(
        len(domains._splits[index][split])
        for index in range(domains.n_domains())
        for split in ("train", "test")
    )
    processed = 0
    started = time.time()
    per_domain = []
    stop = False
    for index in range(domains.n_domains()):
        record = {"domain_index": index, "domain": domains.specs[index].name}
        for split in ("train", "test"):
            count = 0
            for _features, _target, _group in domains.stream(split, index):
                processed += 1
                count += 1
                if progress_every and processed % int(progress_every) == 0:
                    elapsed = time.time() - started
                    rate = processed / max(elapsed, 1e-9)
                    remaining = (planned - processed) / max(rate, 1e-9)
                    print(
                        f"features {processed}/{planned} "
                        f"({rate:.2f} images/s, ETA {remaining / 60:.1f} min)",
                        flush=True,
                    )
                if limit and processed >= int(limit):
                    stop = True
                    break
            record[split] = count
            if stop:
                break
        per_domain.append(record)
        if stop:
            break
    elapsed = time.time() - started
    return {
        "planned_images": planned,
        "processed_images_this_invocation": processed,
        "complete": processed >= planned,
        "limited": bool(limit),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_stream_item": elapsed / max(processed, 1),
        "per_domain": per_domain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-config",
        default=str(ROOT / "configs" / "new_method_kadid_development_v1.json"),
    )
    parser.add_argument(
        "--domain-config",
        default=str(ROOT / "configs" / "domains_iqa_kadid.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol_config).read_text(encoding="utf-8"))
    domain_payload = json.loads(Path(args.domain_config).read_text(encoding="utf-8"))
    assert_kadid_config(args.domain_config, domain_payload, protocol)
    device = resolve_device(args.device)
    domains = IQADomains(
        build_iqa_config(domain_payload),
        backbone=protocol["backbone"],
        img_size=protocol["image_size"],
        max_per_domain=protocol["max_train_per_domain"],
        sample_seed=protocol["sample_seed"],
        device=device,
        cache_dir=args.cache_dir,
    )
    before = len(list(Path(domains.cache_dir).glob("*.npy")))
    result = precompute(
        domains,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    after = len(list(Path(domains.cache_dir).glob("*.npy")))
    payload = {
        "protocol_id": protocol["protocol_id"],
        "evidence_role": "cache-precomputation-only",
        "device": device,
        "cache_dir": domains.cache_dir,
        "atomic_cache_writes": True,
        "protocol_sha256": _sha256(args.protocol_config),
        "domain_config_sha256": _sha256(args.domain_config),
        "cached_files_before": before,
        "cached_files_after": after,
        **result,
    }
    dump_result(Path(domains.cache_dir) / "precompute_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
