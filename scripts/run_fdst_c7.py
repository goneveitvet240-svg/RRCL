#!/usr/bin/env python3
"""Single-shot FDST opportunity--selection confirmation (C7-fdst-v3)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_fdst import C7_PROTOCOL, FDSTDomains, build_from_config
from metrics import forgetting_matrix_stats
from result_io import dump_result
from run_provenance import with_provenance
from scripts import fdst_analysis_core as core
from scripts.preflight_fdst import fixed_temporal_groups, sha as ids_sha

FORMAL_PROTOCOL = C7_PROTOCOL
APPENDIX_PATH = "docs/C7_FDST_APPENDIX_A.json"
CONFIG_PATH = "configs/domains_fdst.json"
SPLITS_DIR = "configs/fdst_splits"
OUT_DIR = "runs_real/fdst_c7"
RESULT_NAME = "fdst_c7.json"

# These are aliases, not new choices: Appendix A verifies exact equality.
BACKBONE = core.BACKBONE
IMG_SIZE = core.IMG_SIZE
LAM = core.LAM
ALPHA = core.ALPHA
PATCH_TARGET = core.PATCH_TARGET
METHOD_IDS = core.METHOD_IDS
ORACLE_GRID = core.ORACLE_GRID
MIN_IMPROVEMENT = core.MIN_IMPROVEMENT


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_appendix(repo):
    path = repo / APPENDIX_PATH
    if not path.is_file():
        raise SystemExit(f"FDST C7 GATE: {APPENDIX_PATH} missing")
    appendix = json.loads(path.read_text())
    if appendix.get("protocol") != FORMAL_PROTOCOL:
        raise SystemExit("FDST C7 GATE: Appendix A protocol mismatch")
    if tuple(appendix.get("method_ids", ())) != METHOD_IDS:
        raise SystemExit("FDST C7 GATE: Appendix A method roster mismatch")
    if [float(value) for value in appendix.get("oracle_grid", [])] != ORACLE_GRID:
        raise SystemExit("FDST C7 GATE: Appendix A oracle grid mismatch")
    expected = {
        "shrinkage_gamma": "bb2e3afe969670f74eee69a4952dbaab8618a8e5",
        "fstar_pred": "609399ac2039c9f280b61d54206da439ff5a53bc",
    }
    if appendix.get("development_source_commits") != expected:
        raise SystemExit("FDST C7 GATE: development source commits mismatch")
    if appendix.get("analysis_code_source") != "scripts/run_worldexpo_c7.py@18b21be":
        raise SystemExit("FDST C7 GATE: shared analysis source is not frozen")
    return appendix, file_sha(path)


def check_manifest(repo, splits_dir=SPLITS_DIR, config_path=CONFIG_PATH):
    manifest_path = repo / splits_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"FDST C7 GATE: {manifest_path} missing; run preflight")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != FORMAL_PROTOCOL:
        raise SystemExit(
            f"FDST C7 GATE: manifest protocol {manifest.get('protocol')!r} "
            f"is not {FORMAL_PROTOCOL}"
        )
    rules = manifest.get("rules", {})
    expected_rules = {
        "eligible_partition": "train_data only",
        "official_test_partition": "excluded",
        "domain_unit": "one video; six distinct scenes",
        "target_scope": "full_frame",
        "min_frames": 120,
        "k": 6,
        "block_size": 10,
        "temporal_blocks": "~60/20/20 contiguous, fixed-block-aligned",
        "order_salt": "order-v3",
        "point_coordinate_policy": "finite; clip to frame if within 1% per axis; reject otherwise",
        "point_tolerance_fraction_per_axis": 0.01,
    }
    if rules != expected_rules:
        raise SystemExit("FDST C7 GATE: manifest rules differ from protocol v3")
    label_audit = manifest.get("label_audit", {})
    expected_policy = expected_rules["point_coordinate_policy"]
    if (
        label_audit.get("scope") != "all paired train_data frames"
        or label_audit.get("coordinate_policy") != expected_policy
        or label_audit.get("tolerance_fraction_per_axis") != 0.01
        or label_audit.get("hard_out_of_bounds_points") != 0
        or label_audit.get("frames_checked") != 9000
        or not isinstance(label_audit.get("clipped_points"), int)
        or label_audit.get("clipped_points", -1) < 0
    ):
        raise SystemExit("FDST C7 GATE: v3 label audit is missing or invalid")
    config = repo / config_path
    if not config.is_file() or file_sha(config) != manifest.get("config_sha256"):
        raise SystemExit("FDST C7 GATE: config is missing or differs from manifest")
    domains = manifest.get("domains", [])
    if len(domains) != 6 or len({row.get("scene_id") for row in domains}) != 6:
        raise SystemExit("FDST C7 GATE: need six domains from six distinct scenes")
    for domain in domains:
        for role in ("fit", "val", "test"):
            entry = domain.get(role, {})
            path = repo / entry.get("file", "")
            if not path.is_file():
                raise SystemExit(f"FDST C7 GATE: split list missing: {path}")
            ids = [line for line in path.read_text().splitlines() if line]
            if len(ids) != entry.get("count") or ids_sha(ids) != entry.get("ids_sha256"):
                raise SystemExit(f"FDST C7 GATE: split list hash mismatch: {path.name}")
            fixed_temporal_groups(ids, 10)
    for path_key, hash_key in (
        ("scene_map", "scene_map_sha256"),
        ("source_record", "source_record_sha256"),
        ("identity_audit", "identity_audit_sha256"),
    ):
        path = repo / manifest.get(path_key, "")
        if not path.is_file() or file_sha(path) != manifest.get(hash_key):
            raise SystemExit(f"FDST C7 GATE: {path_key} missing or hash mismatch")
    return manifest, file_sha(manifest_path)


def enforce_single_shot(repo):
    previous = sorted((repo / "runs_real").rglob(RESULT_NAME)) if (repo / "runs_real").exists() else []
    if previous:
        raise SystemExit(
            f"FDST C7 GATE: result already exists at {previous[0]}; "
            "the formal confirmation is single-shot"
        )


def acquire_single_shot_lock(repo, commit=None):
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "attempt.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"FDST C7 GATE: {lock} already exists; an attempt has started and "
            "cannot be silently repeated"
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({
            "commit": commit,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pid": os.getpid(),
        }, handle)
        handle.write("\n")
    return lock


def require_fdst_groups(ids, where):
    try:
        groups = fixed_temporal_groups(ids, 10)
    except RuntimeError as exc:
        raise SystemExit(f"FDST PROTOCOL HALT in {where}: {exc}") from None
    if len(groups) < 2:
        raise SystemExit(f"FDST PROTOCOL HALT in {where}: fewer than two blocks")
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Frozen single-shot FDST C7 confirmation; no parameters"
    )
    parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    appendix, appendix_sha = load_appendix(repo)
    core.ensure_clean_worktree(allow_dirty=False)
    manifest, manifest_sha = check_manifest(repo)
    enforce_single_shot(repo)
    from run_provenance import _git_output
    acquire_single_shot_lock(repo, commit=_git_output(["rev-parse", "HEAD"]))

    # The shared frozen functions resolve this global at call time.
    core.require_detected_groups = require_fdst_groups

    config = json.loads((repo / CONFIG_PATH).read_text())
    domains = FDSTDomains(build_from_config(config), backbone=BACKBONE, img_size=IMG_SIZE)
    total_domains = domains.n_domains()
    names = [spec.name for spec in domains.domains]
    started = time.time()

    scales = [core.domain_scale(domains, index, "mean") for index in range(total_domains)]
    fixed_runs = {
        float(factor): core.run_fixed_factor(domains, factor, scales)
        for factor in ORACLE_GRID
    }
    f1_run = fixed_runs[1.0]
    rel_f1 = f1_run["balanced_rel_mae"]
    f1_errors = f1_run["per_image"]
    minimum = min(run["balanced_rel_mae"] for run in fixed_runs.values())
    tolerance = 1e-12 * max(1.0, abs(minimum))
    best_factor = max(
        factor for factor, run in fixed_runs.items()
        if run["balanced_rel_mae"] <= minimum + tolerance
    )
    oracle_best = {
        "factor": best_factor,
        "rel_MAE": fixed_runs[best_factor]["balanced_rel_mae"],
    }
    oracle_curve = [
        {"factor": float(factor), "rel_MAE": fixed_runs[float(factor)]["balanced_rel_mae"]}
        for factor in ORACLE_GRID
    ]
    oracle_bootstrap = core.paired_oracle_block_bootstrap(
        {factor: run["per_image"] for factor, run in fixed_runs.items()}, f1_errors
    )
    oracle_improvement = (rel_f1 - oracle_best["rel_MAE"]) / max(rel_f1, 1e-12)
    opportunity_present = bool(
        oracle_improvement >= MIN_IMPROVEMENT
        and oracle_bootstrap["ci95_lower_rel"] > 0
    )

    collected = core.collect_weighting_statistics(domains)
    records, precisions, _ = collected
    selected_gamma, gamma_selections = core.select_shrinkage_gamma(records, precisions)
    boundary_gammas = [row["selected"]["gamma"] for row in gamma_selections]
    shrinkage = core.run_weighted_method(domains, boundary_gammas, collected)
    precision = core.run_weighted_method(domains, 1.0, collected)
    fstar_factor, fstar_curve = core.predict_fixed_factor(records, precisions)
    fstar_run = fixed_runs[float(fstar_factor)]

    method_inputs = {
        "shrinkage_gamma": {
            "rel": shrinkage[0], "M_abs": shrinkage[1], "M_rel": shrinkage[2],
            "errors": shrinkage[3],
            "details": {"selected_gamma": selected_gamma,
                        "boundary_gammas": boundary_gammas,
                        "boundary_selections": gamma_selections,
                        "residual_precisions": precisions},
        },
        "fstar_pred": {
            "rel": fstar_run["balanced_rel_mae"], "M_abs": fstar_run["M_abs"],
            "M_rel": fstar_run["M_rel"], "errors": fstar_run["per_image"],
            "details": {"predicted_factor": fstar_factor,
                        "train_objective_curve": fstar_curve},
        },
        "precision_weighting": {
            "rel": precision[0], "M_abs": precision[1], "M_rel": precision[2],
            "errors": precision[3],
            "details": {"gamma": 1.0, "residual_precisions": precisions},
        },
    }
    method_results = {}
    per_domain_f1 = np.asarray(f1_run["M_rel"])[total_domains - 1, :total_domains]
    for method, record in method_inputs.items():
        bootstrap = core.paired_block_bootstrap(record["errors"], f1_errors)
        improvement = (rel_f1 - record["rel"]) / max(rel_f1, 1e-12)
        harms = core.relative_domain_harms(
            np.asarray(record["M_rel"])[total_domains - 1, :total_domains], per_domain_f1
        )
        verdict = core.compute_verdict(
            improvement, bootstrap, harms, oracle_best["rel_MAE"], rel_f1
        )
        method_results[method] = {
            "balanced_rel_mae": record["rel"],
            "M_abs": np.asarray(record["M_abs"]).tolist(),
            "M_rel": np.asarray(record["M_rel"]).tolist(),
            "nBwT": forgetting_matrix_stats(np.asarray(record["M_abs"]))["nBwT"],
            "per_domain_metrics": dict(zip(names, map(core.domain_metrics, record["errors"]))),
            "details": record["details"],
            "verdict": verdict,
        }
    useful = [name for name, row in method_results.items() if row["verdict"]["positive"]]
    classification = {
        "opportunity_present": opportunity_present,
        "gap_confirmed": bool(opportunity_present and not useful),
        "construction_candidate": bool(opportunity_present and useful),
        "opportunity_absent": not opportunity_present,
        "useful_train_only_methods": useful,
    }
    payload = {
        "protocol": FORMAL_PROTOCOL,
        "frozen_constants": {
            "backbone": BACKBONE, "img_size": IMG_SIZE, "lam": LAM,
            "alpha": ALPHA, "patch_target": PATCH_TARGET,
            "config": CONFIG_PATH, "splits_dir": SPLITS_DIR,
            "block_size": 10, "target_scope": "full_frame",
        },
        "appendix_a": appendix,
        "appendix_a_sha256": appendix_sha,
        "manifest_sha256": manifest_sha,
        "config_sha256": manifest.get("config_sha256"),
        "domain_names": names,
        "scales": scales,
        "data_manifest": domains.data_manifest(),
        "rel_f1": rel_f1,
        "f1_M_abs": f1_run["M_abs"].tolist(),
        "f1_M_rel": f1_run["M_rel"].tolist(),
        "f1_nBwT": forgetting_matrix_stats(f1_run["M_abs"])["nBwT"],
        "f1_per_domain_metrics": dict(zip(names, map(core.domain_metrics, f1_errors))),
        "oracle_curve": oracle_curve,
        "oracle_best": oracle_best,
        "oracle_improvement_rel": oracle_improvement,
        "oracle_selection_adjusted_bootstrap": oracle_bootstrap,
        "methods": method_results,
        "classification": classification,
        "runtime_seconds": time.time() - started,
    }
    out = repo / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    result = out / RESULT_NAME
    dump_result(result, with_provenance(payload, str(repo / CONFIG_PATH), {}))
    print(
        f"f=1={rel_f1:.5f} | oracle={oracle_best['rel_MAE']:.5f} "
        f"({100 * oracle_improvement:+.2f}%) | opportunity={opportunity_present} "
        f"gap={classification['gap_confirmed']} "
        f"construction={classification['construction_candidate']}"
    )
    print(f"saved -> {result}")


if __name__ == "__main__":
    main()
