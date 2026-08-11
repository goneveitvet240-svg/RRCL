#!/usr/bin/env python3
"""Read-only FDST preflight and split freezer for C7-fdst-v3.

No model, feature extractor, prediction, or result file is read.  Formal
selection uses only official-train file identities and annotation
availability:

* a reviewed scene map assigns every eligible video directory to a scene;
* one video per scene is retained (most paired frames, then video id);
* six scenes are retained (most paired frames, then scene id);
* domain order is a salted hash of scene and video identity;
* each video is divided into fixed 10-frame temporal blocks, followed by a
  block-aligned contiguous ~60/20/20 fit/val/test split;
* the official FDST ``test_data`` partition is never enumerated or used.

Formal outputs are single-generation artifacts.  They must be committed on a
clean freeze commit before ``scripts/run_fdst_c7.py`` is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_fdst import C7_PROTOCOL, enumerate_video_pairs, frame_sort_key, read_points

ORDER_SALT = "order-v3"
BLOCK_SIZE_DEFAULT = 10
MIN_FRAMES_DEFAULT = 120
K_DEFAULT = 6
POINT_TOLERANCE_FRACTION = 0.01
CANONICAL_SPLITS_DIR = "configs/fdst_splits"
CANONICAL_CONFIG_OUT = "configs/domains_fdst.json"
SOURCE_RECORD = "docs/FDST_SOURCE_RECORD.json"
IDENTITY_AUDIT = "docs/FDST_SCENE_IDENTITY_AUDIT.json"
PROTOCOL_FORMAL = C7_PROTOCOL
PROTOCOL_TESTING = "C7-fdst-testing"


def sha(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene_map(path):
    """Load an explicitly reviewed ``video directory -> scene`` mapping."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"FDST IDENTITY GATE: scene map does not exist: {path}")
    payload = json.loads(path.read_text())
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, dict) or not videos:
        raise SystemExit(
            "FDST IDENTITY GATE: scene map must be JSON with a non-empty "
            "'videos' object mapping train_data-relative video directories "
            "to scene ids"
        )
    normalized = {}
    for video, scene in videos.items():
        video = Path(str(video)).as_posix().strip("/")
        scene = str(scene).strip()
        if not video or video == "." or not scene:
            raise SystemExit("FDST IDENTITY GATE: empty video or scene id in scene map")
        if scene.upper() == "REVIEW_REQUIRED":
            raise SystemExit(
                f"FDST IDENTITY GATE: {video} still has REVIEW_REQUIRED; "
                "scene identity must be reviewed before split selection"
            )
        if video in normalized:
            raise SystemExit(f"FDST IDENTITY GATE: duplicate video id {video}")
        normalized[video] = scene
    evidence = str(payload.get("evidence", "")).strip()
    if not evidence or "REVIEW_REQUIRED" in evidence.upper():
        raise SystemExit(
            "FDST IDENTITY GATE: scene map needs a reviewed evidence note"
        )
    return {
        "schema": "rrcl-fdst-scene-map-v2",
        "evidence": evidence,
        "videos": dict(sorted(normalized.items())),
    }


def order_hash(scene_id, video_id):
    value = f"fdst|{scene_id}|{video_id}|{ORDER_SALT}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixed_temporal_groups(ids, block_size=BLOCK_SIZE_DEFAULT):
    """Create non-overlapping temporal blocks without crossing a video."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not ids:
        return []
    parents = {Path(item).parent.as_posix() for item in ids}
    if len(parents) != 1:
        raise RuntimeError("fixed temporal groups may not cross video directories")
    ordered = sorted(ids, key=frame_sort_key)
    if ordered != list(ids):
        raise RuntimeError("frame ids are not in natural temporal order")
    groups = [ordered[index:index + block_size] for index in range(0, len(ordered), block_size)]
    if len(groups[-1]) < max(2, block_size // 2):
        raise RuntimeError(
            f"degenerate fixed-block structure: {len(ids)} frames yield "
            f"group sizes {[len(group) for group in groups]}"
        )
    return groups


def temporal_blocks_grouped(groups):
    """Block-aligned contiguous ~60/20/20 split."""
    total = sum(len(group) for group in groups)
    if len(groups) < 5:
        raise RuntimeError(f"need at least five temporal groups, got {len(groups)}")
    fit, val, test = [], [], []
    seen = 0
    for group in groups:
        if seen < 0.6 * total:
            fit.extend(group)
        elif seen < 0.8 * total:
            val.extend(group)
        else:
            test.extend(group)
        seen += len(group)
    if not fit or not val or not test:
        raise RuntimeError("degenerate block-aligned fit/val/test split")
    return fit, val, test


def audit_labels(root, domains):
    """Audit labels under the v3 finite-and-near-frame coordinate policy.

    FDST contains a small number of head points just outside the decoded image
    boundary.  V3 accepts finite coordinates no farther than one percent of
    the corresponding image dimension outside the frame, records every point
    that the frozen loader will clip, and rejects larger excursions.
    """
    import numpy as np
    from PIL import Image

    root = Path(root)
    checked = total_points = empty_frames = clipped_points = 0
    frames_with_clipped_points = hard_out_of_bounds = 0
    max_overflow_x = max_overflow_y = 0.0
    for domain in domains:
        for image_id, label_id in domain["pairs"]:
            try:
                points = read_points(root / label_id)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"FDST LABEL AUDIT: cannot parse {label_id}: {exc}"
                ) from None
            checked += 1
            if points.size == 0:
                empty_frames += 1
                continue
            if not bool(np.all(np.isfinite(points))):
                raise SystemExit(f"FDST LABEL AUDIT: non-finite point in {label_id}")
            with Image.open(root / image_id) as image:
                width, height = image.size
            tolerance_x = POINT_TOLERANCE_FRACTION * width
            tolerance_y = POINT_TOLERANCE_FRACTION * height
            hard_inside = (
                (points[:, 0] >= -tolerance_x)
                & (points[:, 0] <= (width - 1) + tolerance_x)
                & (points[:, 1] >= -tolerance_y)
                & (points[:, 1] <= (height - 1) + tolerance_y)
            )
            strict_inside = (
                (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
                & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
            )
            clipped = ~strict_inside
            total_points += int(points.shape[0])
            clipped_count = int(clipped.sum())
            clipped_points += clipped_count
            frames_with_clipped_points += int(clipped_count > 0)
            hard_out_of_bounds += int((~hard_inside).sum())
            if clipped_count:
                clipped_values = points[clipped]
                x_overflow = np.maximum(
                    np.maximum(-clipped_values[:, 0], 0),
                    np.maximum(clipped_values[:, 0] - (width - 1), 0),
                )
                y_overflow = np.maximum(
                    np.maximum(-clipped_values[:, 1], 0),
                    np.maximum(clipped_values[:, 1] - (height - 1), 0),
                )
                max_overflow_x = max(max_overflow_x, float(x_overflow.max()))
                max_overflow_y = max(max_overflow_y, float(y_overflow.max()))
    if hard_out_of_bounds:
        raise SystemExit(
            "FDST LABEL AUDIT: "
            f"{hard_out_of_bounds} points exceed the frozen one-percent "
            "near-frame tolerance"
        )
    return {
        "scope": "all paired train_data frames",
        "coordinate_policy": "finite; clip to frame if within 1% per axis; reject otherwise",
        "tolerance_fraction_per_axis": POINT_TOLERANCE_FRACTION,
        "frames_checked": checked,
        "total_points": total_points,
        "empty_frames": empty_frames,
        "frames_with_clipped_points": frames_with_clipped_points,
        "clipped_points": clipped_points,
        "max_overflow_x_px": max_overflow_x,
        "max_overflow_y_px": max_overflow_y,
        "hard_out_of_bounds_points": hard_out_of_bounds,
    }


def choose_domains(videos, scene_map, min_frames, k):
    disk_ids = set(videos)
    map_ids = set(scene_map["videos"])
    missing = sorted(disk_ids - map_ids)
    extra = sorted(map_ids - disk_ids)
    if missing or extra:
        raise SystemExit(
            "FDST IDENTITY GATE: scene map and paired train videos differ; "
            f"unmapped_on_disk={missing[:10]}, absent_from_disk={extra[:10]}"
        )
    by_scene = defaultdict(list)
    for video_id, pairs in videos.items():
        if len(pairs) >= min_frames:
            by_scene[scene_map["videos"][video_id]].append((video_id, pairs))
    representatives = []
    for scene_id, candidates in by_scene.items():
        video_id, pairs = sorted(candidates, key=lambda row: (-len(row[1]), row[0]))[0]
        representatives.append({"scene_id": scene_id, "video_id": video_id, "pairs": pairs})
    if len(representatives) < k:
        raise SystemExit(
            f"FDST PROTOCOL FAILURE: only {len(representatives)} distinct scenes "
            f"have a video with >= {min_frames} paired frames; need {k}"
        )
    selected = sorted(
        representatives,
        key=lambda row: (-len(row["pairs"]), row["scene_id"], row["video_id"]),
    )[:k]
    return sorted(
        selected, key=lambda row: order_hash(row["scene_id"], row["video_id"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="extracted FDST dataset root")
    parser.add_argument("--scene-map", required=True, help="reviewed video-to-scene JSON")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES_DEFAULT)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE_DEFAULT)
    parser.add_argument("--splits-dir", default=CANONICAL_SPLITS_DIR)
    parser.add_argument("--config-out", default=CANONICAL_CONFIG_OUT)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--skip-label-audit", action="store_true")
    args = parser.parse_args()

    defaults = (MIN_FRAMES_DEFAULT, K_DEFAULT, BLOCK_SIZE_DEFAULT)
    non_default = (args.min_frames, args.k, args.block_size) != defaults
    if non_default and not args.testing:
        raise SystemExit("non-default FDST protocol parameters require --testing")
    if args.skip_label_audit and not args.testing:
        raise SystemExit("--skip-label-audit is testing-only")
    canonical = args.splits_dir == CANONICAL_SPLITS_DIR or args.config_out == CANONICAL_CONFIG_OUT
    if args.testing and canonical:
        raise SystemExit("--testing must use non-canonical output paths")
    protocol = PROTOCOL_TESTING if args.testing else PROTOCOL_FORMAL

    repo = Path(__file__).resolve().parents[1]

    def rel(path):
        path = Path(path)
        try:
            return path.relative_to(repo).as_posix()
        except ValueError:
            return str(path)

    source_record = repo / SOURCE_RECORD
    if not source_record.is_file():
        raise SystemExit(f"FDST SOURCE GATE: missing {SOURCE_RECORD}")
    source = json.loads(source_record.read_text())
    if source.get("dataset") != "FDST" or source.get("raw_data_redistribution") is not False:
        raise SystemExit("FDST SOURCE GATE: source record is incomplete")
    if source.get("scene_identity_audit") != IDENTITY_AUDIT:
        raise SystemExit("FDST SOURCE GATE: scene identity audit pointer mismatch")
    identity_audit_path = repo / IDENTITY_AUDIT
    if not identity_audit_path.is_file():
        raise SystemExit(f"FDST IDENTITY GATE: missing {IDENTITY_AUDIT}")
    identity_audit = json.loads(identity_audit_path.read_text())
    if (
        identity_audit.get("protocol") != PROTOCOL_FORMAL
        or identity_audit.get("coverage") != "exact"
        or identity_audit.get("raw_images_committed") is not False
    ):
        raise SystemExit("FDST IDENTITY GATE: identity audit is incomplete")

    splits_dir = repo / args.splits_dir
    config_out = repo / args.config_out
    manifest_path = splits_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} already exists; regenerating frozen splits "
            "requires a new protocol version"
        )

    try:
        videos = enumerate_video_pairs(args.root)
    except RuntimeError as exc:
        root = Path(args.root)
        found = ", ".join(sorted(path.name for path in root.iterdir())[:20]) if root.is_dir() else "missing"
        raise SystemExit(
            "FDST DATA GATE: expected extracted train_data/<video>/ with "
            f"same-stem JPG/JSON pairs under {root}; found: {found}; detail: {exc}"
        ) from None
    if not videos:
        raise SystemExit(
            "FDST DATA GATE: no same-directory, same-stem image/JSON pairs "
            "found under train_data; official test_data is intentionally ignored"
        )
    ordered_video_ids = sorted(videos, key=frame_sort_key)
    if not args.testing and (
        identity_audit.get("train_video_count") != len(ordered_video_ids)
        or identity_audit.get("train_video_ids_sha256") != sha(ordered_video_ids)
    ):
        raise SystemExit(
            "FDST IDENTITY GATE: train video ids differ from the reviewed identity audit"
        )
    scene_map = load_scene_map(args.scene_map)
    selected = choose_domains(videos, scene_map, args.min_frames, args.k)

    print(f"paired train videos={len(videos)}; mapped scenes={len(set(scene_map['videos'].values()))}")
    for row in selected:
        print(f"  scene={row['scene_id']} video={row['video_id']} pairs={len(row['pairs'])}")

    audit_domains = [
        {"video_id": video_id, "pairs": pairs}
        for video_id, pairs in sorted(videos.items())
    ]
    label_audit = (
        {"skipped": True}
        if args.skip_label_audit
        else audit_labels(args.root, audit_domains)
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    normalized_map_path = splits_dir / "scene_map.json"
    normalized_map_path.write_text(json.dumps(scene_map, indent=2) + "\n")

    manifest_domains = []
    config_domains = []
    for index, row in enumerate(selected, start=1):
        ids = [pair[0] for pair in row["pairs"]]
        groups = fixed_temporal_groups(ids, args.block_size)
        fit, val, test = temporal_blocks_grouped(groups)
        files = {}
        safe_name = f"d{index:02d}"
        for role, split_ids in (("fit", fit), ("val", val), ("test", test)):
            path = splits_dir / f"{safe_name}_{role}.txt"
            path.write_text("\n".join(split_ids) + "\n")
            files[role] = {"file": rel(path), "count": len(split_ids), "ids_sha256": sha(split_ids)}
        manifest_domains.append({
            "domain_index": index - 1,
            "scene_id": row["scene_id"],
            "video_id": row["video_id"],
            "n_temporal_groups": len(groups),
            **files,
        })
        config_domains.append({
            "name": f"FDST-{row['scene_id']}-{row['video_id']}",
            "kind": "fdst_video",
            "root": str(Path(args.root).resolve()),
            "scene_id": row["scene_id"],
            "video_id": row["video_id"],
            "sample_key": f"FDST-{row['scene_id']}-{row['video_id']}",
            "fit_include_file": files["fit"]["file"],
            "val_include_file": files["val"]["file"],
            "test_include_file": files["test"]["file"],
        })

    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps({"domains": config_domains}, indent=2) + "\n")
    manifest = {
        "protocol": protocol,
        "root": str(Path(args.root).resolve()),
        "rules": {
            "eligible_partition": "train_data only",
            "official_test_partition": "excluded",
            "domain_unit": "one video; six distinct scenes",
            "target_scope": "full_frame",
            "min_frames": args.min_frames,
            "k": args.k,
            "block_size": args.block_size,
            "temporal_blocks": "~60/20/20 contiguous, fixed-block-aligned",
            "order_salt": ORDER_SALT,
            "point_coordinate_policy": "finite; clip to frame if within 1% per axis; reject otherwise",
            "point_tolerance_fraction_per_axis": POINT_TOLERANCE_FRACTION,
        },
        "domains": manifest_domains,
        "scene_map": rel(normalized_map_path),
        "scene_map_sha256": file_sha(normalized_map_path),
        "source_record": SOURCE_RECORD,
        "source_record_sha256": file_sha(source_record),
        "identity_audit": IDENTITY_AUDIT,
        "identity_audit_sha256": file_sha(identity_audit_path),
        "label_audit": label_audit,
        "config": rel(config_out),
        "config_sha256": file_sha(config_out),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"protocol={protocol}")
    print(f"wrote {rel(config_out)} and {rel(manifest_path)}")
    print("Commit these files before the single FDST model run.")


if __name__ == "__main__":
    main()
