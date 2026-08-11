#!/usr/bin/env python3
"""Inspect FDST train identities and emit a scene-map review template.

This utility is outcome-blind: it reads only paths and pair availability,
not labels, pixels, features, or predictions.  It never infers that two
videos share a scene; the generated placeholders require human review using
official dataset metadata or unmistakable directory semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_fdst import enumerate_video_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="configs/fdst_scene_map_candidate.json")
    args = parser.parse_args()

    try:
        videos = enumerate_video_pairs(args.root)
    except RuntimeError as exc:
        raise SystemExit(f"FDST DATA GATE: {exc}") from None
    if not videos:
        raise SystemExit("FDST DATA GATE: no paired train_data image/JSON files found")
    print(f"paired train videos: {len(videos)}")
    for video_id, pairs in sorted(videos.items()):
        print(
            f"{video_id}\tframes={len(pairs)}\t"
            f"first={Path(pairs[0][0]).name}\tlast={Path(pairs[-1][0]).name}"
        )
    output = Path(__file__).resolve().parents[1] / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "rrcl-fdst-scene-map-v2",
        "evidence": "REVIEW_REQUIRED: cite official metadata or directory identity",
        "videos": {video_id: "REVIEW_REQUIRED" for video_id in sorted(videos)},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote review template -> {output}")
    print("Replace every REVIEW_REQUIRED value before formal preflight.")


if __name__ == "__main__":
    main()
