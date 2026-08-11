"""FDST video-domain loader for the C7 unseen confirmation.

Protocol: ``docs/C7_FDST_PROTOCOL_V3.md``.  One selected video is one
continual-learning domain, and the six selected videos must come from six
different FDST scenes.  Only ``train_data`` is eligible; the official FDST
test partition remains untouched.  Targets are full-frame point counts
because FDST does not publish WorldExpo-style ROI polygons.

The preflight materializes dataset-root-relative image ids.  Each image must
have a same-directory, same-stem JSON annotation.  FDST's VIA-style
``regions`` schema is supported, together with a small set of transparent
point-list fallbacks used by public mirrors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cache_io import atomic_save_array, load_array_cache

C7_PROTOCOL = "C7-fdst-v3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class FDSTVideoSpec:
    name: str
    root: str
    scene_id: str
    video_id: str
    sample_key: str
    fit_include_file: str
    val_include_file: str
    test_include_file: str


def build_from_config(cfg):
    domains = []
    for entry in cfg["domains"]:
        if entry.get("kind") != "fdst_video":
            raise ValueError(f"not an fdst_video entry: {entry.get('name')}")
        domains.append(
            FDSTVideoSpec(
                name=entry["name"],
                root=entry["root"],
                scene_id=str(entry["scene_id"]),
                video_id=str(entry["video_id"]),
                sample_key=entry.get(
                    "sample_key", f"FDST-{entry['scene_id']}-{entry['video_id']}"
                ),
                fit_include_file=entry["fit_include_file"],
                val_include_file=entry["val_include_file"],
                test_include_file=entry["test_include_file"],
            )
        )
    return domains


def train_root(root):
    root = Path(root)
    for name in ("train_data", "train"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"no FDST train_data directory found under {root}")


def frame_sort_key(path):
    """Natural temporal order, with the path string as a stable tie break."""
    path = Path(path)
    numbers = tuple(int(chunk) for chunk in re.findall(r"\d+", path.stem))
    return numbers, path.as_posix()


def enumerate_video_pairs(root):
    """Return ``video_id -> [(image_rel, label_rel), ...]`` from train_data.

    A video id is the image's parent directory relative to ``train_data``.
    Pairing is deliberately strict: JSON and image must be side by side with
    the same stem, preventing ambiguous cross-video matching of names such as
    ``001.jpg``.
    """
    root = Path(root).resolve()
    training = train_root(root)
    videos = {}
    frames = sorted(
        (path for path in training.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES),
        key=frame_sort_key,
    )
    for image in frames:
        label = image.with_suffix(".json")
        if not label.is_file():
            continue
        video_id = image.parent.relative_to(training).as_posix()
        if video_id == ".":
            raise RuntimeError(
                "FDST frames are directly under train_data; video identity "
                "cannot be recovered from directories"
            )
        pair = (
            image.relative_to(root).as_posix(),
            label.relative_to(root).as_posix(),
        )
        videos.setdefault(video_id, []).append(pair)
    for pairs in videos.values():
        pairs.sort(key=lambda pair: frame_sort_key(pair[0]))
    return videos


def _xy_from_region(region):
    if not isinstance(region, dict):
        return None
    shape = region.get("shape_attributes", region)
    if not isinstance(shape, dict):
        return None
    for x_key, y_key in (("x", "y"), ("cx", "cy")):
        if x_key in shape and y_key in shape:
            return float(shape[x_key]), float(shape[y_key])
    return None


def read_points(path):
    """Parse FDST point annotations into an ``(n, 2)`` float array."""
    data = json.loads(Path(path).read_text())
    points = []
    if isinstance(data, dict) and "regions" not in data:
        nested = [value for value in data.values()
                  if isinstance(value, dict) and "regions" in value]
        if len(nested) == 1:
            data = nested[0]
    if isinstance(data, dict) and "regions" in data:
        regions = data["regions"]
        regions = regions.values() if isinstance(regions, dict) else regions
        regions = list(regions)
        for region in regions:
            point = _xy_from_region(region)
            if point is None:
                raise RuntimeError(f"unsupported FDST region schema in {path}")
            points.append(point)
    else:
        if isinstance(data, dict):
            if "points" in data:
                raw = data["points"]
            elif "annPoints" in data:
                raw = data["annPoints"]
            else:
                raise RuntimeError(f"unsupported FDST annotation schema in {path}")
        elif isinstance(data, list):
            raw = data
        else:
            raise RuntimeError(f"unsupported FDST annotation schema in {path}")
        for item in raw or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
            else:
                point = _xy_from_region(item)
                if point is None:
                    raise RuntimeError(f"unsupported FDST point schema in {path}")
                points.append(point)
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _read_ids(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _points_to_patch_counts(points, image_path, img_size, n_patches):
    grid = int(round(np.sqrt(n_patches)))
    if grid * grid != n_patches:
        raise ValueError(f"patch count {n_patches} is not a square grid")
    counts = np.zeros((grid, grid), dtype=np.float64)
    if points.size == 0:
        return counts.reshape(-1)
    with Image.open(image_path) as image:
        width, height = image.size
    xs = np.clip(points[:, 0] * (float(img_size) / max(width, 1)), 0, img_size - 1e-6)
    ys = np.clip(points[:, 1] * (float(img_size) / max(height, 1)), 0, img_size - 1e-6)
    gx = np.clip((xs / img_size * grid).astype(int), 0, grid - 1)
    gy = np.clip((ys / img_size * grid).astype(int), 0, grid - 1)
    np.add.at(counts, (gy, gx), 1.0)
    return counts.reshape(-1)


class FDSTDomains:
    """Frozen-protocol FDST video domains (C7-fdst-v3)."""

    def __init__(self, domains, backbone="dinov2_vitb14", img_size=518):
        from features import DinoFeatureExtractor

        self.domains = list(domains)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.cache_dir = os.path.join(".feature_cache_fdst", f"{backbone}_img{img_size}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(backbone=backbone, img_size=img_size)

    def n_domains(self):
        return len(self.domains)

    def _split_ids(self, spec, split):
        if split == "train":
            return _read_ids(spec.fit_include_file) + _read_ids(spec.val_include_file)
        if split == "test":
            return _read_ids(spec.test_include_file)
        raise ValueError(f"unknown split {split!r}")

    def frozen_val_ids(self, d):
        return list(_read_ids(self.domains[d].val_include_file))

    def _selected_pairs(self, split, d):
        spec = self.domains[d]
        ids = self._split_ids(spec, split)
        pairs = []
        for image_id in ids:
            image = Path(spec.root) / image_id
            label = image.with_suffix(".json")
            if not image.is_file() or not label.is_file():
                raise RuntimeError(
                    f"{spec.name}: frozen pair missing for {image_id}; dataset "
                    "layout drifted (do not regenerate splits)"
                )
            pairs.append((str(image), str(label), image_id))
        return pairs

    def stream(self, split, d, with_ids=False):
        for frame, label, image_id in self._selected_pairs(split, d):
            X = self._features(frame)
            points = read_points(label)
            Y = _points_to_patch_counts(points, frame, self.img_size, X.shape[0])
            if with_ids:
                yield X, Y.reshape(-1, 1), X.shape[0], image_id
            else:
                yield X, Y.reshape(-1, 1), X.shape[0]

    def data_manifest(self):
        records = []
        for index, spec in enumerate(self.domains):
            record = {
                "name": spec.name,
                "sample_key": spec.sample_key,
                "kind": "fdst_video",
                "scene_id": spec.scene_id,
                "video_id": spec.video_id,
                "target_scope": "full_frame",
            }
            for split in ("train", "test"):
                identifiers = [
                    f"{image_id}\t{Path(image_id).with_suffix('.json').as_posix()}"
                    for _, _, image_id in self._selected_pairs(split, index)
                ]
                record[split] = {
                    "count": len(identifiers),
                    "ids_sha256": hashlib.sha256(
                        "\n".join(identifiers).encode("utf-8")
                    ).hexdigest(),
                }
            records.append(record)
        return {"protocol": C7_PROTOCOL, "domains": records}

    def _features(self, image_path):
        from features import cache_key

        key = cache_key(image_path, self.backbone, self.img_size)
        path = os.path.join(self.cache_dir, key)
        cached = load_array_cache(path)
        if cached is not None:
            return cached
        image = Image.open(image_path).convert("RGB")
        X = self.extractor(image)
        atomic_save_array(path, X)
        return X
