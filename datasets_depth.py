"""
Depth estimation domains for continual patch-level regression.
Image -> per-patch mean depth (meters).

Same patch-grid pipeline as crowd counting; reuses run_adaptive_f.py unchanged.

Supported formats
  nyu   : NYU Depth V2 individual PNG pairs
          root/rgb/*.png  +  root/depth/*.png  (16-bit, mm -> /1000 -> m)
          optional split file: one filename stem per line (no extension)
  make3d: root/imgs/*.jpg  +  root/depth_sph_corr/*.mat  (variable shape)

Config JSON example:
{
  "domains": [
    {"name": "NYU-Bedroom", "kind": "nyu",
     "root": "/data/nyu_depth_v2",
     "split_file": "splits/bedroom_train.txt",
     "test_split_file": "splits/bedroom_test.txt"},
    {"name": "NYU-Kitchen", "kind": "nyu",
     "root": "/data/nyu_depth_v2",
     "split_file": "splits/kitchen_train.txt",
     "test_split_file": "splits/kitchen_test.txt"}
  ]
}
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cache_io import atomic_save_array, load_array_cache
from features import DinoFeatureExtractor, cache_key
from data_manifest import identifiers_manifest


@dataclass
class DepthSpec:
    name: str
    kind: str       # "nyu" | "make3d" | "diml"
    root: str
    split_file: str = ""       # text file with train stems (no ext)
    test_split_file: str = ""  # text file with test stems; if empty, use all not in train
    max_depth: float = 10.0    # clip depth above this (meters)
    train_split: str = "train"
    test_split: str = "test"


def build_depth_config(cfg):
    out = []
    for d in cfg["domains"]:
        out.append(DepthSpec(
            name=d["name"], kind=d.get("kind", "nyu"), root=d["root"],
            split_file=d.get("split_file", ""),
            test_split_file=d.get("test_split_file", ""),
            max_depth=float(d.get("max_depth", 10.0)),
            train_split=d.get("train_split", "train"),
            test_split=d.get("test_split", "test"),
        ))
    return out


def _nyu_pairs(spec: DepthSpec, split: str):
    rgb_dir   = os.path.join(spec.root, "rgb")
    depth_dir = os.path.join(spec.root, "depth")
    if not os.path.isdir(rgb_dir):   # flat layout: root/*.jpg + root/*_depth.png
        rgb_dir = spec.root; depth_dir = spec.root
    all_rgb = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) +
                     glob.glob(os.path.join(rgb_dir, "*.jpg")))
    # filter by split file if provided
    split_file = spec.split_file if split == "train" else spec.test_split_file
    if split_file and os.path.exists(split_file):
        allowed = set(l.strip() for l in open(split_file) if l.strip())
        all_rgb = [p for p in all_rgb
                   if os.path.splitext(os.path.basename(p))[0] in allowed]
    pairs = []
    for rgb in all_rgb:
        stem = os.path.splitext(os.path.basename(rgb))[0]
        depth = os.path.join(depth_dir, stem + ".png")
        if not os.path.exists(depth):
            depth = os.path.join(depth_dir, stem + "_depth.png")
        if os.path.exists(depth):
            pairs.append((rgb, depth))
    if not pairs:
        raise RuntimeError(f"No NYU pairs found under {spec.root} for split={split}")
    return pairs


def _load_depth_png(path, max_depth):
    """Load 16-bit depth PNG (mm) -> float32 array (m), clipped."""
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float32)
    # NYU: values in mm; some datasets store in 0.001m units
    if arr.max() > 1000:
        arr /= 1000.0        # mm -> m
    arr = np.clip(arr, 0.0, max_depth)
    return arr


def _depth_to_patches(depth_arr, img_size, n_patches, min_valid_fraction=0.5):
    """Resize depth and compute valid-only patch means.

    Invalid zero-depth pixels must not be interpolated into metric targets.
    Resize depth mass and validity mass separately, then divide them.  Patches
    with insufficient valid support are excluded from both fitting and
    evaluation.
    """
    g = int(round(np.sqrt(n_patches)))
    if g * g != n_patches:
        raise ValueError(f"n_patches={n_patches} is not square")
    patch_size = img_size // g
    valid = np.isfinite(depth_arr) & (depth_arr > 1e-3)
    weighted_depth = np.where(valid, depth_arr, 0.0).astype(np.float32)
    valid_float = valid.astype(np.float32)
    resized_depth_mass = np.asarray(
        Image.fromarray(weighted_depth).resize((img_size, img_size), Image.BILINEAR),
        dtype=np.float64,
    )
    resized_valid_mass = np.asarray(
        Image.fromarray(valid_float).resize((img_size, img_size), Image.BILINEAR),
        dtype=np.float64,
    )
    depth_sum = resized_depth_mass.reshape(
        g, patch_size, g, patch_size
    ).sum(axis=(1, 3))
    valid_sum = resized_valid_mass.reshape(
        g, patch_size, g, patch_size
    ).sum(axis=(1, 3))
    patch_area = float(patch_size * patch_size)
    patch_valid = valid_sum >= float(min_valid_fraction) * patch_area
    patch_depth = np.divide(
        depth_sum,
        valid_sum,
        out=np.zeros_like(depth_sum),
        where=valid_sum > 1e-12,
    )
    return patch_depth.reshape(-1), patch_valid.reshape(-1)


class DepthDomains:
    """Domain-incremental depth estimation; compatible with run_adaptive_f.py."""

    def __init__(self, specs, backbone="dinov2_vitb14", img_size=518,
                 max_per_domain=None, sample_seed=42):
        self.specs = list(specs)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.max_per_domain = max_per_domain
        self.sample_seed = int(sample_seed)
        self.cache_dir = os.path.join(".feature_cache_depth", f"{backbone}_img{img_size}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(backbone=backbone, img_size=img_size)

    def n_domains(self):
        return len(self.specs)

    def data_manifest(self):
        records = []
        for domain_index, spec in enumerate(self.specs):
            record = {"name": spec.name, "kind": spec.kind}
            for split in ("train", "test"):
                pairs = _nyu_pairs(spec, split)
                if (
                    self.max_per_domain
                    and split == "train"
                    and len(pairs) > self.max_per_domain
                ):
                    rng = np.random.default_rng(self.sample_seed + domain_index)
                    indices = sorted(
                        rng.choice(
                            len(pairs),
                            size=int(self.max_per_domain),
                            replace=False,
                        ).tolist()
                    )
                    pairs = [pairs[index] for index in indices]
                record[split] = identifiers_manifest(
                    f"{os.path.basename(rgb)}\t{os.path.basename(depth)}"
                    for rgb, depth in pairs
                )
            records.append(record)
        return {
            "sample_seed": self.sample_seed,
            "max_per_domain": self.max_per_domain,
            "domains": records,
        }

    def stream(self, split, d):
        spec = self.specs[d]
        if spec.kind == "nyu":
            pairs = _nyu_pairs(spec, split)
        else:
            raise ValueError(f"Unknown depth kind: {spec.kind}")
        if self.max_per_domain and split == "train" and len(pairs) > self.max_per_domain:
            rng = np.random.default_rng(self.sample_seed + d)
            idx = sorted(rng.choice(len(pairs), size=int(self.max_per_domain),
                                    replace=False).tolist())
            pairs = [pairs[i] for i in idx]
        for rgb_path, depth_path in pairs:
            X = self._feat(rgb_path)
            depth_arr = _load_depth_png(depth_path, spec.max_depth)
            Y, valid = _depth_to_patches(depth_arr, self.img_size, X.shape[0])
            if not np.any(valid):
                continue
            yield X[valid], Y[valid].reshape(-1, 1), int(np.sum(valid))

    def _feat(self, path):
        key = cache_key(path, self.backbone, self.img_size)
        p = os.path.join(self.cache_dir, key)
        cached = load_array_cache(p)
        if cached is not None:
            return cached
        im = Image.open(path).convert("RGB")
        X = self.extractor(im)
        atomic_save_array(p, X)
        return X
