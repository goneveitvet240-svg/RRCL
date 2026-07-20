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

from features import DinoFeatureExtractor, cache_key


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


def _depth_to_patches(depth_arr, img_size, n_patches):
    """Resize depth map and compute per-patch mean depth."""
    g = int(round(np.sqrt(n_patches)))
    if g * g != n_patches:
        raise ValueError(f"n_patches={n_patches} is not square")
    patch_size = img_size // g
    # resize depth map to img_size x img_size
    im = Image.fromarray(depth_arr).resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(im, dtype=np.float64)
    # compute per-patch mean
    patches = arr.reshape(g, patch_size, g, patch_size).mean(axis=(1, 3))
    return patches.reshape(-1)   # (n_patches,)


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
            Y = _depth_to_patches(depth_arr, self.img_size, X.shape[0])
            yield X, Y.reshape(-1, 1), X.shape[0]

    def _feat(self, path):
        key = cache_key(path, self.backbone, self.img_size)
        p = os.path.join(self.cache_dir, key)
        if os.path.exists(p):
            return np.load(p)
        im = Image.open(path).convert("RGB")
        X = self.extractor(im)
        np.save(p, X)
        return X
