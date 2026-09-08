"""
IQA (Image Quality Assessment) domains for continual scalar regression.
Image -> scalar MOS (Mean Opinion Score).

Reuses the same DINOv2 feature extractor as crowd counting / age estimation.
Interface matches datasets_age.py so run_age_cl.py works unchanged.

Supported dataset formats
  csv   : generic CSV with img_col and mos_col  (KonIQ-10k, SPAQ, KADID-10k)
  tid   : TID2013 / TID2008 - mos_with_names.txt in root
  live  : LIVE IQA - one subfolder per distortion type, dmos.mat in each

Config JSON example:
{
  "domains": [
    {"name": "KonIQ",  "kind": "csv",  "root": "/path/koniq10k/images",
     "csv": "/path/koniq10k/koniq10k_scores_and_distributions.csv",
     "img_col": "image_name", "mos_col": "MOS"},
    {"name": "KADID-Noise", "kind": "csv",
     "root": "/path/kadid10k/images",
     "csv": "/path/kadid10k/dmos.csv",
     "img_col": "dist_img", "mos_col": "dmos",
     "filter_col": "dist_type", "filter_val": "1,2,3"},
    {"name": "TID2013", "kind": "tid", "root": "/path/tid2013"}
  ]
}
"""
import csv
import glob
import hashlib
import os
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from cache_io import atomic_save_array, load_array_cache
from features import DinoFeatureExtractor
from data_manifest import identifiers_manifest


@dataclass
class IQASpec:
    name: str
    kind: str        # "csv" | "tid" | "live"
    root: str
    csv: str = ""
    img_col: str = "image_name"
    mos_col: str = "MOS"
    filter_col: str = ""
    filter_val: str = ""   # comma-separated allowed values
    group_col: str = ""    # keep related contents in the same train/test split
    train_ratio: float = 0.8
    split_seed: int = 42


def build_iqa_config(cfg):
    def expand_path(value):
        return os.path.abspath(os.path.expandvars(os.path.expanduser(str(value))))

    out = []
    for d in cfg["domains"]:
        out.append(IQASpec(
            name=d["name"], kind=d.get("kind", "csv"), root=expand_path(d["root"]),
            csv=expand_path(d["csv"]) if d.get("csv") else "",
            img_col=d.get("img_col", "image_name"),
            mos_col=d.get("mos_col", "MOS"),
            filter_col=d.get("filter_col", ""),
            filter_val=d.get("filter_val", ""),
            group_col=d.get("group_col", ""),
            train_ratio=float(d.get("train_ratio", 0.8)),
            split_seed=int(d.get("split_seed", 42)),
        ))
    return out


def _derived_csv_value(row, column, img_col):
    """Return a CSV value, deriving KADID's distortion type when needed.

    The official KADID ``dmos.csv`` has no ``dist_type`` column.  Its distorted
    image names follow ``I{reference}_{distortion_type}_{level}.png``.  Keeping
    the derivation here makes the dataset reader work with the official file
    instead of requiring an undocumented, server-local CSV rewrite.
    """
    value = row.get(column, "").strip()
    if value or column != "dist_type":
        return value
    filename = row.get(img_col, "").strip()
    match = re.match(r"^[Ii]\d+_(\d+)_\d+\.[^.]+$", os.path.basename(filename))
    return str(int(match.group(1))) if match else ""


def _load_csv_pairs(spec: IQASpec):
    """Return ``(image_path, score, split_group)`` records."""
    pairs = []
    filter_vals = set(spec.filter_val.split(",")) if spec.filter_val else None
    with open(spec.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filter_value = _derived_csv_value(row, spec.filter_col, spec.img_col)
            if filter_vals and filter_value not in filter_vals:
                continue
            fname = row[spec.img_col].strip()
            mos = float(row[spec.mos_col])
            # try root/fname directly, or just fname as absolute
            p = os.path.join(spec.root, fname) if not os.path.isabs(fname) else fname
            if not os.path.exists(p):
                # try without extension
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    if os.path.exists(p + ext):
                        p = p + ext
                        break
            if os.path.exists(p):
                group = row.get(spec.group_col, "").strip() if spec.group_col else p
                if not group:
                    raise ValueError(
                        f"Missing split group '{spec.group_col}' for {fname} in {spec.csv}"
                    )
                pairs.append((p, mos, group))
    return pairs


def _load_tid_pairs(spec: IQASpec):
    """TID2013/2008: distorted/ folder + mos_with_names.txt"""
    score_file = os.path.join(spec.root, "mos_with_names.txt")
    if not os.path.exists(score_file):
        score_file = os.path.join(spec.root, "mos.txt")
    dist_dir = os.path.join(spec.root, "distorted_images")
    if not os.path.isdir(dist_dir):
        dist_dir = spec.root
    pairs = []
    with open(score_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            mos, fname = float(parts[0]), parts[1]
            p = os.path.join(dist_dir, fname)
            if os.path.exists(p):
                pairs.append((p, mos, p))
    return pairs


def _load_pairs(spec: IQASpec):
    if spec.kind == "csv":
        return _load_csv_pairs(spec)
    elif spec.kind == "tid":
        return _load_tid_pairs(spec)
    else:
        raise ValueError(f"Unknown IQA kind: {spec.kind}")


def _train_test_split(pairs, ratio, seed):
    """Deterministically split whole content groups, never individual variants."""
    if not pairs:
        raise RuntimeError("No IQA samples matched the dataset configuration")
    groups = sorted({group for _, _, group in pairs})
    if len(groups) < 2:
        raise RuntimeError("IQA train/test splitting requires at least two content groups")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(groups))
    n_train = min(len(groups) - 1, max(1, int(len(groups) * ratio)))
    train_groups = {groups[i] for i in permutation[:n_train]}
    test_groups = set(groups) - train_groups
    train = [record for record in pairs if record[2] in train_groups]
    test = [record for record in pairs if record[2] in test_groups]
    if train_groups & test_groups:
        raise AssertionError("Content-group leakage in IQA train/test split")
    return train, test


class IQADomains:
    """Domain-incremental IQA: each domain = one IQA database or distortion family."""

    def __init__(self, specs: List[IQASpec], backbone="dinov2_vitb14",
                 img_size=518, max_per_domain=None, sample_seed=42,
                 device=None, cache_dir=None):
        self.specs = list(specs)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.max_per_domain = max_per_domain
        self.sample_seed = int(sample_seed)
        if cache_dir is None:
            cache_root = os.environ.get("RRCL_FEATURE_CACHE_ROOT")
            cache_dir = (
                os.path.join(cache_root, "iqa", f"{backbone}_img{img_size}")
                if cache_root
                else os.path.join(".feature_cache_iqa", f"{backbone}_img{img_size}")
            )
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(
            backbone=backbone, img_size=img_size, device=device
        )
        # pre-split all domains
        self._splits = {}
        for d, spec in enumerate(self.specs):
            pairs = _load_pairs(spec)
            tr, te = _train_test_split(pairs, spec.train_ratio, spec.split_seed)
            if self.max_per_domain and len(tr) > self.max_per_domain:
                rng = np.random.default_rng(self.sample_seed + d)
                idx = sorted(rng.choice(len(tr), size=int(self.max_per_domain), replace=False).tolist())
                tr = [tr[i] for i in idx]
            self._splits[d] = {"train": tr, "test": te}
            train_groups = {record[2] for record in tr}
            test_groups = {record[2] for record in te}
            if train_groups & test_groups:
                raise RuntimeError(f"Reference-content leakage in domain {spec.name}")
            print(
                f"  Domain {spec.name}: train={len(tr)} ({len(train_groups)} groups), "
                f"test={len(te)} ({len(test_groups)} groups)"
            )

    def n_domains(self):
        return len(self.specs)

    def stream(self, split, d):
        for img_path, mos, group in self._splits[d][split]:
            feat = self._feat(img_path)
            yield feat, np.array([[mos]], dtype=np.float64), group

    def groups(self, split, d):
        return [group for _, _, group in self._splits[d][split]]

    def data_manifest(self):
        records = []
        for index, spec in enumerate(self.specs):
            record = {"name": spec.name, "kind": spec.kind}
            for split in ("train", "test"):
                rows = self._splits[index][split]
                record[split] = identifiers_manifest(
                    f"{os.path.basename(path)}\t{group}" for path, _, group in rows
                )
                record[f"{split}_groups"] = identifiers_manifest(
                    group for _, _, group in rows
                )
            records.append(record)
        return {"split_seed": [spec.split_seed for spec in self.specs], "domains": records}

    def _feat(self, path):
        key = hashlib.md5((path + self.backbone + str(self.img_size)).encode()).hexdigest()
        cache = os.path.join(self.cache_dir, key + ".npy")
        cached = load_array_cache(cache)
        if cached is not None:
            return cached
        im = Image.open(path).convert("RGB")
        f = self.extractor(im)
        # mean-pool patches -> single image descriptor
        vec = f.mean(axis=0, keepdims=True)   # (1, D)
        atomic_save_array(cache, vec)
        return vec
