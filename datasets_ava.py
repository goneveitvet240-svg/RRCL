"""
AVA Aesthetics domains for continual scalar regression.
Image -> mean aesthetic score (1-10).

Three domains defined by aesthetic score tiers (p33 / p67 percentile split):
  Low   (score < score_hi  of domain boundary)
  Mid
  High

Interface is identical to datasets_age.py / datasets_iqa.py so run_age_cl.py
works unchanged.

Config JSON example:
{
  "domains": [
    {"name": "AVA-Low",  "kind": "ava",
     "root": "/path/AVA_Aesthetics_10pct/processed/images",
     "csv":  "/path/AVA_Aesthetics_10pct/processed/aesthetic_scores.csv",
     "score_min": 0.0, "score_max": 5.09},
    {"name": "AVA-Mid",  "kind": "ava", ...
     "score_min": 5.09, "score_max": 5.69},
    {"name": "AVA-High", "kind": "ava", ...
     "score_min": 5.69, "score_max": 10.0}
  ]
}
"""
import csv
import hashlib
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from features import DinoFeatureExtractor


@dataclass
class AVASpec:
    name: str
    root: str          # directory containing JPEG images
    csv: str           # path to aesthetic_scores.csv
    score_min: float = 0.0
    score_max: float = 10.0
    score_col: str = "mean_score"
    img_col: str = "image_name"
    train_ratio: float = 0.8
    split_seed: int = 42


def build_ava_config(cfg):
    out = []
    for d in cfg["domains"]:
        out.append(AVASpec(
            name=d["name"],
            root=d["root"],
            csv=d["csv"],
            score_min=float(d.get("score_min", 0.0)),
            score_max=float(d.get("score_max", 10.0)),
            score_col=d.get("score_col", "mean_score"),
            img_col=d.get("img_col", "image_name"),
            train_ratio=float(d.get("train_ratio", 0.8)),
            split_seed=int(d.get("split_seed", 42)),
        ))
    return out


def _load_ava_pairs(spec: AVASpec):
    """Load (image_path, score) pairs within [score_min, score_max)."""
    pairs = []
    with open(spec.csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            score = float(row[spec.score_col])
            if not (spec.score_min <= score < spec.score_max):
                continue
            fname = row[spec.img_col].strip()
            p = os.path.join(spec.root, fname)
            if not os.path.exists(p):
                continue
            pairs.append((p, score))
    if not pairs:
        raise RuntimeError(
            f"No AVA images found in [{spec.score_min}, {spec.score_max}) "
            f"under {spec.root}"
        )
    return pairs


class AVADomains:
    """Domain-incremental aesthetic assessment; compatible with run_age_cl.py."""

    def __init__(self, specs, backbone="dinov2_vitb14",
                 img_size=518, max_per_domain=None, sample_seed=42):
        self.specs = list(specs)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.max_per_domain = max_per_domain
        self.sample_seed = int(sample_seed)
        self.cache_dir = os.path.join(".feature_cache_ava", f"{backbone}_img{img_size}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(backbone=backbone, img_size=img_size)

        # Pre-split all domains
        self._splits = {}
        for d, spec in enumerate(self.specs):
            pairs = _load_ava_pairs(spec)
            rng = np.random.default_rng(spec.split_seed)
            perm = rng.permutation(len(pairs)).tolist()
            n_test = max(1, int(len(pairs) * (1 - spec.train_ratio)))
            test_pairs  = [pairs[perm[i]] for i in range(n_test)]
            train_pairs = [pairs[perm[i]] for i in range(n_test, len(pairs))]
            if self.max_per_domain and len(train_pairs) > self.max_per_domain:
                rng2 = np.random.default_rng(self.sample_seed + d)
                idx = sorted(rng2.choice(len(train_pairs),
                                         size=int(self.max_per_domain),
                                         replace=False).tolist())
                train_pairs = [train_pairs[i] for i in idx]
            self._splits[d] = {"train": train_pairs, "test": test_pairs}
            print(f"  Domain {spec.name}: train={len(train_pairs)}, test={len(test_pairs)}")

    def n_domains(self):
        return len(self.specs)

    def stream(self, split, d):
        for img_path, score in self._splits[d][split]:
            feat = self._feat(img_path).reshape(1, -1)
            yield feat, np.array([[score]], dtype=np.float64), 1

    def _ck(self, path):
        h = hashlib.md5((path + self.backbone + str(self.img_size)).encode()).hexdigest()
        return os.path.join(self.cache_dir, h + "_mp.npy")

    def _feat(self, path):
        cp = self._ck(path)
        if os.path.exists(cp):
            return np.load(cp)
        im = Image.open(path).convert("RGB")
        X = np.asarray(self.extractor(im), dtype=np.float64)
        mp = X.mean(0)
        np.save(cp, mp)
        return mp
