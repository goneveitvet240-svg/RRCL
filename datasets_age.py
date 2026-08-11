"""Age-estimation domains for the continual-regression transfer test.

image -> scalar age. Reuses the frozen DINOv2 extractor (features.py), mean-pools
patch tokens into one image descriptor, and caches it. Same streaming interface
as datasets_real so the continual-regression machinery can be reused unchanged.

Filename parsing:
  UTKFace : "<age>_<gender>_<race>_<datetime>.jpg"    -> age = field 0
  AgeDB   : "<id>_<name>_<age>_<gender>.jpg"           -> age = field 2

Config (domains_age_*.json) per domain:
  {"name","kind":"utkface"|"agedb","root","age_min"?,"age_max"?}
age_min/age_max let you carve magnitude sub-domains (e.g. young vs old) from one set.
"""
import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cache_io import atomic_save_array, load_array_cache
from features import DinoFeatureExtractor
from data_manifest import identifiers_manifest


@dataclass
class AgeSpec:
    name: str
    kind: str
    root: str
    age_min: float = 0.0
    age_max: float = 200.0


def build_age_config(cfg):
    out = []
    for d in cfg["domains"]:
        out.append(AgeSpec(name=d["name"], kind=d.get("kind", "utkface"), root=d["root"],
                           age_min=float(d.get("age_min", 0)), age_max=float(d.get("age_max", 200))))
    return out


def _parse_age(kind, path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    try:
        if kind == "utkface":
            return float(parts[0])
        if kind == "agedb":
            return float(parts[2])
    except (ValueError, IndexError):
        return None
    return None


class AgeDomains:
    def __init__(self, domains, backbone="vit_base_patch14_dinov2.lvd142m",
                 img_size=518, max_per_domain=None, test_frac=0.2):
        self.domains = list(domains)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.max_per_domain = max_per_domain
        self.test_frac = test_frac
        self.cache_dir = os.path.join(".feature_cache_age", f"{backbone}_img{img_size}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(backbone=backbone, img_size=img_size)

    def n_domains(self):
        return len(self.domains)

    def _pairs(self, spec):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            imgs.extend(glob.glob(os.path.join(spec.root, "**", ext), recursive=True))
        out = []
        for p in sorted(set(imgs)):
            a = _parse_age(spec.kind, p)
            if a is None or a < spec.age_min or a > spec.age_max:
                continue
            out.append((p, a))
        if not out:
            raise RuntimeError(f"No age images under {spec.root} (kind={spec.kind}, "
                               f"age∈[{spec.age_min},{spec.age_max}])")
        return out

    def stream(self, split, d):
        for path, age in self._selected_pairs(split, d):
            yield self._feat(path).reshape(1, -1), np.asarray([[float(age)]]), 1

    def _selected_pairs(self, split, d):
        spec = self.domains[d]
        pairs = self._pairs(spec)
        import random
        if spec.kind == "agedb":
            # AgeDB names are <id>_<person>_<age>_<gender>.jpg.  Split whole
            # identities so photographs of the same person cannot leak across
            # train and test.
            grouped = {}
            for pair in pairs:
                fields = os.path.splitext(os.path.basename(pair[0]))[0].split("_")
                identity = fields[1] if len(fields) >= 4 else fields[0]
                grouped.setdefault(identity, []).append(pair)
            identities = sorted(grouped)
            random.Random(12345).shuffle(identities)
            ntest = max(1, int(len(identities) * self.test_frac))
            selected_identities = (
                identities[ntest:] if split == "train" else identities[:ntest]
            )
            selected = [
                pair
                for identity in selected_identities
                for pair in grouped[identity]
            ]
        else:
            random.Random(12345).shuffle(pairs)
            ntest = max(1, int(len(pairs) * self.test_frac))
            selected = pairs[ntest:] if split == "train" else pairs[:ntest]
        if self.max_per_domain is not None and split == "train":
            selected = selected[: int(self.max_per_domain)]
        return selected

    def data_manifest(self):
        return {
            "split_seed": 12345,
            "domains": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "split_group": (
                        "subject_identity" if spec.kind == "agedb" else "image"
                    ),
                    "train": identifiers_manifest(
                        os.path.basename(path)
                        for path, _ in self._selected_pairs("train", index)
                    ),
                    "test": identifiers_manifest(
                        os.path.basename(path)
                        for path, _ in self._selected_pairs("test", index)
                    ),
                }
                for index, spec in enumerate(self.domains)
            ],
        }

    def _ck(self, path):
        h = hashlib.md5((path + self.backbone + str(self.img_size)).encode()).hexdigest()
        return os.path.join(self.cache_dir, h + "_mp.npy")

    def _feat(self, path):
        cp = self._ck(path)
        cached = load_array_cache(cp)
        if cached is not None:
            return cached
        im = Image.open(path).convert("RGB")
        X = np.asarray(self.extractor(im), dtype=np.float64)
        mp = X.mean(0)
        atomic_save_array(cp, mp)
        return mp
