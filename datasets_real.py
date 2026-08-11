import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cache_io import atomic_save_array, load_array_cache
from features import DinoFeatureExtractor, cache_key


def _stable_domain_seed(sample_key: str, base_seed: int) -> int:
    """Derive a position- and path-independent seed from a logical dataset key.

    Using the sequence position index ``d`` as a seed offset makes the
    selected training subset depend on *where* the dataset appears in the
    sequence, which can confound ordering experiments (e.g. JHU-first vs
    JHU-last would draw different images).  Hashing the filesystem root is
    also insufficient because equivalent dataset copies may live at different
    paths.  ``sample_key`` is therefore a stable logical identity such as
    ``JHU-v2.0`` or ``ShanghaiTech-A``.
    """
    if not sample_key:
        raise ValueError("sample_key must be a non-empty logical dataset identity")
    digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return (base_seed + offset) & 0xFFFF_FFFF_FFFF_FFFF


@dataclass
class DomainSpec:
    name: str
    kind: str
    root: str
    sample_key: str = ""
    train_split: str = "train"
    test_split: str = "test"
    include_file: str = ""
    train_include_file: str = ""
    test_include_file: str = ""


def build_from_config(cfg):
    domains = []
    for d in cfg["domains"]:
        domains.append(
            DomainSpec(
                name=d["name"],
                kind=d.get("kind", "jhu"),
                root=d["root"],
                sample_key=d.get("sample_key", d["name"]),
                train_split=d.get("train_split", "train"),
                test_split=d.get("test_split", "test"),
                include_file=d.get("include_file", ""),
                train_include_file=d.get("train_include_file", ""),
                test_include_file=d.get("test_include_file", ""),
            )
        )
    return domains


class RealCountingDomains:
    def __init__(self, domains, backbone="dinov2_vitb14", img_size=518,
                 max_per_domain=None, sample_seed=42):
        self.domains = list(domains)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.max_per_domain = max_per_domain
        self.sample_seed = int(sample_seed)
        self.cache_dir = os.path.join(".feature_cache", f"{backbone}_img{img_size}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.extractor = DinoFeatureExtractor(backbone=backbone, img_size=img_size)

    def n_domains(self):
        return len(self.domains)

    def stream(self, split, d, with_ids=False):
        """Yield ``(X, Y, n)`` per image, or ``(X, Y, n, image_id)`` when
        ``with_ids`` is true.  ``image_id`` is the image file basename — a
        stable identity for the holdout split that does not depend on stream
        order or filesystem location."""
        pairs, point_reader = self._selected_pairs(split, d)
        for image_path, gt_path in pairs:
            X = self._features(image_path)
            points = point_reader(gt_path)
            Y = _points_to_patch_counts(points, image_path, self.img_size, X.shape[0])
            if with_ids:
                yield X, Y.reshape(-1, 1), X.shape[0], os.path.basename(image_path)
            else:
                yield X, Y.reshape(-1, 1), X.shape[0]

    def _selected_pairs(self, split, d):
        spec = self.domains[d]
        actual = spec.train_split if split == "train" else spec.test_split
        if spec.kind == "jhu":
            pairs = _jhu_pairs(spec.root, actual)
            point_reader = _read_jhu_points
        elif spec.kind in {"shanghai", "shanghaitech"}:
            pairs = _shanghai_pairs(spec.root, actual)
            point_reader = _read_shanghai_points
        elif spec.kind in {"qnrf", "ucf_qnrf"}:
            pairs = _qnrf_pairs(spec.root, actual)
            point_reader = _read_qnrf_points
        else:
            raise ValueError(f"Unsupported domain kind: {spec.kind}")
        include_file = spec.include_file
        if split == "train" and spec.train_include_file:
            include_file = spec.train_include_file
        if split != "train" and spec.test_include_file:
            include_file = spec.test_include_file
        if include_file:
            allowed = _read_include_ids(include_file)
            pairs = [
                (img, gt) for img, gt in pairs
                if os.path.splitext(os.path.basename(img))[0] in allowed
            ]
            if not pairs:
                raise RuntimeError(f"No pairs left after filtering with {include_file}")
        # Random stratified sampling for train split only.
        # Test split always uses the full set to keep evaluation unbiased.
        if self.max_per_domain is not None and split == "train":
            n = int(self.max_per_domain)
            if len(pairs) > n:
                sample_key = spec.sample_key or spec.name
                rng = np.random.default_rng(
                    _stable_domain_seed(sample_key, self.sample_seed)
                )
                idx = sorted(rng.choice(len(pairs), size=n, replace=False).tolist())
                pairs = [pairs[i] for i in idx]
        return pairs, point_reader

    def data_manifest(self):
        """Return path-independent hashes of the exact selected train/test files."""
        records = []
        for domain_index, spec in enumerate(self.domains):
            domain_record = {
                "name": spec.name,
                "sample_key": spec.sample_key or spec.name,
                "kind": spec.kind,
            }
            for split in ("train", "test"):
                pairs, _ = self._selected_pairs(split, domain_index)
                identifiers = [
                    f"{os.path.basename(image)}\t{os.path.basename(target)}"
                    for image, target in pairs
                ]
                digest = hashlib.sha256(
                    "\n".join(identifiers).encode("utf-8")
                ).hexdigest()
                domain_record[split] = {
                    "count": len(identifiers),
                    "ids_sha256": digest,
                }
            records.append(domain_record)
        return {
            "sample_seed": self.sample_seed,
            "max_per_domain": self.max_per_domain,
            "domains": records,
        }

    def _features(self, image_path):
        key = cache_key(image_path, self.backbone, self.img_size)
        path = os.path.join(self.cache_dir, key)
        cached = load_array_cache(path)
        if cached is not None:
            return cached
        im = Image.open(image_path).convert("RGB")
        X = self.extractor(im)
        atomic_save_array(path, X)
        return X


def _jhu_pairs(root, split):
    img_dir = os.path.join(root, split, "images")
    gt_dir = os.path.join(root, split, "gt")
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)
    pairs = []
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        gt = os.path.join(gt_dir, stem + ".txt")
        if os.path.exists(gt):
            pairs.append((img, gt))
    if not pairs:
        raise RuntimeError(f"No JHU image/gt pairs found under {root}/{split}")
    return pairs


def _read_include_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            ids.add(os.path.splitext(os.path.basename(item))[0])
    return ids


def _shanghai_pairs(root, split):
    split_dir = "train_data" if split in {"train", "train_data"} else "test_data"
    img_dir = os.path.join(root, split_dir, "images")
    gt_dir = os.path.join(root, split_dir, "ground-truth")
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(root, split_dir, "ground_truth")
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)
    pairs = []
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        candidates = [
            os.path.join(gt_dir, "GT_" + stem + ".mat"),
            os.path.join(gt_dir, stem + ".mat"),
            os.path.join(gt_dir, "GT_" + stem + ".txt"),
            os.path.join(gt_dir, stem + ".txt"),
        ]
        gt = next((p for p in candidates if os.path.exists(p)), None)
        if gt is not None:
            pairs.append((img, gt))
    if not pairs:
        raise RuntimeError(f"No ShanghaiTech image/gt pairs found under {root}/{split_dir}")
    return pairs


def _qnrf_pairs(root, split):
    """UCF-QNRF: images and <stem>_ann.mat live in the same Train/ or Test/ dir."""
    img_dir = None
    for c in (split, split.capitalize(), split.title(), split.upper()):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            img_dir = d
            break
    if img_dir is None:
        raise RuntimeError(f"QNRF split dir not found under {root} for split={split}")
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)
    pairs = []
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        candidates = [
            os.path.join(img_dir, stem + "_ann.mat"),
            os.path.join(img_dir, stem + ".mat"),
            os.path.join(img_dir, "GT_" + stem + ".mat"),
        ]
        gt = next((p for p in candidates if os.path.exists(p)), None)
        if gt is not None:
            pairs.append((img, gt))
    if not pairs:
        raise RuntimeError(f"No QNRF image/ann pairs found under {img_dir}")
    return pairs


def _read_qnrf_points(path):
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("Missing scipy for QNRF .mat labels. pip install scipy") from e
    mat = loadmat(path)
    for key in ("annPoints", "annpoints", "points", "annPoint", "loc"):
        if key in mat:
            arr = np.asarray(mat[key], dtype=np.float64)
            if arr.size == 0:
                return np.zeros((0, 2), dtype=np.float64)
            return arr.reshape(-1, arr.shape[-1])[:, :2]
    if "image_info" in mat:  # ShanghaiTech-style nesting fallback
        info = mat["image_info"]
        try:
            return np.asarray(info[0, 0][0, 0][0], dtype=np.float64).reshape(-1, 2)
        except Exception:
            pass
    raise RuntimeError(f"Cannot find QNRF annotations in {path}; keys={list(mat.keys())}")


def _read_jhu_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    return np.asarray(pts, dtype=np.float64)


def _read_shanghai_points(path):
    if path.endswith(".txt"):
        pts = []
        with open(path) as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
        return np.asarray(pts, dtype=np.float64)

    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("Missing scipy for ShanghaiTech .mat labels. Install with: pip install scipy") from e

    mat = loadmat(path)
    if "image_info" in mat:
        info = mat["image_info"]
        try:
            pts = info[0, 0][0, 0][0]
        except Exception:
            pts = info[0][0][0][0][0]
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    for key in ("annPoints", "points", "gt"):
        if key in mat:
            arr = np.asarray(mat[key], dtype=np.float64)
            return arr.reshape(-1, arr.shape[-1])[:, :2]
    raise RuntimeError(f"Cannot find point annotations in {path}; keys={list(mat.keys())}")


def _points_to_patch_counts(points, image_path, img_size, n_patches):
    g = int(round(np.sqrt(n_patches)))
    if g * g != n_patches:
        raise ValueError(f"patch count {n_patches} is not a square grid")
    counts = np.zeros((g, g), dtype=np.float64)
    if points.size == 0:
        return counts.reshape(-1)
    with Image.open(image_path) as im:
        w, h = im.size
    xs = np.clip(points[:, 0] * (float(img_size) / max(w, 1)), 0, img_size - 1e-6)
    ys = np.clip(points[:, 1] * (float(img_size) / max(h, 1)), 0, img_size - 1e-6)
    gx = np.clip((xs / img_size * g).astype(int), 0, g - 1)
    gy = np.clip((ys / img_size * g).astype(int), 0, g - 1)
    np.add.at(counts, (gy, gx), 1.0)
    return counts.reshape(-1)
