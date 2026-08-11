"""Stable, order-independent fit/validation holdout split for training data.

Motivation (pilot audit, 2026-07-26)
------------------------------------
The previous split rule assigned every ``val_every``-th image of the *stream
order* to validation (``i % val_every == val_every - 1``).  Stream order is
sorted-filename order, and domains that never trigger max-per-domain
subsampling (SHA: 300 train images, SHB: exactly 400) therefore received an
IDENTICAL fit/validation split for every sample seed.  Multi-seed runs would
consequently overstate selector-split stability.

This module provides the replacement rule:

    sha256(f"{sample_key}|{image_id}|{split_seed}") -> int
    validation  <=>  int % val_every == 0

Properties:
  * independent of stream order and file paths;
  * changes with ``split_seed`` for EVERY domain, including SHA/SHB;
  * deterministic given (sample_key, image_id, split_seed);
  * expected validation fraction 1/val_every.

Every consumer must record the returned split manifest in its result JSON so
the exact partition is auditable.
"""

from __future__ import annotations

import hashlib

SPLIT_ALGORITHM_VERSION = "holdout-hash-v1"
SPLIT_ALGORITHM = (
    "sha256(sample_key|image_id|split_seed) little-endian uint64 "
    "% val_every == 0 -> validation"
)

ID_SOURCE_IMAGE = "image_basename"
ID_SOURCE_INDEX = "stream_index"

ROLE_FIT = "fit"
ROLE_PRECISION_VAL = "precision_val"
ROLE_GAMMA_VAL = "gamma_val"


def _hash_bucket(sample_key, image_id, split_seed):
    if not sample_key:
        raise ValueError("sample_key must be a non-empty logical dataset identity")
    if image_id is None or image_id == "":
        raise ValueError("image_id must be non-empty")
    token = f"{sample_key}|{image_id}|{int(split_seed)}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "little")


def is_validation(sample_key, image_id, split_seed, val_every=5):
    """Stable hash assignment of one training image to fit or validation."""
    val_every = int(val_every)
    if val_every < 2:
        raise ValueError(f"val_every={val_every} must be >= 2")
    return _hash_bucket(sample_key, image_id, split_seed) % val_every == 0


def holdout_role(sample_key, image_id, split_seed, val_every=5):
    """Three-way role for the shrinkage diagnostic (no validation reuse).

    The two-way validation set (``is_validation``) is split in half by the
    SAME hash: bucket ``0`` of ``2*val_every`` estimates residual precision,
    bucket ``val_every`` selects gamma, everything else fits.  Guarantees:

      * role != fit  <=>  is_validation(...) is True  (strict compatibility
        with the two-way selector split: the runners' validation set is the
        union of the two diagnostic validation roles);
      * precision estimation and gamma selection never share an image
        (fixes the pilot-audit double-dipping finding).
    """
    val_every = int(val_every)
    if val_every < 2:
        raise ValueError(f"val_every={val_every} must be >= 2")
    bucket = _hash_bucket(sample_key, image_id, split_seed) % (2 * val_every)
    if bucket == 0:
        return ROLE_PRECISION_VAL
    if bucket == val_every:
        return ROLE_GAMMA_VAL
    return ROLE_FIT


def domain_sample_key(domains, index):
    """Best-effort logical dataset identity for domain ``index``."""
    specs = getattr(domains, "domains", None)
    if specs is not None:
        try:
            spec = specs[index]
        except (IndexError, TypeError):
            spec = None
        if spec is not None:
            key = getattr(spec, "sample_key", "") or getattr(spec, "name", "")
            if key:
                return str(key)
    return f"domain:{int(index)}"


def iter_stream_with_ids(domains, split, index):
    """Yield ``(X, Y, n, image_id, id_source)`` from a domains object.

    Real dataset loaders expose ``stream(split, d, with_ids=True)``; test
    doubles that only implement ``stream(split, d)`` fall back to synthetic
    positional ids.  The id source is reported so result manifests can flag
    runs that did not use true image identities.
    """
    try:
        iterator = domains.stream(split, index, with_ids=True)
    except TypeError:
        iterator = None
    if iterator is not None:
        for item in iterator:
            X, Y, n, image_id = item
            yield X, Y, n, str(image_id), ID_SOURCE_IMAGE
        return
    for position, (X, Y, n) in enumerate(domains.stream(split, index)):
        yield X, Y, n, f"index:{position}", ID_SOURCE_INDEX


def split_manifest(sample_key, split_seed, val_every, fit_ids, val_ids, id_source):
    """Auditable record of one domain's realized fit/validation partition."""
    fit_ids = sorted(str(item) for item in fit_ids)
    val_ids = sorted(str(item) for item in val_ids)
    overlap = set(fit_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"fit/val overlap for {sample_key}: {sorted(overlap)[:3]}")
    return {
        "sample_key": str(sample_key),
        "split_seed": int(split_seed),
        "val_every": int(val_every),
        "algorithm": SPLIT_ALGORITHM,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "id_source": str(id_source),
        "fit_count": len(fit_ids),
        "val_count": len(val_ids),
        "fit_ids_sha256": hashlib.sha256("\n".join(fit_ids).encode("utf-8")).hexdigest(),
        "val_ids_sha256": hashlib.sha256("\n".join(val_ids).encode("utf-8")).hexdigest(),
    }


def roles_manifest(sample_key, split_seed, val_every, role_ids, id_source):
    """Auditable record of a multi-role partition (three-way diagnostic split).

    ``role_ids``: mapping role -> iterable of image ids.  Roles must be
    pairwise disjoint.
    """
    seen = {}
    for role, ids in role_ids.items():
        for image_id in ids:
            image_id = str(image_id)
            if image_id in seen:
                raise ValueError(
                    f"id {image_id} assigned to both {seen[image_id]} and {role}"
                )
            seen[image_id] = role
    record = {
        "sample_key": str(sample_key),
        "split_seed": int(split_seed),
        "val_every": int(val_every),
        "algorithm": SPLIT_ALGORITHM,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "id_source": str(id_source),
        "roles": {},
    }
    for role, ids in role_ids.items():
        ids = sorted(str(item) for item in ids)
        record["roles"][str(role)] = {
            "count": len(ids),
            "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        }
    return record


def require_roles_nonempty(manifest, roles, where):
    """Fail fast when any required role of a multi-role partition is empty."""
    for role in roles:
        count = manifest.get("roles", {}).get(str(role), {}).get("count", 0)
        if count <= 0:
            raise RuntimeError(
                f"{where}: degenerate role partition (role={role} is empty, "
                f"seed={manifest.get('split_seed')})"
            )
    return manifest


def require_both_partitions(manifest, where):
    """Fail fast when a domain produced an empty fit or validation partition."""
    if manifest["fit_count"] <= 0 or manifest["val_count"] <= 0:
        raise RuntimeError(
            f"{where}: degenerate holdout split "
            f"(fit={manifest['fit_count']}, val={manifest['val_count']}, "
            f"seed={manifest['split_seed']})"
        )
    return manifest
