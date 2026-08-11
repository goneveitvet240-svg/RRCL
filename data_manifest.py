"""Path-independent data identity records for release artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def identifiers_manifest(identifiers):
    values = sorted(str(value) for value in identifiers)
    return {
        "count": len(values),
        "ids_sha256": hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest(),
    }


def path_identifier(path):
    return Path(path).name
