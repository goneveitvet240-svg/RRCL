"""Crash-safe NumPy cache I/O for generated feature arrays.

Feature caches are reproducible artifacts, but a direct ``np.save(path, x)``
can leave a truncated file when the disk fills or the process is interrupted.
Readers that only check ``path.exists()`` then mistake that partial file for a
valid cache entry.  This module writes to a unique file in the destination
directory, fsyncs it, and atomically replaces the final path only after the
write succeeds.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def load_array_cache(path):
    """Return a cached array, or ``None`` when absent/corrupt.

    A malformed cache entry is removed because it is generated data and cannot
    become valid on a later read.  Filesystem/permission errors are not hidden.
    """
    source = Path(path)
    if not source.exists():
        return None
    try:
        return np.load(source)
    except (EOFError, ValueError):
        try:
            source.unlink()
        except FileNotFoundError:
            pass
        return None


def atomic_save_array(path, array):
    """Persist ``array`` without ever exposing a partial destination file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
