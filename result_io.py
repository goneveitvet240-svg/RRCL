"""Strict, atomic JSON output for paper-facing RRCL artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


def _json_safe(value):
    """Replace undefined matrix entries with JSON null and reject infinities."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError(f"non-finite result value is not serializable: {value}")
    return value


def dump_result(path, payload):
    """Write standards-compliant JSON atomically.

    ``allow_nan=False`` is deliberately retained after sanitization so any
    future unhandled non-finite value fails the run rather than creating a
    Python-only JSON artifact.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                safe_payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
