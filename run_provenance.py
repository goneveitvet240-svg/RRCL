"""Machine-readable provenance attached to every paper-facing result file."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = "rrcl-result-v2"


def _git_output(arguments):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_provenance(config_path, arguments):
    config = Path(config_path).expanduser().resolve()
    config_bytes = config.read_bytes()
    status = _git_output(["status", "--porcelain"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "arguments": arguments,
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_snapshot": json.loads(config_bytes),
    }


def with_provenance(payload, config_path, arguments):
    output = dict(payload)
    output["_provenance"] = build_provenance(config_path, arguments)
    return output
