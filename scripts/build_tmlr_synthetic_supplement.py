#!/usr/bin/env python3
"""Build a deterministic supplement for the formal TMLR synthetic-v1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_real" / "tmlr_synthetic_v1"
FIXED_ZIP_TIME = (2026, 8, 7, 0, 0, 0)
PATH_REPLACEMENTS = {
    str(ROOT).encode(): b"${RRCL_ROOT}",
}
FORBIDDEN_PUBLIC_TOKENS = (
    b"/Users/",
    b"/home/",
    b"/root/",
    b"pangwei",
    b"Pang Wei",
    b"Qingdao",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def add_file(archive, source: Path, archive_name: str):
    source_payload = source.read_bytes()
    payload = source_payload
    replacements = []
    for private, public in PATH_REPLACEMENTS.items():
        if private in payload:
            payload = payload.replace(private, public)
            replacements.append(public.decode("utf-8"))
    leaked = [token.decode("utf-8") for token in FORBIDDEN_PUBLIC_TOKENS if token in payload]
    if leaked:
        raise SystemExit(
            f"refusing to package private identity/path token(s) in {archive_name}: {leaked}"
        )
    info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)
    record = {
        "path": archive_name,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    if replacements:
        record.update({
            "path_redacted": True,
            "redaction_tokens": replacements,
            "source_sha256_before_redaction": sha256_bytes(source_payload),
        })
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT / "release" /
            "rrcl_tmlr_synthetic_v1_submission_2026-08-10.zip"
        ),
    )
    args = parser.parse_args()
    validation = subprocess.run(
        [sys.executable, "scripts/validate_tmlr_synthetic_v1.py"],
        cwd=ROOT,
        check=False,
    )
    if validation.returncode:
        raise SystemExit("refusing to package: synthetic-v1 validation failed")

    sources = []
    for source in sorted(RUNS.rglob("*")):
        if source.is_file():
            sources.append((source, f"artifacts/{source.relative_to(RUNS).as_posix()}"))
    repository_files = (
        "configs/tmlr_synthetic_v1.json",
        "tmlr_synthetic.py",
        "scripts/run_tmlr_synthetic_v1.py",
        "scripts/summarize_tmlr_synthetic_v1.py",
        "scripts/validate_tmlr_synthetic_v1.py",
        "paper/figs/table_tmlr_synthetic.csv",
        "docs/RRCL_TMLR_SYNTHETIC_PROTOCOL.md",
    )
    for relative in repository_files:
        sources.append((ROOT / relative, relative))
    missing = [str(source) for source, _ in sources if not source.is_file()]
    if missing:
        raise SystemExit("missing supplement inputs:\n  " + "\n  ".join(missing))

    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(destination, "w") as archive:
        for source, archive_name in sorted(sources, key=lambda item: item[1]):
            entries.append(add_file(archive, source, archive_name))
        manifest = {
            "schema_version": "rrcl-tmlr-synthetic-supplement-v1",
            "protocol": "rrcl-tmlr-synthetic-v1",
            "formal_run_commit": "46a015c1af52c24cd722cfb5998df61596a2e161",
            "formal_scenarios": 320,
            "development_or_synthetic_evidence_only": True,
            "diagnostic_oracles_not_deployable": True,
            "submission_anonymized": True,
            "publication_paths_redacted": True,
            "entries": entries,
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = zipfile.ZipInfo("MANIFEST.json", date_time=FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)

    with zipfile.ZipFile(destination, "r") as archive:
        for name in archive.namelist():
            payload = archive.read(name)
            leaked = [
                token.decode("utf-8")
                for token in FORBIDDEN_PUBLIC_TOKENS
                if token in payload
            ]
            if leaked:
                raise SystemExit(
                    f"private identity/path token leaked into {name}: {leaked}"
                )

    archive_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    hash_path = destination.with_suffix(destination.suffix + ".sha256")
    hash_path.write_text(
        f"{archive_hash}  {destination.name}\n", encoding="utf-8"
    )
    print(f"built {destination}")
    print(f"files {len(entries) + 1} (including MANIFEST.json)")
    print(f"sha256 {archive_hash}")


if __name__ == "__main__":
    main()
