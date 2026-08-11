#!/usr/bin/env python3
"""Build the deterministic anonymous selector-v2 supplement component."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_real" / "tmlr_selector_v2"
FIXED_ZIP_TIME = (2026, 8, 10, 0, 0, 0)
REPLACEMENTS = (
    (str(ROOT).encode(), b"${RRCL_ROOT}"),
    (b"/Users/", b"${LOCAL_USERS_ROOT}/"),
    (b"/home/", b"${REMOTE_HOME_ROOT}/"),
    (b"/root/", b"${REMOTE_ROOT}/"),
    (b"pangwei", b"anonymous_user"),
    (b"Pang Wei", b"Anonymous Author"),
    (b"Qingdao", b"Anonymous Institution"),
)
FORBIDDEN = tuple(source for source, _ in REPLACEMENTS)
TEXT_SUFFIXES = {".csv", ".json", ".lock", ".md", ".py"}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publication_payload(source: Path) -> tuple[bytes, str | None]:
    original = source.read_bytes()
    payload = original
    if source.suffix.lower() in TEXT_SUFFIXES or source.name == "attempt.lock":
        for private, public in REPLACEMENTS:
            payload = payload.replace(private, public)
    leaks = [token.decode() for token in FORBIDDEN if token in payload]
    if leaks:
        raise SystemExit(f"private token(s) in {source}: {leaks}")
    return payload, digest(original) if payload != original else None


def add_file(archive, source: Path, archive_name: str):
    payload, original_hash = publication_payload(source)
    info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)
    record = {"path": archive_name, "size_bytes": len(payload), "sha256": digest(payload)}
    if original_hash:
        record.update(
            {
                "publication_redacted": True,
                "source_sha256_before_redaction": original_hash,
            }
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "release" / "rrcl_tmlr_selector_v2_submission_2026-08-10.zip",
    )
    args = parser.parse_args()
    validation = subprocess.run(
        [sys.executable, "scripts/validate_tmlr_selector_v2.py"],
        cwd=ROOT,
        check=False,
    )
    if validation.returncode:
        raise SystemExit("refusing to package: selector-v2 validation failed")

    sources = []
    for source in sorted(RUNS.iterdir()):
        if source.is_file():
            sources.append((source, f"artifacts/{source.name}"))
    sources.append(
        (
            ROOT / "runs_real" / "tmlr_selector_v2_compute_audit.json",
            "artifacts/compute_audit.json",
        )
    )
    for relative in (
        "configs/tmlr_selector_v2.json",
        "docs/RRCL_TMLR_SELECTOR_V2_PROTOCOL.md",
        "docs/RRCL_TMLR_SELECTOR_V2_RESULTS_2026-08-10.md",
        "paper/figs/table_tmlr_selector_v2.csv",
        "tmlr_selector_v2.py",
        "analytic_forgetting_baselines.py",
        "scripts/run_tmlr_selector_v2.py",
        "scripts/validate_tmlr_selector_v2.py",
        "scripts/benchmark_tmlr_selector_v2.py",
        "tests/test_tmlr_selector_v2.py",
        "tests/test_analytic_forgetting_baselines.py",
    ):
        sources.append((ROOT / relative, relative))
    missing = [str(source) for source, _ in sources if not source.is_file()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))

    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(destination, "w") as archive:
        for source, name in sorted(sources, key=lambda item: item[1]):
            entries.append(add_file(archive, source, name))
        manifest = {
            "schema_version": "rrcl-tmlr-selector-v2-supplement-v1",
            "submission_anonymized": True,
            "publication_paths_redacted": True,
            "separate_project_artifacts_included": False,
            "fdst_reused": False,
            "formal_result_sha256": "329de9484f0ac337e543176834dc3f08437f0673b46643290c6aa3cba2f5d2a5",
            "entries": entries,
        }
        info = zipfile.ZipInfo("MANIFEST.json", date_time=FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        )

    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"corrupt archive member: {bad}")
        for name in archive.namelist():
            payload = archive.read(name)
            leaks = [token.decode() for token in FORBIDDEN if token in payload]
            if leaks:
                raise SystemExit(f"private token(s) leaked into {name}: {leaks}")
    archive_hash = digest(destination.read_bytes())
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{archive_hash}  {destination.name}\n", encoding="utf-8"
    )
    print(f"built {destination}")
    print(f"files {len(entries) + 1} (including MANIFEST.json)")
    print(f"sha256 {archive_hash}")


if __name__ == "__main__":
    main()
