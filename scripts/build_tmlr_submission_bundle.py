#!/usr/bin/env python3
"""Build the deterministic, anonymous TMLR submission supplement bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
DESTINATION = RELEASE / "rrcl_tmlr_submission_supplement_2026-08-10.zip"
FIXED_ZIP_TIME = (2026, 8, 10, 0, 0, 0)
FORBIDDEN_TOKENS = (
    b"/Users/",
    b"/home/",
    b"/root/",
    b"pangwei",
    b"Pang Wei",
    b"Qingdao",
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_builder(script: str, output: Path) -> None:
    subprocess.run(
        [sys.executable, script, "--out", str(output)],
        cwd=ROOT,
        check=True,
    )


def audit_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"corrupt archive member in {path.name}: {bad}")
        for name in archive.namelist():
            payload = archive.read(name)
            leaks = [token.decode() for token in FORBIDDEN_TOKENS if token in payload]
            if leaks:
                raise SystemExit(
                    f"anonymous submission audit failed in {path.name}:{name}: {leaks}"
                )


def add_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> dict[str, object]:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)
    return {"path": name, "size_bytes": len(payload), "sha256": digest(payload)}


def main() -> int:
    RELEASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rrcl-tmlr-submission-") as temporary:
        temporary_root = Path(temporary)
        component_specs = (
            ("scripts/build_tmlr_supplement.py", temporary_root / "core.zip"),
            (
                "scripts/build_tmlr_synthetic_supplement.py",
                temporary_root / "synthetic.zip",
            ),
            ("scripts/build_fdst_c7_supplement.py", temporary_root / "fdst.zip"),
            (
                "scripts/build_tmlr_selector_v2_supplement.py",
                temporary_root / "selector_v2.zip",
            ),
        )
        for script, output in component_specs:
            run_builder(script, output)
            audit_archive(output)

        readme = """# RRCL anonymous TMLR supplement

This bundle contains four deterministic, manifest-bearing archives:

- retained development diagnostics (`core.zip`);
- the preregistered controlled synthetic boundary study (`synthetic.zip`);
- the frozen single-shot FDST boundary result (`fdst.zip`);
- the preregistered held-out learned-selector and matched-baseline study
  (`selector_v2.zip`).

Diagnostic Oracles use held-out labels only to measure opportunity and are not
deployable methods. FDST raw images, videos, and annotations are not
redistributed. The FDST archive contains only derived results, split metadata,
protocol records, and validation code. Identity fields and private local paths
are replaced by explicit anonymous or symbolic tokens for double-blind review.
The selector-v2 component uses disjoint synthetic tasks and does not reuse
FDST.
""".encode()

        entries: list[dict[str, object]] = []
        with zipfile.ZipFile(DESTINATION, "w") as archive:
            entries.append(add_bytes(archive, "README.md", readme))
            for _, source in component_specs:
                entries.append(add_bytes(archive, source.name, source.read_bytes()))
            manifest = {
                "schema_version": "rrcl-tmlr-anonymous-submission-bundle-v1",
                "submission_anonymized": True,
                "raw_fdst_data_included": False,
                "component_archives": entries,
            }
            add_bytes(
                archive,
                "MANIFEST.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            )

    audit_archive(DESTINATION)
    if DESTINATION.stat().st_size > 100 * 1024 * 1024:
        raise SystemExit("submission supplement exceeds the TMLR 100 MB limit")
    archive_hash = digest(DESTINATION.read_bytes())
    hash_path = DESTINATION.with_suffix(DESTINATION.suffix + ".sha256")
    hash_path.write_text(f"{archive_hash}  {DESTINATION.name}\n", encoding="utf-8")
    print(f"built {DESTINATION}")
    print(f"size_bytes {DESTINATION.stat().st_size}")
    print(f"sha256 {archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
