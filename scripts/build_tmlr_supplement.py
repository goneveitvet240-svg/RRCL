#!/usr/bin/env python3
"""Build a deterministic archive for the retained RRCL TMLR claim set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXED_ZIP_TIME = (2026, 8, 7, 0, 0, 0)
TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".nohup", ".txt"}
PUBLICATION_REPLACEMENTS = (
    (str(ROOT).encode(), b"${RRCL_ROOT}"),
    (b"/Users/", b"${LOCAL_USERS_ROOT}/"),
    (b"/home/", b"${REMOTE_HOME_ROOT}/"),
    (b"/root/", b"${REMOTE_ROOT}/"),
    (b"pangwei", b"anonymous_user"),
    (b"Pang Wei", b"Anonymous Author"),
    (b"Qingdao", b"Anonymous Institution"),
)
FORBIDDEN_TOKENS = tuple(source for source, _ in PUBLICATION_REPLACEMENTS)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
) -> dict[str, object]:
    payload = source.read_bytes()
    if source.suffix.lower() in TEXT_SUFFIXES:
        for private, public in PUBLICATION_REPLACEMENTS:
            payload = payload.replace(private, public)
    info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)
    return {
        "path": archive_name,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def tracked_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT / "release" /
            "rrcl_tmlr_core_submission_2026-08-10.zip"
        ),
    )
    parser.add_argument(
        "--c4-log",
        type=Path,
        default=Path(
            "/Users/pangwei/Downloads/c4_gamma_bb2e3af_results/"
            "c4_gamma_bb2e3af.nohup"
        ),
    )
    parser.add_argument(
        "--fstar-log",
        type=Path,
        default=Path(
            "/Users/pangwei/Downloads/fstar_calibration_609399a_results/"
            "root/autodl-tmp/fstar_calibration_609399a.nohup"
        ),
    )
    args = parser.parse_args()

    validation = subprocess.run(
        [sys.executable, "scripts/validate_tmlr_claim_set.py"],
        cwd=ROOT,
        check=False,
    )
    if validation.returncode != 0:
        raise SystemExit("refusing to package: TMLR claim-set validation failed")

    sources: list[tuple[Path, str]] = []
    for source in sorted((ROOT / "runs_real" / "c4_gamma").rglob("*")):
        if source.is_file():
            relative = source.relative_to(ROOT / "runs_real" / "c4_gamma")
            sources.append((source, f"artifacts/c4_gamma/{relative.as_posix()}"))
    for source in sorted((ROOT / "runs_real" / "fstar_calibration").rglob("*")):
        if source.is_file():
            relative = source.relative_to(
                ROOT / "runs_real" / "fstar_calibration"
            )
            sources.append(
                (source, f"artifacts/fstar_calibration/{relative.as_posix()}")
            )

    repository_files = (
        "paper/figs/table_c4_material_decision.csv",
        "paper/figs/table_fstar_calibration.csv",
        "paper/figs/fstar_calibration_scatter.pdf",
        "paper/figs/fstar_calibration_scatter.png",
        "docs/EXPERIMENTS.md",
    )
    for relative in repository_files:
        sources.append((ROOT / relative, relative))
    sources.extend(
        (
            (args.c4_log, "logs/c4_gamma_bb2e3af.nohup"),
            (args.fstar_log, "logs/fstar_calibration_609399a.nohup"),
        )
    )

    missing = [str(source) for source, _ in sources if not source.is_file()]
    if missing:
        raise SystemExit("missing supplement inputs:\n  " + "\n  ".join(missing))

    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(destination, "w") as archive:
        for source, archive_name in sorted(sources, key=lambda item: item[1]):
            entries.append(add_file(archive, source, archive_name))

        manifest = {
            "schema_version": "rrcl-tmlr-supplement-v1",
            "archive_role": "retained-development-claim-set",
            "development_data_only": True,
            "diagnostic_oracles_not_deployable": True,
            "submission_anonymized": True,
            "publication_paths_redacted": True,
            "c4_source_commit": "bb2e3afe969670f74eee69a4952dbaab8618a8e5",
            "fixed_factor_source_commit": "609399ac2039c9f280b61d54206da439ff5a53bc",
            "packaging_repository_head": tracked_head(),
            "entries": entries,
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = zipfile.ZipInfo("MANIFEST.json", date_time=FIXED_ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)

    with zipfile.ZipFile(destination, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"corrupt archive member: {bad}")
        for name in archive.namelist():
            member = archive.read(name)
            leaks = [token.decode() for token in FORBIDDEN_TOKENS if token in member]
            if leaks:
                raise SystemExit(f"anonymous audit failed in {name}: {leaks}")

    archive_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    hash_file = destination.with_suffix(destination.suffix + ".sha256")
    hash_file.write_text(f"{archive_hash}  {destination.name}\n", encoding="utf-8")
    print(f"built {destination}")
    print(f"files {len(entries) + 1} (including MANIFEST.json)")
    print(f"sha256 {archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
