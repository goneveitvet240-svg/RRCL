#!/usr/bin/env python3
"""Build a deterministic, raw-data-free supplement for C7-fdst-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXED_ZIP_TIME = (2026, 8, 9, 0, 0, 0)
FREEZE_COMMIT = "a398bc908c32670ac68f8e38b6fbb9ba404ccf3e"
RESULT_SHA256 = "0dfa51f7bf1008be170b43daf0969bb4c037f35c1cf50936d314770384bccdc3"
PATH_REPLACEMENTS = {
    b"/Users/pangwei/Documents/ai/datasets/FDST": b"${FDST_ROOT}",
    b"/Users/pangwei/Documents/ai/RRCL": b"${RRCL_ROOT}",
    b"/tmp/fdst_train_first_frames_contact.png": b"${LOCAL_CONTACT_SHEET}",
    b'"frozen_by": "pangwei"': b'"frozen_by": "anonymous_protocol_owner"',
    b'"reviewer": "pangwei_with_codex"': b'"reviewer": "anonymous_protocol_reviewer"',
}
FORBIDDEN_PUBLIC_TOKENS = (
    b"/Users/",
    b"/home/",
    b"/root/",
    b"Documents/ai/datasets",
    b"pangwei",
    b"Pang Wei",
    b"Qingdao",
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
) -> dict[str, object]:
    source_payload = source.read_bytes()
    payload = source_payload
    replacements: list[str] = []
    for private, public in PATH_REPLACEMENTS.items():
        if private in payload:
            payload = payload.replace(private, public)
            replacements.append(public.decode("utf-8"))
    leaked = [token.decode("utf-8") for token in FORBIDDEN_PUBLIC_TOKENS if token in payload]
    if leaked:
        raise SystemExit(
            f"refusing to package private absolute path(s) in {archive_name}: {leaked}"
        )
    info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)
    record: dict[str, object] = {
        "path": archive_name,
        "size_bytes": len(payload),
        "sha256": sha256(payload),
    }
    if replacements:
        record.update({
            "path_redacted": True,
            "redaction_tokens": replacements,
            "source_sha256_before_redaction": sha256(source_payload),
        })
    return record


def head() -> str:
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
            "rrcl_tmlr_fdst_c7_submission_2026-08-10.zip"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "runs_real" / "fdst_c7" / "fdst_c7.json",
    )
    args = parser.parse_args()

    validation = subprocess.run(
        [sys.executable, "scripts/validate_fdst_c7_result.py", "--result", str(args.result)],
        cwd=ROOT,
        check=False,
    )
    if validation.returncode != 0:
        raise SystemExit("refusing to package: FDST C7 result validation failed")

    sources: list[tuple[Path, str]] = [
        (args.result, "artifacts/fdst_c7/fdst_c7.path_redacted.json"),
        (args.result.parent / "attempt.lock", "artifacts/fdst_c7/attempt.lock"),
    ]
    repository_files = (
        "configs/fdst_scene_map_candidate.json",
        "datasets_fdst.py",
        "docs/C7_FDST_APPENDIX_A.json",
        "docs/C7_FDST_PROTOCOL_V2.md",
        "docs/C7_FDST_PROTOCOL_V3.md",
        "docs/FDST_SCENE_IDENTITY_AUDIT.json",
        "docs/FDST_SOURCE_RECORD.json",
        "docs/RRCL_TMLR_FDST_RESULTS_2026-08-09.md",
        "paper/figs/table_fdst_c7.csv",
        "scripts/preflight_fdst.py",
        "scripts/fdst_analysis_core.py",
        "scripts/run_fdst_c7.py",
        "scripts/validate_fdst_c7_result.py",
    )
    sources.extend((ROOT / name, name) for name in repository_files)
    sources.append(
        (ROOT / "configs" / "domains_fdst.json", "configs/domains_fdst.path_redacted.json")
    )
    for source in sorted((ROOT / "configs" / "fdst_splits").rglob("*")):
        if source.is_file():
            relative = source.relative_to(ROOT)
            archive_name = relative.as_posix()
            if archive_name == "configs/fdst_splits/manifest.json":
                archive_name = "configs/fdst_splits/manifest.path_redacted.json"
            sources.append((source, archive_name))

    missing = [str(source) for source, _ in sources if not source.is_file()]
    if missing:
        raise SystemExit("missing FDST supplement inputs:\n  " + "\n  ".join(missing))

    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(destination, "w") as archive:
        for source, archive_name in sorted(sources, key=lambda item: item[1]):
            entries.append(add_file(archive, source, archive_name))

        manifest = {
            "schema_version": "rrcl-tmlr-fdst-c7-supplement-v1",
            "archive_role": "single-shot-unseen-source-opportunity-boundary",
            "protocol": "C7-fdst-v3",
            "formal_result_sha256": RESULT_SHA256,
            "split_freeze_commit": FREEZE_COMMIT,
            "packaging_repository_head": head(),
            "external_source_count": 1,
            "selected_scene_count": 6,
            "selected_scenes_are_not_independent_replications": True,
            "raw_dataset_files_included": False,
            "publication_paths_redacted": True,
            "submission_anonymized": True,
            "official_fdst_test_partition_used": False,
            "diagnostic_oracle_not_deployable": True,
            "post_result_tuning_permitted": False,
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
                    f"private absolute path leaked into built archive entry {name}: {leaked}"
                )

    archive_hash = sha256(destination.read_bytes())
    hash_path = destination.with_suffix(destination.suffix + ".sha256")
    hash_path.write_text(f"{archive_hash}  {destination.name}\n", encoding="utf-8")
    print(f"built {destination}")
    print(f"files {len(entries) + 1} (including MANIFEST.json)")
    print(f"sha256 {archive_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
