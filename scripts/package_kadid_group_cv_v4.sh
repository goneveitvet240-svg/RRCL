#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${RRCL_PYTHON:-python3}"
RESULT_DIR="runs_real/kadid_group_cv_v4"
RESULT_FILE="$RESULT_DIR/main/kadid_group_cv_v4.json"
EXPORT_ROOT="${RRCL_EXPORT_ROOT:-/root/autodl-tmp/rrcl_exports}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$EXPORT_ROOT/rrcl_kadid_group_cv_v4_${STAMP}.tar.gz"

if [[ ! -f "$RESULT_FILE" ]]; then
  echo "Missing result: $RESULT_FILE" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/validate_kadid_group_cv_v4.py "$RESULT_FILE" \
  --protocol-config configs/kadid_group_cv_v4.json

mkdir -p "$EXPORT_ROOT"
tar -czf "$ARCHIVE" \
  "$RESULT_DIR" \
  configs/kadid_group_cv_v4.json \
  configs/domains_iqa_kadid.json \
  docs/RRCL_KADID_GROUP_CV_V4.md
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "Export archive: $ARCHIVE"
echo "Checksum: $ARCHIVE.sha256"
