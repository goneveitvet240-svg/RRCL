#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RESULT_DIR="runs_real/new_method_kadid_development_v1"
EXPORT_ROOT="${RRCL_EXPORT_ROOT:-/root/autodl-tmp/rrcl_exports}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$EXPORT_ROOT/rrcl_kadid_development_v1_${STAMP}.tar.gz"

if [[ ! -f "$RESULT_DIR/main/kadid_baseline_audit.json" ]]; then
  echo "Missing validated result: $RESULT_DIR/main/kadid_baseline_audit.json" >&2
  exit 1
fi

mkdir -p "$EXPORT_ROOT"
tar -czf "$ARCHIVE" \
  "$RESULT_DIR" \
  configs/new_method_kadid_development_v1.json \
  configs/domains_iqa_kadid.json
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "Export archive: $ARCHIVE"
echo "Checksum: $ARCHIVE.sha256"
