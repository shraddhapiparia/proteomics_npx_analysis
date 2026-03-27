#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${1:?Usage: bash scripts/00_list_fields.sh <dataset_id> [outdir]}"
OUTDIR="${2:-data/interim}"

mkdir -p "$OUTDIR"

dx extract_dataset "$DATASET_ID" \
  --list-fields \
  > "$OUTDIR/ukb_field_dictionary.txt"

echo "[00] Field dictionary written to $OUTDIR/ukb_field_dictionary.txt"