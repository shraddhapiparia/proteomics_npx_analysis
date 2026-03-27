#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${1:?Usage: bash scripts/01_extract_olink_participants.sh <dataset_id> [outdir]}"
OUTDIR="${2:-data/interim}"

mkdir -p "$OUTDIR"

dx extract_dataset "$DATASET_ID" \
  --fields participant.eid,participant.p30900_i0 \
  --delimiter $'\t' \
  -o "$OUTDIR/participants_olink_raw.tsv"

awk -F'\t' 'NR==1 || $2 > 0' \
  "$OUTDIR/participants_olink_raw.tsv" \
  > "$OUTDIR/participants_olink_present.tsv"

cut -f1 "$OUTDIR/participants_olink_present.tsv" \
  > "$OUTDIR/eids_olink_present.tsv"

N=$(( $(wc -l < "$OUTDIR/eids_olink_present.tsv") - 1 ))

echo "[01] Olink participant list written to $OUTDIR/eids_olink_present.tsv"
echo "[01] Participants with Olink measurements: $N"