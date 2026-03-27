#!/usr/bin/env bash
set -euo pipefail

METADATA_DIR="${1:-data/metadata}"
OUTDIR="${2:-data/interim}"

PHENO_BASE="$METADATA_DIR/phenotype_base_fields.txt"
SYMPTOM_FIELDS="$METADATA_DIR/symptom_fields_selected.txt"
OLINK_FIELDS="$METADATA_DIR/olink_proteins_fields.txt"

mkdir -p "$OUTDIR"

[[ -f "$PHENO_BASE" ]] || { echo "Missing $PHENO_BASE" >&2; exit 1; }
[[ -f "$SYMPTOM_FIELDS" ]] || { echo "Missing $SYMPTOM_FIELDS" >&2; exit 1; }
[[ -f "$OLINK_FIELDS" ]] || { echo "Missing $OLINK_FIELDS" >&2; exit 1; }

{
  echo "participant.eid"
  cat "$PHENO_BASE"
  cat "$SYMPTOM_FIELDS"
} | sed '/^[[:space:]]*$/d' | sort -u > "$OUTDIR/all_pheno_fields.txt"

awk '{print "olink_instance_0." $1}' "$OLINK_FIELDS" \
  > "$OUTDIR/olink_fields_prefixed.txt"

{
  echo "participant.eid"
  cat "$OUTDIR/olink_fields_prefixed.txt"
} > "$OUTDIR/olink_fields_with_eid.txt"

echo "[02] Built phenotype field list: $OUTDIR/all_pheno_fields.txt"
echo "[02] Built Olink field list:     $OUTDIR/olink_fields_with_eid.txt"
echo "[02] Number of phenotype fields: $(wc -l < "$OUTDIR/all_pheno_fields.txt")"
echo "[02] Number of Olink proteins:    $(wc -l < "$OUTDIR/olink_fields_prefixed.txt")"