#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${1:?Usage: bash scripts/03_generate_sql.sh <dataset_id> [interim_dir] [sql_dir]}"
INTERIM_DIR="${2:-data/interim}"
SQL_DIR="${3:-sql}"

PHENO_FIELDS="$INTERIM_DIR/all_pheno_fields.txt"
OLINK_FIELDS="$INTERIM_DIR/olink_fields_with_eid.txt"

mkdir -p "$SQL_DIR"

[[ -f "$PHENO_FIELDS" ]] || { echo "Missing $PHENO_FIELDS" >&2; exit 1; }
[[ -f "$OLINK_FIELDS" ]] || { echo "Missing $OLINK_FIELDS" >&2; exit 1; }

dx extract_dataset "$DATASET_ID" \
  --entities participant \
  --fields-file "$PHENO_FIELDS" \
  --sql \
  -o "$SQL_DIR/pheno_query.sql"

dx extract_dataset "$DATASET_ID" \
  --fields-file "$OLINK_FIELDS" \
  --sql \
  -o "$SQL_DIR/olink_query.sql"

echo "[03] Generated $SQL_DIR/pheno_query.sql"
echo "[03] Generated $SQL_DIR/olink_query.sql"