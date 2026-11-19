"""
Pediatric cohort preparation: loading, group assignment, age and sex coding,
total abundance, and the total-abundance outlier rule.

These functions were moved verbatim from primary_regression_analysis.py
(pre-refactor copy: archive/primary_regression_analysis_pre_refactor.py).
Behaviour is unchanged; only load_data() gained optional path arguments
(defaults are the original files) and three small helpers (add_sex_coding,
exclude_total_abundance_outliers, make_expanded_group_frame) wrap logic that
was previously inline in primary_regression_analysis.main().

Used by:
  01_cohort_qc.py, 02_protein_associations.py, 03_crosscenter_analysis.py,
  05_pathways_figures.py (via figures.py).

Group definitions (assign_groups)
---------------------------------
  Healthy              SampleID starts with HC
  LC_Neuro             phenotype Group == LC and NEURO == 1
  LC_NonNeuro_strict   phenotype Group == LC and NEURO != 1
  CACTUS               SampleID starts with CACTUS
  Unknown              anything else
  LC_NonNeuro_expanded = LC_NonNeuro_strict + CACTUS (derived per comparison)

Covariate coding
----------------
  harmonized_age  years, derived from DOB and Data Vial for every subject
  sex_01          1.0 = female (Gender == "F"), 0.0 = otherwise
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
# Script is at scripts/pediatric/; parent.parent.parent is the repo root.
ROOT       = Path(__file__).resolve().parent.parent.parent
NPX_FILE   = ROOT / "data" / "raw" / "Inflammation_NPX_AgeM.txt"
PHENO_FILE = ROOT / "data" / "raw" / "phenotype.txt"
OUTDIR     = ROOT / "results" / "pediatric"

# ── Date format constants ──────────────────────────────────────────────────────
# LC subjects use D/M/YYYY or D/M/YY; Data Vial has some M/D/YY rows.
LC_DOB_FORMATS  = ["%d/%m/%Y", "%d/%m/%y"]
LC_VIAL_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%y"]

# Healthy and CACTUS subjects use M/D/YY for both DOB and Data Vial.
HC_DOB_FORMATS  = ["%m/%d/%y", "%m/%d/%Y"]
HC_VIAL_FORMATS = ["%m/%d/%y", "%m/%d/%Y"]

# Derived-age plausibility bounds (years) applied to Healthy and CACTUS subjects.
AGE_PLAUSIBLE_MIN = 2.0
AGE_PLAUSIBLE_MAX = 21.0

# Maximum acceptable difference (years) between derived and provided age for LC subjects.
AGE_TOLERANCE = 0.1  # ~36 days


# Total-abundance outlier rule applied before regression (not to QC tables):
# |z| > 3 of the row-wise nansum of all NPX values, z computed over ALL samples.
TOTAL_ABUNDANCE_Z = 3.0

# ── Date parsing ───────────────────────────────────────────────────────────────

def try_parse_date(
    date_str: str, formats: list[str]
) -> tuple[datetime | None, str | None, bool]:
    """
    Try formats in order; return (datetime, format_used, is_ambiguous).
    is_ambiguous = True when more than one format succeeds and they produce
    different dates (i.e. the parse is not uniquely determined).
    """
    if not date_str or pd.isna(date_str):
        return None, None, False
    date_str = str(date_str).strip()
    parsed: list[tuple[datetime, str]] = []
    for fmt in formats:
        try:
            parsed.append((datetime.strptime(date_str, fmt), fmt))
        except ValueError:
            pass
    if not parsed:
        return None, None, False
    dt0, fmt0 = parsed[0]
    is_ambiguous = len({p[0] for p in parsed}) > 1
    return dt0, fmt0, is_ambiguous


def age_in_years(dob: datetime, vial: datetime) -> float:
    return (vial - dob).days / 365.25


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(
    verbose: bool = True,
    npx_file: Path | None = None,
    pheno_file: Path | None = None,
):
    """Load NPX and phenotype; validate required columns; return raw frames.

    npx_file / pheno_file default to the repository files NPX_FILE / PHENO_FILE.
    Returns (npx, pheno, protein_cols); protein_cols = every NPX column except
    SampleID, in file order.
    """
    npx_file   = Path(npx_file)   if npx_file   is not None else NPX_FILE
    pheno_file = Path(pheno_file) if pheno_file is not None else PHENO_FILE
    required_pheno = {"SampleID", "Gender", "DOB", "Data Vial"}

    for path in (npx_file, pheno_file):
        if not path.exists():
            sys.exit(f"ERROR: file not found: {path}")

    npx   = pd.read_csv(npx_file,   sep="\t")
    pheno = pd.read_csv(pheno_file, sep="\t")

    npx["SampleID"]   = npx["SampleID"].str.strip()
    pheno["SampleID"] = pheno["SampleID"].str.strip()

    missing = required_pheno - set(pheno.columns)
    if missing:
        sys.exit(f"ERROR: phenotype.txt missing required columns: {sorted(missing)}")

    protein_cols = [c for c in npx.columns if c != "SampleID"]

    if verbose:
        print(f"\n{'='*60}")
        print("DATA LOADING")
        print(f"{'='*60}")
        print(f"  NPX:    {npx_file}  ({len(npx)} samples, {len(protein_cols)} proteins)")
        print(f"  Pheno:  {pheno_file}  ({len(pheno)} rows, {len(pheno.columns)} cols)")

    return npx, pheno, protein_cols


# ── Group assignment ───────────────────────────────────────────────────────────

def assign_groups(npx: pd.DataFrame, pheno: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Merge NPX with phenotype on SampleID and assign groups:
      Healthy              SampleID starts with HC
      LC_Neuro             Group==LC and NEURO==1
      LC_NonNeuro_strict   Group==LC and NEURO==0
      CACTUS               SampleID starts with CACTUS; NEURO is NaN
      Unknown              anything else

    Returns merged DataFrame with all protein columns plus metadata and 'group'.
    """
    lc_rows  = pheno[pheno["Group"] == "LC"][["SampleID", "NEURO"]]
    neuro_map = dict(zip(lc_rows["SampleID"], lc_rows["NEURO"]))

    meta_cols = [c for c in [
        "SampleID", "Gender", "DOB", "Data Vial", "Age", "Group", "NEURO", "Time",
        "Acute severity", "Variante", "known DURATION of SYMPTOMS",
        "Data SarsCov2 Symptoms Onset",
    ] if c in pheno.columns]
    df = npx.merge(pheno[meta_cols], on="SampleID", how="left")

    def _group(sid: str) -> str:
        if sid.startswith("HC"):
            return "Healthy"
        if sid in neuro_map:
            return "LC_Neuro" if neuro_map[sid] == 1 else "LC_NonNeuro_strict"
        if sid.startswith("CACTUS"):
            return "CACTUS"
        return "Unknown"

    df["group"] = df["SampleID"].apply(_group)

    if verbose:
        counts = df["group"].value_counts()
        print(f"\n{'='*60}")
        print("GROUP ASSIGNMENT")
        print(f"{'='*60}")
        for grp in ["Healthy", "LC_Neuro", "LC_NonNeuro_strict", "CACTUS", "Unknown"]:
            n   = counts.get(grp, 0)
            tag = "  (contributes to LC_NonNeuro_expanded)" if grp == "CACTUS" else ""
            print(f"  {grp:<26}: n={n}{tag}")
        n_exp = counts.get("LC_NonNeuro_strict", 0) + counts.get("CACTUS", 0)
        print(f"  {'LC_NonNeuro_expanded (derived)':<26}: n={n_exp}")

    return df


# ── Age derivation and QC ──────────────────────────────────────────────────────

def derive_ages(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive harmonized_age (years) from DOB and Data Vial for every subject.

    LC subjects: DOB in D/M format; derived age validated against provided Age.
    Healthy / CACTUS: DOB in M/D format; derived age checked for plausibility.

    Returns df with 'harmonized_age' and 'date_parse_warning' columns,
    plus a QC DataFrame with one row per subject.
    """
    qc_rows: list[dict] = []
    harmonized: list[float | None] = []
    parse_warnings: list[bool] = []

    for _, row in df.iterrows():
        grp      = row["group"]
        dob_raw  = str(row.get("DOB",        "")).strip()
        vial_raw = str(row.get("Data Vial",  "")).strip()
        provided = row.get("Age")  # float for LC, NaN for Healthy/CACTUS

        is_lc = grp in ("LC_Neuro", "LC_NonNeuro_strict")
        dob_fmts  = LC_DOB_FORMATS  if is_lc else HC_DOB_FORMATS
        vial_fmts = LC_VIAL_FORMATS if is_lc else HC_VIAL_FORMATS

        dob_dt,  dob_fmt,  dob_amb  = try_parse_date(dob_raw,  dob_fmts)
        vial_dt, vial_fmt, vial_amb = try_parse_date(vial_raw, vial_fmts)

        derived: float | None = None
        if dob_dt is not None and vial_dt is not None:
            derived = age_in_years(dob_dt, vial_dt)

        warn = (dob_dt is None) or (vial_dt is None) or dob_amb or vial_amb

        # LC: validate against provided age
        age_diff: float | None = None
        if is_lc and pd.notna(provided) and derived is not None:
            age_diff = derived - float(provided)
            if abs(age_diff) > AGE_TOLERANCE:
                warn = True

        # Healthy / CACTUS: plausibility check
        if not is_lc and derived is not None:
            if not (AGE_PLAUSIBLE_MIN <= derived <= AGE_PLAUSIBLE_MAX):
                warn = True

        harmonized.append(derived)
        parse_warnings.append(warn)

        fmt_str = f"DOB:{dob_fmt or 'FAILED'}|Vial:{vial_fmt or 'FAILED'}"
        qc_rows.append({
            "SampleID":           row["SampleID"],
            "group":              grp,
            "DOB":                dob_raw,
            "Data_Vial":          vial_raw,
            "provided_age":       f"{provided:.4f}" if pd.notna(provided) else "",
            "derived_age":        f"{derived:.4f}"  if derived  is not None else "",
            "age_difference":     f"{age_diff:.4f}" if age_diff is not None else "",
            "date_format_used":   fmt_str,
            "age_source":         "derived",
            "date_parse_warning": warn,
        })

    df = df.copy()
    df["harmonized_age"]    = harmonized
    df["date_parse_warning"] = parse_warnings

    qc_df = pd.DataFrame(qc_rows)

    if verbose:
        n_total  = len(harmonized)
        n_failed = sum(a is None for a in harmonized)
        n_warn   = int(qc_df["date_parse_warning"].sum())
        print(f"\n{'='*60}")
        print("AGE DERIVATION QC")
        print(f"{'='*60}")
        print(f"  Subjects total:            {n_total}")
        print(f"  Age derived successfully:  {n_total - n_failed}")
        print(f"  Failed to derive age:      {n_failed}")
        print(f"  Any QC warning flagged:    {n_warn}")
        if n_warn:
            flagged = qc_df[qc_df["date_parse_warning"]]["SampleID"].tolist()
            for sid in flagged:
                row_q = qc_df[qc_df["SampleID"] == sid].iloc[0]
                print(f"    {sid}  derived={row_q['derived_age']}  "
                      f"provided={row_q['provided_age']}  "
                      f"fmt={row_q['date_format_used']}")

    return df, qc_df


# ── Total abundance ────────────────────────────────────────────────────────────

def compute_total_abundance(
    df: pd.DataFrame, protein_cols: list[str], verbose: bool = True
) -> pd.DataFrame:
    """total_abundance = row-wise nansum of all NPX protein values per sample."""
    df = df.copy()
    df["total_abundance"] = df[protein_cols].apply(np.nansum, axis=1)
    if verbose:
        print(f"\n{'='*60}")
        print("TOTAL ABUNDANCE (by group)")
        print(f"{'='*60}")
        ta = df.groupby("group")["total_abundance"].agg(["mean", "std", "min", "max"])
        print(ta.round(1).to_string())
    return df



# ── Sex coding ─────────────────────────────────────────────────────────────────

def add_sex_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Add sex_01: 1.0 = female (Gender == "F"), 0.0 otherwise (incl. missing).

    Identical to the inline coding previously in primary_regression_analysis.main().
    """
    df = df.copy()
    df["sex_01"] = (df["Gender"] == "F").astype(float)
    return df


# ── Total-abundance outlier rule (regressions only) ───────────────────────────

def exclude_total_abundance_outliers(
    df: pd.DataFrame, z_threshold: float = TOTAL_ABUNDANCE_Z, verbose: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Drop samples with |z(total_abundance)| > z_threshold.

    z is computed over ALL rows of df (every group), with sample SD (ddof=1),
    exactly as in the previous primary_regression_analysis.main(). Apply this to
    the full prepared cohort, never to a pre-subset comparison frame, or the
    z-scores (and therefore the excluded samples) change.

    Returns (kept_df, excluded_sample_ids).
    """
    if "total_abundance" not in df.columns:
        sys.exit("ERROR: total_abundance missing; run compute_total_abundance() first.")
    ta = df["total_abundance"]
    ta_z = (ta - ta.mean()) / ta.std(ddof=1)
    is_outlier = ta_z.abs() > z_threshold
    excluded = df.loc[is_outlier, "SampleID"].tolist()
    kept = df[ta_z.abs() <= z_threshold].copy()
    if verbose and excluded:
        print(f"\n  Outlier exclusion: {len(excluded)} sample(s) with "
              f"|total_abundance z|>{z_threshold:g} removed before regression: {excluded}")
    return kept, excluded


# ── Expanded comparator group ─────────────────────────────────────────────────

def make_expanded_group_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with LC_NonNeuro_strict and CACTUS relabelled
    LC_NonNeuro_expanded (used only for the expanded-reference comparison)."""
    df_expanded = df.copy()
    df_expanded.loc[
        df_expanded["group"].isin(["LC_NonNeuro_strict", "CACTUS"]), "group"
    ] = "LC_NonNeuro_expanded"
    return df_expanded


# ── Full preparation pipeline ─────────────────────────────────────────────────

def prepare_cohort(
    verbose: bool = True,
    npx_file: Path | None = None,
    pheno_file: Path | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load -> assign groups -> derive ages -> sex_01 -> total_abundance.

    Same order of operations as the previous primary_regression_analysis.main().
    No outlier exclusion and no file writes happen here.

    Returns (df, protein_cols, age_qc):
      df            one row per NPX sample: proteins + metadata + group,
                    harmonized_age, date_parse_warning, sex_01, total_abundance
      protein_cols  NPX protein columns tested
      age_qc        per-subject age-derivation QC table
    """
    npx, pheno, protein_cols = load_data(verbose=verbose, npx_file=npx_file,
                                         pheno_file=pheno_file)
    df = assign_groups(npx, pheno, verbose=verbose)
    df, age_qc = derive_ages(df, verbose=verbose)
    df = add_sex_coding(df)
    df = compute_total_abundance(df, protein_cols, verbose=verbose)
    return df, protein_cols, age_qc
