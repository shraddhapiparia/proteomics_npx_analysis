"""
Entry point 1 of 5 — Cohort preparation and QC.

  01_cohort_qc.py            cohort, QC, demographics, missingness, outlier  (this file)
  02_protein_associations.py LC-only regression, Firth, FGF21, permutation, sex
  03_crosscenter_analysis.py ComBat and cross-center comparisons, diagnostics
  04_discrimination.py       repeated stratified CV discrimination (AUC)
  05_pathways_figures.py     pathway enrichment and manuscript figures


What this does (full cohort, BEFORE the regression-only outlier exclusion):
  1. Load NPX (data/raw/Inflammation_NPX_AgeM.txt) and phenotype (data/raw/phenotype.txt)
  2. Assign analysis groups (Healthy, LC_Neuro, LC_NonNeuro_strict, CACTUS)
  3. Derive harmonized_age from DOB + Data Vial; code sex_01 (1 = female)
  4. Compute total_abundance and report |z|>3 samples (excluded later by step 2)
  5. Write QC and demographics tables, and the harmonized metadata used by
     downstream scripts (AUC, Firth, heatmaps, sex check)

Outputs (under --outdir, default results/pediatric/):
  age_qc.tsv
  protein_missingness_by_group.tsv
  protein_missingness_by_group_all_proteins.tsv
  subtype_demographics_slide_table.tsv
  subtype_demographics_full.tsv
  subtype_demographics_missing_variables.tsv
  timepoint_qc.tsv
and --metadata_path (default data/interim/harmonized_metadata.tsv).

Existing files are NEVER overwritten unless --overwrite is given; the check runs
before anything is written.

Usage
-----
  python scripts/pediatric/01_cohort_qc.py --outdir results/pediatric/rerun \\
      --metadata_path results/pediatric/rerun/harmonized_metadata.tsv
  python scripts/pediatric/01_cohort_qc.py --overwrite     # regenerate in place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact, ttest_ind  # noqa: E402

import cohort  # noqa: E402
from cohort import LC_DOB_FORMATS, LC_VIAL_FORMATS, try_parse_date  # noqa: E402
from output_safety import check_outputs_writable  # noqa: E402

# ── Output filenames ───────────────────────────────────────────────────────────
# Listed up front so that existing files can be checked BEFORE anything is written.
AGE_QC_FILE = "age_qc.tsv"
DEMOGRAPHICS_FILES = [
    "subtype_demographics_slide_table.tsv",
    "subtype_demographics_full.tsv",
    "subtype_demographics_missing_variables.tsv",
]
MISSINGNESS_FILES = [
    "protein_missingness_by_group.tsv",
    "protein_missingness_by_group_all_proteins.tsv",
]
TIMEPOINT_FILE = "timepoint_qc.tsv"
DEFAULT_METADATA_PATH = cohort.ROOT / "data" / "interim" / "harmonized_metadata.tsv"

# Columns exported for downstream scripts: single source of truth for group,
# harmonized_age and sex_01 (same list as the previous primary script).
METADATA_COLS = [
    "SampleID", "group", "harmonized_age", "sex_01", "Gender",
    "Group", "NEURO", "Time", "Acute severity", "Variante",
    "date_parse_warning",
]


# Summary tables below were moved verbatim from the pre-refactor
# primary_regression_analysis.py. All are computed on the FULL cohort, before the
# total-abundance outlier exclusion applied to the regressions.

# ── Timepoint QC ──────────────────────────────────────────────────────────────

def make_timepoint_table(df: pd.DataFrame, save_path: Path | None = None) -> pd.DataFrame:
    """
    Count T0 / T1 / missing_time by group.
    LC_NonNeuro_expanded = LC_NonNeuro_strict + CACTUS (derived here from original df).
    Runs Fisher exact test for T0/T1 imbalance between LC_Neuro and LC_NonNeuro_strict.
    Time is not added as a model covariate; this is QC only.
    """
    if "Time" not in df.columns:
        print("\n[INFO] Time column not found in phenotype.txt — timepoint QC skipped.")
        return pd.DataFrame()

    def _counts(sub: pd.DataFrame) -> dict:
        t = sub["Time"].fillna("").str.strip()
        return {
            "n_T0":         int((t == "T0").sum()),
            "n_T1":         int((t == "T1").sum()),
            "missing_time": int((~t.isin(["T0", "T1"])).sum()),
        }

    rows = []
    for grp in ["LC_Neuro", "LC_NonNeuro_strict", "CACTUS", "Healthy"]:
        rows.append({"group": grp, **_counts(df[df["group"] == grp])})

    # LC_NonNeuro_expanded counts from original groups before renaming
    exp_sub = df[df["group"].isin(["LC_NonNeuro_strict", "CACTUS"])]
    rows.append({"group": "LC_NonNeuro_expanded", **_counts(exp_sub)})

    table = pd.DataFrame(rows, columns=["group", "n_T0", "n_T1", "missing_time"])

    print(f"\n{'='*60}")
    print("TIMEPOINT QC")
    print(f"{'='*60}")
    print(table.to_string(index=False))

    # Fisher exact: T0/T1 balance between LC_Neuro and LC_NonNeuro_strict
    ne = table[table["group"] == "LC_Neuro"].iloc[0]
    nn = table[table["group"] == "LC_NonNeuro_strict"].iloc[0]
    t0_ne, t1_ne = int(ne["n_T0"]), int(ne["n_T1"])
    t0_nn, t1_nn = int(nn["n_T0"]), int(nn["n_T1"])

    if t1_ne + t1_nn == 0:
        print("\n  All LC_Neuro and LC_NonNeuro_strict samples are T0 — no imbalance.")
    elif t0_ne + t0_nn > 0:
        _, p_fisher = fisher_exact([[t0_ne, t0_nn], [t1_ne, t1_nn]],
                                    alternative="two-sided")
        print(f"\n  Fisher exact T0/T1 balance (LC_Neuro vs LC_NonNeuro_strict): "
              f"p={p_fisher:.3f}")
        if p_fisher < 0.05:
            print(f"  [WARN] Timepoint is significantly imbalanced between LC_Neuro and "
                  f"LC_NonNeuro_strict (p={p_fisher:.3f}). A T0-only sensitivity analysis "
                  f"is recommended before interpreting subtype protein differences.")
        else:
            print(f"  T1 samples present but timepoint is not significantly imbalanced. "
                  f"Monitor if sample composition changes.")
    else:
        print("\n  Insufficient T0 or T1 counts for Fisher exact test.")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, sep="\t", index=False)
        print(f"  Saved: {save_path.name}")

    return table


# ── Full demographics table ────────────────────────────────────────────────────

def make_full_demographics_table(
    df: pd.DataFrame, save_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a manuscript/slide demographics table for the five analysis groups.

    Statistical tests (all pairwise vs LC_Neuro):
      Age:              Welch t-test
      Sex:              Fisher exact 2×2
      Severity, Days, Variant:  LC_NonNeuro_strict vs LC_Neuro only
                                (CACTUS and Healthy rows show NA)

    Saves three TSV files under save_dir (if provided):
      subtype_demographics_slide_table.tsv   wide, formatted strings
      subtype_demographics_full.tsv          same layout (alias for downstream use)
      subtype_demographics_missing_variables.tsv  unavailable-variable log
    """
    SEVERITY_COL = "Acute severity"
    VARIANT_COL  = "Variante"
    ONSET_COL    = "Data SarsCov2 Symptoms Onset"

    has_severity = SEVERITY_COL in df.columns
    has_variant  = VARIANT_COL  in df.columns
    has_onset    = ONSET_COL    in df.columns

    # -- Group subsets
    neuro    = df[df["group"] == "LC_Neuro"]
    nonneuro = df[df["group"] == "LC_NonNeuro_strict"]
    expanded = df[df["group"].isin(["LC_NonNeuro_strict", "CACTUS"])]
    cactus   = df[df["group"] == "CACTUS"]
    healthy  = df[df["group"] == "Healthy"]

    all_grps = [
        ("LC_Neuro",             neuro),
        ("LC_NonNeuro_strict",   nonneuro),
        ("LC_NonNeuro_expanded", expanded),
        ("CACTUS",               cactus),
        ("Healthy",              healthy),
    ]
    non_ref  = all_grps[1:]
    LC_ONLY  = {"CACTUS", "Healthy"}   # rows restricted to LC groups show NA here

    # -- Helpers
    def fmt_ms(vals):
        v = pd.Series(vals).dropna()
        return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else "—"

    def fmt_np(n, tot):
        return f"{n} ({100.0 * n / tot:.1f}%)" if tot > 0 else "—"

    def pfmt(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "—"
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    def age_p(sub):
        a = sub["harmonized_age"].dropna()
        r = neuro["harmonized_age"].dropna()
        return ttest_ind(a, r, equal_var=False)[1] if len(a) > 1 and len(r) > 1 else None

    def sex_p(sub):
        nf = int((sub["Gender"] == "F").sum())
        nm = int((sub["Gender"] == "M").sum())
        rf = int((neuro["Gender"] == "F").sum())
        rm = int((neuro["Gender"] == "M").sum())
        return fisher_exact([[nf, nm], [rf, rm]], "two-sided")[1] if nf + nm > 0 else None

    def sev_counts(sub):
        if not has_severity:
            return None, None
        s = sub[SEVERITY_COL].fillna("").str.strip().str.lower()
        return int((s == "mild").sum()), int(s.isin(["moderate", "moderatel"]).sum())

    def sev_p():
        ne_ml, ne_md = sev_counts(neuro)
        nn_ml, nn_md = sev_counts(nonneuro)
        if ne_ml is None or (ne_ml + ne_md == 0) or (nn_ml + nn_md == 0):
            return None
        return fisher_exact([[ne_ml, ne_md], [nn_ml, nn_md]], "two-sided")[1]

    def onset_days(sub):
        if not has_onset:
            return pd.Series(dtype=float)
        days = []
        for _, row in sub.iterrows():
            od, _, _ = try_parse_date(str(row.get(ONSET_COL, "")).strip(), LC_DOB_FORMATS)
            vd, _, _ = try_parse_date(str(row.get("Data Vial", "")).strip(), LC_VIAL_FORMATS)
            if od is not None and vd is not None and (vd - od).days >= 0:
                days.append(float((vd - od).days))
            else:
                days.append(np.nan)
        return pd.Series(days, dtype=float)

    neuro_days    = onset_days(neuro)
    nonneuro_days = onset_days(nonneuro)

    def days_p():
        a = neuro_days.dropna()
        b = nonneuro_days.dropna()
        return ttest_ind(a, b, equal_var=False)[1] if len(a) > 1 and len(b) > 1 else None

    def var_counts(sub):
        if not has_variant:
            return None, None
        v = sub[VARIANT_COL].fillna("").str.strip().str.lower()
        return int((v == "omicron").sum()), int(v.isin(["delta", "alpha", "originale"]).sum())

    def var_p():
        ne_om, ne_oth = var_counts(neuro)
        nn_om, nn_oth = var_counts(nonneuro)
        if ne_om is None or (ne_om + ne_oth == 0) or (nn_om + nn_oth == 0):
            return None
        return fisher_exact([[ne_om, ne_oth], [nn_om, nn_oth]], "two-sided")[1]

    # -- Build table rows
    rows: list[dict] = []

    # N
    row: dict = {"Variable": "N", "p_vs_LC_Neuro": "—"}
    for gn, gs in all_grps:
        row[gn] = str(len(gs))
    rows.append(row)

    # Age
    row = {"Variable": "Age (years), mean ± SD"}
    for gn, gs in all_grps:
        row[gn] = fmt_ms(gs["harmonized_age"])
    row["p_vs_LC_Neuro"] = "; ".join(f"{gn}: {pfmt(age_p(gs))}" for gn, gs in non_ref)
    rows.append(row)

    # Female n (%)
    row = {"Variable": "Female, n (%)"}
    for gn, gs in all_grps:
        row[gn] = fmt_np(int((gs["Gender"] == "F").sum()), len(gs))
    row["p_vs_LC_Neuro"] = "; ".join(f"{gn}: {pfmt(sex_p(gs))}" for gn, gs in non_ref)
    rows.append(row)

    # Male n (%)
    row = {"Variable": "Male, n (%)"}
    for gn, gs in all_grps:
        row[gn] = fmt_np(int((gs["Gender"] == "M").sum()), len(gs))
    row["p_vs_LC_Neuro"] = "(see Female)"
    rows.append(row)

    # Severity: mild
    row = {"Variable": "Severity: mild, n (%)"}
    for gn, gs in all_grps:
        if gn in LC_ONLY:
            row[gn] = "NA"
        else:
            nm, _ = sev_counts(gs)
            row[gn] = fmt_np(nm, len(gs)) if nm is not None else "NA"
    row["p_vs_LC_Neuro"] = (f"LC_NonNeuro_strict: {pfmt(sev_p())} (mild vs moderate)"
                             if has_severity else "NA (column absent)")
    rows.append(row)

    # Severity: moderate
    row = {"Variable": "Severity: moderate, n (%)"}
    for gn, gs in all_grps:
        if gn in LC_ONLY:
            row[gn] = "NA"
        else:
            _, nmd = sev_counts(gs)
            row[gn] = fmt_np(nmd, len(gs)) if nmd is not None else "NA"
    row["p_vs_LC_Neuro"] = "(see mild)"
    rows.append(row)

    # Days since COVID-19 symptom onset
    row = {"Variable": "Days since COVID-19 symptom onset, mean ± SD"}
    for gn, gs in all_grps:
        if gn in LC_ONLY:
            row[gn] = "NA"
        elif gn == "LC_Neuro":
            row[gn] = fmt_ms(neuro_days) if has_onset else "NA"
        elif gn == "LC_NonNeuro_strict":
            row[gn] = fmt_ms(nonneuro_days) if has_onset else "NA"
        elif gn == "LC_NonNeuro_expanded":
            # CACTUS subjects have no onset date; value shown is for LC_NonNeuro_strict subset only
            row[gn] = (fmt_ms(nonneuro_days) + " (LC subset)" if has_onset else "NA")
        else:
            row[gn] = "NA"
    row["p_vs_LC_Neuro"] = (f"LC_NonNeuro_strict: {pfmt(days_p())}"
                             if has_onset else "NA (onset column absent)")
    rows.append(row)

    # Variant: Omicron n (% of known)
    row = {"Variable": "Variant: Omicron, n (% of known)"}
    for gn, gs in all_grps:
        if gn in LC_ONLY:
            row[gn] = "NA"
        else:
            # For LC_NonNeuro_expanded: CACTUS has no variant data; use nonneuro subset
            src = nonneuro if gn == "LC_NonNeuro_expanded" else gs
            nom, noth = var_counts(src)
            if nom is None:
                row[gn] = "NA"
            else:
                known = nom + noth
                suffix = " (LC subset)" if gn == "LC_NonNeuro_expanded" else ""
                row[gn] = (fmt_np(nom, known) + suffix) if known > 0 else "—"
    row["p_vs_LC_Neuro"] = (f"LC_NonNeuro_strict: {pfmt(var_p())} (Omicron vs other)"
                             if has_variant else "NA (column absent)")
    rows.append(row)

    # -- Assemble slide table
    col_order = ["Variable", "LC_Neuro", "LC_NonNeuro_strict", "LC_NonNeuro_expanded",
                 "CACTUS", "Healthy", "p_vs_LC_Neuro"]
    slide_df = pd.DataFrame(rows, columns=col_order)

    # -- Missing-variables log
    miss_rows: list[dict] = [
        {
            "variable":    "known DURATION of SYMPTOMS",
            "column_name": "known DURATION of SYMPTOMS",
            "status":      "found" if "known DURATION of SYMPTOMS" in df.columns else "not found",
            "note":        ("Values are week categories (3/6/12/18), not individual days. "
                            "Not used for 'Days since onset'; onset date column used instead."),
        },
        {
            "variable":    "Days since COVID-19 symptom onset",
            "column_name": ONSET_COL,
            "status":      "found" if has_onset else "not found",
            "note":        ("Computed as (Data Vial – onset date) in days using LC date formats. "
                            "Negative values set to NA."
                            if has_onset else
                            f"Column '{ONSET_COL}' absent from phenotype.txt. "
                            "Row shows NA. 'known DURATION of SYMPTOMS' (week categories) is available."),
        },
    ]
    if not has_severity:
        miss_rows.append({
            "variable":    "Acute severity",
            "column_name": SEVERITY_COL,
            "status":      "not found",
            "note":        "Column absent from phenotype.txt. All severity cells set to NA.",
        })
    if not has_variant:
        miss_rows.append({
            "variable":    "Variant/Omicron",
            "column_name": VARIANT_COL,
            "status":      "not found",
            "note":        "Column absent from phenotype.txt. All variant cells set to NA.",
        })
    miss_df = pd.DataFrame(miss_rows)

    # -- Verbose print
    print(f"\n{'='*60}")
    print("FULL DEMOGRAPHICS TABLE")
    print(f"{'='*60}")
    print(slide_df.to_string(index=False))

    print(f"\n  Sex detail (counts, %, Fisher exact p vs LC_Neuro):")
    for gn, gs in all_grps:
        nf  = int((gs["Gender"] == "F").sum())
        nm  = int((gs["Gender"] == "M").sum())
        tot = len(gs)
        pf  = 100.0 * nf / tot if tot > 0 else 0.0
        pm  = 100.0 * nm / tot if tot > 0 else 0.0
        p   = sex_p(gs) if gn != "LC_Neuro" else None
        ptxt = f"  p vs LC_Neuro = {pfmt(p)}" if p is not None else ""
        print(f"  {gn:<26}: F={nf} ({pf:.1f}%), M={nm} ({pm:.1f}%){ptxt}")

    # -- Save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for tbl, fname in [
            (slide_df, "subtype_demographics_slide_table.tsv"),
            (slide_df, "subtype_demographics_full.tsv"),
            (miss_df,  "subtype_demographics_missing_variables.tsv"),
        ]:
            out = save_dir / fname
            tbl.to_csv(out, sep="\t", index=False)
            print(f"  Saved: {fname}")

    return slide_df, miss_df


# ── Protein missingness by group ──────────────────────────────────────────────

def make_missingness_by_group_table(
    df: pd.DataFrame,
    protein_cols: list[str],
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    For every protein, count missing values per analysis group.

    Groups: LC_Neuro, LC_NonNeuro_strict, LC_NonNeuro_expanded, CACTUS, Healthy.
    LC_NonNeuro_expanded = LC_NonNeuro_strict + CACTUS (derived without mutating df).

    Saves two TSV files:
      protein_missingness_by_group.tsv           proteins with ≥1 missing value
      protein_missingness_by_group_all_proteins.tsv  all 87 proteins
    """
    GROUPS = ["LC_Neuro", "LC_NonNeuro_strict", "CACTUS", "Healthy"]

    group_masks = {g: df["group"] == g for g in GROUPS}
    group_masks["LC_NonNeuro_expanded"] = df["group"].isin(["LC_NonNeuro_strict", "CACTUS"])
    group_n = {g: int(mask.sum()) for g, mask in group_masks.items()}

    col_order = [
        "protein", "total_missing",
        "missing_LC_Neuro", "missing_LC_NonNeuro_strict",
        "missing_LC_NonNeuro_expanded", "missing_CACTUS", "missing_Healthy",
        "pct_missing_LC_Neuro", "pct_missing_LC_NonNeuro_strict",
        "pct_missing_LC_NonNeuro_expanded", "pct_missing_CACTUS", "pct_missing_Healthy",
    ]

    rows = []
    for prot in protein_cols:
        null_mask = df[prot].isnull()
        total = int(null_mask.sum())
        row: dict = {"protein": prot, "total_missing": total}
        for g, gmask in group_masks.items():
            n_miss = int((null_mask & gmask).sum())
            n_grp  = group_n[g]
            row[f"missing_{g}"]     = n_miss
            row[f"pct_missing_{g}"] = round(100.0 * n_miss / n_grp, 1) if n_grp > 0 else 0.0
        rows.append(row)

    all_df  = pd.DataFrame(rows, columns=col_order)
    miss_df = all_df[all_df["total_missing"] > 0].reset_index(drop=True)

    print(f"\n{'='*60}")
    print("PROTEIN MISSINGNESS BY GROUP")
    print(f"{'='*60}")
    if miss_df.empty:
        print("  No missing protein values.")
    else:
        for _, r in miss_df.iterrows():
            parts = [f"total missing={r['total_missing']}"]
            for g in ["LC_Neuro", "LC_NonNeuro_strict", "CACTUS", "Healthy"]:
                parts.append(f"{g}={int(r[f'missing_{g}'])}")
            print(f"  {r['protein']}: {'; '.join(parts)}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        miss_path = save_dir / "protein_missingness_by_group.tsv"
        all_path  = save_dir / "protein_missingness_by_group_all_proteins.tsv"
        miss_df.to_csv(miss_path, sep="\t", index=False)
        all_df.to_csv(all_path,  sep="\t", index=False)
        print(f"  Saved: {miss_path.name}")
        print(f"  Saved: {all_path.name}")

    return miss_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=cohort.OUTDIR,
                        help="Directory for QC/demographics TSVs (default: results/pediatric)")
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH,
                        help="Harmonized metadata TSV (default: data/interim/harmonized_metadata.tsv)")
    parser.add_argument("--npx_file", type=Path, default=cohort.NPX_FILE)
    parser.add_argument("--pheno_file", type=Path, default=cohort.PHENO_FILE)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir: Path = args.outdir

    planned = (
        [outdir / AGE_QC_FILE]
        + [outdir / f for f in MISSINGNESS_FILES]
        + [outdir / f for f in DEMOGRAPHICS_FILES]
        + [outdir / TIMEPOINT_FILE]
        + [args.metadata_path]
    )
    check_outputs_writable(planned, overwrite=args.overwrite)

    # ── Prepare cohort (no exclusions) ───────────────────────────────────────
    df, protein_cols, age_qc = cohort.prepare_cohort(
        verbose=True, npx_file=args.npx_file, pheno_file=args.pheno_file)

    # ── Protein missingness by group ─────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    make_missingness_by_group_table(df, protein_cols, save_dir=outdir)

    # ── Age QC ───────────────────────────────────────────────────────────────
    n_age_null = int(df["harmonized_age"].isna().sum())
    if n_age_null:
        print(f"\n[WARN] {n_age_null} subject(s) have no derived age and will be "
              f"dropped from all models by listwise deletion.")
    age_qc.to_csv(outdir / AGE_QC_FILE, sep="\t", index=False)
    print(f"\nAge QC table saved: {outdir / AGE_QC_FILE}")
    n_flagged = int(age_qc["date_parse_warning"].sum())
    if n_flagged:
        print(f"  [WARN] {n_flagged} row(s) flagged — review {AGE_QC_FILE} before "
              f"interpreting regression results.")

    # ── Harmonized metadata for downstream scripts ───────────────────────────
    meta_cols = [c for c in METADATA_COLS if c in df.columns]
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df[meta_cols].to_csv(args.metadata_path, sep="\t", index=False)
    print(f"Harmonized metadata saved: {args.metadata_path}")

    # ── Demographics and timepoint QC ────────────────────────────────────────
    make_full_demographics_table(df, save_dir=outdir)
    make_timepoint_table(df, save_path=outdir / TIMEPOINT_FILE)

    # ── Preview of the regression-only outlier rule (nothing removed here) ───
    _, excluded = cohort.exclude_total_abundance_outliers(df, verbose=False)
    print(f"\nTotal-abundance |z|>{cohort.TOTAL_ABUNDANCE_Z:g} samples "
          f"(excluded from all regressions, kept in tables above): {excluded or 'none'}")
    print(f"Proteins in panel: {len(protein_cols)}")


if __name__ == "__main__":
    main()
