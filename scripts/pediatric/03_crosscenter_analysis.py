"""
Entry point 3 of 5 — Cross-center exploratory analyses (do NOT replace the
primary within-center comparison in 02_protein_associations.py).

LC participants were recruited at one center (Gemelli, LC_source); Healthy and
CACTUS (acute COVID / MIS-C) at another (HC_CACTUS_source), so every
cross-center contrast is confounded with center. Run every stage in order, or
pick with --stage:

  raw_crosscenter   Uncorrected NPX, outcome ~ protein + harmonized_age + sex_01:
                    LC_Neuro vs LC_NonNeuro_expanded, vs CACTUS, vs Healthy,
                    vs Healthy + total_abundance; plus the expanded comparison
                    with a CACTUS indicator (Model A) and its summary.
  combat            sva::ComBat via combat_adjustment_sva.R (batch = center,
                    mod = ~ is_LC_Neuro, protein-wise mean imputation, NA mask
                    restored) -> combat_adjusted_npx.tsv, combat_metadata.tsv
  combat_regression Regressions on ComBat NPX: vs strict, expanded, CACTUS,
                    expanded + CACTUS indicator, Healthy.
  acc_misc          LC_Neuro vs acute COVID (ACC) and vs MIS-C separately,
                    4 candidates, ComBat NPX, BH within each comparison.
  outlier_retained  CACTUS154_T0 retained: ComBat refit on 112 samples (same
                    procedure as combat), expanded comparison refit.
  gpca              gPCA batch diagnostic (gpca_combat_residual_check.R),
                    1,000 permutations, seed 1 (stored results were unseeded).
  permanova         PERMANOVA R2 pre/post ComBat (z-scored proteins, seed 0,
                    9,999 permutations).

Stages after 'combat' read the ComBat matrix from --results_dir (default
results/pediatric, i.e. the validated matrix). For a completely fresh run use
the same directory for --outdir and --results_dir.

Existing outputs are never overwritten unless --overwrite is given.

Usage
-----
  python scripts/pediatric/03_crosscenter_analysis.py --outdir results/pediatric/rerun
  python scripts/pediatric/03_crosscenter_analysis.py --stage combat_regression acc_misc --outdir /tmp/x
  python scripts/pediatric/03_crosscenter_analysis.py --outdir X --results_dir X   # fully fresh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import cohort  # noqa: E402
from logistic_models import PRIMARY_COV, SENSITIVITY_COV, run_comparison  # noqa: E402
from output_safety import check_outputs_writable  # noqa: E402

ROOT = cohort.ROOT
ADJUSTED_FNAME = "combat_adjusted_npx.tsv"
CANDIDATE_PROTEINS = ["FGF21", "CCL2", "IL17C", "TNFRSF11B"]
GROUPS_IN_COMBAT = ["LC_Neuro", "LC_NonNeuro_strict", "Healthy", "CACTUS"]
CACTUS_COV = ["harmonized_age", "sex_01", "cactus_indicator"]

STAGE_OUTPUTS = {
    "raw_crosscenter": [
        "regression_LC_NonNeuro_expanded_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_CACTUS_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_Healthy_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_Healthy_vs_LC_Neuro_age_sex_total_abundance_adjusted.tsv",
        "regression_LC_NonNeuro_expanded_vs_LC_Neuro_age_sex_cactus_indicator_adjusted.tsv",
        "cactus_indicator_model_summary.tsv",
    ],
    "combat": ["combat_batch_group_crosstab.tsv", ADJUSTED_FNAME,
               "combat_metadata.tsv", "combat_pca_before_after.pdf"],
    "combat_regression": [
        "regression_combat_LC_NonNeuro_strict_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_combat_LC_NonNeuro_expanded_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_combat_CACTUS_vs_LC_Neuro_age_sex_adjusted.tsv",
        "regression_combat_LC_NonNeuro_expanded_vs_LC_Neuro_age_sex_cactus_indicator_adjusted.tsv",
        "regression_combat_Healthy_vs_LC_Neuro_age_sex_adjusted.tsv",
        "combat_regression_summary.tsv",
    ],
    "acc_misc": ["lc_neuro_vs_acc_misc_candidate_proteins_combat.tsv"],
    "outlier_retained": ["outlier_retained_expanded"],
    "gpca": ["gpca_contrasts"],
    "permanova": ["permanova_combat_diagnostic_summary.tsv"],
}


def run_r_helper(script: str, args: list[str]) -> None:
    cmd = ["Rscript", str(HERE / script)] + args
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def prepare_with_cactus_indicator(verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Prepared cohort + cactus_indicator (1 = CACTUS), then the regression-only
    |total-abundance z| > 3 exclusion (z over all 112 samples)."""
    df, protein_cols, _ = cohort.prepare_cohort(verbose=verbose)
    df["cactus_indicator"] = (df["group"] == "CACTUS").astype(float)
    df, _ = cohort.exclude_total_abundance_outliers(df, verbose=True)
    return df, protein_cols


# ═════════════════════════════════════════════════════════════════════════════
# Stage: raw_crosscenter — uncorrected NPX (site-confounded, exploratory)
# ═════════════════════════════════════════════════════════════════════════════

def build_cactus_summary_table(
    primary_df: pd.DataFrame,
    cactus_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge primary (age+sex) and cactus-indicator results into a per-protein
    summary.

    Columns:
      protein, primary_beta, primary_p, primary_FDR,
      cactus_beta, cactus_p, cactus_FDR,
      direction_same, significant_primary, significant_cactus_adjusted
    """
    p = primary_df[["protein", "beta", "p_value", "FDR", "evaluable"]].rename(
        columns={
            "beta":      "primary_beta",
            "p_value":   "primary_p",
            "FDR":       "primary_FDR",
            "evaluable": "primary_evaluable",
        }
    )
    c = cactus_df[["protein", "beta", "p_value", "FDR", "evaluable"]].rename(
        columns={
            "beta":      "cactus_beta",
            "p_value":   "cactus_p",
            "FDR":       "cactus_FDR",
            "evaluable": "cactus_evaluable",
        }
    )
    merged = p.merge(c, on="protein", how="outer")

    def _same_direction(row: pd.Series) -> bool:
        pb, cb = row["primary_beta"], row["cactus_beta"]
        if pd.isna(pb) or pd.isna(cb):
            return False
        if not (np.isfinite(float(pb)) and np.isfinite(float(cb))):
            return False
        return bool(np.sign(float(pb)) == np.sign(float(cb)))

    merged["direction_same"] = merged.apply(_same_direction, axis=1)
    merged["significant_primary"] = (
        merged["primary_evaluable"].astype(bool)
        & (merged["primary_p"] < 0.05)
    )
    merged["significant_cactus_adjusted"] = (
        merged["cactus_evaluable"].astype(bool)
        & (merged["cactus_p"] < 0.05)
    )

    col_order = [
        "protein",
        "primary_beta", "primary_p", "primary_FDR",
        "cactus_beta",  "cactus_p",  "cactus_FDR",
        "direction_same",
        "significant_primary",
        "significant_cactus_adjusted",
    ]
    return merged[col_order].sort_values("primary_p")


def print_cactus_summary_stats(summary: pd.DataFrame) -> None:
    n_sig_p = int(summary["significant_primary"].sum())
    n_sig_c = int(summary["significant_cactus_adjusted"].sum())
    n_both  = int((summary["significant_primary"] & summary["significant_cactus_adjusted"]).sum())
    n_dir   = int(
        (summary["significant_primary"]
         & summary["significant_cactus_adjusted"]
         & summary["direction_same"]).sum()
    )
    n_drop  = int((summary["significant_primary"] & ~summary["significant_cactus_adjusted"]).sum())
    n_gain  = int((~summary["significant_primary"] & summary["significant_cactus_adjusted"]).sum())

    print(f"\n  Significant (p<0.05) in primary (age+sex) model:          {n_sig_p}")
    print(f"  Significant (p<0.05) in cactus-indicator model:           {n_sig_c}")
    print(f"  Significant in both models:                                {n_both}")
    print(f"  Significant in both + same effect direction:               {n_dir}")
    print(f"  Lost significance after adding cactus_indicator:           {n_drop}")
    print(f"  Gained significance after adding cactus_indicator:         {n_gain}")

    if n_drop:
        dropped = summary.loc[
            summary["significant_primary"] & ~summary["significant_cactus_adjusted"],
            "protein",
        ].tolist()
        print(f"\n  [NOTE] Dropped proteins: {', '.join(dropped)}")
        print("         These associations may be driven by CACTUS source composition.")

    if n_gain:
        gained = summary.loc[
            ~summary["significant_primary"] & summary["significant_cactus_adjusted"],
            "protein",
        ].tolist()
        print(f"\n  [NOTE] Gained proteins: {', '.join(gained)}")
        print("         Emerging after cactus adjustment — may reflect residual confounding.")


def run_raw_crosscenter(outdir: Path) -> None:
    """Uncorrected-NPX cross-center comparisons (outcome 1 = LC_Neuro), and the
    CACTUS-indicator model of the expanded comparison:
      Model A: outcome ~ protein + harmonized_age + sex_01 + cactus_indicator
      Model B: the expanded age+sex model fitted here."""
    df, protein_cols = prepare_with_cactus_indicator()
    df_expanded = cohort.make_expanded_group_frame(df)   # cactus_indicator kept
    names = STAGE_OUTPUTS["raw_crosscenter"]
    runs = [
        (df_expanded, "LC_NonNeuro_expanded", "LC_Neuro", PRIMARY_COV, names[0]),
        (df,          "CACTUS",               "LC_Neuro", PRIMARY_COV, names[1]),
        (df,          "Healthy",              "LC_Neuro", PRIMARY_COV, names[2]),
        (df,          "Healthy",              "LC_Neuro", SENSITIVITY_COV, names[3]),
        (df_expanded, "LC_NonNeuro_expanded", "LC_Neuro", CACTUS_COV, names[4]),
    ]
    results = {}
    for d, ref, test, covs, fname in runs:
        results[fname] = run_comparison(d, protein_cols, ref, test, covs, verbose=True)
        results[fname].to_csv(outdir / fname, sep="\t", index=False)
        print(f"    → Saved: {fname}")

    # Model B is read back from the TSV just written, as the validated script did
    # (it loaded the existing expanded table); this keeps the summary byte-identical.
    model_b = pd.read_csv(outdir / names[0], sep="\t")
    summary = build_cactus_summary_table(model_b, results[names[4]])
    summary.to_csv(outdir / names[5], sep="\t", index=False)
    print(f"  Saved: {names[5]}")
    print_cactus_summary_stats(summary)


# ═════════════════════════════════════════════════════════════════════════════
# Stage: combat — validated sva::ComBat (R)
# ═════════════════════════════════════════════════════════════════════════════

def run_combat(outdir: Path) -> None:
    run_r_helper("combat_adjustment_sva.R", ["--outdir", str(outdir)])


# ═════════════════════════════════════════════════════════════════════════════
# Stage: combat_regression — logistic regressions on ComBat NPX
# ═════════════════════════════════════════════════════════════════════════════

def load_and_apply_existing_combat(
    df_cb: pd.DataFrame,
    protein_cols: list[str],
    adjusted_path: Path,
) -> pd.DataFrame:
    """
    Load the ComBat-adjusted NPX written by combat_adjustment_sva.R
    (combat_adjusted_npx.tsv) and replace protein columns in df_cb with
    the adjusted values.

    Only samples present in BOTH df_cb and the adjusted file are retained.
    Samples that appear in the file but not in df_cb are ignored.
    Samples in df_cb but not in the file are dropped with a warning — this is
    would happen if the R and Python outlier exclusions produced different sets.

    Returns a copy of df_cb restricted to matched samples, with protein columns
    replaced by ComBat-adjusted values.
    """
    if not adjusted_path.exists():
        sys.exit(
            f"ERROR: ComBat-adjusted NPX not found at:\n  {adjusted_path}\n"
            "Generate it first with sva::ComBat (writes results/pediatric/"
            f"{ADJUSTED_FNAME}):\n"
            "  Rscript scripts/pediatric/combat_adjustment_sva.R\n"
            "or pass --combat_file <path> to an existing ComBat-adjusted matrix."
        )

    print(f"\n{'='*60}")
    print("LOADING EXISTING ComBat-ADJUSTED NPX")
    print(f"{'='*60}")
    print(f"  Source: {adjusted_path}")

    adj_npx = pd.read_csv(adjusted_path, sep="\t")

    if "SampleID" not in adj_npx.columns:
        sys.exit(
            f"ERROR: {ADJUSTED_FNAME} does not have a 'SampleID' column. "
            "Expected format: SampleID + one column per protein."
        )

    adj_npx["SampleID"] = adj_npx["SampleID"].astype(str).str.strip()

    adj_proteins = [c for c in adj_npx.columns if c != "SampleID"]
    print(f"  Loaded: {len(adj_npx)} samples × {len(adj_proteins)} proteins")

    # Report column mismatches without failing — the R and Python protein
    # column sets should be identical (same source file), but flag if not.
    missing_in_adj = [p for p in protein_cols if p not in adj_proteins]
    extra_in_adj   = [p for p in adj_proteins  if p not in protein_cols]
    if missing_in_adj:
        print(
            f"  [WARN] {len(missing_in_adj)} protein(s) expected by primary script "
            f"not found in adjusted file: "
            f"{missing_in_adj[:5]}{'...' if len(missing_in_adj) > 5 else ''}"
        )
    if extra_in_adj:
        print(
            f"  [INFO] {len(extra_in_adj)} extra protein(s) in adjusted file "
            f"(not in primary protein list) — ignored."
        )

    # Sample matching
    meta_sids = set(df_cb["SampleID"])
    adj_sids  = set(adj_npx["SampleID"])
    matched          = meta_sids & adj_sids
    in_meta_not_adj  = meta_sids - adj_sids
    in_adj_not_meta  = adj_sids  - meta_sids

    print(f"\n  Metadata samples (4 groups, post-outlier exclusion): {len(meta_sids)}")
    print(f"  Samples in adjusted NPX file:                         {len(adj_sids)}")
    print(f"  Matched (used for regression):                        {len(matched)}")

    if in_meta_not_adj:
        dropped = sorted(in_meta_not_adj)
        print(
            f"\n  [WARN] {len(dropped)} metadata sample(s) not found in adjusted NPX "
            f"and will be excluded from regressions."
        )
        # Show at most 10 IDs to keep output readable
        for sid in dropped[:10]:
            print(f"    {sid}")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped) - 10} more")
        print(
            "  This is expected if the R and Python outlier-exclusion sets differ "
            "slightly (same threshold, potential floating-point differences)."
        )

    if in_adj_not_meta:
        print(
            f"  [INFO] {len(in_adj_not_meta)} adjusted NPX sample(s) not in "
            f"metadata — ignored."
        )

    if len(matched) < 2:
        sys.exit(
            f"ERROR: Only {len(matched)} sample(s) matched between metadata and "
            f"adjusted NPX. Cannot run regressions."
        )

    # Restrict to matched samples and replace protein values
    df_adj      = df_cb[df_cb["SampleID"].isin(matched)].copy()
    adj_indexed = adj_npx.set_index("SampleID")

    proteins_to_update = [p for p in protein_cols if p in adj_proteins]
    for prot in proteins_to_update:
        sid_mask = df_adj["SampleID"].isin(adj_indexed.index)
        df_adj.loc[sid_mask, prot] = (
            df_adj.loc[sid_mask, "SampleID"].map(adj_indexed[prot]).values
        )

    print(
        f"\n  {len(proteins_to_update)} protein column(s) replaced with "
        f"ComBat-adjusted values."
    )
    return df_adj


def build_regression_summary(
    runs: list[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build a per-comparison fit-summary row. runs: (label, covariates_str, results_df)."""
    rows = []
    for label, covariates, res in runs:
        ev     = res["evaluable"].astype(bool)
        n_eval = int(ev.sum())
        rows.append({
            "comparison":        label,
            "covariates":        covariates,
            "total_proteins":    len(res),
            "evaluable":         n_eval,
            "fit_warning":       int(res["fit_warning"].sum()),
            "nominal_p_lt_0_05": int((res.loc[ev, "p_value"] < 0.05).sum()),
            "FDR_lt_0_05":       int((res.loc[ev, "FDR"] < 0.05).sum()),
        })
    return pd.DataFrame(rows)


def run_combat_regression(outdir: Path, combat_file: Path) -> None:
    """Five regressions on the ComBat matrix (111 samples, outlier excluded)."""
    df, protein_cols = prepare_with_cactus_indicator()
    df_cb = df[df["group"].isin(GROUPS_IN_COMBAT)].copy()
    df_adj = load_and_apply_existing_combat(df_cb, protein_cols, combat_file)
    # LC_NonNeuro_strict + CACTUS -> LC_NonNeuro_expanded; cactus_indicator stays.
    df_adj_expanded = cohort.make_expanded_group_frame(df_adj)

    names = STAGE_OUTPUTS["combat_regression"]
    runs = [  # (label for summary, frame, reference, covariates, output)
        ("LC_NonNeuro_strict_vs_LC_Neuro [combat, age+sex]", df_adj,
         "LC_NonNeuro_strict", PRIMARY_COV, names[0]),
        ("LC_NonNeuro_expanded_vs_LC_Neuro [combat, age+sex]", df_adj_expanded,
         "LC_NonNeuro_expanded", PRIMARY_COV, names[1]),
        ("CACTUS_vs_LC_Neuro [combat, age+sex]", df_adj,
         "CACTUS", PRIMARY_COV, names[2]),
        ("LC_NonNeuro_expanded_vs_LC_Neuro [combat, age+sex+cactus]", df_adj_expanded,
         "LC_NonNeuro_expanded", CACTUS_COV, names[3]),
        ("Healthy_vs_LC_Neuro [combat, age+sex]", df_adj,
         "Healthy", PRIMARY_COV, names[4]),
    ]
    completed = []
    for label, d, ref, covs, fname in runs:
        print(f"\n{label}")
        res = run_comparison(d, protein_cols, ref_label=ref, test_label="LC_Neuro",
                             covariates=covs, verbose=True)
        completed.append((label, ",".join(covs), res))
        res.to_csv(outdir / fname, sep="\t", index=False)
        print(f"    → Saved: {fname}")

    summary = build_regression_summary(completed)
    summary.to_csv(outdir / names[5], sep="\t", index=False)
    print(summary[["comparison", "evaluable", "fit_warning",
                   "nominal_p_lt_0_05", "FDR_lt_0_05"]].to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# Stage: acc_misc — LC_Neuro vs acute COVID (ACC) and vs MIS-C, separately
# ═════════════════════════════════════════════════════════════════════════════
# Conventions of this analysis (kept exactly as validated):
#   LC age = provided Age column; ACC/MIS-C age = (Data Vial - DOB)/365.25 from
#   parse_date_col (Excel serials or day-first strings); sex_01 = 0 F / 1 M;
#   CACTUS154_T0 is excluded implicitly (absent from the 111-sample ComBat file);
#   BH across the 4 candidates within each comparison.

ACC_MISC_REFERENCE = {"ACC": "SARS-COV2", "MIS-C": "MIS-C"}   # label -> phenotype Group


def parse_date_col(series):
    out = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()

    # Excel serial dates
    if numeric_mask.any():
        out.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    # Normal date strings
    string_mask = (~numeric_mask) & series.notna()

    if string_mask.any():
        out.loc[string_mask] = pd.to_datetime(
            series.loc[string_mask],
            errors="coerce",
            format="mixed",
            dayfirst=True
        )

    return out


def load_acc_misc_phenotype(pheno_file: Path) -> pd.DataFrame:
    """Phenotype with harmonized_age, sex_01 (0 F / 1 M) and LC_Neuro flag."""
    pheno = pd.read_csv(pheno_file, sep="\t")
    pheno["Age"] = pd.to_numeric(pheno["Age"], errors="coerce")
    pheno["dob_parsed"] = parse_date_col(pheno["DOB"])
    pheno["vial_parsed"] = parse_date_col(pheno["Data Vial"])
    pheno["age_from_dates"] = (pheno["vial_parsed"] - pheno["dob_parsed"]).dt.days / 365.25

    pheno["harmonized_age"] = np.nan
    lc_mask = pheno["Group"] == "LC"
    pheno.loc[lc_mask, "harmonized_age"] = pheno.loc[lc_mask, "Age"]
    acc_misc_mask = pheno["Group"].isin(["SARS-COV2", "MIS-C"])
    pheno.loc[acc_misc_mask, "harmonized_age"] = pheno.loc[acc_misc_mask, "age_from_dates"]

    print("\nAge availability:")
    print(pheno.groupby("Group")["harmonized_age"].agg(
        count="count", mean="mean", std="std", min="min", max="max"))
    for g in ["SARS-COV2", "MIS-C"]:
        n_group = int((pheno["Group"] == g).sum())
        n_age = int(pheno.loc[pheno["Group"] == g, "harmonized_age"].notna().sum())
        print(f"{g}: age available {n_age}/{n_group}")
        if n_age == 0:
            raise ValueError(f"Age calculation failed for {g}. Check DOB and Data Vial formats.")

    pheno["sex_01"] = pheno["Gender"].astype(str).str.strip().str.upper().map({"F": 0, "M": 1})
    if pheno["sex_01"].isna().any():
        raise ValueError("Some Gender values could not be encoded.")
    pheno["LC_Neuro"] = (pheno["Group"] == "LC") & (pheno["NEURO"] == 1)
    print("\nLC-Neuro count:", int(pheno["LC_Neuro"].sum()))
    return pheno


def fit_acc_misc_logistic(df: pd.DataFrame, ref_name: str, ref_group: str, protein: str) -> dict:
    """case (1 = LC-Neuro, 0 = reference) ~ protein + harmonized_age + sex_01."""
    is_neuro = (df["Group"] == "LC") & (df["NEURO"] == 1)
    is_ref = df["Group"] == ref_group
    sub = df.loc[is_neuro | is_ref, ["Group", "NEURO", protein, "harmonized_age", "sex_01"]].copy()
    sub["case"] = ((sub["Group"] == "LC") & (sub["NEURO"] == 1)).astype(int)
    sub = sub.dropna(subset=[protein, "harmonized_age", "sex_01", "case"])
    n_neuro = int(sub["case"].sum())
    n_ref = int((sub["case"] == 0).sum())
    print(f"{ref_name} | {protein}: Neuro={n_neuro}, Ref={n_ref}")
    if n_neuro == 0 or n_ref == 0:
        raise ValueError(f"{ref_name} | {protein}: invalid group counts after dropping "
                         f"missing values. Neuro={n_neuro}, Ref={n_ref}")

    X = sm.add_constant(sub[[protein, "harmonized_age", "sex_01"]].astype(float), has_constant="add")
    fit = sm.Logit(sub["case"].astype(int), X).fit(disp=False, maxiter=200)
    beta = fit.params[protein]
    ci = fit.conf_int().loc[protein]
    return {
        "comparison": f"LC-Neuro vs {ref_name}", "protein": protein,
        "n_neuro": n_neuro, "n_reference": n_ref,
        "beta": beta, "SE": fit.bse[protein], "OR": np.exp(beta),
        "CI_low": np.exp(ci.iloc[0]), "CI_high": np.exp(ci.iloc[1]),
        "p_value": fit.pvalues[protein],
    }


def run_acc_misc(outdir: Path, combat_file: Path, pheno_file: Path) -> None:
    pheno = load_acc_misc_phenotype(pheno_file)
    combat = pd.read_csv(combat_file, sep="\t")
    df = pheno.merge(combat[["SampleID"] + CANDIDATE_PROTEINS], on="SampleID", how="inner")
    print("\nMerged group counts:")
    print(df["Group"].value_counts(dropna=False))

    res = pd.DataFrame([fit_acc_misc_logistic(df, ref_name, ref_group, protein)
                        for ref_name, ref_group in ACC_MISC_REFERENCE.items()
                        for protein in CANDIDATE_PROTEINS])
    res["BH_q"] = np.nan
    for comparison in res["comparison"].unique():
        idx = res["comparison"] == comparison
        res.loc[idx, "BH_q"] = multipletests(res.loc[idx, "p_value"], method="fdr_bh")[1]

    res.to_csv(outdir / STAGE_OUTPUTS["acc_misc"][0], sep="\t", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ═════════════════════════════════════════════════════════════════════════════
# Stage: outlier_retained — CACTUS154_T0 kept; ComBat refit on 112 samples
# ═════════════════════════════════════════════════════════════════════════════
# Same ComBat preprocessing as combat_adjustment_sva.R: batch = site_source,
# mod = ~ is_LC_Neuro, par.prior = TRUE, protein-wise mean imputation for ComBat
# only, original NA mask restored. Given the original 111 samples this code
# reproduces combat_adjusted_npx.tsv exactly.

OUTLIER_ID = "CACTUS154_T0"

R_COMBAT_RETAINED = r'''
suppressPackageStartupMessages(library(sva))
a <- commandArgs(trailingOnly=TRUE)
input <- a[1]; output <- a[2]
raw <- read.delim(input, check.names=FALSE, stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(raw$SampleID))
protein_cols <- setdiff(colnames(raw), c("SampleID", "group", "site_source", "is_LC_Neuro"))
original <- t(as.matrix(raw[, protein_cols, drop=FALSE]))
storage.mode(original) <- "double"
colnames(original) <- raw$SampleID
rownames(original) <- protein_cols
na_mask <- is.na(original)
cat("ComBat: input", nrow(original), "proteins x", ncol(original), "samples;",
    sum(na_mask), "missing NPX cells\n")
filled <- original
if (any(na_mask)) {
  means <- rowMeans(original, na.rm=TRUE)
  if (any(!is.finite(means))) stop("At least one protein is entirely missing")
  for (i in which(rowSums(na_mask)>0)) filled[i, na_mask[i,]] <- means[i]
}
mod <- model.matrix(~ is_LC_Neuro, data=raw)
corrected <- sva::ComBat(dat=filled, batch=raw$site_source, mod=mod,
                         par.prior=TRUE, prior.plots=FALSE)
corrected[na_mask] <- NA_real_
out <- data.frame(SampleID=raw$SampleID,
                  t(corrected), check.names=FALSE)
write.table(out, file=output, sep="\t", row.names=FALSE, quote=FALSE, na="NA")
cat("ComBat: saved", nrow(out), "samples, missing values restored:",
    sum(is.na(out[, -1, drop=FALSE])), "\n")
'''


def prepare_outlier_retained_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """All 112 samples; checks the |aggregate NPX z| > 3 rule flags only the outlier."""
    npx, pheno, protein_cols = cohort.load_data(verbose=False)
    df = cohort.assign_groups(npx, pheno, verbose=False)
    df, age_qc = cohort.derive_ages(df, verbose=False)
    df[protein_cols] = df[protein_cols].apply(pd.to_numeric, errors="coerce")
    df["sex_01"] = df["Gender"].map({"F": 1.0, "M": 0.0})

    if df.SampleID.duplicated().any():
        raise ValueError("Duplicate SampleID entries in NPX / phenotype merge")
    if len(df) != 112:
        raise ValueError(f"Expected 112 total samples before outlier exclusion; found {len(df)}")
    if df.loc[df.SampleID.eq(OUTLIER_ID), "group"].tolist() != ["CACTUS"]:
        raise ValueError(f"{OUTLIER_ID} missing or not in the CACTUS group")
    if (df.group == "Unknown").any():
        raise ValueError("Unclassified samples found; check phenotype group mapping")

    df["aggregate_npx_signal"] = df[protein_cols].sum(axis=1, skipna=True)
    df["aggregate_npx_z"] = ((df.aggregate_npx_signal - df.aggregate_npx_signal.mean())
                             / df.aggregate_npx_signal.std(ddof=1))
    flagged = df.loc[df.aggregate_npx_z.abs() > 3, ["SampleID", "group", "aggregate_npx_z"]]
    print("Samples flagged by original |aggregate NPX z| > 3 rule:\n", flagged.to_string(index=False))
    if set(flagged.SampleID) != {OUTLIER_ID}:
        raise ValueError("Flagged sample IDs differ from the original single-outlier exclusion")
    if df[["harmonized_age", "sex_01"]].isna().any().any():
        raise ValueError("Age or sex missing: check before comparing model sample sizes")
    expected = {"LC_Neuro": 16, "LC_NonNeuro_strict": 18, "CACTUS": 59, "Healthy": 19}
    if df.group.value_counts().to_dict() != expected:
        raise ValueError(f"Unexpected group counts: {df.group.value_counts().to_dict()}")
    return df, age_qc, protein_cols


def run_retained_combat(df: pd.DataFrame, protein_cols: list[str], adjusted_path: Path) -> pd.DataFrame:
    """Write the ComBat input, run R_COMBAT_RETAINED, check order/columns/NA mask."""
    df["site_source"] = np.where(df.group.isin(["LC_Neuro", "LC_NonNeuro_strict"]),
                                 "LC_source", "HC_CACTUS_source")
    df["is_LC_Neuro"] = df.group.eq("LC_Neuro").astype(int)
    combat_input = df[["SampleID", "group", "site_source", "is_LC_Neuro"] + protein_cols].copy()

    with tempfile.TemporaryDirectory() as td:
        input_path, r_path = Path(td) / "combat_input.tsv", Path(td) / "run_combat.R"
        combat_input.to_csv(input_path, sep="\t", index=False, na_rep="NA")
        r_path.write_text(R_COMBAT_RETAINED)
        subprocess.run(["Rscript", str(r_path), str(input_path), str(adjusted_path)],
                       check=True, text=True)

    adjusted = pd.read_csv(adjusted_path, sep="\t")
    if adjusted.SampleID.tolist() != df.SampleID.tolist():
        raise ValueError("ComBat changed sample ordering")
    if set(adjusted.columns) != {"SampleID", *protein_cols}:
        raise ValueError("ComBat output protein columns differ from input")
    if not adjusted[protein_cols].isna().equals(df[protein_cols].isna()):
        raise ValueError("ComBat did not preserve original missing-value pattern")
    return adjusted


def run_outlier_retained(outdir: Path) -> None:
    retained_dir = outdir / STAGE_OUTPUTS["outlier_retained"][0]
    retained_dir.mkdir(parents=True, exist_ok=True)
    df, age_qc, protein_cols = prepare_outlier_retained_frame()
    df[["SampleID", "group", "aggregate_npx_signal", "aggregate_npx_z"]].to_csv(
        retained_dir / "outlier_qc.tsv", sep="\t", index=False)
    age_qc.to_csv(retained_dir / "age_qc.tsv", sep="\t", index=False)

    adjusted = run_retained_combat(df, protein_cols, retained_dir / "combat_adjusted_npx_112.tsv")

    merged = adjusted.merge(df[["SampleID", "group", "harmonized_age", "sex_01"]],
                            on="SampleID", how="left", validate="one_to_one")
    merged.loc[merged.group.isin(["LC_NonNeuro_strict", "CACTUS"]), "group"] = "LC_NonNeuro_expanded"
    comp = merged.loc[merged.group.isin(["LC_Neuro", "LC_NonNeuro_expanded"])].copy()
    print("\nExpanded comparison group counts (outlier retained):")
    print(comp.group.value_counts().to_string())
    assert comp.group.eq("LC_Neuro").sum() == 16
    assert comp.group.eq("LC_NonNeuro_expanded").sum() == 77

    results = run_comparison(comp, protein_cols, ref_label="LC_NonNeuro_expanded",
                             test_label="LC_Neuro", covariates=["harmonized_age", "sex_01"],
                             verbose=True)
    results.to_csv(retained_dir / "expanded_retained_all_proteins.tsv", sep="\t", index=False)
    candidates = results.loc[results.protein.isin(CANDIDATE_PROTEINS)].copy()
    candidates.to_csv(retained_dir / "expanded_retained_four_candidates.tsv", sep="\t", index=False)
    print(candidates[["protein", "OR", "CI_lower", "CI_upper", "p_value", "FDR", "n",
                      "fit_warning", "evaluable"]].to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# Stage: gpca — validated gPCA diagnostic (R)
# ═════════════════════════════════════════════════════════════════════════════
# Permutation p-values: the stored results/pediatric/gpca_contrasts/ were made
# WITHOUT a seed (1,000 permutations; p = 0.115-0.531, Monte Carlo SE ~0.01-0.016).
# This stage passes a fixed seed (default 1, set before each contrast) so reruns
# are reproducible; seeded values differ from the stored ones only by Monte
# Carlo noise (seeds 1-5 gave p = 0.105-0.542; no conclusion changes). The gPCA
# statistics (delta, variances, scores) do not depend on the seed.
GPCA_DEFAULT_SEED = 1


def run_gpca(outdir: Path, combat_dir: Path, nperm: int, seed: int | None) -> None:
    args = ["--nperm", str(nperm), "--combat-dir", str(combat_dir),
            "--outdir", str(outdir / STAGE_OUTPUTS["gpca"][0])]
    if seed is not None:
        args += ["--seed", str(seed)]
    run_r_helper("gpca_combat_residual_check.R", args)


# ═════════════════════════════════════════════════════════════════════════════
# Stage: permanova — multivariate batch-removal diagnostic (not DE analysis)
# ═════════════════════════════════════════════════════════════════════════════
# R2 = SS_among / SS_total on Euclidean distances, pre vs post ComBat;
# protein-wise mean imputation within each matrix; CACTUS154_T0 dropped.
# Proteins are z-scored within each matrix and contrast (standardize=True): the
# validated results/pediatric/permanova_combat_diagnostic_summary.tsv was produced
# with the original script's --standardize option (verified byte-identical).
PERMANOVA_STANDARDIZE = True

def _ss(D2, labels):
    """Anderson (2001) sums of squares from a squared-distance matrix."""
    N = D2.shape[0]
    sst = D2.sum() / (2.0 * N)
    ssw = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        sub = D2[np.ix_(idx, idx)]
        ssw += sub.sum() / (2.0 * len(idx))
    return sst - ssw, ssw, sst


def permanova_r2(X, labels, nperm=9999, standardize=False, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if standardize:
        sd = X.std(0, ddof=1)
        keep = sd > 0                      # drop proteins constant within THIS subset
        if not keep.all():
            print(f"    [standardize] dropped {int((~keep).sum())} "
                  f"zero-variance protein(s) in this subset")
        X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    labels = np.asarray(labels)
    N, a = X.shape[0], len(np.unique(labels))
    if a < 2:
        raise ValueError("PERMANOVA needs >= 2 groups.")
    if not np.isfinite(X).all():          # guard: never emit a spurious p
        raise ValueError("Non-finite NPX after preprocessing; aborting.")

    D2 = squareform(pdist(X, metric="euclidean")) ** 2
    ssa, ssw, sst = _ss(D2, labels)
    F_obs = (ssa / (a - 1)) / (ssw / (N - a))
    r2 = ssa / sst

    count = 1
    for _ in range(nperm):
        ssa_p, ssw_p, _ = _ss(D2, rng.permutation(labels))
        F_p = (ssa_p / (a - 1)) / (ssw_p / (N - a))
        if F_p >= F_obs:
            count += 1
    return {"pseudo_F": F_obs, "R2": r2, "p_value": count / (nperm + 1),
            "n": N, "n_groups": a}


def assign_group_from_sid(sid, neuro_map):
    if sid.startswith("HC"):
        return "Healthy"
    if sid in neuro_map:
        return "LC_Neuro" if neuro_map[sid] == 1 else "LC_NonNeuro_strict"
    if sid.startswith("CACTUS"):
        return "CACTUS"
    return "Unknown"


PERMANOVA_SITE = {
    "LC_Neuro": "LC_source",
    "LC_NonNeuro_strict": "LC_source",
    "Healthy": "HC_CACTUS_source",
    "CACTUS": "HC_CACTUS_source",
}

# contrast name -> (groups kept, label function)
PERMANOVA_CONTRASTS = {
    "biomatched_crosssite": ({"Healthy", "LC_NonNeuro_strict"}, lambda g: g),
    "within_gemelli": ({"LC_Neuro", "LC_NonNeuro_strict"}, lambda g: g),
    "within_bambino": ({"Healthy", "CACTUS"}, lambda g: g),
    "site_pooled": ({"Healthy", "CACTUS", "LC_Neuro", "LC_NonNeuro_strict"},
                    lambda g: PERMANOVA_SITE[g]),
}


def load_matrix(path):
    df = pd.read_csv(path, sep="\t")
    df["SampleID"] = df["SampleID"].astype(str).str.strip()
    return df.set_index("SampleID")


def run_permanova(outdir: Path, combat_file: Path, nperm: int = 9999, seed: int = 0,
                  standardize: bool = PERMANOVA_STANDARDIZE) -> None:
    npx_raw = load_matrix(cohort.NPX_FILE)
    npx_adj = load_matrix(combat_file)
    pheno = pd.read_csv(cohort.PHENO_FILE, sep="\t")
    pheno["SampleID"] = pheno["SampleID"].astype(str).str.strip()
    neuro_map = (pheno.loc[pheno["Group"] == "LC", ["SampleID", "NEURO"]]
                 .set_index("SampleID")["NEURO"].astype(int).to_dict())

    prot_cols = [c for c in npx_raw.columns if c in npx_adj.columns]
    shared = npx_raw.index.intersection(npx_adj.index).drop(OUTLIER_ID, errors="ignore")
    groups = pd.Series({s: assign_group_from_sid(s, neuro_map) for s in shared}, name="group")
    groups = groups[groups != "Unknown"]
    shared = groups.index
    npx_raw = npx_raw.loc[shared, prot_cols]
    npx_adj = npx_adj.loc[shared, prot_cols]
    npx_raw = npx_raw.fillna(npx_raw.mean())
    npx_adj = npx_adj.fillna(npx_adj.mean())
    print(f"{len(shared)} samples, {len(prot_cols)} proteins | nperm={nperm} | "
          f"seed={seed} | standardize={standardize}")

    rows = []
    for name, (keep, labeller) in PERMANOVA_CONTRASTS.items():
        sel = groups[groups.isin(keep)].index
        lab = groups.loc[sel].map(labeller).values
        for stage, mat in (("pre_combat", npx_raw), ("post_combat", npx_adj)):
            res = permanova_r2(mat.loc[sel].values, lab, nperm=nperm, standardize=standardize, seed=seed)
            rows.append({"contrast": name, "stage": stage, **res})
            print(f"  {name:22s} {stage:11s}  R2={res['R2']:.4f}  F={res['pseudo_F']:.3f}  "
                  f"p={res['p_value']:.4f}  n={res['n']}")

    summary = pd.DataFrame(rows)
    wide = summary.pivot(index="contrast", columns="stage", values="R2")
    wide["delta_R2"] = wide["post_combat"] - wide["pre_combat"]
    wide["pct_change_R2"] = 100 * wide["delta_R2"] / wide["pre_combat"]
    summary = summary.merge(wide[["delta_R2", "pct_change_R2"]].reset_index(), on="contrast")
    summary.to_csv(outdir / STAGE_OUTPUTS["permanova"][0], sep="\t", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# Command line
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", choices=list(STAGE_OUTPUTS),
                        default=list(STAGE_OUTPUTS), help="Stages to run (default: all)")
    parser.add_argument("--outdir", type=Path, default=cohort.OUTDIR)
    parser.add_argument("--results_dir", type=Path, default=cohort.OUTDIR,
                        help="Where the ComBat matrix (and metadata) are read from")
    parser.add_argument("--gpca_nperm", type=int, default=1000)
    parser.add_argument("--gpca_seed", type=int, default=GPCA_DEFAULT_SEED,
                        help="Seed for gPCA permutation p-values (default 1); "
                             "use --gpca_unseeded to reproduce the original unseeded behaviour")
    parser.add_argument("--gpca_unseeded", action="store_true")
    parser.add_argument("--permanova_nperm", type=int, default=9999)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir, results_dir = args.outdir.resolve(), args.results_dir.resolve()
    check_outputs_writable([outdir / n for s in args.stage for n in STAGE_OUTPUTS[s]],
                           overwrite=args.overwrite)
    outdir.mkdir(parents=True, exist_ok=True)
    combat_file = results_dir / ADJUSTED_FNAME

    stages = {
        "raw_crosscenter": lambda: run_raw_crosscenter(outdir),
        "combat": lambda: run_combat(outdir),
        "combat_regression": lambda: run_combat_regression(outdir, combat_file),
        "acc_misc": lambda: run_acc_misc(outdir, combat_file, cohort.PHENO_FILE),
        "outlier_retained": lambda: run_outlier_retained(outdir),
        "gpca": lambda: run_gpca(outdir, results_dir, args.gpca_nperm,
                                 None if args.gpca_unseeded else args.gpca_seed),
        "permanova": lambda: run_permanova(outdir, combat_file, args.permanova_nperm),
    }
    for stage in STAGE_OUTPUTS:
        if stage in args.stage:
            print(f"\n{'#' * 70}\n# STAGE: {stage}\n{'#' * 70}")
            stages[stage]()
    print(f"\nOutputs written to: {outdir}")


if __name__ == "__main__":
    main()
