"""
Pediatric Long COVID proteomics — logistic regression analysis.

Produces:
  results/pediatric/regression_Healthy_vs_LC_Neuro_univariate.tsv
  results/pediatric/regression_Healthy_vs_LC_NonNeuro_univariate.tsv
  results/pediatric/regression_Healthy_vs_LC_Neuro_total_abundance_adjusted.tsv
  results/pediatric/regression_Healthy_vs_LC_NonNeuro_total_abundance_adjusted.tsv

Usage:
  python scripts/pediatric_proteomics_regression.py             # full run
  python scripts/pediatric_proteomics_regression.py --max_proteins 3
  python scripts/pediatric_proteomics_regression.py --test_proteins FGF23,TNFRSF11B,HGF
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
NPX_FILE = ROOT / "data/raw/Inflammation_NPX_AgeM.txt"
PHENO_FILE = ROOT / "data/raw/phenotype.txt"
OUTDIR   = ROOT / "results/pediatric"

# ─── Cohort inclusion criteria ────────────────────────────────────────────────
# Healthy:     SampleID starts with "HC"         (n=19, single center)
# LC_Neuro:    SampleID in phenotype AND NEURO==1 (n=16, COPL/CS- prefix)
# LC_NonNeuro: SampleID in phenotype AND NEURO==0 (n=18, COPL/CS- prefix)
# CACTUS*:     Excluded — LC participants without neuro-subtype information
#
# Note: HC* samples have no _T0 suffix in the NPX file.
# Note: COPL/CS* samples have a _T0 or _T1 suffix.
# Note: Healthy vs LC comparisons have NO age/sex covariates because HC* samples
#       do not appear in phenotype.txt and therefore lack metadata.
# Note: Total abundance = sum of all protein NPX values per sample (nansum).
#       This is computed once across all 87 proteins and used as the adjustment
#       covariate in the abundance-adjusted models.
# Note: Outlier (1 CACTUS sample, z=3.21) is in the CACTUS group and is
#       therefore automatically excluded from all regression comparisons.

COMPARISONS = [
    {
        "label":     "Healthy_vs_LC_Neuro",
        "ref":       "Healthy",
        "test":      "LC_Neuro",
        "models":    ["univariate", "total_abundance"],
    },
    {
        "label":     "Healthy_vs_LC_NonNeuro",
        "ref":       "Healthy",
        "test":      "LC_NonNeuro",
        "models":    ["univariate", "total_abundance"],
    },
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(verbose=True):
    npx   = pd.read_csv(NPX_FILE,   sep="\t", index_col="SampleID")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")

    protein_cols = [c for c in npx.columns if c not in ("SampleID",)]

    if verbose:
        print(f"\n{'='*60}")
        print("DATA LOADING")
        print(f"{'='*60}")
        print(f"NPX file:       {NPX_FILE.name}")
        print(f"Pheno file:     {PHENO_FILE.name}")
        print(f"Total samples:  {len(npx)}")
        print(f"Protein cols:   {len(protein_cols)}")

    return npx, pheno, protein_cols


# ─── Group assignment ─────────────────────────────────────────────────────────

def assign_groups(npx, pheno, verbose=True):
    """
    Assign each sample to Healthy, LC_Neuro, LC_NonNeuro, CACTUS, or Unknown.
    CACTUS and Unknown are excluded from all regression comparisons.
    """
    pheno_map = dict(zip(pheno["SampleID"].str.strip(), pheno["NEURO"].astype(int)))

    groups = {}
    for sid in npx.index:
        sid_clean = sid.strip()
        if sid_clean.startswith("HC"):
            groups[sid] = "Healthy"
        elif sid_clean in pheno_map:
            groups[sid] = "LC_Neuro" if pheno_map[sid_clean] == 1 else "LC_NonNeuro"
        elif sid_clean.startswith("CACTUS"):
            groups[sid] = "CACTUS"
        else:
            groups[sid] = "Unknown"

    npx["group"] = pd.Series(groups)

    if verbose:
        counts = npx["group"].value_counts()
        print(f"\n{'='*60}")
        print("COHORT INCLUSION")
        print(f"{'='*60}")
        print(f"  HC* → Healthy:       n={counts.get('Healthy',0)}")
        print(f"  COPL/CS* NEURO=1:    n={counts.get('LC_Neuro',0)}")
        print(f"  COPL/CS* NEURO=0:    n={counts.get('LC_NonNeuro',0)}")
        print(f"  CACTUS* (excluded):  n={counts.get('CACTUS',0)}")
        if counts.get("Unknown", 0):
            print(f"  Unknown (excluded):  n={counts.get('Unknown',0)}")

    return npx


# ─── Total abundance ──────────────────────────────────────────────────────────

def compute_total_abundance(npx, protein_cols, verbose=True):
    """
    total_abundance = nansum of all protein NPX values per sample.
    Computed once from the full panel and used as covariate in adjusted models.
    """
    npx["total_abundance"] = npx[protein_cols].apply(np.nansum, axis=1)

    if verbose:
        print(f"\n{'='*60}")
        print("TOTAL ABUNDANCE")
        print(f"{'='*60}")
        ta = npx.groupby("group")["total_abundance"].agg(["mean", "std", "min", "max"])
        print(ta.round(1).to_string())

    return npx


# ─── Single-protein logistic regression ──────────────────────────────────────

def run_logit(df_comp, protein, covariates, verbose=False):
    """
    Fit logistic regression: outcome ~ protein [+ covariates].

    Reference group is coded 0, test group is coded 1.
    Returns a dict with beta, OR, CI_lower, CI_upper, p_value, n, n_ref, n_test,
    fit_warning.  CI_lower and CI_upper are on the OR scale (not log-OR).
    """
    sub = df_comp[["outcome", protein] + covariates].dropna()
    n       = len(sub)
    n_ref   = (sub["outcome"] == 0).sum()
    n_test  = (sub["outcome"] == 1).sum()

    predictors = [protein] + covariates
    X = sm.add_constant(sub[predictors], has_constant="add")

    fit_warning = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sm.Logit(sub["outcome"], X).fit(
                method="newton", disp=False, maxiter=100, warn_convergence=False
            )
            if caught:
                fit_warning = True

        if not result.mle_retvals.get("converged", True):
            fit_warning = True

        beta  = result.params[protein]
        pval  = result.pvalues[protein]
        ci_lo = np.exp(result.conf_int().loc[protein, 0])
        ci_hi = np.exp(result.conf_int().loc[protein, 1])
        OR    = np.exp(beta)

    except Exception:
        beta, pval, OR, ci_lo, ci_hi = np.nan, np.nan, np.nan, np.nan, np.nan
        fit_warning = True

    return {
        "beta":        beta,
        "OR":          OR,
        "CI_lower":    ci_lo,
        "CI_upper":    ci_hi,
        "p_value":     pval,
        "n":           n,
        "n_ref":       n_ref,
        "n_test":      n_test,
        "fit_warning": fit_warning,
    }


# ─── Full comparison runner ───────────────────────────────────────────────────

def run_comparison(npx, protein_cols, comp, model_type, verbose=True):
    """
    Run one comparison (e.g. Healthy_vs_LC_Neuro) under one model type
    (univariate or total_abundance).

    Model formulas
    ──────────────
    univariate:       outcome ~ protein
    total_abundance:  outcome ~ protein + total_abundance

    Outcome coding: reference group = 0, test group = 1.
    FDR: Benjamini-Hochberg within comparison.
    Missing values: dropped listwise per protein.
    """
    ref_label  = comp["ref"]
    test_label = comp["test"]
    comp_label = comp["label"]

    covariates   = ["total_abundance"] if model_type == "total_abundance" else []
    covar_string = "total_abundance"   if model_type == "total_abundance" else "none"

    df_comp = npx[npx["group"].isin([ref_label, test_label])].copy()
    df_comp["outcome"] = (df_comp["group"] == test_label).astype(int)

    if verbose:
        print(f"\n  Comparison: {comp_label}  [{model_type}]")
        print(f"    Reference: {ref_label}  n={( df_comp['outcome']==0).sum()}")
        print(f"    Test:      {test_label}  n={( df_comp['outcome']==1).sum()}")
        print(f"    Formula:   outcome ~ protein" +
              (" + total_abundance" if covariates else ""))

    rows = []
    for prot in protein_cols:
        r = run_logit(df_comp, prot, covariates)
        r["protein"]    = prot
        r["comparison"] = comp_label
        r["covariates"] = covar_string
        rows.append(r)

    results = pd.DataFrame(rows)

    valid = results["p_value"].notna()
    fdr_vals = np.full(len(results), np.nan)
    if valid.sum() > 0:
        _, fdr_corrected, _, _ = multipletests(
            results.loc[valid, "p_value"], method="fdr_bh"
        )
        fdr_vals[valid.values] = fdr_corrected
    results["FDR"] = fdr_vals

    col_order = [
        "protein", "comparison", "beta", "OR", "CI_lower", "CI_upper",
        "p_value", "FDR", "n", "n_ref", "n_test", "covariates", "fit_warning",
    ]
    results = results[col_order].sort_values("p_value")

    return results


# ─── Validation mode ──────────────────────────────────────────────────────────

def validate_against_existing(results, existing_path, comparison_label,
                               model_type, test_proteins):
    """
    Compare reconstructed betas and p-values against the existing TSV output.
    """
    if not existing_path.exists():
        print(f"  [WARN] Existing file not found: {existing_path}")
        return

    existing = pd.read_csv(existing_path, sep="\t")
    label = f"{comparison_label} [{model_type}]"

    print(f"\n{'─'*60}")
    print(f"VALIDATION — {label}")
    print(f"  Existing file: {existing_path.name}")
    print(f"{'─'*60}")

    tol_beta = 1e-3
    tol_p    = 1e-4

    header = (f"  {'Protein':<14} {'Beta_new':>12} {'Beta_exist':>12} "
              f"{'Diff':>10} {'P_new':>12} {'P_exist':>12} {'Match':>6}")
    print(header)

    all_match = True
    for prot in test_proteins:
        row_new  = results[results["protein"].str.upper() == prot.upper()]
        row_exist = existing[existing["protein"].str.upper() == prot.upper()]

        if row_new.empty or row_exist.empty:
            print(f"  {prot:<14} {'NOT FOUND':>12}")
            all_match = False
            continue

        b_new = row_new.iloc[0]["beta"]
        b_ex  = row_exist.iloc[0]["beta"]
        p_new = row_new.iloc[0]["p_value"]
        p_ex  = row_exist.iloc[0]["p_value"]
        diff  = b_new - b_ex
        match = abs(diff) < tol_beta and abs(p_new - p_ex) < tol_p

        if not match:
            all_match = False

        print(f"  {prot:<14} {b_new:>12.6f} {b_ex:>12.6f} "
              f"{diff:>10.4f} {p_new:>12.6f} {p_ex:>12.6f} "
              f"{'OK' if match else 'MISMATCH':>6}")

    print(f"\n  Overall match: {'YES' if all_match else 'MISMATCH — see notes below'}")

    if not all_match:
        print("""
  Possible causes of mismatch:
    1. Reversed reference/test level encoding
       → check n_ref / n_test match and beta sign
    2. total_abundance computed differently
       → nansum vs. mean, or using per-protein available proteins
    3. Different NaN handling (imputation vs. listwise drop)
    4. Different optimiser (e.g. BFGS vs. Newton) or convergence settings
    5. Protein name case mismatch or extra whitespace in raw file
    6. Different sample subset (outlier inclusion/exclusion)
        """)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max_proteins", type=int, default=None,
        help="Run only the first N proteins (validation shortcut)"
    )
    parser.add_argument(
        "--test_proteins", type=str, default=None,
        help="Comma-separated list of proteins to run and validate "
             "(e.g. FGF23,TNFRSF11B,HGF)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print cohort info and sample counts only; do not fit models"
    )
    args = parser.parse_args()

    validation_mode  = args.test_proteins is not None or args.max_proteins is not None
    test_protein_list = (
        [p.strip().upper() for p in args.test_proteins.split(",")]
        if args.test_proteins else None
    )

    # ── Load and prepare data ─────────────────────────────────────────────────
    npx, pheno, protein_cols = load_data(verbose=True)
    npx = assign_groups(npx, pheno, verbose=True)
    npx = compute_total_abundance(npx, protein_cols, verbose=True)

    # Detect any missing values
    missing_summary = npx[protein_cols].isnull().sum()
    proteins_with_missing = missing_summary[missing_summary > 0]
    if len(proteins_with_missing):
        print(f"\nMissing values detected (dropped per-protein in regression):")
        for p, c in proteins_with_missing.items():
            print(f"  {p}: {c} missing")
    else:
        print("\nNo missing values in protein columns.")

    if args.dry_run:
        print("\n[DRY RUN] Stopping before model fitting.")
        return

    # ── Determine protein list for this run ───────────────────────────────────
    run_proteins = protein_cols
    if test_protein_list is not None:
        run_proteins = [p for p in protein_cols if p.upper() in test_protein_list]
        missing_req  = set(test_protein_list) - {p.upper() for p in run_proteins}
        if missing_req:
            print(f"\n[WARN] Requested proteins not found in NPX file: {missing_req}")
        print(f"\nValidation mode: running {len(run_proteins)} protein(s): {run_proteins}")
    elif args.max_proteins is not None:
        run_proteins = protein_cols[: args.max_proteins]
        print(f"\nValidation mode: running first {len(run_proteins)} protein(s)")

    # ── Run all comparisons and model types ───────────────────────────────────
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL SPECIFICATION")
    print(f"{'='*60}")
    print("  Univariate:        outcome ~ protein")
    print("  Abundance-adj:     outcome ~ protein + total_abundance")
    print("  Outcome coding:    0 = reference (Healthy), 1 = test (LC subtype)")
    print("  FDR method:        Benjamini-Hochberg within comparison")
    print("  Missing values:    listwise deletion per protein")
    print("  Optimiser:         Newton-Raphson (statsmodels Logit)")

    for comp in COMPARISONS:
        for model_type in comp["models"]:
            label      = comp["label"]
            out_suffix = "univariate" if model_type == "univariate" else "total_abundance_adjusted"
            out_path   = OUTDIR / f"regression_{label}_{out_suffix}.tsv"

            results = run_comparison(npx, run_proteins, comp, model_type, verbose=True)

            if validation_mode:
                val_proteins = test_protein_list if test_protein_list else [
                    p.upper() for p in run_proteins
                ]
                validate_against_existing(
                    results, out_path, label, model_type, val_proteins
                )
            else:
                results.to_csv(out_path, sep="\t", index=False)
                n_sig_fdr = (results["FDR"] < 0.05).sum()
                n_sig_p   = (results["p_value"] < 0.05).sum()
                print(f"    → Saved {len(results)} proteins | "
                      f"FDR<0.05: {n_sig_fdr} | p<0.05: {n_sig_p}")
                print(f"       {out_path.name}")

    if validation_mode:
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETE — no output files written.")
        print("Re-run without --test_proteins or --max_proteins to produce outputs.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
