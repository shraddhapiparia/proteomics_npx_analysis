"""
Pediatric Long COVID — LC_Neuro vs LC_NonNeuro subtype regression.

Separate script from pediatric_proteomics_regression.py.
Only handles the within-LC subtype comparison.

Purpose:
  Compare univariate and age+sex-adjusted models to investigate why the
  subtype analysis changed from 2 nominally significant proteins (prior run)
  to 4 nominally significant proteins (current file).

Input files:
  data/raw/Inflammation_NPX_AgeM.txt
  data/raw/phenotype.txt

Output files:
  results/pediatric/regression_LC_NonNeuro_vs_LC_Neuro_univariate.tsv
  results/pediatric/regression_LC_NonNeuro_vs_LC_Neuro_age_sex_adjusted.tsv

Outcome encoding (explicit):
  Reference (outcome = 0): LC_NonNeuro   n = 18
  Test      (outcome = 1): LC_Neuro      n = 16
  Positive beta = higher protein level associated with being LC_Neuro.

  Comparison label: LC_NonNeuro_vs_LC_Neuro
  (left side = reference, consistent with Healthy-vs-LC naming convention)

Age/sex sources:
  Both from phenotype.txt columns Age and Gender (F/M).
  Sex encoded as: F = 1, M = 0  (sex_01).
  All 34 COPL/CS* samples have complete age and sex — no samples dropped.

Usage:
  python scripts/pediatric_subtype_regression.py
  python scripts/pediatric_subtype_regression.py --test_proteins FGF21,CCL2,TNFRSF11B,IL17C
  python scripts/pediatric_subtype_regression.py --validate_existing
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, fisher_exact

ROOT      = Path(__file__).resolve().parent.parent
NPX_FILE  = ROOT / "data/raw/Inflammation_NPX_AgeM.txt"
PHENO_FILE = ROOT / "data/raw/phenotype.txt"
OUTDIR    = ROOT / "results/pediatric"

EXISTING_FILE  = OUTDIR / "regression_LC_NonNeuro_vs_LC_Neuro.tsv"
DEMOG_FILE     = OUTDIR / "subtype_demographics.tsv"


# ─── Demographics table ───────────────────────────────────────────────────────

def make_demographics_table(df, save_path=None):
    """
    Build a paper-style demographic summary for the two LC subtype groups.

    Rows:    n | Age (years) | Female | Male
    Columns: variable | LC_Neuro | LC_NonNeuro | test | statistic | p_value

    Age:  two-sided Welch t-test (unequal variances)
    Sex:  two-sided Fisher exact test on 2×2 count table
    """
    neuro    = df[df["group"] == "LC_Neuro"]
    nonneuro = df[df["group"] == "LC_NonNeuro"]

    n_ne  = len(neuro)
    n_nn  = len(nonneuro)

    # Age
    age_ne  = neuro["age"].dropna()
    age_nn  = nonneuro["age"].dropna()
    t_stat, p_age = ttest_ind(age_ne, age_nn, equal_var=False)
    age_ne_str  = f"{age_ne.mean():.2f} \u00b1 {age_ne.std(ddof=1):.2f}"
    age_nn_str  = f"{age_nn.mean():.2f} \u00b1 {age_nn.std(ddof=1):.2f}"

    # Sex counts
    f_ne = int((neuro["sex_01"]    == 1).sum())
    m_ne = int((neuro["sex_01"]    == 0).sum())
    f_nn = int((nonneuro["sex_01"] == 1).sum())
    m_nn = int((nonneuro["sex_01"] == 0).sum())

    # Fisher exact on 2×2 table: [[F_Neuro, F_NonNeuro], [M_Neuro, M_NonNeuro]]
    table_2x2 = [[f_ne, f_nn], [m_ne, m_nn]]
    odds_ratio, p_sex = fisher_exact(table_2x2, alternative="two-sided")

    f_ne_str = f"{f_ne} ({100*f_ne/n_ne:.1f}%)"
    m_ne_str = f"{m_ne} ({100*m_ne/n_ne:.1f}%)"
    f_nn_str = f"{f_nn} ({100*f_nn/n_nn:.1f}%)"
    m_nn_str = f"{m_nn} ({100*m_nn/n_nn:.1f}%)"

    rows = [
        dict(variable="n",
             LC_Neuro=str(n_ne), LC_NonNeuro=str(n_nn),
             test="—", statistic="—", p_value="—"),
        dict(variable="Age (years)",
             LC_Neuro=age_ne_str, LC_NonNeuro=age_nn_str,
             test="Welch t-test", statistic=f"{t_stat:.3f}", p_value=f"{p_age:.3f}"),
        dict(variable="Female",
             LC_Neuro=f_ne_str, LC_NonNeuro=f_nn_str,
             test="Fisher exact", statistic=f"{odds_ratio:.3f}", p_value=f"{p_sex:.3f}"),
        dict(variable="Male",
             LC_Neuro=m_ne_str, LC_NonNeuro=m_nn_str,
             test="Fisher exact", statistic=f"{odds_ratio:.3f}", p_value=f"{p_sex:.3f}"),
    ]

    table = pd.DataFrame(rows, columns=["variable", "LC_Neuro", "LC_NonNeuro",
                                         "test", "statistic", "p_value"])

    print(f"\n{'='*60}")
    print("DEMOGRAPHIC SUMMARY — LC_Neuro vs LC_NonNeuro")
    print(f"{'='*60}")
    print(f"\n  {'Variable':<16} {'LC_Neuro':>18} {'LC_NonNeuro':>18} "
          f"{'Test':>14} {'Statistic':>10} {'p':>8}")
    print(f"  {'-'*86}")
    for _, r in table.iterrows():
        print(f"  {r['variable']:<16} {r['LC_Neuro']:>18} {r['LC_NonNeuro']:>18} "
              f"{r['test']:>14} {r['statistic']:>10} {r['p_value']:>8}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, sep="\t", index=False)
        print(f"\n  Saved: {save_path}")

    return table, float(p_age), float(p_sex), float(t_stat), float(odds_ratio)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_subtype_data(verbose=True):
    npx   = pd.read_csv(NPX_FILE,  sep="\t", index_col="SampleID")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")

    protein_cols = list(npx.columns)

    # Assign groups: COPL/CS* only, using NEURO flag
    pheno["sex_01"] = (pheno["Gender"] == "F").astype(int)
    pheno_indexed   = pheno.set_index("SampleID")

    # Build merged dataframe for LC subjects only
    lc_ids = [sid for sid in npx.index
               if sid.strip() in pheno_indexed.index]

    df = npx.loc[lc_ids].copy()
    df["group"]   = df.index.map(
        lambda s: "LC_Neuro" if pheno_indexed.loc[s.strip(), "NEURO"] == 1
                  else "LC_NonNeuro"
    )
    df["age"]     = df.index.map(lambda s: pheno_indexed.loc[s.strip(), "Age"])
    df["sex_01"]  = df.index.map(lambda s: pheno_indexed.loc[s.strip(), "sex_01"])

    # outcome: 0 = LC_NonNeuro (reference), 1 = LC_Neuro (test)
    df["outcome"] = (df["group"] == "LC_Neuro").astype(int)

    n_neuro    = (df["group"] == "LC_Neuro").sum()
    n_nonneuro = (df["group"] == "LC_NonNeuro").sum()

    if verbose:
        print(f"\n{'='*60}")
        print("SUBTYPE DATA LOADING")
        print(f"{'='*60}")
        print(f"  Samples included: {len(df)}")
        print(f"    LC_Neuro   (outcome=1): {n_neuro}")
        print(f"    LC_NonNeuro (outcome=0): {n_nonneuro}")
        print(f"  Protein columns: {len(protein_cols)}")
        print(f"  Age range: {df['age'].min():.2f} – {df['age'].max():.2f} years")
        print(f"  Sex (sex_01): F=1, M=0")
        print(f"    Female (sex_01=1): {(df['sex_01']==1).sum()}")
        print(f"    Male   (sex_01=0): {(df['sex_01']==0).sum()}")

        # Check missing
        missing = df[protein_cols].isnull().sum()
        miss_prots = missing[missing > 0]
        if len(miss_prots):
            print(f"\n  Missing values per protein:")
            for p, c in miss_prots.items():
                print(f"    {p}: {c} missing (listwise dropped for that protein)")
        else:
            print(f"\n  No missing values in protein columns.")

        print(f"\n  Missing age:    {df['age'].isna().sum()}")
        print(f"  Missing sex_01: {df['sex_01'].isna().sum()}")

    return df, protein_cols


# ─── Single-protein logistic regression ──────────────────────────────────────

def run_logit(df, protein, covariates):
    sub = df[["outcome", protein] + covariates].dropna()
    n       = len(sub)
    n_ref   = (sub["outcome"] == 0).sum()
    n_test  = (sub["outcome"] == 1).sum()

    X = sm.add_constant(sub[[protein] + covariates], has_constant="add")
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

    return dict(beta=beta, OR=OR, CI_lower=ci_lo, CI_upper=ci_hi,
                p_value=pval, n=n, n_ref=n_ref, n_test=n_test,
                fit_warning=fit_warning)


# ─── Full comparison runner ───────────────────────────────────────────────────

def run_comparison(df, proteins, model_type, verbose=True):
    """
    model_type: 'univariate' or 'age_sex'

    Models
    ──────
    univariate:   outcome ~ protein
    age_sex:      outcome ~ protein + age + sex_01
    """
    covariates   = ["age", "sex_01"] if model_type == "age_sex" else []
    covar_string = "age,sex_01"      if model_type == "age_sex" else "none"
    comp_label   = "LC_NonNeuro_vs_LC_Neuro"

    if verbose:
        formula = "outcome ~ protein" + (" + age + sex_01" if covariates else "")
        print(f"\n  [{model_type}] formula: {formula}")

    rows = []
    for prot in proteins:
        r = run_logit(df, prot, covariates)
        r.update(protein=prot, comparison=comp_label, covariates=covar_string)
        rows.append(r)

    results = pd.DataFrame(rows)

    valid = results["p_value"].notna()
    fdr_vals = np.full(len(results), np.nan)
    if valid.sum() > 0:
        _, fdr_corr, _, _ = multipletests(results.loc[valid, "p_value"],
                                           method="fdr_bh")
        fdr_vals[valid.values] = fdr_corr
    results["FDR"] = fdr_vals

    col_order = ["protein", "comparison", "beta", "OR", "CI_lower", "CI_upper",
                 "p_value", "FDR", "n", "n_ref", "n_test", "covariates", "fit_warning"]
    return results[col_order].sort_values("p_value")


# ─── Comparison table ─────────────────────────────────────────────────────────

def make_comparison_table(uni, age_sex, proteins):
    rows = []
    for prot in proteins:
        r_u  = uni[uni["protein"]    == prot]
        r_as = age_sex[age_sex["protein"] == prot]

        b_u  = r_u.iloc[0]["beta"]    if not r_u.empty  else np.nan
        p_u  = r_u.iloc[0]["p_value"] if not r_u.empty  else np.nan
        b_as = r_as.iloc[0]["beta"]   if not r_as.empty else np.nan
        p_as = r_as.iloc[0]["p_value"]if not r_as.empty else np.nan

        sig_u  = (not np.isnan(p_u))  and (p_u  < 0.05)
        sig_as = (not np.isnan(p_as)) and (p_as < 0.05)

        if not (np.isnan(b_u) or np.isnan(b_as)):
            dir_same = "Yes" if np.sign(b_u) == np.sign(b_as) else "No"
        else:
            dir_same = "NA"

        rows.append(dict(
            protein=prot,
            beta_univariate=b_u,  p_univariate=p_u,
            beta_age_sex=b_as,    p_age_sex=p_as,
            nominal_sig_univariate=sig_u,
            nominal_sig_age_sex=sig_as,
            direction_same=dir_same,
        ))
    return pd.DataFrame(rows)


# ─── Validate against existing file ──────────────────────────────────────────

def validate_against_existing(age_sex_results, test_proteins, tol_beta=1e-3, tol_p=1e-4):
    if not EXISTING_FILE.exists():
        print(f"  [WARN] existing file not found: {EXISTING_FILE}")
        return

    existing = pd.read_csv(EXISTING_FILE, sep="\t")
    print(f"\n{'─'*60}")
    print(f"VALIDATION — age_sex results vs existing TSV")
    print(f"  File: {EXISTING_FILE.name}  (covariates: {existing['covariates'].iloc[0]})")
    print(f"{'─'*60}")
    print(f"  {'Protein':<14} {'Beta_new':>10} {'Beta_old':>10} {'Diff':>9} "
          f"{'P_new':>10} {'P_old':>10} {'Match':>6}")

    all_match = True
    for prot in test_proteins:
        r_new = age_sex_results[age_sex_results["protein"].str.upper() == prot.upper()]
        r_old = existing[existing["protein"].str.upper() == prot.upper()]
        if r_new.empty or r_old.empty:
            print(f"  {prot:<14} {'NOT FOUND':>10}")
            all_match = False
            continue
        b_n, b_o = r_new.iloc[0]["beta"], r_old.iloc[0]["beta"]
        p_n, p_o = r_new.iloc[0]["p_value"], r_old.iloc[0]["p_value"]
        diff = b_n - b_o
        match = abs(diff) < tol_beta and abs(p_n - p_o) < tol_p
        if not match:
            all_match = False
        print(f"  {prot:<14} {b_n:>10.4f} {b_o:>10.4f} {diff:>9.4f} "
              f"{p_n:>10.6f} {p_o:>10.6f} {'OK' if match else 'MISMATCH':>6}")

    print(f"\n  Overall: {'EXACT MATCH' if all_match else 'MISMATCH — check sex encoding or optimizer'}")
    return all_match


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_proteins", type=str, default=None,
                        help="Comma-separated protein names (targeted run)")
    parser.add_argument("--validate_existing", action="store_true",
                        help="After running age+sex model, validate against existing TSV")
    parser.add_argument("--full_proteome", action="store_true",
                        help="Run all 87 proteins (default: targeted proteins only)")
    args = parser.parse_args()

    df, protein_cols = load_subtype_data(verbose=True)

    # ── Demographics table (always written) ────────────────────────────────────
    make_demographics_table(df, save_path=DEMOG_FILE)

    # ── Determine which proteins to run ───────────────────────────────────────
    # Default: the 4 proteins nominally significant in the existing age+sex file,
    # which is the union of the "2 old" + "4 current" candidates we want to probe.
    default_targets = ["FGF21", "CCL2", "TNFRSF11B", "IL17C"]

    if args.full_proteome:
        run_proteins = protein_cols
        print(f"\nFull proteome mode: {len(run_proteins)} proteins")
    elif args.test_proteins:
        requested = [p.strip().upper() for p in args.test_proteins.split(",")]
        run_proteins = [p for p in protein_cols if p.upper() in requested]
        missing = set(requested) - {p.upper() for p in run_proteins}
        if missing:
            print(f"\n[WARN] Requested proteins not in NPX file: {missing}")
        print(f"\nTargeted mode: {len(run_proteins)} protein(s): {run_proteins}")
    else:
        run_proteins = [p for p in protein_cols if p.upper() in
                        [t.upper() for t in default_targets]]
        print(f"\nDefault targeted mode: {len(run_proteins)} proteins "
              f"(nominally significant in existing age+sex file): {run_proteins}")

    # ── Model formulas ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL SPECIFICATION")
    print(f"{'='*60}")
    print("  Outcome: 0 = LC_NonNeuro (reference), 1 = LC_Neuro (test)")
    print("  Positive beta → higher protein → more likely LC_Neuro")
    print("  Comparison label: LC_NonNeuro_vs_LC_Neuro")
    print("  FDR: Benjamini-Hochberg within model")
    print()
    print("  univariate:    outcome ~ protein")
    print("  age_sex:       outcome ~ protein + age + sex_01")
    print("    age:     continuous (years), from phenotype.txt Age column")
    print("    sex_01:  F = 1, M = 0, from phenotype.txt Gender column")

    # ── Run both models ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING MODELS")
    print(f"{'='*60}")

    uni    = run_comparison(df, run_proteins, "univariate",  verbose=True)
    age_sex = run_comparison(df, run_proteins, "age_sex",    verbose=True)

    # ── Validate age+sex against existing file ─────────────────────────────────
    if args.validate_existing or not args.full_proteome:
        validate_against_existing(age_sex, [p for p in run_proteins],
                                   tol_beta=1e-3, tol_p=1e-4)

    # ── Comparison table ───────────────────────────────────────────────────────
    comp_table = make_comparison_table(uni, age_sex, run_proteins)

    print(f"\n{'='*60}")
    print("COMPARISON TABLE — univariate vs age+sex adjusted")
    print(f"{'='*60}")
    print(f"\n  {'Protein':<14} {'Beta_uni':>10} {'P_uni':>10} "
          f"{'Beta_as':>10} {'P_as':>10} "
          f"{'Sig_uni':>8} {'Sig_as':>8} {'DirSame':>8}")
    for _, r in comp_table.iterrows():
        def fp(x):
            return f"{x:.4f}" if not (isinstance(x, float) and np.isnan(x)) else "NaN"
        sig_u  = "Yes" if r["nominal_sig_univariate"] else "No"
        sig_as = "Yes" if r["nominal_sig_age_sex"]    else "No"
        print(f"  {r['protein']:<14} {fp(r['beta_univariate']):>10} {fp(r['p_univariate']):>10} "
              f"{fp(r['beta_age_sex']):>10} {fp(r['p_age_sex']):>10} "
              f"{sig_u:>8} {sig_as:>8} {r['direction_same']:>8}")

    n_sig_uni = comp_table["nominal_sig_univariate"].sum()
    n_sig_as  = comp_table["nominal_sig_age_sex"].sum()
    print(f"\n  p<0.05 in univariate: {n_sig_uni}/{len(run_proteins)}")
    print(f"  p<0.05 in age+sex:    {n_sig_as}/{len(run_proteins)}")

    gained = comp_table[comp_table["nominal_sig_age_sex"] & ~comp_table["nominal_sig_univariate"]]
    lost   = comp_table[~comp_table["nominal_sig_age_sex"] & comp_table["nominal_sig_univariate"]]
    both   = comp_table[comp_table["nominal_sig_age_sex"] & comp_table["nominal_sig_univariate"]]

    if not both.empty:
        print(f"\n  Significant in BOTH models: {both['protein'].tolist()}")
    if not gained.empty:
        print(f"  Gained with age+sex (not in univariate): {gained['protein'].tolist()}")
    if not lost.empty:
        print(f"  Lost with age+sex (significant in univariate only): {lost['protein'].tolist()}")

    # ── Save outputs (full proteome only) ────────────────────────────────────
    if args.full_proteome:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        uni_path    = OUTDIR / "regression_LC_NonNeuro_vs_LC_Neuro_univariate.tsv"
        as_path     = OUTDIR / "regression_LC_NonNeuro_vs_LC_Neuro_age_sex_adjusted.tsv"
        uni.to_csv(uni_path,     sep="\t", index=False)
        age_sex.to_csv(as_path,  sep="\t", index=False)
        print(f"\n  Saved: {uni_path.name}")
        print(f"  Saved: {as_path.name}")
        n_sig_fdr_uni = (uni["FDR"] < 0.05).sum()
        n_sig_fdr_as  = (age_sex["FDR"] < 0.05).sum()
        print(f"\n  FDR<0.05 univariate: {n_sig_fdr_uni}")
        print(f"  FDR<0.05 age+sex:    {n_sig_fdr_as}")
    else:
        print(f"\n  [Targeted mode — no files written]")
        print(f"  Run with --full_proteome to write output TSVs.")


if __name__ == "__main__":
    main()
