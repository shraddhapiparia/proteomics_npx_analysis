"""
Per-protein logistic regression with Benjamini-Hochberg FDR.

Moved verbatim from primary_regression_analysis.py (refactor, Task 1).

Model:   outcome ~ protein [+ covariates], statsmodels Logit, Newton-Raphson.
Coding:  outcome = 0 for reference_group, 1 for test_group (LC_Neuro in all
         current comparisons), so beta > 0 means higher NPX in test_group.
Missing: listwise deletion per protein (protein + covariates).
FDR:     BH within one comparison, computed over EVALUABLE proteins only
         (p-value present, no fit warning, finite beta). Non-evaluable proteins
         get FDR = NaN and do not count towards the number of tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Covariate sets used by 02_protein_associations.py and 03_crosscenter_analysis.py.
PRIMARY_COV     = ["harmonized_age", "sex_01"]
SENSITIVITY_COV = ["harmonized_age", "sex_01", "total_abundance"]

# ── Single-protein logistic regression ────────────────────────────────────────

def run_logit(df_comp: pd.DataFrame, protein: str, covariates: list[str]) -> dict:
    """
    Fit logistic regression: outcome ~ protein [+ covariates].
    Reference group = 0; test group = 1.
    CI_lower / CI_upper are on the OR scale (exp of log-OR confidence interval).
    Missing values dropped listwise.
    """
    cols = ["outcome", protein] + covariates
    sub  = df_comp[cols].dropna()
    n      = len(sub)
    n_ref  = int((sub["outcome"] == 0).sum())
    n_test = int((sub["outcome"] == 1).sum())

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


# ── Comparison runner ──────────────────────────────────────────────────────────

def run_comparison(
    df: pd.DataFrame,
    protein_cols: list[str],
    ref_label: str,
    test_label: str,
    covariates: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run one comparison (ref vs test) across all proteins.
    Outcome: 0 = ref_label, 1 = test_label.
    FDR: Benjamini-Hochberg within this comparison.
    Missing values dropped listwise per protein.

    Output columns:
      protein, comparison, reference_group, test_group,
      beta, OR, CI_lower, CI_upper, p_value, FDR,
      n, n_ref, n_test, covariates, fit_warning
    """
    comp_label = f"{ref_label}_vs_{test_label}"
    covar_str  = ",".join(covariates) if covariates else "none"

    mask   = df["group"].isin([ref_label, test_label])
    df_sub = df[mask].copy()
    df_sub["outcome"] = (df_sub["group"] == test_label).astype(int)

    n_ref_tot  = int((df_sub["outcome"] == 0).sum())
    n_test_tot = int((df_sub["outcome"] == 1).sum())

    if verbose:
        print(f"\n  {comp_label}  [{covar_str}]")
        print(f"    Reference (0): {ref_label:<26} n={n_ref_tot}")
        print(f"    Test      (1): {test_label:<26} n={n_test_tot}")

    rows = []
    for prot in protein_cols:
        r = run_logit(df_sub, prot, covariates)
        r.update(protein=prot, comparison=comp_label,
                 reference_group=ref_label, test_group=test_label,
                 covariates=covar_str)
        rows.append(r)

    results = pd.DataFrame(rows)

    # evaluable: p_value present, no fit_warning, and beta is finite
    evaluable = (
        results["p_value"].notna()
        & ~results["fit_warning"].astype(bool)
        & results["beta"].apply(lambda x: bool(np.isfinite(x)) if pd.notna(x) else False)
    )
    results["evaluable"] = evaluable

    fdr_vals = np.full(len(results), np.nan)
    if evaluable.sum() > 0:
        _, fdr_corr, _, _ = multipletests(
            results.loc[evaluable, "p_value"], method="fdr_bh"
        )
        fdr_vals[evaluable.values] = fdr_corr
    results["FDR"] = fdr_vals

    col_order = [
        "protein", "comparison", "reference_group", "test_group",
        "beta", "OR", "CI_lower", "CI_upper",
        "p_value", "FDR", "n", "n_ref", "n_test", "covariates",
        "fit_warning", "evaluable",
    ]
    results = results[col_order].sort_values("p_value")

    if verbose:
        n_eval    = int(evaluable.sum())
        n_warn    = int(results["fit_warning"].sum())
        n_sig_p   = int((results.loc[evaluable, "p_value"] < 0.05).sum())
        n_sig_fdr = int((results.loc[evaluable, "FDR"] < 0.05).sum())
        print(f"    Proteins: {len(results)} | evaluable: {n_eval} | "
              f"fit_warning: {n_warn} | p<0.05 (eval): {n_sig_p} | "
              f"FDR<0.05 (eval): {n_sig_fdr}")

    return results
