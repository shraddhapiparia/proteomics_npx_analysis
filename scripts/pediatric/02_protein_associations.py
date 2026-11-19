"""
Entry point 2 of 5 — Protein associations within the Long COVID cohort.

All analyses here are within one recruitment center (Gemelli) on uncorrected
(raw) Olink NPX, except where stated. Run every stage, or pick with --stage:

  primary      Primary analysis. LC_Neuro (n=16) vs LC_NonNeuro_strict (n=18),
               outcome ~ protein + harmonized_age + sex_01 (logistic), all 87
               proteins, BH FDR over evaluable proteins.
               -> regression_LC_NonNeuro_strict_vs_LC_Neuro_age_sex_adjusted.tsv
  firth        Firth logistic regression for CCL2, IL17C, TNFRSF11B, FGF21
               (firth_regression.R): LC strict comparison on raw NPX, plus
               LC_Neuro vs Healthy on ComBat NPX.
               -> firth_candidate_sensitivity.tsv
  fgf21        FGF21 adjusted for fatigue and for non-neurological symptom
               burden (N SYSTEMS); fatigue prevalence by subtype.
               -> fgf21_fatigue_prevalence.tsv, fgf21_sensitivity_logistic.tsv
  permutation  LC-only Freedman–Lane permutation test (linear model
               NPX ~ age + sex + group, 10,000 permutations, seed 42).
               -> permutation_lc_neuro_strict_raw_npx.tsv
  sex          Sex effects on the key proteins (reviewer request).
               -> sex_effect_check/

Inputs: data/raw/Inflammation_NPX_AgeM.txt, data/raw/phenotype.txt,
data/interim/harmonized_metadata.tsv (from 01_cohort_qc.py; used by firth, sex)
and, for firth only, <results_dir>/combat_adjusted_npx.tsv (from
03_crosscenter_analysis.py --stage combat).

Existing outputs are never overwritten unless --overwrite is given; the check
runs for all selected stages before anything is computed.

Usage
-----
  python scripts/pediatric/02_protein_associations.py --outdir results/pediatric/rerun
  python scripts/pediatric/02_protein_associations.py --stage primary permutation --outdir /tmp/x
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import cohort  # noqa: E402
from logistic_models import PRIMARY_COV, run_comparison  # noqa: E402
from output_safety import check_outputs_writable  # noqa: E402

ROOT = cohort.ROOT
DEFAULT_METADATA = ROOT / "data" / "interim" / "harmonized_metadata.tsv"
CANDIDATE_PROTEINS = ["FGF21", "CCL2", "IL17C", "TNFRSF11B"]

# Output names per stage (files, or a sub-directory for 'sex').
STAGE_OUTPUTS = {
    "primary":     ["regression_LC_NonNeuro_strict_vs_LC_Neuro_age_sex_adjusted.tsv"],
    "firth":       ["firth_candidate_sensitivity.tsv"],
    "fgf21":       ["fgf21_fatigue_prevalence.tsv", "fgf21_sensitivity_logistic.tsv"],
    "permutation": ["permutation_lc_neuro_strict_raw_npx.tsv"],
    "sex":         ["sex_effect_check"],
}


# ═════════════════════════════════════════════════════════════════════════════
# Stage: primary — LC_Neuro vs LC_NonNeuro_strict, age/sex-adjusted logistic
# ═════════════════════════════════════════════════════════════════════════════

# Proteins with near-constant NPX in the LC arms (reported as NE); diagnostic only.
SPARSE_PROTEIN_CHECK = ["FGF23", "IL20", "IL22RA1", "ARTN"]


def print_sparse_protein_check(df: pd.DataFrame) -> None:
    """Per-group distribution of SPARSE_PROTEIN_CHECK proteins in the LC arms."""
    check = df[df["group"].isin(["LC_Neuro", "LC_NonNeuro_strict"])][
        ["SampleID", "group", "harmonized_age", "sex_01"] + SPARSE_PROTEIN_CHECK
    ]
    print(check.to_string(index=False))
    for protein in SPARSE_PROTEIN_CHECK:
        print("\n", "=" * 60)
        print(protein)
        print(check.groupby("group")[protein].agg(
            ["count", "mean", "std", "min", "median", "max"]))
        print("Unique values:", check[protein].nunique())


def run_primary(outdir: Path, npx_file: Path, pheno_file: Path,
                max_proteins: int | None = None, sparse_check: bool = False) -> None:
    """Primary model on the prepared cohort after the |total-abundance z| > 3
    exclusion (z over all 112 samples; removes CACTUS154_T0 only).
    Outcome 0 = LC_NonNeuro_strict, 1 = LC_Neuro; sex_01 1 = female."""
    df, protein_cols, _ = cohort.prepare_cohort(verbose=True, npx_file=npx_file,
                                                pheno_file=pheno_file)
    df, _ = cohort.exclude_total_abundance_outliers(df, verbose=True)
    proteins = protein_cols if max_proteins is None else protein_cols[:max_proteins]
    print(f"\nPRIMARY: outcome ~ protein + harmonized_age + sex_01 "
          f"({len(proteins)} proteins, BH FDR over evaluable proteins)")
    results = run_comparison(df, proteins, "LC_NonNeuro_strict", "LC_Neuro",
                             PRIMARY_COV, verbose=True)
    if max_proteins is None:
        out_path = outdir / STAGE_OUTPUTS["primary"][0]
        results.to_csv(out_path, sep="\t", index=False)
        print(f"    → Saved: {out_path}")
    else:
        print("    → [--max_proteins] file write skipped")
    if sparse_check:
        print_sparse_protein_check(df)


# ═════════════════════════════════════════════════════════════════════════════
# Stage: firth — validated R implementation (logistf)
# ═════════════════════════════════════════════════════════════════════════════

def run_firth(outdir: Path, metadata_file: Path, combat_file: Path) -> None:
    """Firth logistic regression via firth_regression.R (run from the repo root)."""
    cmd = ["Rscript", str(HERE / "firth_regression.R"),
           "--outdir", str(outdir), "--meta-file", str(metadata_file),
           "--combat-file", str(combat_file)]
    print("\nFIRTH: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# Stage: fgf21 — fatigue and non-neurological symptom-burden sensitivity
# ═════════════════════════════════════════════════════════════════════════════

FGF21 = "FGF21"
FGF21_DOB_FORMATS = ["%d/%m/%Y", "%d/%m/%y"]
FGF21_VIAL_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%y"]


def parse_first_date(date_str, formats):
    """Return the first successfully parsed date (None if none parse)."""
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def load_fgf21_lc_frame(npx_file: Path, pheno_file: Path) -> pd.DataFrame:
    """The 34 LC participants with FGF21, fatigue, N SYSTEMS, sex and age.

    harmonized_age = (Data Vial - DOB) / 365.25 with the LC date formats.
    non_neuro_systems = N SYSTEMS - LC_Neuro (the neurological system is counted
    in N SYSTEMS for LC-Neuro participants, so one system is subtracted).
    """
    npx = pd.read_csv(npx_file, sep="\t")
    pheno = pd.read_csv(pheno_file, sep="\t")
    npx["SampleID"] = npx["SampleID"].str.strip()
    pheno["SampleID"] = pheno["SampleID"].str.strip()

    required = {"SampleID", "Group", "NEURO", "FATIGUE", "N SYSTEMS", "Gender", "DOB", "Data Vial"}
    missing = required - set(pheno.columns)
    if missing:
        raise ValueError(f"Missing phenotype columns: {sorted(missing)}")
    if FGF21 not in npx.columns:
        raise ValueError(f"{FGF21} not found in NPX file")

    lc_meta = pheno.loc[pheno["Group"].eq("LC"),
                        ["SampleID", "NEURO", "FATIGUE", "N SYSTEMS", "Gender", "DOB", "Data Vial"]].copy()
    df = lc_meta.merge(npx[["SampleID", FGF21]], on="SampleID", how="inner")

    df["LC_Neuro"] = pd.to_numeric(df["NEURO"], errors="coerce")       # 1 = LC-Neuro
    df["fatigue"] = pd.to_numeric(df["FATIGUE"], errors="coerce")
    df["n_systems"] = pd.to_numeric(df["N SYSTEMS"], errors="coerce")
    df["female"] = df["Gender"].astype(str).str.upper().str.startswith("F").astype(int)
    df["non_neuro_systems"] = df["n_systems"] - df["LC_Neuro"]

    harmonized_age = []
    for _, row in df.iterrows():
        dob = parse_first_date(row["DOB"], FGF21_DOB_FORMATS)
        vial = parse_first_date(row["Data Vial"], FGF21_VIAL_FORMATS)
        harmonized_age.append(np.nan if dob is None or vial is None
                              else (vial - dob).days / 365.25)
    df["harmonized_age"] = harmonized_age
    return df


def summarize_fatigue_and_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Fatigue prevalence by subtype (Fisher exact test); prints symptom burden."""
    fatigue_table = pd.crosstab(df["LC_Neuro"], df["fatigue"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0)
    nonneuro_not = int(fatigue_table.loc[0, 0])
    nonneuro_yes = int(fatigue_table.loc[0, 1])
    neuro_not = int(fatigue_table.loc[1, 0])
    neuro_yes = int(fatigue_table.loc[1, 1])
    _, fisher_p = fisher_exact([[neuro_yes, neuro_not], [nonneuro_yes, nonneuro_not]])

    fatigue_summary = pd.DataFrame([
        {"group": "LC-Neuro", "fatigue_n": neuro_yes, "total_n": neuro_yes + neuro_not,
         "fatigue_percent": 100 * neuro_yes / (neuro_yes + neuro_not)},
        {"group": "LC-NonNeuro", "fatigue_n": nonneuro_yes, "total_n": nonneuro_yes + nonneuro_not,
         "fatigue_percent": 100 * nonneuro_yes / (nonneuro_yes + nonneuro_not)},
    ])
    fatigue_summary["fisher_exact_p"] = fisher_p
    print("\nFatigue prevalence")
    print(fatigue_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    burden_summary = (df.groupby("LC_Neuro")[["n_systems", "non_neuro_systems"]]
                      .agg(["mean", "std", "median", "min", "max"]))
    print("\nSymptom burden")
    print(burden_summary.round(2).to_string())
    return fatigue_summary


def fit_fgf21_logistic(df: pd.DataFrame, model_name: str, extra_covariates: list[str]) -> dict:
    """LC_Neuro ~ FGF21 + [extra covariates] + harmonized_age + female (Newton)."""
    predictors = [FGF21] + extra_covariates + ["harmonized_age", "female"]
    analysis_df = df[["LC_Neuro"] + predictors].dropna().copy()
    y = analysis_df["LC_Neuro"].astype(int)
    X = sm.add_constant(analysis_df[predictors].astype(float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X).fit(method="newton", maxiter=100, disp=False)
    beta = model.params[FGF21]
    ci_low, ci_high = model.conf_int().loc[FGF21]
    return {
        "model": model_name,
        "FGF21_beta": beta,
        "FGF21_OR": np.exp(beta),
        "FGF21_CI_low": np.exp(ci_low),
        "FGF21_CI_high": np.exp(ci_high),
        "FGF21_p_value": model.pvalues[FGF21],
        "n": int(model.nobs),
        "df_resid": int(model.df_resid),
        "converged": bool(model.mle_retvals.get("converged", True)),
    }


def run_fgf21(outdir: Path, npx_file: Path, pheno_file: Path) -> None:
    df = load_fgf21_lc_frame(npx_file, pheno_file)
    print("\nFGF21 SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Total Long COVID participants: {len(df)}")
    print(f"LC-Neuro: {(df['LC_Neuro'] == 1).sum()}")
    print(f"LC-NonNeuro: {(df['LC_Neuro'] == 0).sum()}")

    summarize_fatigue_and_burden(df).to_csv(
        outdir / "fgf21_fatigue_prevalence.tsv", sep="\t", index=False)

    results = pd.DataFrame([
        fit_fgf21_logistic(df, "Primary: FGF21 + age + sex", []),
        fit_fgf21_logistic(df, "Sensitivity: + fatigue", ["fatigue"]),
        fit_fgf21_logistic(df, "Sensitivity: + non-neuro symptom burden", ["non_neuro_systems"]),
    ])
    results.to_csv(outdir / "fgf21_sensitivity_logistic.tsv", sep="\t", index=False)

    print("\nFGF21 association with LC-Neuro")
    print("=" * 60)
    cols = ["model", "FGF21_OR", "FGF21_CI_low", "FGF21_CI_high", "FGF21_p_value", "n", "converged"]
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ═════════════════════════════════════════════════════════════════════════════
# Stage: permutation — LC-only Freedman–Lane permutation (linear model)
# ═════════════════════════════════════════════════════════════════════════════

PERMUTATION_N = 10000
PERMUTATION_SEED = 42


def load_lc_permutation_frame(npx_file: Path, pheno_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """34 LC participants: group = NEURO (1/0), age = provided Age column,
    sex = 0 female / 1 male, plus all 87 raw NPX proteins."""
    npx = pd.read_csv(npx_file, sep="\t")
    pheno = pd.read_csv(pheno_file, sep="\t")

    pheno = pheno.loc[pheno["Group"] == "LC"].copy()
    pheno["group"] = pd.to_numeric(pheno["NEURO"], errors="coerce")
    pheno["age"] = pd.to_numeric(pheno["Age"], errors="coerce")
    pheno["sex"] = pheno["Gender"].str.strip().str.upper().map({"F": 0, "M": 1})
    proteins = [c for c in npx.columns if c != "SampleID"]

    assert pheno["SampleID"].is_unique
    assert npx["SampleID"].is_unique
    df = pheno.merge(npx[["SampleID"] + proteins], on="SampleID", how="inner",
                     validate="one_to_one")
    assert len(df) == 34, f"Expected 34 LC participants, found {len(df)}"
    assert (df["group"] == 1).sum() == 16
    assert (df["group"] == 0).sum() == 18
    assert df[["group", "age", "sex"]].notna().all().all()
    assert len(proteins) == 87, f"Expected 87 proteins, found {len(proteins)}"
    return df, proteins


def freedman_lane_group_test(y: np.ndarray, Z: np.ndarray, X: np.ndarray,
                             rng: np.random.Generator, n_perm: int):
    """Freedman–Lane test of the last column of X (group).

    Full model y ~ X (= Z + group); reduced model y ~ Z. Reduced-model
    residuals are permuted and added back to its fitted values; the full model
    is refit to every permuted response (vectorised). Statistic: t of the group
    coefficient. Two-sided p = (1 + #{|t*| >= |t_obs|}) / (n_perm + 1).
    Returns (full_fit, observed_t, reduced_residuals, p_perm).
    """
    fit = sm.OLS(y, X).fit()
    obs_t = float(fit.tvalues[-1])

    reduced = sm.OLS(y, Z).fit()
    fitted = np.asarray(reduced.fittedvalues)
    resid = np.asarray(reduced.resid)

    perm_idx = np.argsort(rng.random((n_perm, len(y))), axis=1)
    Y_perm = fitted[:, None] + resid[perm_idx].T

    X_inv = np.linalg.pinv(X)
    betas = X_inv @ Y_perm
    errors = Y_perm - X @ betas
    dof = len(y) - np.linalg.matrix_rank(X)
    sigma2 = np.sum(errors**2, axis=0) / dof
    var_group = np.linalg.pinv(X.T @ X)[-1, -1]
    perm_t = betas[-1] / np.sqrt(sigma2 * var_group)

    p_perm = (1 + np.sum(np.abs(perm_t) >= abs(obs_t))) / (n_perm + 1)
    return fit, obs_t, resid, p_perm


def run_permutation(outdir: Path, npx_file: Path, pheno_file: Path) -> None:
    """One generator stream (seed 42) is shared by all proteins in file order,
    so results depend on that order. BH across all 87 permutation p-values."""
    rng = np.random.default_rng(PERMUTATION_SEED)
    df, proteins = load_lc_permutation_frame(npx_file, pheno_file)
    print("Groups:", df["group"].value_counts().to_dict())
    print("Proteins:", len(proteins))

    results = []
    for protein in proteins:
        sub = df[["group", "age", "sex", protein]].copy()
        sub[protein] = pd.to_numeric(sub[protein], errors="coerce")
        sub = sub.dropna()

        y = sub[protein].to_numpy(float)
        Z = sm.add_constant(sub[["age", "sex"]].to_numpy(float))
        X = np.column_stack([Z, sub["group"].to_numpy(float)])
        fit, obs_t, resid, p_perm = freedman_lane_group_test(y, Z, X, rng, PERMUTATION_N)

        is_neuro = sub["group"].to_numpy() == 1
        results.append({
            "protein": protein,
            "n_neuro": int(sub["group"].sum()),
            "n_nonneuro": int((sub["group"] == 0).sum()),
            "beta": fit.params[-1],
            "t_stat": obs_t,
            "p_ols": fit.pvalues[-1],
            "p_perm": p_perm,
            "sd_neuro": resid[is_neuro].std(ddof=1),        # reduced-model residual SDs
            "sd_non": resid[sub["group"].to_numpy() == 0].std(ddof=1),
        })

    res = pd.DataFrame(results)
    res["q_perm_BH"] = multipletests(res["p_perm"], method="fdr_bh")[1]
    res = res.sort_values("p_perm")
    res.to_csv(outdir / STAGE_OUTPUTS["permutation"][0], sep="\t", index=False)

    print("\nFour candidate proteins:")
    print(res[res["protein"].isin(CANDIDATE_PROTEINS)].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nProteins with permutation BH q < 0.05:", (res["q_perm_BH"] < 0.05).sum())


# ═════════════════════════════════════════════════════════════════════════════
# Stage: sex — do the key proteins differ by sex? (reviewer comment "Others 2")
# ═════════════════════════════════════════════════════════════════════════════

SEX_CHECK_PROTEINS = ["FGF21", "TNFRSF11B", "IL17C", "CCL2"]
SEX_MIN_PER_SEX = 5          # within-group models need >= 5 of each sex
SEX_FEMALE_CODE = 1          # sex_01: 1 = female


def load_sex_check_frame(npx_file: Path, metadata_file: Path,
                         proteins: list[str]) -> pd.DataFrame:
    """All groups in the harmonized metadata (each participant once), with
    age = harmonized_age and female = (sex_01 == 1)."""
    npx = pd.read_csv(npx_file, sep="\t")
    meta = pd.read_csv(metadata_file, sep="\t")
    npx["SampleID"] = npx["SampleID"].astype(str).str.strip()
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()

    missing = [p for p in proteins if p not in npx.columns]
    if missing:
        raise ValueError(f"Proteins not in NPX file: {missing}")
    dup = meta["SampleID"].duplicated(keep=False)
    if dup.any():
        raise ValueError("Some SampleIDs appear in more than one group:\n"
                         f"{meta.loc[dup, 'group'].value_counts().to_string()}")

    df = meta.merge(npx[["SampleID"] + proteins], on="SampleID", how="inner")
    df["age"] = pd.to_numeric(df["harmonized_age"], errors="coerce")
    df["female"] = pd.to_numeric(df["sex_01"], errors="coerce")
    df = df.dropna(subset=["age", "female", "group"])
    df["female"] = (df["female"] == SEX_FEMALE_CODE).astype(int)
    return df


def sex_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("group")
              .agg(n=("SampleID", "size"), n_female=("female", "sum"), mean_age=("age", "mean"))
              .assign(pct_female=lambda d: 100 * d["n_female"] / d["n"])
              .reset_index())


def pooled_sex_effect(df: pd.DataFrame, proteins: list[str], safe: dict) -> pd.DataFrame:
    """NPX ~ female + age + C(group), HC3 SEs; BH across the proteins tested."""
    rows = []
    for p in proteins:
        col = safe[p]
        sub = df.dropna(subset=[col])
        fit = smf.ols(f"{col} ~ female + age + C(group)", data=sub).fit(cov_type="HC3")
        b = fit.params["female"]
        lo, hi = fit.conf_int().loc["female"]
        rows.append({
            "protein": p, "n": int(fit.nobs), "n_female": int(sub["female"].sum()),
            "female_minus_male_NPX": b, "ci_low": lo, "ci_high": hi,
            "fold_difference": 2 ** b, "p": fit.pvalues["female"],
            "protein_sd_NPX": sub[col].std(), "difference_in_SD_units": b / sub[col].std(),
        })
    pooled = pd.DataFrame(rows)
    pooled["p_BH"] = multipletests(pooled["p"], method="fdr_bh")[1]
    return pooled


def sex_group_interaction(df: pd.DataFrame, proteins: list[str], safe: dict) -> pd.DataFrame:
    """F-test: NPX ~ female * C(group) + age  vs  NPX ~ female + age + C(group)."""
    rows = []
    for p in proteins:
        col = safe[p]
        sub = df.dropna(subset=[col])
        reduced = smf.ols(f"{col} ~ female + age + C(group)", data=sub).fit()
        full = smf.ols(f"{col} ~ female * C(group) + age", data=sub).fit()
        f_stat, p_val, df_diff = full.compare_f_test(reduced)
        rows.append({"protein": p, "F": f_stat, "df_interaction": int(df_diff),
                     "p_interaction": p_val})
    return pd.DataFrame(rows)


def within_group_sex_effect(df: pd.DataFrame, proteins: list[str], safe: dict) -> pd.DataFrame:
    """Descriptive: OLS NPX ~ female + age (HC3) and Mann-Whitney within each group."""
    rows = []
    for g, gdf in df.groupby("group"):
        n_f = int(gdf["female"].sum())
        n_m = int(len(gdf) - n_f)
        for p in proteins:
            col = safe[p]
            sub = gdf.dropna(subset=[col])
            row = {"group": g, "protein": p, "n_female": n_f, "n_male": n_m}
            if min(n_f, n_m) < SEX_MIN_PER_SEX:
                row["note"] = f"skipped: fewer than {SEX_MIN_PER_SEX} of one sex"
            else:
                fit = smf.ols(f"{col} ~ female + age", data=sub).fit(cov_type="HC3")
                lo, hi = fit.conf_int().loc["female"]
                row.update({
                    "female_minus_male_NPX": fit.params["female"],
                    "ci_low": lo, "ci_high": hi,
                    "p_adjusted_age": fit.pvalues["female"],
                    "p_mannwhitney": mannwhitneyu(
                        sub.loc[sub["female"] == 1, col], sub.loc[sub["female"] == 0, col],
                        alternative="two-sided").pvalue,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def run_sex_check(outdir: Path, npx_file: Path, metadata_file: Path) -> None:
    sex_dir = outdir / STAGE_OUTPUTS["sex"][0]
    sex_dir.mkdir(parents=True, exist_ok=True)
    proteins = SEX_CHECK_PROTEINS
    df = load_sex_check_frame(npx_file, metadata_file, proteins)

    counts = sex_group_counts(df)
    counts.to_csv(sex_dir / "sex_effect_group_counts.tsv", sep="\t", index=False)
    print("\nSEX EFFECT CHECK")
    print("=" * 64)
    print(counts.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    safe = {p: f"p_{i}" for i, p in enumerate(proteins)}   # patsy-safe column names
    df = df.rename(columns=safe)
    pooled = pooled_sex_effect(df, proteins, safe)
    pooled.to_csv(sex_dir / "sex_effect_pooled.tsv", sep="\t", index=False)
    interaction = sex_group_interaction(df, proteins, safe)
    interaction.to_csv(sex_dir / "sex_by_group_interaction.tsv", sep="\t", index=False)
    within = within_group_sex_effect(df, proteins, safe)
    within.to_csv(sex_dir / "sex_effect_within_group.tsv", sep="\t", index=False)

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print("\n1. Pooled: NPX ~ female + age + group  (female - male, log2 NPX)")
    print(pooled[["protein", "n", "female_minus_male_NPX", "ci_low", "ci_high",
                  "fold_difference", "difference_in_SD_units", "p", "p_BH"]]
          .to_string(index=False, float_format=fmt))
    print("\n2. Sex x group interaction")
    print(interaction.to_string(index=False, float_format=fmt))
    print("\n3. Within-group (descriptive)")
    print(within.to_string(index=False, float_format=fmt))


# ═════════════════════════════════════════════════════════════════════════════
# Command line
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", choices=list(STAGE_OUTPUTS),
                        default=list(STAGE_OUTPUTS), help="Stages to run (default: all)")
    parser.add_argument("--outdir", type=Path, default=cohort.OUTDIR,
                        help="Output directory (default: results/pediatric)")
    parser.add_argument("--results_dir", type=Path, default=cohort.OUTDIR,
                        help="Where upstream outputs are read (combat_adjusted_npx.tsv)")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="Harmonized metadata from 01_cohort_qc.py")
    parser.add_argument("--npx_file", type=Path, default=cohort.NPX_FILE)
    parser.add_argument("--pheno_file", type=Path, default=cohort.PHENO_FILE)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing output files")
    parser.add_argument("--max_proteins", type=int, default=None,
                        help="primary only: first N proteins, no file written")
    parser.add_argument("--sparse_protein_check", action="store_true",
                        help=f"primary only: print NPX summaries for {SPARSE_PROTEIN_CHECK}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    planned = [outdir / name for stage in args.stage for name in STAGE_OUTPUTS[stage]
               if not (stage == "primary" and args.max_proteins is not None)]
    check_outputs_writable(planned, overwrite=args.overwrite)
    outdir.mkdir(parents=True, exist_ok=True)

    stages = {
        "primary": lambda: run_primary(outdir, args.npx_file, args.pheno_file,
                                       args.max_proteins, args.sparse_protein_check),
        "firth": lambda: run_firth(outdir, args.metadata.resolve(),
                                   (args.results_dir / "combat_adjusted_npx.tsv").resolve()),
        "fgf21": lambda: run_fgf21(outdir, args.npx_file, args.pheno_file),
        "permutation": lambda: run_permutation(outdir, args.npx_file, args.pheno_file),
        "sex": lambda: run_sex_check(outdir, args.npx_file, args.metadata),
    }
    for stage in STAGE_OUTPUTS:              # fixed order, whatever order was given
        if stage in args.stage:
            print(f"\n{'#' * 70}\n# STAGE: {stage}\n{'#' * 70}")
            stages[stage]()
    print(f"\nOutputs written to: {outdir}")


if __name__ == "__main__":
    main()
