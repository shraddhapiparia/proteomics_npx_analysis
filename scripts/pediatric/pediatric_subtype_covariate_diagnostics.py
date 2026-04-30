"""
Diagnostic script: why do TNFRSF11B and CCL2 gain nominal significance after
age/sex adjustment in the LC_Neuro vs LC_NonNeuro subtype comparison?

Uses the same 34 LC subtype samples as pediatric_subtype_regression.py.

Outputs
-------
results/pediatric/subtype_covariate_diagnostics/
  tnfrsf11b_scatter_age.png
  tnfrsf11b_boxplot_subtype.png
  tnfrsf11b_boxplot_sex.png
  tnfrsf11b_residualized.png
  ccl2_scatter_age.png
  ccl2_boxplot_subtype.png
  ccl2_boxplot_sex.png
  ccl2_residualized.png
  covariate_effect_summary.tsv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
import warnings

ROOT     = Path(__file__).resolve().parent.parent
NPX_FILE  = ROOT / "data/raw/Inflammation_NPX_AgeM.txt"
PHENO_FILE = ROOT / "data/raw/phenotype.txt"
OUTDIR   = ROOT / "results/pediatric/subtype_covariate_diagnostics"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROTEINS = ["TNFRSF11B", "CCL2"]

# ── Palette ───────────────────────────────────────────────────────────────────
C_NEURO    = "#d62728"   # red  – LC_Neuro
C_NONNEURO = "#1f77b4"   # blue – LC_NonNeuro
C_F        = "#e377c2"   # pink – Female
C_M        = "#17becf"   # teal – Male


# ── Data loading (same logic as pediatric_subtype_regression.py) ──────────────
def load_data():
    npx   = pd.read_csv(NPX_FILE,   sep="\t", index_col="SampleID")
    pheno = pd.read_csv(PHENO_FILE, sep="\t")

    pheno["sex_01"] = (pheno["Gender"] == "F").astype(int)
    pheno_idx = pheno.set_index("SampleID")

    lc_ids = [sid for sid in npx.index if sid.strip() in pheno_idx.index]
    df = npx.loc[lc_ids, PROTEINS].copy()
    df["group"]  = [("LC_Neuro" if pheno_idx.loc[s.strip(), "NEURO"] == 1
                      else "LC_NonNeuro") for s in df.index]
    df["age"]    = [pheno_idx.loc[s.strip(), "Age"]    for s in df.index]
    df["sex_01"] = [pheno_idx.loc[s.strip(), "sex_01"] for s in df.index]
    df["sex_label"] = df["sex_01"].map({1: "F", 0: "M"})
    df["outcome"] = (df["group"] == "LC_Neuro").astype(int)
    return df


# ── Descriptive summaries ─────────────────────────────────────────────────────
def descriptive_summary(df):
    print(f"\n{'='*60}")
    print("DESCRIPTIVE SUMMARIES")
    print(f"{'='*60}")

    # Age and sex by subtype
    print("\n  Age by subtype:")
    age_tab = df.groupby("group")["age"].agg(["mean","std","min","max"]).round(2)
    print(age_tab.to_string())

    t_stat, t_p = stats.ttest_ind(
        df[df["group"]=="LC_Neuro"]["age"],
        df[df["group"]=="LC_NonNeuro"]["age"]
    )
    print(f"  Age t-test: t={t_stat:.3f}, p={t_p:.4f}")

    print("\n  Sex by subtype:")
    sex_tab = pd.crosstab(df["group"], df["sex_label"])
    print(sex_tab.to_string())
    chi2, chi_p, _, _ = stats.chi2_contingency(sex_tab)
    print(f"  Chi-squared: chi2={chi2:.3f}, p={chi_p:.4f}")

    summaries = {}
    for prot in PROTEINS:
        print(f"\n  {prot}")
        sub = df[["group","age","sex_01","sex_label",prot]].dropna()

        # By subtype
        grp = sub.groupby("group")[prot].agg(["mean","std"])
        print(f"    LC_Neuro mean={grp.loc['LC_Neuro','mean']:.3f}  "
              f"SD={grp.loc['LC_Neuro','std']:.3f}")
        print(f"    LC_NonNeuro mean={grp.loc['LC_NonNeuro','mean']:.3f}  "
              f"SD={grp.loc['LC_NonNeuro','std']:.3f}")

        # Correlation with age
        r_age, p_age = stats.pearsonr(sub["age"], sub[prot])
        r_age_n, p_age_n = stats.pearsonr(
            sub[sub["group"]=="LC_Neuro"]["age"],
            sub[sub["group"]=="LC_Neuro"][prot]
        )
        r_age_nn, p_age_nn = stats.pearsonr(
            sub[sub["group"]=="LC_NonNeuro"]["age"],
            sub[sub["group"]=="LC_NonNeuro"][prot]
        )
        print(f"    Pearson r (protein vs age, all): r={r_age:.3f}, p={p_age:.4f}")
        print(f"    Pearson r (protein vs age, Neuro): r={r_age_n:.3f}, p={p_age_n:.4f}")
        print(f"    Pearson r (protein vs age, NonNeuro): r={r_age_nn:.3f}, p={p_age_nn:.4f}")

        # By sex
        f_mean = sub[sub["sex_01"]==1][prot].mean()
        m_mean = sub[sub["sex_01"]==0][prot].mean()
        t_s, p_s = stats.ttest_ind(sub[sub["sex_01"]==1][prot],
                                    sub[sub["sex_01"]==0][prot])
        print(f"    Mean F={f_mean:.3f}  M={m_mean:.3f}")
        print(f"    Sex t-test: t={t_s:.3f}, p={p_s:.4f}")

        summaries[prot] = dict(
            r_age=r_age, p_age=p_age,
            r_age_neuro=r_age_n, r_age_nonneuro=r_age_nn,
            mean_F=f_mean, mean_M=m_mean, p_sex=p_s
        )
    return summaries


# ── Logistic regression helpers (same as subtype script) ─────────────────────
def logit_protein(df, prot, covariates):
    sub = df[["outcome", prot] + covariates].dropna()
    X = sm.add_constant(sub[[prot] + covariates], has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(sub["outcome"], X).fit(
                method="newton", disp=False, maxiter=100, warn_convergence=False)
        return res.params[prot], res.pvalues[prot]
    except Exception:
        return np.nan, np.nan


# ── Residualization (OLS protein ~ age + sex) ─────────────────────────────────
def residualize(df, prot):
    sub = df[["age","sex_01",prot]].dropna()
    X   = sm.add_constant(sub[["age","sex_01"]], has_constant="add")
    res = sm.OLS(sub[prot], X).fit()
    return sub.index, res.resid.values


# ── Plotting ──────────────────────────────────────────────────────────────────
def jitter(n, width=0.06, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def plot_scatter_age(df, prot, ax=None):
    """Scatter: protein vs age, colored by subtype, with regression lines."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    for grp, color, label in [("LC_Neuro", C_NEURO, "LC Neuro"),
                                ("LC_NonNeuro", C_NONNEURO, "LC Non-Neuro")]:
        sub = df[df["group"] == grp][["age", prot]].dropna()
        ax.scatter(sub["age"], sub[prot], color=color, alpha=0.7,
                   s=40, label=label, zorder=3)
        # regression line
        m, b = np.polyfit(sub["age"], sub[prot], 1)
        xr = np.linspace(sub["age"].min(), sub["age"].max(), 100)
        ax.plot(xr, m*xr + b, color=color, linewidth=1.5, alpha=0.8)

    r, p = stats.pearsonr(df["age"].dropna(), df[prot].dropna())
    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_ylabel(f"{prot} (NPX)", fontsize=10)
    ax.set_title(f"{prot} vs Age\nr = {r:.2f}, p = {p:.3f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_boxplot_subtype(df, prot, color_by_sex=False, ax=None):
    """Violin + strip: protein by subtype, optionally colored by sex."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    groups  = ["LC_Neuro", "LC_NonNeuro"]
    xlabels = ["LC Neuro", "LC Non-Neuro"]
    colors  = [C_NEURO, C_NONNEURO]
    positions = [1, 2]

    data_grps = [df[df["group"] == g][prot].dropna().values for g in groups]

    # Violin
    parts = ax.violinplot(data_grps, positions=positions, showmedians=True,
                          showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.35)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1)

    # Strip
    for pos, grp, col in zip(positions, groups, colors):
        sub  = df[df["group"] == grp][["age","sex_01","sex_label",prot]].dropna()
        yvals = sub[prot].values
        xvals = pos + jitter(len(yvals))
        if color_by_sex:
            pt_colors = [C_F if s == 1 else C_M for s in sub["sex_01"].values]
            ax.scatter(xvals, yvals, c=pt_colors, s=35, zorder=4, alpha=0.9,
                       edgecolors="white", linewidths=0.4)
        else:
            ax.scatter(xvals, yvals, color=col, s=35, zorder=4, alpha=0.9,
                       edgecolors="white", linewidths=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(f"{prot} (NPX)", fontsize=10)

    if color_by_sex:
        ax.set_title(f"{prot} by Subtype\n(points colored by sex)", fontsize=11)
        f_patch = mpatches.Patch(color=C_F, label="Female")
        m_patch = mpatches.Patch(color=C_M, label="Male")
        ax.legend(handles=[f_patch, m_patch], fontsize=8)
    else:
        ax.set_title(f"{prot} by Subtype", fontsize=11)

    # t-test annotation
    t, p = stats.ttest_ind(data_grps[0], data_grps[1])
    ymax = max(np.nanmax(v) for v in data_grps)
    ax.text(1.5, ymax * 1.02, f"t={t:.2f}, p={p:.3f}",
            ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    return ax


def plot_residualized(df, prot, ax=None):
    """Violin + strip of residuals (protein ~ age + sex) by subtype."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    idx, resids = residualize(df, prot)
    resid_df = pd.DataFrame({"residual": resids, "group": df.loc[idx, "group"].values})

    groups    = ["LC_Neuro", "LC_NonNeuro"]
    xlabels   = ["LC Neuro", "LC Non-Neuro"]
    colors    = [C_NEURO, C_NONNEURO]
    positions = [1, 2]
    data_grps = [resid_df[resid_df["group"] == g]["residual"].values
                  for g in groups]

    parts = ax.violinplot(data_grps, positions=positions, showmedians=True,
                          showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.35)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1)

    for pos, grp, col in zip(positions, groups, colors):
        yvals = resid_df[resid_df["group"] == grp]["residual"].values
        ax.scatter(pos + jitter(len(yvals)), yvals, color=col,
                   s=35, zorder=4, alpha=0.9, edgecolors="white", linewidths=0.4)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(f"{prot} residual\n(after age + sex)", fontsize=10)
    ax.set_title(f"{prot} — Age+Sex Residuals\nby Subtype", fontsize=11)

    t, p = stats.ttest_ind(data_grps[0], data_grps[1])
    ymax = max(np.nanmax(v) for v in data_grps)
    ax.text(1.5, ymax * 1.02, f"t={t:.2f}, p={p:.3f}",
            ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    return ax


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print("SUBTYPE COVARIATE DIAGNOSTICS")
    print("Proteins: TNFRSF11B, CCL2")
    print(f"{'='*60}")

    df = load_data()
    print(f"\nSamples: n={len(df)}  "
          f"LC_Neuro={( df['group']=='LC_Neuro').sum()}  "
          f"LC_NonNeuro={(df['group']=='LC_NonNeuro').sum()}")

    summaries = descriptive_summary(df)

    # ── Logistic regression: univariate vs age+sex ────────────────────────────
    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION SUMMARY")
    print(f"{'='*60}")

    regression_rows = []
    for prot in PROTEINS:
        b_u, p_u = logit_protein(df, prot, [])
        b_as, p_as = logit_protein(df, prot, ["age", "sex_01"])
        pct_change = 100 * (abs(b_as) - abs(b_u)) / abs(b_u)

        s = summaries[prot]
        # Classify likely explanation based on magnitude of age/sex associations
        age_corr  = abs(s["r_age"])
        sex_diff_p = s["p_sex"]

        if age_corr >= 0.25 and sex_diff_p <= 0.20:
            expl = "age and sex both contribute"
        elif age_corr >= 0.25:
            expl = "primarily age-related"
        elif sex_diff_p <= 0.20:
            expl = "primarily sex-related"
        else:
            expl = "age/sex together reduce residual variance"

        print(f"\n  {prot}")
        print(f"    univariate:   beta={b_u:.4f}  p={p_u:.4f}")
        print(f"    age+sex adj:  beta={b_as:.4f}  p={p_as:.4f}")
        print(f"    |beta| change: {pct_change:+.1f}%")
        print(f"    likely explanation: {expl}")

        regression_rows.append(dict(
            protein=prot,
            beta_univariate=round(b_u,  6),
            beta_age_sex=round(b_as, 6),
            percent_change_in_beta_magnitude=round(pct_change, 1),
            p_univariate=round(p_u,  6),
            p_age_sex=round(p_as, 6),
            r_protein_age=round(s["r_age"],  3),
            p_protein_age=round(s["p_age"],  4),
            mean_female=round(s["mean_F"], 3),
            mean_male=round(s["mean_M"],   3),
            p_sex_diff=round(s["p_sex"],   4),
            likely_explanation=expl,
        ))

    summary_df = pd.DataFrame(regression_rows)
    tsv_path = OUTDIR / "covariate_effect_summary.tsv"
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Saved: {tsv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    for prot in PROTEINS:
        plow = prot.lower()

        # 1. Scatter vs age
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_scatter_age(df, prot, ax)
        fig.tight_layout()
        p1 = OUTDIR / f"{plow}_scatter_age.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p1.name}")

        # 2. Boxplot by subtype (no sex color)
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_boxplot_subtype(df, prot, color_by_sex=False, ax=ax)
        fig.tight_layout()
        p2 = OUTDIR / f"{plow}_boxplot_subtype.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p2.name}")

        # 3. Boxplot by subtype, points colored by sex
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_boxplot_subtype(df, prot, color_by_sex=True, ax=ax)
        fig.tight_layout()
        p3 = OUTDIR / f"{plow}_boxplot_sex.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p3.name}")

        # 4. Residualized by subtype
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_residualized(df, prot, ax)
        fig.tight_layout()
        p4 = OUTDIR / f"{plow}_residualized.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p4.name}")

    # ── Print full summary table ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COVARIATE EFFECT SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("ALL OUTPUTS")
    print(f"{'='*60}")
    for f in sorted(OUTDIR.iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
