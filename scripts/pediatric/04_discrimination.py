#!/usr/bin/env python3
"""
Entry point 4 of 5 — Discrimination (AUC) of LC-Neuro vs LC-NonNeuro-strict.

This is the validated revised analysis (formerly auc_validation.py), kept
verbatim: repeated stratified 5-fold CV (200 repeats), protein selection inside
training folds, L2 logistic regression with fixed C = 1.0, participant-level
out-of-fold predictions, bootstrap CIs, permutation tests, protein-selection
frequencies, descriptive candidate AUCs and ROC curves.
Input: data/interim/harmonized_metadata.tsv from 01_cohort_qc.py.
Expensive at default settings; use --n-repeats / --n-perm / --n-bootstrap for
quick checks, and --outdir to avoid touching results/pediatric/auc_validation_revised.

  python scripts/pediatric/04_discrimination.py --outdir results/pediatric/rerun_auc

Cross-validated discrimination of LC-Neuro vs LC-NonNeuro-strict, with
protein selection performed inside each training fold.

Revision of the AUC validation analysis in response to Reviewer comment 3.

Question
--------
Do plasma proteins discriminate LC-Neuro from LC-NonNeuro-strict, and do they
add discrimination beyond age and sex?

Design
------
- Outer procedure: repeated stratified 5-fold cross-validation (200 repeats).
- In each training fold: rank all eligible proteins using training participants
  only, keep the top K, fit an L2-penalised logistic regression with a FIXED
  penalty (C = 1.0). No hyperparameter is tuned, so there is no inner loop and
  the procedure is not "nested" CV.
- Held-out participants are used only to generate predictions.
- Point estimate: each child's out-of-fold probabilities are averaged across
  the repeats, and AUC is computed from these participant-level predictions.
  The mean of per-repeat AUCs is also reported as a check.
- Uncertainty: stratified participant-level bootstrap of the out-of-fold
  predictions. This reflects sampling of evaluated participants conditional on
  the fitted models; it does not refit the selection/modelling procedure.
- Chance level: permutation of the full procedure (selection + fitting + CV).
    Test A: permute outcome labels           -> do proteins discriminate at all?
    Test B: permute protein rows only,       -> do proteins add beyond age+sex?
            keeping outcome, age, sex linked
- Sensitivity: rank proteins by age- and sex-adjusted partial correlation
  instead of the unadjusted Mann-Whitney statistic.

Outputs (OUTDIR)
----------------
    model_performance.tsv              AUC, bootstrap CI, delta-AUC, Brier, per-repeat AUC
    permutation_tests.tsv              observed AUC, null mean/quantiles, permutation p
    permutation_null_distribution.tsv  raw null AUCs
    selection_check.tsv                fast ranking vs scipy Mann-Whitney agreement
    selected_protein_frequency.tsv     selection stability (primary ranking)
    sensitivity_adjusted_ranking.tsv   AUCs and selection with adjusted ranking
    cv_predictions.tsv                 every out-of-fold prediction
    subject_level_oof_predictions.tsv  repeat-averaged prediction per child
    descriptive_candidate_auc.tsv      apparent (NOT validated) AUCs of the 4 candidates
    model_comparison_roc.png/.pdf
    permutation_null.png/.pdf
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--npx", default=ROOT / "data" / "raw" / "Inflammation_NPX_AgeM.txt")
parser.add_argument("--meta", default=ROOT / "data" / "interim" / "harmonized_metadata.tsv")
parser.add_argument("--outdir", default=ROOT / "results" / "pediatric" / "auc_validation_revised")
parser.add_argument("--top-k", type=int, default=1)
parser.add_argument("--n-splits", type=int, default=5)
parser.add_argument("--n-repeats", type=int, default=200)
parser.add_argument("--n-repeats-perm", type=int, default=20,
                    help="Repeats per run in the permutation tests (observed and null use the same value).")
parser.add_argument("--n-perm", type=int, default=1000)
parser.add_argument("--n-bootstrap", type=int, default=5000)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--overwrite", action="store_true",
                    help="Allow writing into an existing, non-empty --outdir")
args = parser.parse_args()

NPX_FILE = Path(args.npx)
META_FILE = Path(args.meta)
OUTDIR = Path(args.outdir)
# Overwrite guard (not part of the analysis): never replace existing outputs silently.
if OUTDIR.exists() and any(OUTDIR.iterdir()) and not args.overwrite:
    sys.exit(f"ERROR: {OUTDIR} already contains outputs; nothing was written.\n"
             "  Use --outdir <new directory>, or --overwrite to replace them deliberately.")
OUTDIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = ["FGF21", "IL17C", "TNFRSF11B", "CCL2"]
COVARIATES = ["harmonized_age", "sex_01"]
MODELS = ["Age + sex", "Proteins", "Age + sex + proteins"]

TOP_K = args.top_k
N_SPLITS = args.n_splits
N_REPEATS = args.n_repeats
N_REPEATS_PERM = args.n_repeats_perm
N_PERM = args.n_perm
N_BOOTSTRAP = args.n_bootstrap
SEED = args.seed
C_FIXED = 1.0


# ---------------------------------------------------------------------
# 1. Load the primary LC-Neuro vs LC-NonNeuro-strict comparison
# ---------------------------------------------------------------------

npx = pd.read_csv(NPX_FILE, sep="\t")
meta = pd.read_csv(META_FILE, sep="\t")
npx["SampleID"] = npx["SampleID"].astype(str).str.strip()
meta["SampleID"] = meta["SampleID"].astype(str).str.strip()

missing_meta = {"SampleID", "group", *COVARIATES} - set(meta.columns)
if missing_meta:
    raise ValueError(f"Missing metadata columns: {sorted(missing_meta)}")

missing_candidates = [p for p in CANDIDATES if p not in npx.columns]
if missing_candidates:
    raise ValueError(f"Candidate proteins missing from NPX file: {missing_candidates}")

meta = meta[meta["group"].isin(["LC_Neuro", "LC_NonNeuro_strict"])].copy()
df = meta.merge(npx, on="SampleID", how="inner").reset_index(drop=True)
df["outcome"] = (df["group"] == "LC_Neuro").astype(int)

for col in COVARIATES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[COVARIATES].isna().any().any():
    raise ValueError("Missing harmonized age or sex in the primary comparison.")

protein_cols = [c for c in npx.columns if c != "SampleID"]
protein_pool = [p for p in protein_cols if df[p].notna().all() and df[p].nunique() >= 3]

X_prot = df[protein_pool].to_numpy(dtype=float)
X_cov = df[COVARIATES].to_numpy(dtype=float)
y = df["outcome"].to_numpy(dtype=int)
sample_ids = df["SampleID"].to_numpy()

print("\nAUC VALIDATION (selection inside training folds)")
print("=" * 64)
print(f"Participants: {len(df)}  (LC-Neuro {y.sum()}, LC-NonNeuro {len(y) - y.sum()})")
print(f"Proteins eligible for selection: {len(protein_pool)}")
print(f"Top-K per training fold: {TOP_K};  fixed C = {C_FIXED}")
print(f"CV: {N_SPLITS}-fold x {N_REPEATS} repeats;  permutations: {N_PERM} "
      f"x {N_REPEATS_PERM} repeats")


# ---------------------------------------------------------------------
# 2. Selection rules (training data only)
# ---------------------------------------------------------------------

def rank_mannwhitney(Xp, y_tr):
    """
    Two-sided Mann-Whitney |z| (tie-corrected, continuity-corrected),
    matching scipy.stats.mannwhitneyu(method="asymptotic").
    Larger = smaller p-value.
    """
    n = len(y_tr)
    n1 = y_tr.sum()
    n0 = n - n1
    r = rankdata(Xp, axis=0)
    U = r[y_tr == 1].sum(axis=0) - n1 * (n1 + 1) / 2
    mu = n1 * n0 / 2

    # tie correction, computed per protein
    tie_term = np.empty(Xp.shape[1])
    for j in range(Xp.shape[1]):
        _, t = np.unique(Xp[:, j], return_counts=True)
        tie_term[j] = (t**3 - t).sum()
    sd = np.sqrt(n1 * n0 / 12 * ((n + 1) - tie_term / (n * (n - 1))))

    z = np.zeros_like(sd)
    np.divide(np.abs(U - mu) - 0.5, sd, out=z, where=sd > 0)
    return z


def rank_adjusted(Xp, y_tr, Xc_tr):
    """
    Sensitivity ranking: |partial correlation| between each protein and the
    outcome after linearly removing age and sex (Frisch-Waugh-Lovell).
    """
    Z = np.column_stack([np.ones(len(y_tr)), Xc_tr])
    H = Z @ np.linalg.pinv(Z)
    rp = Xp - H @ Xp
    ry = y_tr - H @ y_tr
    num = ry @ rp
    den = np.sqrt((ry @ ry) * (rp * rp).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0, num / den, 0.0)
    return np.abs(corr)


def select_top(Xp, y_tr, Xc_tr, k, rule):
    score = rank_mannwhitney(Xp, y_tr) if rule == "mannwhitney" else rank_adjusted(Xp, y_tr, Xc_tr)
    # stable sort on -score so ties break by column order deterministically
    return np.argsort(-score, kind="stable")[:k]


def make_logistic_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=C_FIXED, l1_ratio=0, solver="lbfgs", max_iter=5000)),
    ])


# ---------------------------------------------------------------------
# 3. Check: vectorised ranking reproduces scipy Mann-Whitney selection
# ---------------------------------------------------------------------

def check_selection_equivalence(n_folds_to_check=50):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for i, (tr, _) in enumerate(cv.split(X_prot, y)):
        if i >= n_folds_to_check:
            break
        fast = set(np.array(protein_pool)[select_top(X_prot[tr], y[tr], X_cov[tr], TOP_K, "mannwhitney")])
        pvals = [
            mannwhitneyu(X_prot[tr][y[tr] == 1, j], X_prot[tr][y[tr] == 0, j],
                         alternative="two-sided").pvalue
            for j in range(len(protein_pool))
        ]
        slow = set(np.array(protein_pool)[np.argsort(pvals, kind="stable")[:TOP_K]])
        rows.append({"fold": i, "fast": ",".join(sorted(fast)),
                     "scipy": ",".join(sorted(slow)), "identical": fast == slow})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "selection_check.tsv", sep="\t", index=False)
    print(f"\nSelection check: fast ranking matches scipy Mann-Whitney in "
          f"{out['identical'].sum()}/{len(out)} folds")
    return out


check_selection_equivalence()


# ---------------------------------------------------------------------
# 4. Core CV routine (used for the main analysis and every permutation)
# ---------------------------------------------------------------------

def run_cv(Xp, Xc, y_vec, n_repeats, rule="mannwhitney", models=MODELS, keep_details=False):
    """
    Returns participant-level averaged out-of-fold predictions per model,
    per-repeat pooled AUCs, and optionally fold-level details.
    """
    n = len(y_vec)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=SEED)

    pred_sum = {m: np.zeros(n) for m in models}
    pred_cnt = np.zeros(n)
    repeat_pred = {m: np.full(n, np.nan) for m in models}
    repeat_aucs = {m: [] for m in models}
    selection_counts = np.zeros(Xp.shape[1], dtype=int)
    details = []

    for fold_id, (tr, te) in enumerate(cv.split(Xp, y_vec)):
        sel = select_top(Xp[tr], y_vec[tr], Xc[tr], TOP_K, rule)
        selection_counts[sel] += 1

        design = {
            "Age + sex": (Xc[tr], Xc[te]),
            "Proteins": (Xp[tr][:, sel], Xp[te][:, sel]),
            "Age + sex + proteins": (np.hstack([Xc[tr], Xp[tr][:, sel]]),
                                     np.hstack([Xc[te], Xp[te][:, sel]])),
        }

        for m in models:
            Xtr, Xte = design[m]
            prob = make_logistic_model().fit(Xtr, y_vec[tr]).predict_proba(Xte)[:, 1]
            pred_sum[m][te] += prob
            repeat_pred[m][te] = prob
            if keep_details:
                details.extend({
                    "SampleID": sample_ids[i], "outcome": int(y_vec[i]), "model": m,
                    "repeat": fold_id // N_SPLITS, "fold_id": fold_id,
                    "predicted_probability": p,
                    "selected_proteins": ",".join(np.array(protein_pool)[sel]),
                } for i, p in zip(te, prob))
        pred_cnt[te] += 1

        # end of a repeat: every child has exactly one prediction in repeat_pred
        if (fold_id + 1) % N_SPLITS == 0:
            for m in models:
                repeat_aucs[m].append(roc_auc_score(y_vec, repeat_pred[m]))
                repeat_pred[m][:] = np.nan

    avg_pred = {m: pred_sum[m] / pred_cnt for m in models}
    return {
        "avg_pred": avg_pred,
        "auc": {m: roc_auc_score(y_vec, avg_pred[m]) for m in models},
        "repeat_aucs": repeat_aucs,
        "selection_counts": selection_counts,
        "n_folds": N_SPLITS * n_repeats,
        "details": details,
    }


# ---------------------------------------------------------------------
# 5. Main analysis (200 repeats)
# ---------------------------------------------------------------------

t0 = time.time()
main = run_cv(X_prot, X_cov, y, N_REPEATS, keep_details=True)
print(f"\nMain CV finished in {time.time() - t0:.1f}s")

pd.DataFrame(main["details"]).to_csv(OUTDIR / "cv_predictions.tsv", sep="\t", index=False)

subject_predictions = pd.DataFrame({"SampleID": sample_ids, "outcome": y})
for m in MODELS:
    subject_predictions[m] = main["avg_pred"][m]
subject_predictions.to_csv(OUTDIR / "subject_level_oof_predictions.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------
# 6. Participant-level bootstrap (paired across models)
# ---------------------------------------------------------------------

def bootstrap_auc(pred_by_model, y_vec, n_boot, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_vec == 1), np.flatnonzero(y_vec == 0)
    boot = {m: np.empty(n_boot) for m in MODELS}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m in MODELS:
            boot[m][b] = roc_auc_score(y_vec[idx], pred_by_model[m][idx])
    return boot


boot = bootstrap_auc(main["avg_pred"], y, N_BOOTSTRAP, SEED)

perf_rows = []
for m in MODELS:
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    row = {
        "model": m,
        "oof_auc": main["auc"][m],
        "auc_ci_low": lo,
        "auc_ci_high": hi,
        "mean_per_repeat_auc": np.mean(main["repeat_aucs"][m]),
        "per_repeat_auc_2.5pct": np.percentile(main["repeat_aucs"][m], 2.5),
        "per_repeat_auc_97.5pct": np.percentile(main["repeat_aucs"][m], 97.5),
        "brier_score": brier_score_loss(y, main["avg_pred"][m]),
    }
    if m != "Age + sex":
        d = boot[m] - boot["Age + sex"]
        row["delta_auc_vs_age_sex"] = main["auc"][m] - main["auc"]["Age + sex"]
        row["delta_ci_low"], row["delta_ci_high"] = np.percentile(d, [2.5, 97.5])
    perf_rows.append(row)

performance = pd.DataFrame(perf_rows)
performance.to_csv(OUTDIR / "model_performance.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------
# 7. Selection stability (primary ranking)
# ---------------------------------------------------------------------

selection_frequency = (
    pd.DataFrame({"protein": protein_pool, "times_selected": main["selection_counts"]})
    .assign(selection_frequency=lambda d: d["times_selected"] / main["n_folds"])
    .sort_values("selection_frequency", ascending=False)
)
selection_frequency.to_csv(OUTDIR / "selected_protein_frequency.tsv", sep="\t", index=False)

n_other = int(((selection_frequency["times_selected"] > 0)
               & ~selection_frequency["protein"].isin(CANDIDATES)).sum())


# ---------------------------------------------------------------------
# 8. Permutation tests of the full procedure
# ---------------------------------------------------------------------
#   A: permute y                      -> Proteins model vs its null
#   B: permute protein rows only      -> Age + sex + proteins vs its null
#      (outcome, age and sex stay linked; the age+sex model and the CV
#       splits are unchanged, so the combined AUC is the test statistic)

obs_A = run_cv(X_prot, X_cov, y, N_REPEATS_PERM, models=["Proteins"])["auc"]["Proteins"]
obs_B = run_cv(X_prot, X_cov, y, N_REPEATS_PERM,
               models=["Age + sex", "Age + sex + proteins"])["auc"]
obs_age_sex_perm = obs_B["Age + sex"]
obs_B = obs_B["Age + sex + proteins"]


def one_permutation(seed):
    rng = np.random.default_rng(seed)
    a = run_cv(X_prot, X_cov, rng.permutation(y), N_REPEATS_PERM,
               models=["Proteins"])["auc"]["Proteins"]
    b = run_cv(X_prot[rng.permutation(len(y))], X_cov, y, N_REPEATS_PERM,
               models=["Age + sex + proteins"])["auc"]["Age + sex + proteins"]
    return a, b


t0 = time.time()
perm_seeds = np.random.default_rng(SEED + 1).integers(0, 2**31 - 1, size=N_PERM)
null = np.array(Parallel(n_jobs=args.n_jobs)(delayed(one_permutation)(s) for s in perm_seeds))
print(f"Permutation tests finished in {time.time() - t0:.1f}s")

pd.DataFrame(null, columns=["null_auc_proteins_permute_y",
                            "null_auc_combined_permute_proteins"]).to_csv(
    OUTDIR / "permutation_null_distribution.tsv", sep="\t", index=False)


def perm_summary(name, observed, null_vec, null_desc):
    return {
        "test": name,
        "null": null_desc,
        "observed_auc": observed,
        "null_mean": null_vec.mean(),
        "null_2.5pct": np.percentile(null_vec, 2.5),
        "null_97.5pct": np.percentile(null_vec, 97.5),
        "p_one_sided": (np.sum(null_vec >= observed) + 1) / (len(null_vec) + 1),
        "n_perm": len(null_vec),
        "n_repeats_per_run": N_REPEATS_PERM,
    }


perm_table = pd.DataFrame([
    perm_summary("A: Proteins", obs_A, null[:, 0], "outcome labels permuted"),
    perm_summary("B: Age + sex + proteins", obs_B, null[:, 1],
                 f"protein rows permuted (age+sex AUC fixed at {obs_age_sex_perm:.3f})"),
])
perm_table.to_csv(OUTDIR / "permutation_tests.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------
# 9. Sensitivity: age- and sex-adjusted ranking
# ---------------------------------------------------------------------

sens = run_cv(X_prot, X_cov, y, N_REPEATS, rule="adjusted")
sens_boot = bootstrap_auc(sens["avg_pred"], y, N_BOOTSTRAP, SEED)
sens_freq = pd.Series(sens["selection_counts"] / sens["n_folds"], index=protein_pool)

sens_rows = []
for m in MODELS:
    lo, hi = np.percentile(sens_boot[m], [2.5, 97.5])
    sens_rows.append({"ranking": "age/sex-adjusted partial correlation", "model": m,
                      "oof_auc": sens["auc"][m], "auc_ci_low": lo, "auc_ci_high": hi})
sens_table = pd.DataFrame(sens_rows)
sens_table_freq = (sens_freq.sort_values(ascending=False).head(10)
                   .rename("selection_frequency").reset_index()
                   .rename(columns={"index": "protein"}))
with open(OUTDIR / "sensitivity_adjusted_ranking.tsv", "w") as fh:
    sens_table.to_csv(fh, sep="\t", index=False)
    fh.write("\n")
    sens_table_freq.to_csv(fh, sep="\t", index=False)


# ---------------------------------------------------------------------
# 10. Descriptive (apparent, NOT validated) AUCs for the 4 candidates
# ---------------------------------------------------------------------
# These proteins were chosen from the full dataset, so their AUCs here are
# optimistic by construction. Report only as descriptive, if at all.

desc_rows = []
for p in CANDIDATES:
    v = df[p].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    desc_rows.append({
        "protein": p,
        "apparent_auc": roc_auc_score(y[ok], v[ok]),
        "direction_in_LC_Neuro": "higher" if roc_auc_score(y[ok], v[ok]) > 0.5 else "lower",
        "n": int(ok.sum()),
        "note": "Apparent AUC on full data; candidate chosen from same data; not a validation estimate.",
    })
pd.DataFrame(desc_rows).to_csv(OUTDIR / "descriptive_candidate_auc.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------
# 11. Figures
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2, 6.2))
for m in MODELS:
    fpr, tpr, _ = roc_curve(y, main["avg_pred"][m])
    r = performance.set_index("model").loc[m]

    # Smooth ROC curve for visualization only.
    # Collapse duplicate FPR values while retaining the maximum TPR.
    roc_df = (
        pd.DataFrame({"fpr": fpr, "tpr": tpr})
        .groupby("fpr", as_index=False)["tpr"]
        .max()
    )

    fpr_unique = roc_df["fpr"].to_numpy()
    tpr_unique = np.maximum.accumulate(roc_df["tpr"].to_numpy())

    # Monotone cubic interpolation avoids the oscillations produced by
    # ordinary cubic splines.
    interp = PchipInterpolator(fpr_unique, tpr_unique)
    fpr_smooth = np.linspace(0, 1, 300)
    tpr_smooth = np.clip(interp(fpr_smooth), 0, 1)

    ax.plot(
        fpr_smooth,
        tpr_smooth,
        linewidth=2,
        label=f"{m}: AUC {r.oof_auc:.2f} "
              f"({r.auc_ci_low:.2f}–{r.auc_ci_high:.2f})"
    )
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
ax.set_xlabel("1 − Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_aspect("equal", adjustable="box")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUTDIR / "model_comparison_roc.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "model_comparison_roc.pdf", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (_, r), col in zip(axes, perm_table.iterrows(), [0, 1]):
    ax.hist(null[:, col], bins=40, color="#9aa5b1", edgecolor="white")
    ax.axvline(r.observed_auc, color="#c0392b", linewidth=2,
               label=f"Observed {r.observed_auc:.3f}")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="0.5")
    ax.set_title(f"{r.test}\npermutation p = {r.p_one_sided:.3f}", fontsize=10)
    ax.set_xlabel("Cross-validated AUC under null")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Permutations")
fig.tight_layout()
fig.savefig(OUTDIR / "permutation_null.png", dpi=300, bbox_inches="tight")
fig.savefig(OUTDIR / "permutation_null.pdf", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------
# 12. Console summary
# ---------------------------------------------------------------------

fmt = lambda x: f"{x:.3f}"
print("\nModel performance (selection within training folds)")
print("=" * 64)
print(performance.to_string(index=False, float_format=fmt))

print("\nPermutation tests of the full procedure")
print("=" * 64)
print(perm_table.drop(columns=["null"]).to_string(index=False, float_format=fmt))

print("\nSelection frequency (top 10)")
print("=" * 64)
print(selection_frequency.head(10).to_string(index=False, float_format=fmt))
print(f"Selection total = {TOP_K * 100}% of folds; "
      f"{n_other} non-candidate proteins were selected at least once.")

print("\nSensitivity: age/sex-adjusted ranking")
print("=" * 64)
print(sens_table.to_string(index=False, float_format=fmt))

print(f"\nOutputs written to: {OUTDIR}")