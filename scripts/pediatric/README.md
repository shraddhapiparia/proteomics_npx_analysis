# Pediatric analysis — five entry points

Run from the repository root. Inputs: `data/raw/Inflammation_NPX_AgeM.txt`
(Olink NPX) and `data/raw/phenotype.txt`. Outputs go to `results/pediatric/`
by default. **Every entry point refuses to overwrite existing outputs**; use
`--outdir <new dir>` for a fresh copy or `--overwrite` to regenerate in place.
Expensive stages can be run on their own with `--stage`.

| # | Script | Stages (`--stage`) | Main outputs |
|---|---|---|---|
| 1 | `01_cohort_qc.py` | (single) | `age_qc.tsv`, `subtype_demographics_*.tsv`, `timepoint_qc.tsv`, `protein_missingness_by_group*.tsv`, `data/interim/harmonized_metadata.tsv` |
| 2 | `02_protein_associations.py` | `primary` `firth` `fgf21` `permutation` `sex` | `regression_LC_NonNeuro_strict_vs_LC_Neuro_age_sex_adjusted.tsv`, `firth_candidate_sensitivity.tsv`, `fgf21_*.tsv`, `permutation_lc_neuro_strict_raw_npx.tsv`, `sex_effect_check/` |
| 3 | `03_crosscenter_analysis.py` | `raw_crosscenter` `combat` `combat_regression` `acc_misc` `outlier_retained` `gpca` `permanova` | uncorrected cross-center and CACTUS-indicator tables, `combat_adjusted_npx.tsv`, `regression_combat_*.tsv`, `lc_neuro_vs_acc_misc_candidate_proteins_combat.tsv`, `outlier_retained_expanded/`, `gpca_contrasts/`, `permanova_combat_diagnostic_summary.tsv` |
| 4 | `04_discrimination.py` | (single; `--n-repeats`, `--n-perm`, `--n-bootstrap` for quick checks) | `auc_validation_revised/` |
| 5 | `05_pathways_figures.py` | `pathways` `fig_candidates` `fig_heatmap` `fig_pathways` `fig_venn` `fig_combat_diagnostics` | `pathway/`, figure PNG/PDFs and their source-data tables |

Order for a complete fresh run (use the same directory for `--outdir` and
`--results_dir`): 01 → 02 `primary fgf21 permutation sex` → 03 `combat` → 03 rest
→ 02 `firth` → 04 → 05.

## Analysis hierarchy

- **Primary (02 `primary`)**: within one center (Gemelli), uncorrected NPX,
  LC_Neuro (16) vs LC_NonNeuro_strict (18),
  `outcome ~ protein + harmonized_age + sex_01` (logistic; outcome 1 = LC_Neuro,
  sex_01 1 = female), 87 proteins, BH FDR over evaluable proteins. FGF23, IL20,
  IL22RA1, ARTN and IL2RB are non-evaluable (NE). CACTUS154_T0
  (|total-abundance z| > 3) is excluded from all regressions only.
- **Sensitivity (02)**: Firth (`firth_regression.R`); FGF21 + fatigue and +
  non-neurological symptom burden (N SYSTEMS − neuro); LC-only permutation; sex
  effects (reviewer).
- **Cross-center, exploratory (03)**: these never replace the primary analysis.
  ComBat (`combat_adjustment_sva.R`: batch = center, `mod = ~ is_LC_Neuro`,
  protein-wise mean imputation, NA mask restored); comparisons vs Healthy,
  pooled CACTUS, expanded, and ACC and MIS-C separately; outlier-retained ComBat
  refit; gPCA and PERMANOVA batch diagnostics.

## Method notes kept from the validated scripts

- **LC-only permutation (02 `permutation`)**: 34 LC participants, raw NPX,
  **linear** model `NPX ~ age + sex + group` (provided `Age` column; sex
  F = 0 / M = 1). Freedman–Lane: reduced-model (`NPX ~ age + sex`) residuals
  are permuted and the full model refit. Statistic: t of the group term.
  Two-sided p = (1 + #|t*| ≥ |t_obs|)/(10,000 + 1), seed 42 (one generator
  stream over proteins in file order). BH across all 87 p-values. `sd_neuro` /
  `sd_non` are residual SDs. ARTN and IL22RA1 are constant across all 34 LC
  participants, so their t / p values are numerical noise.
- **ACC / MIS-C (03 `acc_misc`)**: LC age = provided `Age`; ACC/MIS-C age from
  dates; sex F = 0 / M = 1; ComBat NPX (111 samples, so CACTUS154_T0 is
  absent); BH over the 4 candidates within each comparison.
- **Outlier-retained (03 `outlier_retained`)**: all 112 samples, same ComBat
  procedure as `combat_adjustment_sva.R`. Given the original 111 samples it
  reproduces `combat_adjusted_npx.tsv` exactly.
- **PERMANOVA (03 `permanova`)**: proteins are z-scored within each matrix and
  contrast (`standardize=True`). The stored
  `permanova_combat_diagnostic_summary.tsv` was generated with standardization
  enabled (the original script's `--standardize` option; reproduced
  byte-for-byte). Euclidean distances, 9,999 permutations, seed 0.
- **gPCA (03 `gpca`)**: the stored `gpca_contrasts/` results were generated
  **without** a random seed (1,000 permutations), so their permutation p-values
  (0.115–0.531) cannot be reproduced digit-for-digit (Monte Carlo SE ≈ 0.01–0.016).
  The stage now passes `--seed 1` (set before each contrast), which makes reruns
  reproducible. Seeds 1–5 gave p = 0.105–0.542 across all contrasts, so no
  conclusion depends on the seed. All other gPCA outputs (delta, variance
  fractions, scores, counts) are seed-independent and match the stored files.
  `--gpca_unseeded` restores the original behaviour.
- **Pathways (05 `pathways`)**: see "Pathway inputs" below.

## Pathway inputs

The protein lists are read from `--results_dir`. No script writes them; each
was checked against the stored regression tables and equals the set below
(file order of names is irrelevant; lists are sorted before testing):

| File | Content | Derivation (verified exactly) |
|---|---|---|
| `panel_87_proteins.txt` | 87 proteins | All Olink assays in `Inflammation_NPX_AgeM.txt` — the enrichment background |
| `core4.txt` | FGF21, CCL2, TNFRSF11B, IL17C | Nominal p < 0.05 (evaluable) in the primary raw-NPX LC_Neuro vs LC_NonNeuro_strict model |
| `expanded11.txt` | 11 proteins | Nominal p < 0.05 in the ComBat LC_Neuro vs LC_NonNeuro_expanded model |
| `expanded11_no_cactus.txt` | same 4 proteins as `core4.txt` | `expanded11` ∩ primary nominal hits; its results are therefore identical to `core` |
| `healthy23.txt` | 23 proteins | Nominal p < 0.05 in the ComBat LC_Neuro vs Healthy model (`regression_combat_Healthy_vs_LC_Neuro_age_sex_adjusted.tsv`) |

`core` and `expanded_no_cactus` have fewer than 5 proteins and are run
deliberately; interpret them descriptively. LC_Neuro vs Healthy is a
cross-center, exploratory analysis.

Gene-set libraries are downloaded from Enrichr at run time with
`gseapy.get_library` (network required): `GO_Biological_Process_2023`,
`Reactome_2022`, `KEGG_2021_Human` (the year is part of the Enrichr library
name; no other version identifier is available). Every gene set is restricted to
the 87-protein background (terms with ≥ 3 panel members), and BH is applied
across all terms of the three libraries per list (`fdr_across_libraries`).
Record of acquisition:
- The stored `pathway/enrichment_*.csv` files were written on 2026-08-11 (file
  modification time; libraries were fetched at that run).
- A rerun on 2026-11-19 with gseapy 1.1.13 downloaded 5,406 / 1,816 / 320 terms
  and reproduced every term, p-value, adjusted p-value, odds ratio and
  `fdr_across_libraries` value exactly. The gseapy version used in August was
  not recorded.
- Gene order inside the `Genes` column can change between runs (Python set
  order); the gene sets themselves do not.
- `enrichment_all_lists.csv` written by this workflow contains all four lists;
  the stored copy contains only `core` (it was overwritten by the last of four
  separate August runs).

## Helper files (required)

| File | Used by |
|---|---|
| `cohort.py` | loading, group assignment, age derivation, sex coding, outlier rule (01, 02, 03, 05) |
| `figures.py` | the five figure functions (moved unchanged from the former plot scripts) (05) |
| `logistic_models.py` | `run_logit`, `run_comparison` (02, 03) |
| `combat_adjustment_sva.R` | 03 `combat` (validated sva::ComBat) |
| `firth_regression.R` | 02 `firth` (logistf) |
| `gpca_combat_residual_check.R` | 03 `gpca` |
| `../shared/output_safety.py` | overwrite guard (01, 02, 03, 05) |
| `../shared/npx_io.py` | `map_to_gene_symbols` (05 `pathways`) |

The three R helpers accept `--outdir` (and input-path) arguments; their defaults
are unchanged. `gpca_combat_residual_check.R` also accepts `--seed`.

## Superseded scripts

The scripts consolidated into the five entry points were moved to a private
archive (`archive/`, git-ignored, not part of the public repository):
`01_prepare_cohort.py`, `02_primary_regression.py`,
`combat_adjusted_regression.py`, `sensitivity_expanded.py`,
`sensitivity_outlier_retained.py`, `neuro_signature_candidate_diagnostics.py`,
`fgf21_fatigue_sensitivity.py`, `sex_check.py`, `permanova_combat_diagnostic.py`,
`pathway_enrichment.py`, `auc_validation.py` and five `plot_*.py` scripts. Two
earlier analyses that are not part of this workflow are archived with them:
`auc_discrimination.py` (earlier descriptive AUC; outputs in `auc/`,
`auc_no_covariates/`) and `plot_combat_top_hit_prepost.py` (pre/post ComBat
top-hit check).
