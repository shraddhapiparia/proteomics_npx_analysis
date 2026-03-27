# Long COVID Proteomics Analysis: Pediatric Cohort and UK Biobank Adult Cohort

This repository contains analysis workflows for comparing proteomic signatures of Long COVID across two cohorts:

1. **Pediatric cohort**: a small Olink inflammation panel dataset with pediatric Long COVID cases and healthy controls.
2. **Adult UK Biobank cohort**: a larger population-based cohort used to identify and validate proteomic differences associated with Long COVID.

The overall goal is to study whether circulating inflammatory protein patterns differ between Long COVID cases neurocognitive subgroups and controls, and whether similar signals appear across a small clinical pediatric cohort and a large adult population cohort.

---

## Project goals

This repository is organized around four main goals:

- Define and document the **two study cohorts**
- Build a reproducible **UK Biobank Long COVID selection pipeline**
- Perform **logistic regression subgroup analyses** using proteomics data
- Prepare for later **protein–protein interaction (PPI)** analysis on top-ranked proteins

---

## Cohorts

### 1) Pediatric cohort

This cohort consists of:

- **Long COVID pediatric cases**
- **Healthy pediatric controls**

#### Special considerations

This cohort requires extra caution in interpretation because:

- all **cases come from one center**
- all **controls come from another center**
- the sample size is very small (**n = 53**)
- only **86 proteins** are available after filtering

Because case/control status is confounded with recruitment site, it is not possible to statistically separate:

- biological disease effects
- site-specific effects (sample handling, processing, batch effects)
- population differences between centers

As a result, standard covariate adjustment for site is not identifiable in this dataset.
Modeling strategy

Given these constraints, regression models in the pediatric cohort are adjusted only for:

- total protein abundance (sample-level summary)

This adjustment is used as a proxy to control for:

- global intensity shifts
- sample loading differences
- broad technical variation

Other covariates (e.g., site) are not included because they are perfectly collinear with case/control status.

---

### 2) Adult UK Biobank cohort

This cohort will be derived from **UK Biobank proteomics and phenotype data**.

The UK Biobank analysis will be used to:

- define Long COVID-related or post-COVID subgroups using diagnosis-based selection logic
- identify proteomic differences using regression models
- compare whether pediatric observations are supported in a larger adult cohort

Because UK Biobank is much larger and more heterogeneous, it provides an opportunity to test whether signals seen in the pediatric cohort are reproducible at scale.

---

## Pediatric proteomics data and prior QC

The pediatric plasma samples were measured using the **Olink Inflammation 96-plex panel**, which uses proximity extension assay technology.

According to the source study:

- plasma samples were analyzed on the Olink inflammation panel
- data underwent normalization to reduce inter- and intra-run variation
- the processed data were reported as **Normalized Protein Expression (NPX)**
- NPX values are on a **log2 scale**
- proteins with values **below the limit of detection (LOD) in more than 80% of cases** were excluded

This means the pediatric cohort data already come with major assay-level preprocessing applied.

### Implication for this repository

For the pediatric cohort, this repository will treat the provided NPX matrix as:

- already normalized by the assay workflow
- already filtered for extreme low-detection proteins

Additional analysis-level QC may still be performed, such as:

- missingness review
- outlier inspection
- sample-level abundance summaries
- group-wise visualization
- sensitivity analyses

---

## UK Biobank proteomics considerations

UK Biobank also provides Olink-based proteomic measurements, and one of the first tasks in this repository is to verify:

- whether the exported values are already on an NPX-like scale
- whether normalization has already been applied
- whether LOD-related filtering is needed or should be repeated
- whether protein availability matches the pediatric panel proteins

This repository will include explicit checks for:

- protein distributions
- value ranges
- missingness patterns
- potential LOD-based exclusion criteria

At this stage, LOD handling for UK Biobank is treated as a **planned QC step**, not an assumed preprocessing step.

---

## Planned analyses

### 1) Cohort description

For both cohorts, the repository will include demographic summaries such as:

- age
- sex
- subgroup counts
- case/control definitions
- available proteins

For the pediatric cohort, subgroup reporting will emphasize the small sample size and site imbalance.

For UK Biobank, cohort description will document how cases and controls were selected from diagnosis and phenotype fields.

---

### 2) UK Biobank Long COVID selection

A dedicated script will be used to define Long COVID-related cases in UK Biobank using diagnosis-based logic.

Planned components include:

- extracting relevant UK Biobank fields
- selecting participants with COVID history
- defining Long COVID or post-COVID groups using diagnosis information
- selecting matched or comparable controls
- documenting inclusion/exclusion logic clearly

This selection script is an important part of the repository because cohort definition strongly affects downstream results.

---

### 3) Logistic regression subgroup analysis

For both cohorts, protein-wise regression models will be used to test associations between protein levels and case/control status.

Planned modeling framework:

- outcome: subgroup or disease status
- predictor: one protein at a time
- covariates: age, sex, and other relevant available covariates
- multiple-testing correction across proteins

Possible outputs include:

- beta coefficients
- odds ratios
- confidence intervals
- p-values
- FDR-adjusted p-values
- volcano plots and ranked result tables

Because the pediatric cohort is small, model stability and over-interpretation will be carefully monitored.

---

### 4) Later-stage protein–protein interaction analysis

After identifying top proteins from regression analyses, the repository will later expand to protein–protein interaction analysis.

This future step may include:

- selecting significant or top-ranked proteins
- mapping proteins to gene symbols
- querying interaction databases
- identifying enriched pathways or network modules
- comparing interaction structure across pediatric and adult findings

PPI is intentionally planned as a later stage after the cohort selection and regression framework are finalized.

---

## Expected repository structure

```text
proteomics_npx_analysis/
├── README.md
├── .gitignore
├── environment.yml
├── requirements.txt
├── configs/
│   ├── dataset.env.example
│   └── ukbb_extract_config.yaml
├── data/
│   ├── raw/                       # gitignored
│   ├── interim/                   # gitignored
│   ├── processed/                 # gitignored
│   └── metadata/
│       ├── phenotype_base_fields.txt
│       ├── symptom_fields_selected.txt
│       ├── olink_proteins_fields.txt
│       └── README.md
├── docs/
│   ├── cohort_notes.md
│   ├── ukbb_field_selection.md
│   ├── ukbb_group_definition.md
│   └── analysis_plan.md
├── results/
│   ├── pediatric/
│   ├── ukbb/
│   └── comparative/
├── scripts/
│   ├── 00_list_fields.sh
│   ├── 01_extract_olink_participants.sh
│   ├── 02_build_field_lists.sh
│   ├── 03_generate_sql.sh
│   ├── 04_merge_pheno_olink.py
│   ├── 05_define_groups.py
│   ├── 06_run_logistic_regression.R
│   └── utils/
│       ├── common.sh
│       └── spark_helpers.py
└── sql/
    ├── pheno_query.sql
    └── olink_query.sql

```

## Key caveats

### Pediatric cohort

- very small sample size
- potential center/site confounding
- limited protein panel after filtering

### UK Biobank cohort

- case definitions depend heavily on phenotype/diagnosis selection logic
- adult population biology may not directly mirror pediatric Long COVID biology

## Repository workflow

- List all available UK Biobank fields
- Identify participants with Olink proteomics measurements
- Build phenotype and symptom field lists
- Generate Spark SQL queries for phenotype and Olink extraction
- Execute SQL in DNAnexus RAP Spark environment
- Merge phenotype and Olink tables on participant eid
- Define COVID / Long COVID subgroup labels
- Run protein-wise regression analyses

## UK Biobank extraction pipeline
bash scripts/00_list_fields.sh <dataset_id> data/interim
bash scripts/01_extract_olink_participants.sh <dataset_id> data/interim
bash scripts/02_build_field_lists.sh
bash scripts/03_generate_sql.sh <dataset_id>
python scripts/04_merge_pheno_olink.py \
  --pheno-sql sql/pheno_query.sql \
  --olink-sql sql/olink_query.sql \
  --eids data/interim/eids_olink_present.tsv \
  --out data/processed/merged_pheno_olink.parquet
python scripts/05_define_groups.py \
  --input data/processed/merged_pheno_olink.parquet \
  --out data/processed/merged_pheno_olink_grouped.parquet
Rscript scripts/06_run_logistic_regression.R