# Quality Control Notes

This document summarizes quality control steps applied to both the pediatric Olink NPX cohort and the UK Biobank Olink proteomics cohort.

---

# Shared QC Steps

## Sample-level QC

The following checks are performed for each participant:

- Calculate proportion of missing proteins per sample
- Compute overall protein abundance per sample (`total_abundance`)
- Identify outlier samples using:
  - distribution plots
  - boxplots
  - z-score or IQR-based thresholds
- Inspect whether missingness or total abundance differs across analysis groups

## Protein-level QC

For each protein:

- Calculate proportion of missing samples
- Inspect NPX value distribution
- Check for extreme skew or very low variance
- Remove proteins with poor detection or insufficient variability

Typical summary metrics include:

- percent missing
- mean NPX
- standard deviation
- detection rate

---

# Pediatric Cohort QC

## Existing preprocessing

The pediatric cohort data were already provided as:

- Olink NPX values
- log2-scaled and normalized
- proteins filtered to retain those detected in >80% of samples

## Additional checks performed

- Confirm no samples show extreme missingness
- Evaluate whether `total_abundance` differs between study groups
- Inspect missingness and abundance by:
  - Healthy
  - Long COVID neurological
  - Long COVID non-neurological

## Covariate used in downstream models

```text
total_abundance = mean or sum of NPX values across measured proteins
```

- total_abundance is included as an adjustment covariate in regression models to reduce the influence of global abundance shifts across samples.

----

# UK Biobank QC

## Required preprocessing checks

Because UK Biobank Olink data are extracted directly from RAP, the following QC steps should be performed before downstream analysis:

- Confirm extracted values are on an NPX-like scale
- Verify expected number of measured proteins per participant
- Examine missingness across both samples and proteins
- Compare protein distributions to the pediatric cohort when possible

## Protein filtering

Potential filtering rules:

- Remove proteins with >80% missing values
- Remove proteins with near-zero variance

Example:

```text
keep protein if missing_rate <= 0.80
```

## Sample filtering

Potential sample exclusion criteria:

- Excessive missing proteins
- Extreme total abundance
- Missing key phenotype or subgroup labels
- Additional technical variables

## The following UK Biobank variables may be evaluated as technical covariates:

- Olink plate (p30901_i0)
- Olink well (p30902_i0)
- Number of proteins measured (p30900_i0)

These may be used for:

- QC visualization
- Batch effect assessment
- Adjustment in regression models if needed