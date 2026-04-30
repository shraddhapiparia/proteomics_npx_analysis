# Proteomics NPX Analysis

This repository contains analysis code and documentation for Olink NPX proteomics analyses in pediatric Long COVID and related UK Biobank validation analyses.

The project focuses on identifying proteins associated with Long COVID subgroups, evaluating robustness to covariate adjustment, and comparing pediatric findings with adult UK Biobank proteomics results where possible.

## Current status

This repository is under active development.

The pediatric proteomics analysis scripts include differential protein regression, age/sex sensitivity analyses, covariate diagnostics, and downstream protein-protein interaction analysis.

The UK Biobank component is partially implemented using slef reported symptom based cohort definition but ICD10 code based diagnosis cohort is temporarily blocked because the UK Biobank Research Analysis Platform (RAP) is currently unavailable. Once RAP access is restored, the UKB extraction, cohort definition, and regression scripts will be rechecked and rerun.

## Repository structure

```text
proteomics_npx_analysis/
├── data/
│   └── metadata/              # Field IDs and metadata notes
├── docs/                      # Analysis notes and interpretation documents
├── scripts/
│   ├── config/                # Configuration files
│   ├── pediatric/             # Pediatric proteomics analysis scripts
│   ├── downstream/            # Downstream interpretation, including PPI
│   └── ukbb/                  # UK Biobank RAP workflow scripts
├── LICENSE
└── README.md
```
## Analysis modules

### 1. Pediatric subtype proteomics regression

The pediatric analysis evaluates protein-level associations across Long COVID subgroups using regression-based models.

Current scripts include:

    scripts/pediatric/pediatric_proteomics_regression.py
    scripts/pediatric/pediatric_subtype_regression.py
    scripts/pediatric/pediatric_subtype_covariate_diagnostics.py

These scripts support:

- protein-wise regression analyses
- subgroup comparisons with healthy cohort
- age and sex sensitivity analyses
- covariate diagnostics
- summary tables for interpretation

### 2. Pediatric covariate sensitivity analysis

Two proteins (TNFRSF11B and CCL2) reached nominal significance in the subtype comparison only after adjustment for age and sex. To characterize this discrepancy, a diagnostic analysis was run to understand whether the gain in significance reflects confounding, suppression, or improved precision.

### 3. UK Biobank RAP analysis

The UK Biobank component is designed to define adult Long COVID-related groups and compare adult Olink proteomics signals with pediatric findings.

Current UKB workflow scripts include:

    scripts/ukbb/00_list_fields.sh
    scripts/ukbb/01_extract_olink_participants.sh
    scripts/ukbb/02_build_field_lists.sh
    scripts/ukbb/03_generate_sql.sh
    scripts/ukbb/04_merge_pheno_olink.py
    scripts/ukbb/05_define_groups.py
    scripts/ukbb/06_run_logistic_regression.py

Planned UKB workflow:

    field selection → phenotype/Olink extraction → phenotype-proteomics merge → group definition → logistic regression → pediatric comparison

**Note:** UK Biobank RAP is currently down, so this part of the workflow is pending rerun and validation.

### 4. Downstream PPI analysis

Downstream scripts are used to interpret significant or prioritized proteins through protein-protein interaction analysis and network visualization.

    scripts/downstream/ppi_analysis.py
    scripts/downstream/ppi_network_plots.py

These analyses are intended to help evaluate whether prioritized proteins cluster into biologically interpretable pathways or interaction networks.

## How to run

The exact command-line workflow is still being finalized.

For now, scripts are organized by analysis stage:

    # Pediatric regression
    python scripts/pediatric/pediatric_proteomics_regression.py

    # Pediatric subtype regression
    python scripts/pediatric/pediatric_subtype_regression.py

    # Pediatric covariate diagnostics
    python scripts/pediatric/pediatric_subtype_covariate_diagnostics.py

    # Downstream PPI analysis
    python scripts/downstream/ppi_analysis.py
    python scripts/downstream/ppi_network_plots.py

A future update should add a single driver script or documented workflow that runs the analysis end to end.

## Outputs

Generated outputs are expected to include:

    results/
    ├── tables/
    ├── figures/
    ├── diagnostics/
    └── ppi/

Large result files and intermediate outputs are not committed by default. 

## Notes and limitations

- Pediatric analyses are based on a smaller cohort and should be interpreted cautiously.
- Some models may be sensitive to covariate adjustment because of limited sample size.
- UK Biobank analyses require RAP access and are currently pending rerun because RAP is unavailable.
- Protein identifiers and Olink panel naming conventions may require normalization before cross-cohort comparison.


## License

This project is released under the MIT License.
