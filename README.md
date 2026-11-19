# Proteomics NPX Analysis

Reproducible analysis workflows for Olink NPX proteomics in pediatric Long COVID and UK Biobank.

The repository includes protein association analyses, sensitivity analyses, cross-center comparisons, predictive performance evaluation, and exploratory pathway analysis.

## Repository structure

```text
scripts/
├── pediatric/
│   ├── 01_cohort_qc.py
│   ├── 02_protein_associations.py
│   ├── 03_crosscenter_analysis.py
│   ├── 04_discrimination.py
│   ├── 05_pathways_figures.py
│   └── README.md
├── ukbb/                 # UK Biobank RAP workflow
├── downstream/           # PPI analysis
└── shared/               # Reusable functions

docs/                     # Analysis and methodology notes
data/                     # Local inputs and reference metadata
results/                  # Generated tables and figures
```

## Running the analyses

Run commands from the repository root. Install the required Python and R packages before execution. The pediatric workflows use controlled-access input data that are not included in this repository.

### Pediatric Long COVID

```bash
# 1. Cohort preparation and QC
python scripts/pediatric/01_cohort_qc.py --outdir results/pediatric/rerun \
  --metadata_path results/pediatric/rerun/harmonized_metadata.tsv

# 2. Protein association and sensitivity analyses
python scripts/pediatric/02_protein_associations.py --outdir results/pediatric/rerun

# 3. Cross-center analyses and batch diagnostics
python scripts/pediatric/03_crosscenter_analysis.py --outdir results/pediatric/rerun

# 4. Cross-validated discrimination
python scripts/pediatric/04_discrimination.py --outdir results/pediatric/rerun/auc_validation_revised

# 5. Pathway enrichment and figures
python scripts/pediatric/05_pathways_figures.py --outdir results/pediatric/rerun
```

Individual analyses can be run using `--stage` where supported. Existing outputs are protected against accidental overwriting; use `--outdir` for new runs.

See [Pediatric analysis README](scripts/pediatric/README.md) for individual stages, input requirements, and methodological details.

### UK Biobank

The UK Biobank workflow is implemented in `scripts/ukbb/` and requires access to the UK Biobank Research Analysis Platform (RAP).

See [UKBB workflow](docs/ukbb_rap_workflow.md) for details.

### Downstream PPI analysis

```bash
python scripts/downstream/ppi_analysis.py
python scripts/downstream/ppi_network_plots.py
```

## Data and outputs

Individual-level proteomics and phenotype data are not distributed with the repository. Generated results are written to `results/`; large and intermediate files are excluded from version control.

## License

MIT License.