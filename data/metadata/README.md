# Metadata files

This directory stores small, version-controlled inputs used to generate UK Biobank extraction queries.

## Files

- `phenotype_base_fields.txt`  
  Core phenotype and covariate field identifiers.

- `symptom_fields_selected.txt`  
  Selected symptom-related field identifiers used for Long COVID subgrouping.

- `olink_proteins_fields.txt`  
  Olink protein field identifiers to be extracted from the UK Biobank proteomics dataset.

## Notes

These files are maintained manually and serve as the main user-edited inputs to the extraction pipeline.
Generated files derived from them are written to `data/interim/`.