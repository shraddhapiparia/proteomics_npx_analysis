# UK Biobank group definition notes

## Goal

The goal is to define analysis groups for downstream proteomic comparison in UK Biobank, including:

- participants with COVID history
- participants with prolonged symptom burden consistent with Long COVID
- comparison groups such as COVID without Long COVID and healthy controls

## Planned logic

Long COVID-related subgrouping is based on combinations of:

- COVID history fields
- current symptom fields
- symptom duration fields

A common working rule is:

- current symptom present = coded as `1`
- prolonged duration = coded as `4` for more than 12 weeks

## Important caveat

Group definitions in UK Biobank are sensitive to the exact phenotype and symptom logic used. For this reason, grouping code is versioned in the repository and should be documented whenever logic changes.