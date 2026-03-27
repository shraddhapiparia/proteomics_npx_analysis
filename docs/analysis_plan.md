# Analysis Plan

## 1. Cohort definition

### Pediatric
- Use provided labels: Healthy, Long COVID
- Subgroups defined based on neurocognitive symptoms: headache, brain fog, difficult concentration, unusual behavior, unusual memory problems.

### UK Biobank
- Define COVID / Long COVID using:
  - diagnosis fields
  - antibody / test data (as needed)
- Define Long COVID using WHO equivalent symptoms in UKBB as mentioned in the article: https://www.nature.com/articles/s41467-025-62354-0
- Subgroups defined based on neurocognitive symptoms: problems communicating, headaches, numbness/tingling

---

## 2. QC and filtering

- Apply shared QC pipeline
- Ensure comparable protein sets across cohorts

---

## 3. Logistic regression

### Model

For each protein:

outcome ~ protein + age + sex (+ total_abundance in pediatric cohort)

### Outputs
- beta
- odds ratio (OR)
- 95% CI
- p-value
- FDR (BH)

---

## 4. Subgroup analysis

- LC-Neuro vs Healthy
- LC-NonNeuro vs Healthy
- LC-Neuro vs LC-NonNeuro

---

## 5. Cross-cohort comparison

- Identify overlapping significant proteins
- Compare direction of effects
- Highlight consistent signals

---

## 6. Future work

- Protein–protein interaction (PPI) analysis
- Pathway enrichment