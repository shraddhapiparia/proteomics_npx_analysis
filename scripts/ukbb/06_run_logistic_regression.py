import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# ============================================================
# 0) Paths
# ============================================================
# Cohort parquet written by 05_define_groups.py — contains
# analysis_group, all WHO flag columns, age, sex, and proteins.
PARQUET_IN = "cohort_with_groups_age_sex_shared_proteins.parquet"
TSV_OUT    = "protein_logistic_regression_results.tsv"

GROUP_COL      = "analysis_group"
AGE_FIELD      = "participant.p21022"
SEX_FIELD      = "participant.p31"
PROTEIN_PREFIX = "olink_instance_0."

COMPARISONS = [
    ("Healthy", "LC_Neuro"),
    ("Healthy", "LC_NonNeuro"),
    ("LC_Neuro", "LC_NonNeuro"),
]


# ============================================================
# 1) Read grouped cohort from 05_define_groups.py
# ============================================================
merged_df = spark.read.parquet(PARQUET_IN)
print("rows:", merged_df.count(), "| cols:", len(merged_df.columns))
merged_df.groupBy("analysis_group").count().orderBy("analysis_group").show(truncate=False)


# ============================================================
# 2) Select columns needed for regression
# ============================================================
protein_cols = [c for c in merged_df.columns if c.startswith(PROTEIN_PREFIX)]

print(f"Found {len(protein_cols)} protein columns")
print("Example proteins:", protein_cols[:5])

model_spark_df = merged_df.select(
    col("eid"),
    col("analysis_group"),
    col(f"`{AGE_FIELD}`").alias("age"),
    col(f"`{SEX_FIELD}`").alias("sex"),
    *[col(f"`{c}`") for c in protein_cols]
)

model_df_pd = model_spark_df.toPandas()

print("Pandas shape:", model_df_pd.shape)
print(model_df_pd[["analysis_group", "age", "sex"]].head())


# ============================================================
# 3) Protein-wise logistic regression
# ============================================================
def run_protein_logistic_regression_pd(df_pd, protein, group_a, group_b):
    comparison = f"{group_a}_vs_{group_b}"

    sub = (
        df_pd[df_pd["analysis_group"].isin([group_a, group_b])]
        [["analysis_group", "age", "sex", protein]]
        .dropna()
        .copy()
    )

    n_obs = len(sub)

    if n_obs < 50:
        return None

    sub["label"] = (sub["analysis_group"] == group_b).astype(int)

    protein_sd = sub[protein].std()
    if protein_sd == 0 or np.isnan(protein_sd):
        return None

    if sub["sex"].dtype == object:
        sex_map = {"Female": 0, "Male": 1, "F": 0, "M": 1}
        sub["sex"] = sub["sex"].map(sex_map)

    X = sub[[protein, "age", "sex"]].copy()
    X = sm.add_constant(X)
    y = sub["label"]

    try:
        model = sm.Logit(y, X).fit(disp=False)

        coef = model.params[protein]
        se   = model.bse[protein]
        p_value = model.pvalues[protein]

        return {
            "comparison": comparison,
            "protein":    protein,
            "coef":       float(coef),
            "SE":         float(se),
            "OR":         float(np.exp(coef)),
            "CI_lo":      float(np.exp(coef - 1.96 * se)),
            "CI_hi":      float(np.exp(coef + 1.96 * se)),
            "p_value":    float(p_value),
            "n":          int(n_obs),
        }

    except Exception as e:
        return {
            "comparison": comparison,
            "protein":    protein,
            "error":      str(e),
            "n":          int(n_obs),
        }


# ============================================================
# 4) Run comparisons
# ============================================================
all_results = []

for group_a, group_b in COMPARISONS:
    print(f"Running {group_a} vs {group_b}")

    for protein in protein_cols:
        res = run_protein_logistic_regression_pd(
            model_df_pd,
            protein=protein,
            group_a=group_a,
            group_b=group_b,
        )

        if res is not None and "p_value" in res:
            all_results.append(res)

results_df = pd.DataFrame(all_results)

print(results_df.columns)
print(results_df.head())


# ============================================================
# 5) FDR correction within comparison
# ============================================================
results_df["FDR"] = np.nan

for comp in results_df["comparison"].unique():
    mask = results_df["comparison"] == comp
    results_df.loc[mask, "FDR"] = multipletests(
        results_df.loc[mask, "p_value"],
        method="fdr_bh",
    )[1]


# ============================================================
# 6) Save regression results
# ============================================================
results_df = results_df.sort_values(["comparison", "p_value"])
results_df.to_csv(TSV_OUT, sep="\t", index=False)

print(f"Saved results: {TSV_OUT}")
print("Final result shape:", results_df.shape)
