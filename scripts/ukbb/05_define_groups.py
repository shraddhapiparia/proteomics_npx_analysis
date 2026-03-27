from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, sum as Fsum, count, avg,
    array_contains
)
from functools import reduce
import re

spark = SparkSession.builder.getOrCreate()

# ================================================================
# FIELD DEFINITIONS
# All fields you want to assess, explicitly listed and documented
# ================================================================

# Core fields
EID          = "participant.eid"
SEX          = "participant.p31"           # 0=Female, 1=Male
AGE          = "participant.p21022"        # Age at recruitment
ICD10        = "participant.p41270"        # Diagnoses - ICD10 (array)
ICD9         = "participant.p41271"        # Diagnoses - ICD9 (array)
COVID_FIRST  = "participant.p29157"        # Date first had COVID-19
COVID_LAST   = "participant.p29159"        # Date most recently had COVID-19
ANTIBODY     = "participant.p27981_i0"     # Antibody test result (instance 0)
COVID_TESTS  = "participant.p40100"        # Records of COVID-19 test results (array)

# Proteomics (Olink)
OLINK_PLATES = "participant.p30900_i0"    # Olink plates measured (instance 0)
OLINK_BATCH  = "participant.p30901_i0"    # Olink batch (instance 0)
OLINK_QC     = "participant.p30902_i0"    # Olink QC flag (instance 0)

# Backtick-quote a column name that may contain dots/special chars
def q(field):
    return col(f"`{field}`")


# ================================================================
# 1. READ PARQUET & CACHE
# Cache once — avoids repeated full scans for every action below.
# ================================================================
raw_df = spark.read.parquet("merged_pheno_olink_parquet")
raw_df.cache()

n_rows = raw_df.count()
n_cols = len(raw_df.columns)
print(f"Rows: {n_rows:,}  |  Cols: {n_cols:,}")


# ================================================================
# 2. VALIDATE EXPECTED FIELDS EXIST
# Fail early with a clear message rather than a cryptic KeyError.
# ================================================================
EXPECTED_FIELDS = [
    EID, SEX, AGE, ICD10, ICD9,
    COVID_FIRST, COVID_LAST, ANTIBODY, COVID_TESTS,
    OLINK_PLATES, OLINK_BATCH, OLINK_QC,
] + SYMPTOM_FIELDS_RAW

missing_fields = [f for f in EXPECTED_FIELDS if f not in raw_df.columns]
if missing_fields:
    print(f"WARNING — {len(missing_fields)} expected field(s) missing from schema:")
    for f in missing_fields:
        print(f"  ✗ {f}")
else:
    print("All expected fields present ✓")

# Work only with fields actually present
present_fields = [f for f in EXPECTED_FIELDS if f in raw_df.columns]


# ================================================================
# 3. SINGLE-PASS MISSINGNESS SUMMARY
# One Spark job for all fields instead of one job per field.
# ================================================================
def missingness_summary(df, fields, total_rows):
    """Return a DataFrame with null counts and pct for every field."""
    agg_exprs = [
        count(when(q(f).isNull(), 1)).alias(f)
        for f in fields
    ]
    null_counts = df.agg(*agg_exprs).collect()[0].asDict()
    rows = []
    for f in fields:
        nc = null_counts[f]
        rows.append((f, nc, total_rows, round(nc / total_rows * 100, 2)))
    return spark.createDataFrame(rows, ["field", "null_count", "total", "pct_null"])

miss_df = missingness_summary(raw_df, present_fields, n_rows)
miss_df.orderBy("pct_null", ascending=False).show(50, truncate=False)


# ================================================================
# 4. VALUE DISTRIBUTIONS (demographic / key fields only)
# One groupBy per field — kept separate so output is readable.
# ================================================================
def show_counts(df, field, n=20):
    if field not in df.columns:
        print(f"  (skipping {field} — not in schema)")
        return
    print(f"\n===== {field} =====")
    df.groupBy(q(field)).count().orderBy("count", ascending=False).show(n, truncate=False)

for f in [SEX, AGE, ANTIBODY]:
    show_counts(raw_df, f)


# ================================================================
# 5. SYMPTOM FIELDS 
# Pairs are (current_field, duration_field) where current is symptom
# and duration is current+1.
# ================================================================
SYMPTOM_FIELDS_RAW = [
    "participant.p28606", "participant.p28607",
    "participant.p28609", "participant.p28610",
    "participant.p28612", "participant.p28613",
    "participant.p28615", "participant.p28616",
    "participant.p28624", "participant.p28625",
    "participant.p28627", "participant.p28628",
    "participant.p28630", "participant.p28631",
    "participant.p28633", "participant.p28634",
    "participant.p28642", "participant.p28643",
    "participant.p28645", "participant.p28646",
    "participant.p28648", "participant.p28649",
    "participant.p28654", "participant.p28655",
    "participant.p28657", "participant.p28658",
    "participant.p28663", "participant.p28664",
    "participant.p28678", "participant.p28679",
    "participant.p28681", "participant.p28682",
    "participant.p28684", "participant.p28685",
    "participant.p28687", "participant.p28688",
    "participant.p28693", "participant.p28694",
    "participant.p28696", "participant.p28697",
    "participant.p28699", "participant.p28700",
    "participant.p28702", "participant.p28703",
    "participant.p28711", "participant.p28712",
    "participant.p28714", "participant.p28715",
    "participant.p28720", "participant.p28721",
    "participant.p28723", "participant.p28724",
    "participant.p28726", "participant.p28727",
    "participant.p28732", "participant.p28733",
]

# Validate all symptom fields exist in the schema
missing_symptom = [f for f in SYMPTOM_FIELDS_RAW if f not in raw_df.columns]
if missing_symptom:
    print(f"WARNING: {len(missing_symptom)} symptom field(s) missing from schema:")
    for f in missing_symptom:
        print(f"  ✗ {f}")

symptom_fields = [f for f in SYMPTOM_FIELDS_RAW if f in raw_df.columns]

# Build (current, duration) pairs — consecutive even/odd field numbers
def get_base_field_num(colname):
    """Returns field number only for non-instance-qualified fields."""
    m = re.fullmatch(r"participant\.p(\d+)", colname)
    return int(m.group(1)) if m else None

field_map = {}
for c in symptom_fields:
    num = get_base_field_num(c)
    if num is not None:
        field_map[num] = c

pairs = []
for num in sorted(field_map):
    if (num + 1) in field_map:
        pairs.append((field_map[num], field_map[num + 1]))

# Sanity check — every field should be in exactly one pair
paired_fields = {f for pair in pairs for f in pair}
unpaired = [f for f in symptom_fields if f not in paired_fields]
if unpaired:
    print(f"WARNING: {len(unpaired)} symptom field(s) not paired (check for missing partner):")
    for f in unpaired:
        print(f"  ✗ {f}")

print(f"\nSymptom fields loaded  : {len(symptom_fields)}")
print(f"Current/duration pairs : {len(pairs)}")   # expect 28


# ================================================================
# 6. COVID POSITIVE FLAG
# Positive = has a COVID-first date
# ================================================================
covid_pos_expr = (
    q(COVID_FIRST).isNotNull()
    | (q(ANTIBODY) == 1)
)
# Add COVID_TESTS check only if it's an array-type column
if COVID_TESTS in raw_df.columns:
    # array_contains works if COVID_TESTS is ArrayType; adjust value as needed
    try:
        covid_pos_expr = covid_pos_expr | array_contains(q(COVID_TESTS), "Positive")
    except Exception:
        pass  # Non-array column — skip this condition

analysis_df = raw_df.withColumn(
    "covid_positive_flag",
    when(covid_pos_expr, 1).when(~covid_pos_expr, 0).otherwise(lit(None))
)


# ================================================================
# 7. SYMPTOM FLAGS
# use reduce() to combine Column expressions correctly.
# ================================================================

# Helper: safely reduce a list of Column bool expressions with OR or ADD
def reduce_or(exprs):
    return reduce(lambda a, b: a | b, exprs) if exprs else lit(False)

def reduce_add(exprs):
    return reduce(lambda a, b: a + b, exprs) if exprs else lit(0)


# -- Any current symptom (current == 1) --
any_current_exprs = [q(cur) == 1 for cur, _ in pairs]
any_current_col = when(reduce_or(any_current_exprs), 1).otherwise(0) if pairs else lit(0)

# -- Long-COVID flag: current == 1 AND duration in (3=4–12wk, 4=>12wk) --
lc_exprs = [(q(cur) == 1) & q(lng).isin(3, 4) for cur, lng in pairs]
lc_col = when(reduce_or(lc_exprs), 1).otherwise(0) if pairs else lit(0)

# -- Neuro symptoms (fixed field numbers per your schema) --
NEURO_CURRENT_FIELDS = [
    "participant.p28720",   # Problems thinking / concentrating
    "participant.p28723",   # Problems communicating
    "participant.p28633",   # Headaches
    "participant.p28732",   # Numbness / tingling
]
neuro_pairs = [
    (cur, field_map[get_base_field_num(cur) + 1])
    for cur in NEURO_CURRENT_FIELDS
    if cur in raw_df.columns
    and get_base_field_num(cur) is not None
    and (get_base_field_num(cur) + 1) in field_map
]
print(f"\nNeuro current/duration pairs: {len(neuro_pairs)}")
for p in neuro_pairs:
    print(f"  {p[0]}  →  {p[1]}")

neuro_lc_exprs = [(q(cur) == 1) & q(lng).isin(3, 4) for cur, lng in neuro_pairs]
neuro_lc_col = when(reduce_or(neuro_lc_exprs), 1).otherwise(0) if neuro_pairs else lit(0)

# -- Symptom burden counts (FIX: was Python sum(), now reduce_add) --
n_current_col = reduce_add([when(q(cur) == 1, 1).otherwise(0) for cur, _ in pairs]) if pairs else lit(0)
n_lc_col      = reduce_add([when((q(cur) == 1) & q(lng).isin(3, 4), 1).otherwise(0) for cur, lng in pairs]) if pairs else lit(0)
n_neuro_lc_col= reduce_add([when((q(cur) == 1) & q(lng).isin(3, 4), 1).otherwise(0) for cur, lng in neuro_pairs]) if neuro_pairs else lit(0)

# Add all derived columns in one pass (Spark 3.3+: withColumns avoids plan explosion)
derived_cols = {
    "any_current_symptom"   : any_current_col,
    "long_covid_symptom_flag": lc_col,
    "neuro_long_covid_flag"  : neuro_lc_col,
    "n_current_symptoms"     : n_current_col,
    "n_lc_symptoms"          : n_lc_col,
    "n_neuro_lc_symptoms"    : n_neuro_lc_col,
}

try:
    # Spark 3.3+
    analysis_df = analysis_df.withColumns(derived_cols)
except AttributeError:
    # Fallback for Spark < 3.3
    for name, expr in derived_cols.items():
        analysis_df = analysis_df.withColumn(name, expr)


# ================================================================
# 8. ANALYSIS GROUPS
# FIX: explicit null branch first so nulls never silently become
# "Unclassified". Healthy = confirmed COVID-negative (flag == 0).
# ================================================================
analysis_df = analysis_df.withColumn(
    "analysis_group",
    when(col("covid_positive_flag").isNull(),                                                          "Unknown_COVID_status")
    .when(col("covid_positive_flag") == 0,                                                             "Healthy")
    .when((col("covid_positive_flag") == 1) & (col("long_covid_symptom_flag") == 0),                  "COVID_no_LC")
    .when((col("covid_positive_flag") == 1) & (col("long_covid_symptom_flag") == 1)
          & (col("neuro_long_covid_flag") == 1),                                                       "LC_Neuro")
    .when((col("covid_positive_flag") == 1) & (col("long_covid_symptom_flag") == 1)
          & (col("neuro_long_covid_flag") == 0),                                                       "LC_NonNeuro")
    .otherwise("Unclassified")   # Should be empty — log if not
)


# ================================================================
# 9. DISTRIBUTIONS OF DERIVED FIELDS
# ================================================================
for f in ["covid_positive_flag", "any_current_symptom",
          "long_covid_symptom_flag", "neuro_long_covid_flag", "analysis_group"]:
    show_counts(analysis_df, f)


# ================================================================
# 10. GROUP SUMMARIES — demographics & proteomics
# One aggregation per group instead of separate jobs.
# ================================================================
agg_exprs = [count("*").alias("n")]

if AGE in analysis_df.columns:
    agg_exprs += [
        avg(q(AGE)).alias("mean_age"),
        Fsum(q(AGE)).alias("sum_age"),   # kept for back-compat
    ]

if OLINK_PLATES in analysis_df.columns:
    agg_exprs.append(Fsum(q(OLINK_PLATES)).alias("sum_olink_plates"))

if OLINK_QC in analysis_df.columns:
    agg_exprs.append(avg(q(OLINK_QC)).alias("mean_olink_qc"))

print("\n=== Group summary ===")
analysis_df.groupBy("analysis_group").agg(*agg_exprs).orderBy("analysis_group").show(truncate=False)

# Sex breakdown by group
if SEX in analysis_df.columns:
    print("\n=== Sex × group ===")
    analysis_df.groupBy("analysis_group", q(SEX)).count().orderBy("analysis_group", q(SEX)).show(truncate=False)


# ================================================================
# 11. UNCLASSIFIED ROW AUDIT
# Should be empty. If not, surface why for investigation.
# ================================================================
unclassified = analysis_df.filter(col("analysis_group") == "Unclassified")
n_unclass = unclassified.count()
if n_unclass > 0:
    print(f"\nWARNING: {n_unclass:,} rows landed in 'Unclassified' — review logic:")
    unclassified.select(
        "covid_positive_flag", "long_covid_symptom_flag",
        "neuro_long_covid_flag", "analysis_group"
    ).show(20, truncate=False)


# ================================================================
# 12. RELEASE CACHE
# ================================================================
raw_df.unpersist()
print("\nDone. Cache released.")