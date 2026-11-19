# ============================================================
# Firth logistic regression sensitivity analysis
# Candidate proteins:
#   CCL2, IL17C, TNFRSF11B, FGF21
#
# Models:
#   LC-Neuro vs LC-NonNeuro  -> raw NPX
#   LC-Neuro vs Healthy      -> ComBat-corrected NPX
#
# Each model:
#   neuro_status ~ protein + age + sex
#
# OR = odds ratio per 1 NPX-unit increase
# ============================================================

library(logistf)
library(dplyr)
library(readr)

# Optional path overrides (paths are relative to the repository root):
#   Rscript scripts/pediatric/firth_regression.R [--outdir DIR] [--meta-file F] [--combat-file F]
cli_args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  i <- match(flag, cli_args)
  if (is.na(i) || i == length(cli_args)) default else cli_args[i + 1L]
}

candidate_proteins <- c(
  "CCL2",
  "IL17C",
  "TNFRSF11B",
  "FGF21"
)

# ------------------------------------------------------------
# EDIT THESE PATHS/COLUMN NAMES TO MATCH YOUR FILES
# ------------------------------------------------------------

meta_file <- arg_value("--meta-file", "data/interim/harmonized_metadata.tsv")

# Raw sample x protein NPX matrix
raw_npx_file <- "data/raw/Inflammation_NPX_AgeM.txt"

# Your already-generated ComBat corrected matrix
combat_file <- arg_value("--combat-file", "results/pediatric/combat_adjusted_npx.tsv")
out_file <- file.path(arg_value("--outdir", "results/pediatric"), "firth_candidate_sensitivity.tsv")

id_col    <- "SampleID"
group_col <- "group"
age_col   <- "harmonized_age"
sex_col   <- "sex_01"

neuro_group    <- "LC_Neuro"
nonneuro_group <- "LC_NonNeuro_strict"
healthy_group  <- "Healthy"


# ============================================================
# Helpers
# ============================================================

read_any <- function(path) {
  read.delim(path, check.names = FALSE)
}


fit_firth_one <- function(df, protein, comparison) {

  d <- df %>%
    select(
      y,
      all_of(protein),
      age,
      sex
    ) %>%
    rename(
      protein_value = all_of(protein)
    ) %>%
    filter(
      complete.cases(.)
    )

  # Ensure numeric covariates
  d$y <- as.integer(d$y)
  d$protein_value <- as.numeric(d$protein_value)
  d$age <- as.numeric(d$age)

  # Keep sex treatment consistent with your primary analysis
  # If sex is already 0/1 numeric, this is fine.
  d$sex <- as.numeric(d$sex)

  if (sd(d$protein_value) == 0) {
    return(
      data.frame(
        comparison = comparison,
        protein = protein,
        n = nrow(d),
        cases = sum(d$y == 1),
        controls = sum(d$y == 0),
        beta_firth = NA,
        OR_firth = NA,
        CI_low = NA,
        CI_high = NA,
        p_firth = NA
      )
    )
  }

  fit <- logistf(
    y ~ protein_value + age + sex,
    data = d,
    pl = TRUE
  )

  beta <- coef(fit)["protein_value"]

  out <- data.frame(
    comparison = comparison,
    protein = protein,
    n = nrow(d),
    cases = sum(d$y == 1),
    controls = sum(d$y == 0),

    beta_firth = beta,

    # OR per 1 NPX-unit increase
    OR_firth = exp(beta),

    CI_low = exp(fit$ci.lower["protein_value"]),
    CI_high = exp(fit$ci.upper["protein_value"]),

    p_firth = fit$prob["protein_value"]
  )

  return(out)
}


run_firth_set <- function(df, comparison_name) {

  results <- lapply(
    candidate_proteins,
    function(p) fit_firth_one(
      df = df,
      protein = p,
      comparison = comparison_name
    )
  )

  bind_rows(results)
}


# ============================================================
# Load metadata
# ============================================================

meta <- read_any(meta_file)

meta[[id_col]] <- trimws(
  as.character(meta[[id_col]])
)


# ============================================================
# 1. PRIMARY:
#    LC-Neuro vs LC-NonNeuro
#    RAW / UNCORRECTED NPX
# ============================================================

raw_npx <- read_any(raw_npx_file)

# Assumes wide matrix:
# first column = SampleID
#
# If your raw file is long Olink format,
# use your existing processed raw matrix instead.

names(raw_npx)[1] <- id_col

raw_npx[[id_col]] <- trimws(
  as.character(raw_npx[[id_col]])
)

primary <- meta %>%
  filter(
    .data[[group_col]] %in%
      c(neuro_group, nonneuro_group)
  ) %>%
  select(
    all_of(c(
      id_col,
      group_col,
      age_col,
      sex_col
    ))
  ) %>%
  inner_join(
    raw_npx,
    by = id_col
  )

primary$y <- ifelse(
  primary[[group_col]] == neuro_group,
  1,
  0
)

primary$age <- primary[[age_col]]
primary$sex <- primary[[sex_col]]

primary_results <- run_firth_set(
  primary,
  "LC-Neuro vs LC-NonNeuro"
)


# ============================================================
# 2. CROSS-CENTER:
#    LC-Neuro vs Healthy
#    COMBAT-CORRECTED NPX
# ============================================================

combat <- read.delim(
  combat_file,
  check.names = FALSE
)

combat[[id_col]] <- trimws(
  as.character(combat[[id_col]])
)

healthy_comp <- meta %>%
  filter(
    .data[[group_col]] %in%
      c(neuro_group, healthy_group)
  ) %>%
  select(
    all_of(c(
      id_col,
      group_col,
      age_col,
      sex_col
    ))
  ) %>%
  inner_join(
    combat,
    by = id_col
  )

healthy_comp$y <- ifelse(
  healthy_comp[[group_col]] == neuro_group,
  1,
  0
)

healthy_comp$age <- healthy_comp[[age_col]]
healthy_comp$sex <- healthy_comp[[sex_col]]

healthy_results <- run_firth_set(
  healthy_comp,
  "LC-Neuro vs Healthy"
)
expanded_comp <- meta %>%
  filter(
    .data[[group_col]] %in%
      c(neuro_group, nonneuro_group, "CACTUS")
  ) %>%
  select(
    all_of(c(
      id_col,
      group_col,
      age_col,
      sex_col
    ))
  ) %>%
  inner_join(
    combat,
    by = id_col
  )

expanded_comp$y <- ifelse(
  expanded_comp[[group_col]] == neuro_group,
  1,
  0
)

expanded_comp$age <- expanded_comp[[age_col]]
expanded_comp$sex <- expanded_comp[[sex_col]]

expanded_results <- run_firth_set(
  expanded_comp,
  "LC-Neuro vs Expanded reference"
)


# ============================================================
# Combine and save
# ============================================================

all_results <- bind_rows(
  primary_results,
  healthy_results,
  expanded_results
)

all_results <- all_results %>%
  mutate(
    across(
      c(
        beta_firth,
        OR_firth,
        CI_low,
        CI_high,
        p_firth
      ),
      ~ round(.x, 4)
    )
  )

print(all_results)

write_tsv(
  all_results,
  out_file
)

cat(
  "\nSaved:\n",
  out_file, "\n"
)