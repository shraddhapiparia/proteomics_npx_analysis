# ---------------------------------------------------------------------------
# run_one_protein()
#
# Fit one binary logistic regression for a single protein.
#
# df          : data frame already subset to the two comparison groups,
#               with a 'group_binary' column (ref = 0, test = 1).
# protein     : name of the protein column to test.
# group_col   : name of the binary outcome column (typically "group_binary").
# covariates  : character vector of additional covariate column names.
#
# Returns a one-row data frame matching the result schema:
#   protein, beta, OR, CI_lower, CI_upper, p_value, FDR, n, fit_warning
#
# Positive beta = higher protein abundance in the test group vs. reference.
# FDR is left as NA here; filled downstream by run_all_proteins().
#
# Warning handling  : warnings (e.g. convergence, separation) are caught,
#                     estimates are kept if available, fit_warning = TRUE.
# Error handling    : on model error, all estimates return NA, fit_warning = TRUE.
# Missing data      : na.omit inside glm(); proteins with < 10 non-missing
#                     observations in either group return the NA row directly.
# ---------------------------------------------------------------------------
run_one_protein <- function(df, protein, group_col, covariates) {

  na_row <- data.frame(
    protein     = protein,
    beta        = NA_real_,
    OR          = NA_real_,
    CI_lower    = NA_real_,
    CI_upper    = NA_real_,
    p_value     = NA_real_,
    FDR         = NA_real_,
    n           = NA_integer_,
    fit_warning = TRUE,
    stringsAsFactors = FALSE
  )

  # Guard: require both groups present and at least 10 non-missing observations
  # each. length() < 2 catches the case where one group is entirely absent from
  # df[[group_col]], which causes tapply() to silently return only one entry.
  group_nonmissing <- tapply(!is.na(df[[protein]]), df[[group_col]], sum)
  if (length(group_nonmissing) < 2 || any(group_nonmissing < 10)) return(na_row)

  fmla <- as.formula(
    paste(group_col, "~", paste(c(protein, covariates), collapse = " + "))
  )

  # Fit model — warnings are caught and muffled so glm() can still return
  # estimates; errors cause an immediate NULL return.
  had_warning <- FALSE

  fit <- tryCatch(
    withCallingHandlers(
      glm(fmla, data = df, family = binomial(), na.action = na.omit),
      warning = function(w) {
        had_warning <<- TRUE
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) NULL
  )

  if (is.null(fit)) return(na_row)

  # Extract tidy estimates; treat a downstream failure as an error case
  tidy_fit <- tryCatch(broom::tidy(fit), error = function(e) NULL)
  if (is.null(tidy_fit)) return(na_row)

  term_row <- tidy_fit[tidy_fit$term == protein, , drop = FALSE]
  if (nrow(term_row) == 0) return(na_row)

  beta  <- term_row$estimate
  p_val <- term_row$p.value
  n_obs <- stats::nobs(fit)

  # Wald confidence intervals (faster and stable at scale vs. profile CIs)
  ci <- tryCatch(
    stats::confint.default(fit, parm = protein, level = 0.95),
    error = function(e) matrix(c(NA_real_, NA_real_), nrow = 1)
  )

  data.frame(
    protein     = protein,
    beta        = beta,
    OR          = exp(beta),
    CI_lower    = exp(ci[1, 1]),
    CI_upper    = exp(ci[1, 2]),
    p_value     = p_val,
    FDR         = NA_real_,
    n           = as.integer(n_obs),
    fit_warning = had_warning,
    stringsAsFactors = FALSE
  )
}


# ---------------------------------------------------------------------------
# run_all_proteins()
#
# Iterate run_one_protein() over all proteins for one pairwise comparison,
# then append BH-adjusted FDR and sort by ascending p_value.
#
# df           : full analysis data frame (all groups present).
# group_col    : name of the group column in df.
# protein_cols : character vector of protein column names to test.
# covariates   : character vector of covariate column names.
# ref_level    : reference group label (encoded as 0 in group_binary).
# test_level   : test group label (encoded as 1 in group_binary).
#
# Stops early with an informative message if either group has fewer than
# 10 rows after subsetting (before per-protein missing data is removed).
#
# Returns a data frame with one row per protein, sorted by p_value ascending.
# FDR is computed only over non-NA p-values; failed proteins receive NA FDR.
# ---------------------------------------------------------------------------
run_all_proteins <- function(df, group_col, protein_cols, covariates,
                             ref_level, test_level) {

  df_sub <- df[df[[group_col]] %in% c(ref_level, test_level), ]

  # Validate overall group sizes before iterating
  group_n <- table(df_sub[[group_col]])

  if (!ref_level %in% names(group_n) || group_n[[ref_level]] < 10) {
    stop(sprintf(
      "Reference group '%s' has fewer than 10 observations (%d). Aborting.",
      ref_level,
      if (ref_level %in% names(group_n)) group_n[[ref_level]] else 0L
    ))
  }
  if (!test_level %in% names(group_n) || group_n[[test_level]] < 10) {
    stop(sprintf(
      "Test group '%s' has fewer than 10 observations (%d). Aborting.",
      test_level,
      if (test_level %in% names(group_n)) group_n[[test_level]] else 0L
    ))
  }

  # Encode binary outcome: reference = 0, test = 1
  # A new column is added so the original group factor is never modified.
  df_sub[["group_binary"]] <- as.integer(df_sub[[group_col]] == test_level)

  results <- purrr::map_dfr(
    protein_cols,
    ~ run_one_protein(
        df         = df_sub,
        protein    = .x,
        group_col  = "group_binary",
        covariates = covariates
      )
  )

  # BH FDR computed within this comparison only, over non-NA p-values
  valid_idx         <- !is.na(results$p_value)
  results$FDR[valid_idx] <- stats::p.adjust(
    results$p_value[valid_idx], method = "BH"
  )

  results[order(results$p_value, na.last = TRUE), ]
}
