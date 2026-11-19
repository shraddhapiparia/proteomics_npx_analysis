#!/usr/bin/env Rscript
# =============================================================================
# combat_adjustment_sva.R — sva::ComBat batch correction
# Pediatric Long COVID proteomics  |  SENSITIVITY ANALYSIS
# =============================================================================
#
# PURPOSE
# -------
# This is a batch-correction sensitivity analysis, not the primary model.
# Because every group belongs to exactly one site/source batch, ComBat-adjusted
# results must always be interpreted alongside:
#   - The raw-NPX primary models (02_protein_associations.py)
#   - The CACTUS-indicator sensitivity (cactus_indicator_sensitivity.py; no longer
#     in the working tree) and the ComBat CACTUS-indicator model in
#     03_crosscenter_analysis.py (stage combat_regression)
#
# BATCH VARIABLE
# --------------
#   LC_source        — LC_Neuro + LC_NonNeuro_strict
#   HC_CACTUS_source — Healthy  + CACTUS
#
# GROUP ASSIGNMENT  (mirrors Python canonical scripts)
# ----------------------------------------------------
#   Healthy            SampleID starts with "HC"
#   LC_Neuro           Group == "LC" & NEURO == 1
#   LC_NonNeuro_strict Group == "LC" & NEURO == 0
#   CACTUS             SampleID starts with "CACTUS"
#   LC_NonNeuro_expanded = LC_NonNeuro_strict + CACTUS  (derived)
#
# BIOLOGICAL MOD COVARIATE
# ------------------------
#   is_LC_Neuro (1 = LC_Neuro, 0 = all others) is passed to ComBat as mod so
#   ComBat tries to preserve LC_Neuro biology. This indicator is NOT perfectly
#   collinear with batch: LC_NonNeuro_strict shares LC_source but has
#   is_LC_Neuro = 0. Use --no-mod to omit mod if ComBat fails numerically.
#
# MISSING VALUES
# --------------
# Missing NPX cells are temporarily imputed with the protein-wise mean across
# samples before ComBat (ComBat requires a complete numeric matrix).
# Original NA positions are restored in the saved output.
#
# INPUTS
# ------
#   data/raw/Inflammation_NPX_AgeM.txt
#   data/raw/phenotype.txt
#
# OUTPUTS  (results/pediatric/)
# --------------------------------
#   combat_batch_group_crosstab.tsv
#   combat_adjusted_npx.tsv          samples × proteins, SampleID column
#   combat_metadata.tsv
#   combat_pca_before_after.pdf
#
# USAGE
# -----
#   Rscript scripts/pediatric/combat_adjustment_sva.R
#   Rscript scripts/pediatric/combat_adjustment_sva.R --no-mod
#   Rscript scripts/pediatric/combat_adjustment_sva.R --outdir <dir>   (default results/pediatric)
# =============================================================================

# ── 0. Dependencies ───────────────────────────────────────────────────────────
if (!requireNamespace("sva", quietly = TRUE)) {
  stop(
    "Package 'sva' is required but not installed.\n",
    "Install via Bioconductor:\n",
    "  if (!requireNamespace('BiocManager', quietly = TRUE)) {\n",
    "    install.packages('BiocManager')\n",
    "  }\n",
    "  BiocManager::install('sva')\n",
    call. = FALSE
  )
}
suppressPackageStartupMessages(library(sva))

# ── 1. Command-line arguments ─────────────────────────────────────────────────
cli_args <- commandArgs(trailingOnly = TRUE)
no_mod   <- "--no-mod" %in% cli_args
outdir_i <- match("--outdir", cli_args)
outdir_arg <- if (!is.na(outdir_i) && outdir_i < length(cli_args)) cli_args[outdir_i + 1L] else NA_character_

# ── 2. Paths ──────────────────────────────────────────────────────────────────
# Derive root from the script's own path so it works from any working directory.
raw_args  <- commandArgs(trailingOnly = FALSE)
file_flag <- raw_args[grep("^--file=", raw_args)]
if (length(file_flag) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_flag[1]))
  root_dir    <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  # Interactive / sourced fallback
  root_dir <- getwd()
  message("[INFO] Script path not detected; using working directory as root: ", root_dir)
}

npx_file   <- file.path(root_dir, "data", "raw", "Inflammation_NPX_AgeM.txt")
pheno_file <- file.path(root_dir, "data", "raw", "phenotype.txt")
out_dir    <- if (is.na(outdir_arg)) file.path(root_dir, "results", "pediatric") else outdir_arg
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (f in c(npx_file, pheno_file)) {
  if (!file.exists(f)) {
    stop("Input file not found: ", f, call. = FALSE)
  }
}

cat("\n", strrep("=", 60), "\n", sep = "")
cat("ComBat BATCH CORRECTION  (sva::ComBat)\n")
cat(strrep("=", 60), "\n")
cat(sprintf("  --no-mod : %s\n", no_mod))
cat(sprintf("  Root     : %s\n", root_dir))
cat(sprintf("  Out dir  : %s\n", out_dir))

# ── 3. Load data ──────────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("LOADING DATA\n")
cat(strrep("=", 60), "\n")

npx   <- read.table(npx_file,   sep = "\t", header = TRUE,
                    stringsAsFactors = FALSE, check.names = FALSE)
pheno <- read.table(pheno_file, sep = "\t", header = TRUE,
                    stringsAsFactors = FALSE, check.names = FALSE)

npx$SampleID   <- trimws(npx$SampleID)
pheno$SampleID <- trimws(pheno$SampleID)

protein_cols <- setdiff(colnames(npx), "SampleID")

cat(sprintf("  NPX:   %d samples, %d proteins\n", nrow(npx), length(protein_cols)))
cat(sprintf("  Pheno: %d rows, %d columns\n",     nrow(pheno), ncol(pheno)))

required_pheno <- c("SampleID", "Group", "NEURO")
missing_req    <- setdiff(required_pheno, colnames(pheno))
if (length(missing_req) > 0) {
  stop("phenotype.txt is missing required columns: ",
       paste(missing_req, collapse = ", "), call. = FALSE)
}

# ── 4. Assign groups ──────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("GROUP ASSIGNMENT\n")
cat(strrep("=", 60), "\n")

lc_rows   <- pheno[!is.na(pheno$Group) & pheno$Group == "LC",
                   c("SampleID", "NEURO"), drop = FALSE]
neuro_map <- setNames(as.integer(lc_rows$NEURO), lc_rows$SampleID)

assign_group <- function(sid) {
  if (startsWith(sid, "HC"))     return("Healthy")
  if (sid %in% names(neuro_map)) {
    return(if (neuro_map[[sid]] == 1L) "LC_Neuro" else "LC_NonNeuro_strict")
  }
  if (startsWith(sid, "CACTUS")) return("CACTUS")
  return("Unknown")
}

npx$group <- vapply(npx$SampleID, assign_group, character(1L))

group_counts <- table(npx$group)
for (g in c("Healthy", "LC_Neuro", "LC_NonNeuro_strict", "CACTUS", "Unknown")) {
  n   <- ifelse(g %in% names(group_counts), as.integer(group_counts[[g]]), 0L)
  tag <- if (g == "CACTUS") "  (contributes to LC_NonNeuro_expanded)" else ""
  cat(sprintf("  %-26s: n=%d%s\n", g, n, tag))
}
n_expanded <- sum(npx$group %in% c("LC_NonNeuro_strict", "CACTUS"))
cat(sprintf("  %-26s: n=%d  (derived)\n", "LC_NonNeuro_expanded", n_expanded))

# ── 5. Merge phenotype ────────────────────────────────────────────────────────
keep_meta <- intersect(c("SampleID", "Gender"), colnames(pheno))
df        <- merge(npx, pheno[, keep_meta, drop = FALSE],
                   by = "SampleID", all.x = TRUE)

# Preserve original row order from npx
df <- df[match(npx$SampleID, df$SampleID), ]

# ── 6. Total-abundance outlier exclusion ──────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("TOTAL-ABUNDANCE OUTLIER EXCLUSION\n")
cat(strrep("=", 60), "\n")

df$total_abundance <- rowSums(df[, protein_cols, drop = FALSE], na.rm = TRUE)
ta_mean <- mean(df$total_abundance)
ta_sd   <- sd(df$total_abundance)
df$ta_z <- (df$total_abundance - ta_mean) / ta_sd

n_excl   <- sum(abs(df$ta_z) > 3)
excl_ids <- df$SampleID[abs(df$ta_z) > 3]
if (n_excl > 0) {
  cat(sprintf("  Excluded %d sample(s) with |total_abundance z| > 3:\n", n_excl))
  cat(sprintf("    %s\n", paste(excl_ids, collapse = ", ")))
  df <- df[abs(df$ta_z) <= 3, ]
} else {
  cat("  No samples excluded (no |z| > 3).\n")
}
cat(sprintf("  Samples after exclusion: %d\n", nrow(df)))

# ── 7. Define batch and biological covariates ─────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("COVARIATES\n")
cat(strrep("=", 60), "\n")

df$site_source <- ifelse(
  df$group %in% c("LC_Neuro", "LC_NonNeuro_strict"), "LC_source",
  ifelse(df$group %in% c("Healthy", "CACTUS"), "HC_CACTUS_source", NA_character_)
)
df$cactus_indicator <- as.integer(df$group == "CACTUS")
df$is_LC_Neuro      <- as.integer(df$group == "LC_Neuro")

cat(sprintf("  site_source:       LC_source=%d  HC_CACTUS_source=%d  NA=%d\n",
            sum(df$site_source == "LC_source",        na.rm = TRUE),
            sum(df$site_source == "HC_CACTUS_source", na.rm = TRUE),
            sum(is.na(df$site_source))))
cat(sprintf("  cactus_indicator:  1 (CACTUS)=%d  0 (other)=%d\n",
            sum(df$cactus_indicator == 1L), sum(df$cactus_indicator == 0L)))
cat(sprintf("  is_LC_Neuro:       1=%d  0=%d\n",
            sum(df$is_LC_Neuro == 1L), sum(df$is_LC_Neuro == 0L)))

# ── 8. Restrict to the 4 analysis groups ─────────────────────────────────────
analysis_groups <- c("LC_Neuro", "LC_NonNeuro_strict", "Healthy", "CACTUS")
n_unknown <- sum(df$group == "Unknown")
if (n_unknown > 0) {
  cat(sprintf("\n  [INFO] %d 'Unknown' sample(s) excluded from ComBat.\n", n_unknown))
}
df_cb <- df[df$group %in% analysis_groups, ]

# ── 9. Save batch × group crosstab ───────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("BATCH x GROUP CROSSTAB\n")
cat(strrep("=", 60), "\n")

crosstab <- table(Group = df_cb$group, Batch = df_cb$site_source)
print(crosstab)

crosstab_df          <- as.data.frame.matrix(crosstab)
crosstab_df$group    <- rownames(crosstab_df)
crosstab_df          <- crosstab_df[, c("group", setdiff(colnames(crosstab_df), "group")),
                                    drop = FALSE]
rownames(crosstab_df) <- NULL
write.table(crosstab_df,
            file  = file.path(out_dir, "combat_batch_group_crosstab.tsv"),
            sep   = "\t", row.names = FALSE, quote = FALSE)
cat("\n  Saved: combat_batch_group_crosstab.tsv\n")

cat(paste0(
  "\n  [NOTE] All groups are uniquely assigned to one batch. ComBat cannot\n",
  "  fully separate batch from group biology. The is_LC_Neuro covariate\n",
  "  partially protects LC_Neuro biology: LC_NonNeuro_strict shares\n",
  "  LC_source but has is_LC_Neuro = 0. ComBat-adjusted results should\n",
  "  be interpreted alongside the unadjusted primary models and the\n",
  "  CACTUS-indicator sensitivity models.\n"
))

# ── 10. Prepare ComBat inputs ─────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("PREPARING ComBat INPUTS\n")
cat(strrep("=", 60), "\n")

# proteins x samples matrix (ComBat convention)
dat_orig           <- t(as.matrix(df_cb[, protein_cols]))
storage.mode(dat_orig) <- "double"
colnames(dat_orig) <- df_cb$SampleID

n_na <- sum(is.na(dat_orig))
cat(sprintf("  Missing NPX cells: %d\n", n_na))

# Record original NA positions; impute with protein-wise mean for ComBat
na_mask    <- is.na(dat_orig)
dat_imputed <- dat_orig
if (n_na > 0) {
  prot_means <- rowMeans(dat_orig, na.rm = TRUE)
  for (i in which(rowSums(is.na(dat_orig)) > 0L)) {
    dat_imputed[i, is.na(dat_imputed[i, ])] <- prot_means[i]
  }
  cat("  NAs imputed with protein-wise mean (restored after ComBat).\n")
}

# Warn if any protein has zero variance within a batch (ComBat will fail)
for (b in unique(df_cb$site_source)) {
  b_cols  <- df_cb$SampleID[df_cb$site_source == b]
  b_vars  <- apply(dat_imputed[, b_cols, drop = FALSE], 1L, var)
  n_zero  <- sum(b_vars == 0, na.rm = TRUE)
  if (n_zero > 0L) {
    cat(sprintf("  [WARN] %d protein(s) with zero within-batch variance in %s.\n",
                n_zero, b))
  }
}

batch <- df_cb$site_source
cat(sprintf("  dat: %d proteins x %d samples\n", nrow(dat_imputed), ncol(dat_imputed)))
cat(sprintf("  batch: %s\n",
            paste(names(table(batch)), table(batch), sep = "=", collapse = ", ")))

if (no_mod) {
  mod <- NULL
  cat("  --no-mod: ComBat running without biological covariate matrix.\n")
} else {
  meta_mod <- data.frame(is_LC_Neuro = df_cb$is_LC_Neuro)
  mod      <- model.matrix(~ is_LC_Neuro, data = meta_mod)
  cat(sprintf("  mod: %d x %d  (intercept + is_LC_Neuro)\n", nrow(mod), ncol(mod)))
}

# ── 11. Run ComBat ────────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("RUNNING sva::ComBat\n")
cat(strrep("=", 60), "\n")

dat_adjusted <- tryCatch(
  sva::ComBat(
    dat         = dat_imputed,
    batch       = batch,
    mod         = mod,
    par.prior   = TRUE,
    prior.plots = FALSE
  ),
  error = function(e) {
    stop(
      "sva::ComBat failed: ", conditionMessage(e), "\n",
      "Possible causes: perfect batch/group confounding, near-zero variance proteins,\n",
      "or a mod matrix that is collinear with batch.\n",
      "Try re-running with --no-mod to omit the biological covariate matrix.",
      call. = FALSE
    )
  }
)

if (!identical(dim(dat_adjusted), dim(dat_imputed))) {
  stop(sprintf(
    "ComBat output shape (%d x %d) differs from input (%d x %d).",
    nrow(dat_adjusted), ncol(dat_adjusted),
    nrow(dat_imputed),  ncol(dat_imputed)
  ), call. = FALSE)
}

# Restore original NA positions
dat_adjusted[na_mask] <- NA
cat(sprintf("  ComBat completed. Output: %d proteins x %d samples.\n",
            nrow(dat_adjusted), ncol(dat_adjusted)))

# ── 12. Save ComBat-adjusted NPX ──────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("SAVING OUTPUTS\n")
cat(strrep("=", 60), "\n")

adj_s_x_p             <- as.data.frame(t(dat_adjusted), check.names = FALSE)
adj_s_x_p             <- cbind(SampleID = rownames(adj_s_x_p), adj_s_x_p)
rownames(adj_s_x_p)   <- NULL
write.table(adj_s_x_p,
            file  = file.path(out_dir, "combat_adjusted_npx.tsv"),
            sep   = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  Saved: combat_adjusted_npx.tsv  (%d samples x %d proteins)\n",
            nrow(adj_s_x_p), ncol(adj_s_x_p) - 1L))

# ── 13. Save metadata ─────────────────────────────────────────────────────────
meta_out <- data.frame(
  SampleID         = df_cb$SampleID,
  group            = df_cb$group,
  site_source      = df_cb$site_source,
  cactus_indicator = df_cb$cactus_indicator,
  is_LC_Neuro      = df_cb$is_LC_Neuro,
  total_abundance  = round(df_cb$total_abundance, 4),
  ta_z             = round(df_cb$ta_z, 4),
  stringsAsFactors = FALSE
)
if ("Gender" %in% colnames(df_cb)) {
  meta_out$sex_01 <- as.integer(df_cb$Gender == "F")
}
write.table(meta_out,
            file  = file.path(out_dir, "combat_metadata.tsv"),
            sep   = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("  Saved: combat_metadata.tsv  (%d rows)\n", nrow(meta_out)))

# ── 14. PCA before/after ComBat ───────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("PCA BEFORE / AFTER ComBat\n")
cat(strrep("=", 60), "\n")

pca_file <- file.path(out_dir, "combat_pca_before_after.pdf")

# Use only samples that are complete in the original data (no NAs in any protein)
complete_sids <- colnames(dat_orig)[!apply(dat_orig, 2L, anyNA)]
n_complete    <- length(complete_sids)

if (n_complete < 3L) {
  cat(sprintf(
    "  [SKIP] Only %d sample(s) have no missing proteins — PCA requires >= 3.\n",
    n_complete
  ))
} else {
  cat(sprintf("  %d / %d samples with complete protein data used for PCA.\n",
              n_complete, ncol(dat_orig)))

  pca_before_dat <- t(dat_orig[, complete_sids])      # samples x proteins
  pca_after_dat  <- t(dat_adjusted[, complete_sids])

  pca_before <- prcomp(pca_before_dat, center = TRUE, scale. = FALSE)
  pca_after  <- prcomp(pca_after_dat,  center = TRUE, scale. = FALSE)

  pct_bef <- round(100 * pca_before$sdev^2 / sum(pca_before$sdev^2), 1)
  pct_aft <- round(100 * pca_after$sdev^2  / sum(pca_after$sdev^2),  1)

  # Annotation vectors aligned to complete_sids
  meta_complete    <- df_cb[match(complete_sids, df_cb$SampleID), ]
  batch_labels     <- meta_complete$site_source
  group_labels     <- meta_complete$group

  # Color = batch; shape = group
  batch_palette <- c(LC_source = "#E63946", HC_CACTUS_source = "#457B9D")
  group_pch     <- c(LC_Neuro = 16L, LC_NonNeuro_strict = 17L,
                     Healthy  = 15L, CACTUS              = 18L)

  col_vec <- batch_palette[batch_labels]
  pch_vec <- group_pch[group_labels]
  pch_vec[is.na(pch_vec)] <- 19L   # fallback for any unlabeled group

  pdf(pca_file, width = 11, height = 5.5)
  par(mfrow = c(1L, 2L), mar = c(5, 5, 4, 2) + 0.1)

  plot(pca_before$x[, 1L], pca_before$x[, 2L],
       col  = col_vec, pch = pch_vec, cex = 1.3,
       xlab = sprintf("PC1 (%.1f%%)", pct_bef[1L]),
       ylab = sprintf("PC2 (%.1f%%)", pct_bef[2L]),
       main = sprintf("Before ComBat  (n=%d)", n_complete),
       bty  = "l", las = 1)

  plot(pca_after$x[, 1L], pca_after$x[, 2L],
       col  = col_vec, pch = pch_vec, cex = 1.3,
       xlab = sprintf("PC1 (%.1f%%)", pct_aft[1L]),
       ylab = sprintf("PC2 (%.1f%%)", pct_aft[2L]),
       main = sprintf("After ComBat  (n=%d)", n_complete),
       bty  = "l", las = 1)

  legend("topright",
         legend = c("batch: LC_source", "batch: HC_CACTUS_source",
                    "LC_Neuro", "LC_NonNeuro_strict", "Healthy", "CACTUS"),
         col    = c(batch_palette["LC_source"], batch_palette["HC_CACTUS_source"],
                    rep("grey40", 4L)),
         pch    = c(19L, 19L,
                    group_pch["LC_Neuro"], group_pch["LC_NonNeuro_strict"],
                    group_pch["Healthy"],  group_pch["CACTUS"]),
         pt.cex = c(1.3, 1.3, 1.1, 1.1, 1.1, 1.1),
         cex    = 0.75, bty = "n")

  invisible(dev.off())
  cat(sprintf("  Saved: combat_pca_before_after.pdf\n"))
}

cat("\n", strrep("=", 60), "\n", sep = "")
cat("COMPLETE\n")
cat(strrep("=", 60), "\n")
cat(sprintf("  Outputs written to: %s\n\n", out_dir))
