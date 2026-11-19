#!/usr/bin/env Rscript
# =============================================================================
# gpca_combat_residual_check.R - gPCA residual/source diagnostic
# Pediatric Long COVID proteomics | BATCH / PHENOTYPE-GUIDED DIAGNOSTIC
# =============================================================================
#
# PURPOSE
# -------
# Runs gPCA pre- and post-ComBat for one or more predefined contrasts and saves
# both contrast-specific outputs and combined all-contrast summary outputs.
#
# Default behavior runs all contrasts:
#   pooled_source       = LC_Neuro + LC_NonNeuro_strict vs Healthy + CACTUS
#   lc_vs_cactus        = LC_Neuro + LC_NonNeuro_strict vs CACTUS
#   lc_vs_healthy       = LC_Neuro + LC_NonNeuro_strict vs Healthy
#   neuro_vs_nonneuro   = LC_Neuro vs LC_NonNeuro_strict
#
# Interpretation note:
#   pooled_source, lc_vs_cactus, and lc_vs_healthy are source/group-aligned
#   diagnostics because source and clinical biology are partially confounded.
#   neuro_vs_nonneuro is within LC_source and is therefore a phenotype-guided
#   neuro-subtype check, not a technical batch test.
#
# INPUTS
# ------
#   data/raw/Inflammation_NPX_AgeM.txt
#   data/raw/phenotype.txt
#   results/pediatric/combat_adjusted_npx.tsv
#   results/pediatric/combat_metadata.tsv       (optional; reconstructed if absent)
#
# OUTPUTS (results/pediatric/gpca_contrasts/)
# --------------------------------------------
#   Per contrast:
#     gpca_<contrast>_summary.tsv
#     gpca_<contrast>_batch_counts.tsv
#     gpca_<contrast>_batch_numeric_key.tsv
#     gpca_<contrast>_scores.tsv
#     gpca_<contrast>_plots.pdf
#
#   Combined:
#     gpca_all_contrasts_summary.tsv
#     gpca_all_contrasts_batch_counts.tsv
#     gpca_all_contrasts_batch_numeric_key.tsv
#     gpca_all_contrasts_scores.tsv
#     gpca_all_contrasts_summary_plots.pdf
#
# USAGE
# -----
#   Rscript scripts/pediatric/gpca_combat_residual_check.R
#   Rscript scripts/pediatric/gpca_combat_residual_check.R --contrast all --nperm 1000
#   Rscript scripts/pediatric/gpca_combat_residual_check.R --contrast lc_vs_healthy --nperm 1000
#   Rscript scripts/pediatric/gpca_combat_residual_check.R --max-proteins 10 --nperm 100
#   Rscript scripts/pediatric/gpca_combat_residual_check.R --complete-cases-only
#   Rscript scripts/pediatric/gpca_combat_residual_check.R --debug-gpca
#   ... --combat-dir <dir with combat_adjusted_npx.tsv + combat_metadata.tsv> --outdir <dir>
#   ... --seed <int>   reproducible permutation p-values (default: unseeded, as for the stored results)
# =============================================================================

# -- 0. Dependencies -----------------------------------------------------------
check_pkg <- function(pkg, install_hint) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed.\n", pkg),
         install_hint, call. = FALSE)
  }
}

check_pkg("gPCA", paste0(
  "Package 'gPCA' is archived on CRAN and may not install with install.packages('gPCA').\n",
  "Install from the CRAN archive using:\n",
  "  Rscript -e \"install.packages('https://cran.r-project.org/src/contrib/Archive/gPCA/gPCA_1.0.tar.gz', repos=NULL, type='source')\"\n"
))
check_pkg("ggplot2", "Install via CRAN:\n  install.packages('ggplot2')\n")
check_pkg("gridExtra", "Install via CRAN:\n  install.packages('gridExtra')\n")

suppressPackageStartupMessages({
  library(gPCA)
  library(ggplot2)
  library(gridExtra)
})

# -- 1. Command-line arguments -------------------------------------------------
cli_args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, flag, default = NA_character_) {
  idx <- which(args == flag)
  if (length(idx) == 0L || idx[1L] == length(args)) return(default)
  args[idx[1L] + 1L]
}

nperm               <- as.integer(parse_arg(cli_args, "--nperm", "1000"))
contrast_arg        <- parse_arg(cli_args, "--contrast", "all")
complete_cases_only <- "--complete-cases-only" %in% cli_args
max_prot_raw        <- parse_arg(cli_args, "--max-proteins", NA_character_)
max_proteins        <- if (!is.na(max_prot_raw)) as.integer(max_prot_raw) else NA_integer_
debug_gpca          <- "--debug-gpca" %in% cli_args
combat_dir_arg      <- parse_arg(cli_args, "--combat-dir", NA_character_)  # default results/pediatric
outdir_arg          <- parse_arg(cli_args, "--outdir", NA_character_)      # default results/pediatric/gpca_contrasts
# Optional RNG seed for the gPCA permutation p-values. Default (no --seed) keeps
# the original unseeded behaviour that produced the stored results; with a seed,
# set.seed(seed) is called before each contrast so every contrast is
# reproducible on its own, whichever contrasts are run.
seed_raw            <- parse_arg(cli_args, "--seed", NA_character_)
gpca_seed           <- if (!is.na(seed_raw)) as.integer(seed_raw) else NA_integer_

allowed_contrasts <- c("pooled_source", "lc_vs_cactus", "lc_vs_healthy", "neuro_vs_nonneuro")
if (contrast_arg == "all") {
  contrasts_to_run <- allowed_contrasts
} else {
  contrasts_to_run <- strsplit(contrast_arg, ",", fixed = TRUE)[[1L]]
  contrasts_to_run <- trimws(contrasts_to_run)
  bad <- setdiff(contrasts_to_run, allowed_contrasts)
  if (length(bad) > 0L) {
    stop("Unknown --contrast value(s): ", paste(bad, collapse = ", "), "\n",
         "Allowed: all, ", paste(allowed_contrasts, collapse = ", "),
         call. = FALSE)
  }
}

if (is.na(nperm) || nperm < 1L) {
  stop("--nperm must be a positive integer.", call. = FALSE)
}
if (!is.na(max_prot_raw) && (is.na(max_proteins) || max_proteins < 1L)) {
  stop("--max-proteins must be a positive integer.", call. = FALSE)
}

# -- 2. Paths ------------------------------------------------------------------
raw_args  <- commandArgs(trailingOnly = FALSE)
file_flag <- raw_args[grep("^--file=", raw_args)]
if (length(file_flag) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_flag[1L]))
  root_dir    <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  root_dir <- getwd()
  message("[INFO] Script path not detected; using working directory as root: ", root_dir)
}

npx_file   <- file.path(root_dir, "data", "raw", "Inflammation_NPX_AgeM.txt")
pheno_file <- file.path(root_dir, "data", "raw", "phenotype.txt")
combat_dir <- if (is.na(combat_dir_arg)) file.path(root_dir, "results", "pediatric") else combat_dir_arg
adj_file   <- file.path(combat_dir, "combat_adjusted_npx.tsv")
meta_file  <- file.path(combat_dir, "combat_metadata.tsv")
out_dir    <- if (is.na(outdir_arg)) file.path(root_dir, "results", "pediatric", "gpca_contrasts") else outdir_arg
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -- 3. Validate input files ---------------------------------------------------
for (f in c(npx_file, pheno_file)) {
  if (!file.exists(f)) stop("Input file not found: ", f, call. = FALSE)
}
if (!file.exists(adj_file)) {
  stop("ComBat-adjusted NPX file not found: ", adj_file, "\n",
       "Run combat_adjustment_sva.R first to generate it.", call. = FALSE)
}

cat("\n", strrep("=", 60), "\n", sep = "")
cat("gPCA MULTI-CONTRAST RESIDUAL / GUIDED DIAGNOSTIC\n")
cat(strrep("=", 60), "\n")
cat(sprintf("  nperm               : %d\n", nperm))
cat(sprintf("  seed                : %s\n", if (is.na(gpca_seed)) "none (unseeded)" else gpca_seed))
cat(sprintf("  contrast            : %s\n", contrast_arg))
cat(sprintf("  contrasts_to_run    : %s\n", paste(contrasts_to_run, collapse = ", ")))
cat(sprintf("  complete_cases_only : %s\n", complete_cases_only))
if (!is.na(max_proteins)) cat(sprintf("  max_proteins        : %d [quick-test mode]\n", max_proteins))
cat(sprintf("  Root dir            : %s\n", root_dir))
cat(sprintf("  Out dir             : %s\n", out_dir))

# -- 4. Load data --------------------------------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("LOADING DATA\n")
cat(strrep("=", 60), "\n")

npx_raw <- read.table(npx_file, sep = "\t", header = TRUE,
                      stringsAsFactors = FALSE, check.names = FALSE)
npx_adj <- read.table(adj_file, sep = "\t", header = TRUE,
                      stringsAsFactors = FALSE, check.names = FALSE)
pheno   <- read.table(pheno_file, sep = "\t", header = TRUE,
                      stringsAsFactors = FALSE, check.names = FALSE)

npx_raw$SampleID <- trimws(npx_raw$SampleID)
npx_adj$SampleID <- trimws(npx_adj$SampleID)
pheno$SampleID   <- trimws(pheno$SampleID)

check_duplicate_ids <- function(df, label) {
  dup <- df$SampleID[duplicated(df$SampleID)]
  if (length(dup) > 0L) {
    stop(sprintf("%s contains duplicated SampleID values: %s",
                 label, paste(unique(dup), collapse = ", ")), call. = FALSE)
  }
}
check_duplicate_ids(npx_raw, "Raw NPX")
check_duplicate_ids(npx_adj, "Adjusted NPX")
check_duplicate_ids(pheno, "Phenotype")

cat(sprintf("  Raw NPX:      %d samples, %d protein columns\n", nrow(npx_raw), ncol(npx_raw) - 1L))
cat(sprintf("  Adjusted NPX: %d samples, %d protein columns\n", nrow(npx_adj), ncol(npx_adj) - 1L))
cat(sprintf("  Phenotype:    %d rows, %d columns\n", nrow(pheno), ncol(pheno)))

# -- 5. Load or reconstruct metadata -----------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("METADATA\n")
cat(strrep("=", 60), "\n")

if (file.exists(meta_file)) {
  meta_all <- read.table(meta_file, sep = "\t", header = TRUE,
                         stringsAsFactors = FALSE, check.names = FALSE)
  meta_all$SampleID <- trimws(meta_all$SampleID)
  cat(sprintf("  Loaded combat_metadata.tsv (%d rows)\n", nrow(meta_all)))
} else {
  cat("  combat_metadata.tsv not found; reconstructing from raw NPX + phenotype.\n")
  required_pheno <- c("SampleID", "Group", "NEURO")
  missing_req    <- setdiff(required_pheno, colnames(pheno))
  if (length(missing_req) > 0L) {
    stop("phenotype.txt is missing required columns: ",
         paste(missing_req, collapse = ", "), call. = FALSE)
  }

  lc_rows   <- pheno[!is.na(pheno$Group) & pheno$Group == "LC",
                     c("SampleID", "NEURO"), drop = FALSE]
  neuro_map <- setNames(as.integer(lc_rows$NEURO), lc_rows$SampleID)

  assign_group <- function(sid) {
    if (startsWith(sid, "HC")) return("Healthy")
    if (sid %in% names(neuro_map)) {
      return(if (neuro_map[[sid]] == 1L) "LC_Neuro" else "LC_NonNeuro_strict")
    }
    if (startsWith(sid, "CACTUS")) return("CACTUS")
    return("Unknown")
  }

  all_sids <- unique(c(npx_raw$SampleID, npx_adj$SampleID))
  meta_all <- data.frame(
    SampleID = all_sids,
    group    = vapply(all_sids, assign_group, character(1L)),
    stringsAsFactors = FALSE
  )
  meta_all$site_source <- ifelse(
    meta_all$group %in% c("LC_Neuro", "LC_NonNeuro_strict"), "LC_source",
    ifelse(meta_all$group %in% c("Healthy", "CACTUS"), "HC_CACTUS_source", NA_character_)
  )
  cat(sprintf("  Reconstructed metadata for %d unique sample IDs.\n", nrow(meta_all)))
}

if (!"SampleID" %in% colnames(meta_all) || !"group" %in% colnames(meta_all)) {
  stop("Metadata must contain SampleID and group columns.", call. = FALSE)
}
check_duplicate_ids(meta_all, "Metadata")

# -- Helper functions ----------------------------------------------------------
make_contrast_labels <- function(meta, contrast) {
  meta$contrast_label <- NA_character_

  if (contrast == "pooled_source") {
    meta$contrast_label[meta$group %in% c("LC_Neuro", "LC_NonNeuro_strict")] <- "LC_source"
    meta$contrast_label[meta$group %in% c("Healthy", "CACTUS")] <- "HC_CACTUS_source"

  } else if (contrast == "lc_vs_cactus") {
    meta$contrast_label[meta$group %in% c("LC_Neuro", "LC_NonNeuro_strict")] <- "LC"
    meta$contrast_label[meta$group == "CACTUS"] <- "CACTUS"

  } else if (contrast == "lc_vs_healthy") {
    meta$contrast_label[meta$group %in% c("LC_Neuro", "LC_NonNeuro_strict")] <- "LC"
    meta$contrast_label[meta$group == "Healthy"] <- "Healthy"

  } else if (contrast == "neuro_vs_nonneuro") {
    meta$contrast_label[meta$group == "LC_Neuro"] <- "LC_Neuro"
    meta$contrast_label[meta$group == "LC_NonNeuro_strict"] <- "LC_NonNeuro_strict"

  } else {
    stop("Unknown contrast: ", contrast, call. = FALSE)
  }

  meta
}

impute_colmean <- function(mat) {
  col_means <- colMeans(mat, na.rm = TRUE)
  for (j in seq_len(ncol(mat))) {
    bad <- is.na(mat[, j])
    if (any(bad)) mat[bad, j] <- col_means[j]
  }
  mat
}

safe_get <- function(obj, nm) {
  val <- tryCatch(obj[[nm]], error = function(e) NULL)
  if (is.null(val) || length(val) == 0L) return(NA_real_)
  val_flat <- unlist(val, use.names = FALSE)
  val_num  <- suppressWarnings(as.numeric(val_flat[1L]))
  if (is.na(val_num)) return(NA_real_)
  val_num
}

safe_pval <- function(obj) {
  val <- tryCatch(obj[["p.val"]], error = function(e) NULL)
  if (!is.null(val) && length(val) > 0L) {
    val_flat <- unlist(val, use.names = FALSE)
    val_num  <- suppressWarnings(as.numeric(val_flat[1L]))
    if (!is.na(val_num)) return(val_num)
  }

  delta_obs  <- safe_get(obj, "delta")
  delta_perm <- tryCatch(obj[["delta.p"]], error = function(e) NULL)
  if (!is.null(delta_perm) && length(delta_perm) > 0L && !is.na(delta_obs)) {
    delta_perm_num <- suppressWarnings(as.numeric(unlist(delta_perm, use.names = FALSE)))
    delta_perm_num <- delta_perm_num[!is.na(delta_perm_num)]
    if (length(delta_perm_num) > 0L) {
      return((sum(delta_perm_num >= delta_obs) + 1) / (length(delta_perm_num) + 1))
    }
  }
  NA_real_
}

fmt_p <- function(p) if (is.na(p)) "NA" else sprintf("%.4f", p)

label_p <- function(p, significant_text, nonsig_text) {
  if (is.na(p)) return("p-value unavailable")
  if (p < 0.05) return(significant_text)
  nonsig_text
}

get_guided_scores <- function(obj, mat, meta, contrast, stage, batch_col, score_label_env) {
  pcg <- tryCatch(obj[["PCg"]], error = function(e) NULL)
  if (!is.null(pcg) && is.matrix(pcg) && nrow(pcg) == nrow(meta)) {
    return(data.frame(
      contrast    = contrast,
      SampleID    = meta$SampleID,
      stage       = stage,
      score1      = pcg[, 1L],
      score2      = if (ncol(pcg) >= 2L) pcg[, 2L] else NA_real_,
      contrast_label = meta[[batch_col]],
      group       = meta$group,
      score_type  = "guided_PC",
      stringsAsFactors = FALSE
    ))
  }

  pca <- prcomp(mat, center = TRUE, scale. = FALSE)
  data.frame(
    contrast    = contrast,
    SampleID    = meta$SampleID,
    stage       = stage,
    score1      = pca$x[, 1L],
    score2      = if (ncol(pca$x) >= 2L) pca$x[, 2L] else NA_real_,
    contrast_label = meta[[batch_col]],
    group       = meta$group,
    score_type  = "fallback_PCA",
    stringsAsFactors = FALSE
  )
}

plot_contrast_pdf <- function(summary_df, scores_df, out_plots_pdf, nperm) {
  ALPHA <- 0.05
  stage_palette <- c(pre_combat = "#E07B54", post_combat = "#4E9E8F")
  group_shapes  <- c(LC_Neuro = 16L, LC_NonNeuro_strict = 17L, Healthy = 15L, CACTUS = 18L)

  p_delta <- ggplot(summary_df, aes(x = stage, y = delta, fill = stage)) +
    geom_col(width = 0.5, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.4f", delta)), vjust = -0.4, size = 3.5) +
    scale_fill_manual(values = stage_palette, guide = "none") +
    scale_x_discrete(labels = c(pre_combat = "Pre-ComBat", post_combat = "Post-ComBat")) +
    labs(title = paste0("gPCA delta: ", unique(summary_df$contrast)),
         subtitle = sprintf("nperm=%d", nperm),
         x = NULL, y = "delta (gPCA batch-detection statistic)") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))

  dfp <- summary_df
  dfp$neg_log_p <- ifelse(is.na(dfp$p_value), NA_real_, -log10(pmax(dfp$p_value, 1e-16)))
  p_pval <- ggplot(dfp, aes(x = stage, y = neg_log_p, fill = stage)) +
    geom_col(width = 0.5, alpha = 0.85, na.rm = TRUE) +
    geom_text(aes(label = ifelse(is.na(p_value), "p=NA", sprintf("p=%.4f", p_value))),
              vjust = -0.4, size = 3.5, na.rm = TRUE) +
    geom_hline(yintercept = -log10(ALPHA), linetype = "dashed", colour = "red", linewidth = 0.8) +
    scale_fill_manual(values = stage_palette, guide = "none") +
    scale_x_discrete(labels = c(pre_combat = "Pre-ComBat", post_combat = "Post-ComBat")) +
    labs(title = "-log10(p-value): Pre vs Post ComBat",
         subtitle = "Dashed line: p = 0.05", x = NULL, y = "-log10(p-value)") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))

  make_score_plot <- function(df_stage, title_prefix) {
    has_score2 <- !all(is.na(df_stage$score2))
    y_vals <- if (has_score2) df_stage$score2 else rep(0, nrow(df_stage))
    y_lab <- if (has_score2) "Score 2" else "Score 2 (not available; y=0)"
    ggplot(df_stage, aes(x = score1, y = y_vals, colour = contrast_label, shape = group)) +
      geom_point(size = 2.5, alpha = 0.85) +
      scale_colour_discrete(name = "contrast_label") +
      scale_shape_manual(values = group_shapes, name = "group", na.value = 19L) +
      labs(title = paste0(title_prefix, " - ", unique(df_stage$score_type)),
           x = "Score 1", y = y_lab) +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", size = 10),
            legend.text = element_text(size = 8),
            legend.title = element_text(size = 8))
  }

  p_pre  <- make_score_plot(scores_df[scores_df$stage == "pre_combat", ], "Pre-ComBat")
  p_post <- make_score_plot(scores_df[scores_df$stage == "post_combat", ], "Post-ComBat")

  pdf(out_plots_pdf, width = 12, height = 10)
  grid.arrange(p_delta, p_pval, p_pre, p_post, ncol = 2L, nrow = 2L)
  invisible(dev.off())
}

plot_combined_summary_pdf <- function(combined_summary, out_pdf, nperm) {
  ALPHA <- 0.05
  stage_palette <- c(pre_combat = "#E07B54", post_combat = "#4E9E8F")
  dfp <- combined_summary
  dfp$neg_log_p <- ifelse(is.na(dfp$p_value), NA_real_, -log10(pmax(dfp$p_value, 1e-16)))

  p_delta <- ggplot(combined_summary, aes(x = contrast, y = delta, fill = stage)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7, alpha = 0.85) +
    geom_text(aes(label = sprintf("%.2f", delta)),
              position = position_dodge(width = 0.75), vjust = -0.35, size = 3) +
    scale_fill_manual(values = stage_palette, name = "Stage") +
    labs(title = "gPCA delta across contrasts",
         subtitle = sprintf("nperm=%d", nperm), x = NULL,
         y = "delta (gPCA batch-detection statistic)") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1),
          plot.title = element_text(face = "bold"))

  p_pval <- ggplot(dfp, aes(x = contrast, y = neg_log_p, fill = stage)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7, alpha = 0.85, na.rm = TRUE) +
    geom_hline(yintercept = -log10(ALPHA), linetype = "dashed", colour = "red", linewidth = 0.8) +
    scale_fill_manual(values = stage_palette, name = "Stage") +
    labs(title = "gPCA permutation p-values across contrasts",
         subtitle = "Dashed line: p = 0.05", x = NULL,
         y = "-log10(p-value)") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1),
          plot.title = element_text(face = "bold"))

  pdf(out_pdf, width = 12, height = 6)
  grid.arrange(p_delta, p_pval, ncol = 2L)
  invisible(dev.off())
}

run_one_contrast <- function(contrast) {
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat(sprintf("CONTRAST: %s\n", contrast))
  cat(strrep("=", 60), "\n")

  prefix <- file.path(out_dir, paste0("gpca_", contrast))
  out_summary   <- paste0(prefix, "_summary.tsv")
  out_counts    <- paste0(prefix, "_batch_counts.tsv")
  out_batch_key <- paste0(prefix, "_batch_numeric_key.tsv")
  out_scores    <- paste0(prefix, "_scores.tsv")
  out_plots_pdf <- paste0(prefix, "_plots.pdf")

  meta <- make_contrast_labels(meta_all, contrast)
  n_before <- nrow(meta)
  meta <- meta[!is.na(meta$contrast_label), ]
  batch_col <- "contrast_label"

  cat(sprintf("  Samples included in contrast: %d / %d\n", nrow(meta), n_before))
  if (nrow(meta) < 3L) stop("Fewer than 3 samples remain for contrast ", contrast, call. = FALSE)

  shared_sids <- Reduce(intersect, list(meta$SampleID, npx_raw$SampleID, npx_adj$SampleID))
  if (length(shared_sids) < 3L) {
    stop("Fewer than 3 shared samples for contrast ", contrast, call. = FALSE)
  }
  meta_c    <- meta[match(shared_sids, meta$SampleID), ]
  raw_c     <- npx_raw[match(shared_sids, npx_raw$SampleID), ]
  adj_c     <- npx_adj[match(shared_sids, npx_adj$SampleID), ]

  if (!all(meta_c$SampleID == raw_c$SampleID) || !all(meta_c$SampleID == adj_c$SampleID)) {
    stop("Sample alignment failed for contrast ", contrast, call. = FALSE)
  }

  raw_prots <- setdiff(colnames(raw_c), "SampleID")
  adj_prots <- setdiff(colnames(adj_c), "SampleID")
  shared_prots <- intersect(raw_prots, adj_prots)
  if (length(shared_prots) == 0L) stop("No shared proteins for contrast ", contrast, call. = FALSE)
  if (!is.na(max_proteins) && max_proteins < length(shared_prots)) {
    shared_prots <- shared_prots[seq_len(max_proteins)]
    cat(sprintf("  [QUICK TEST] Restricted to first %d proteins.\n", max_proteins))
  }

  mat_raw <- as.matrix(raw_c[, shared_prots, drop = FALSE])
  mat_adj <- as.matrix(adj_c[, shared_prots, drop = FALSE])
  storage.mode(mat_raw) <- "double"
  storage.mode(mat_adj) <- "double"
  rownames(mat_raw) <- meta_c$SampleID
  rownames(mat_adj) <- meta_c$SampleID

  n_na_raw <- sum(is.na(mat_raw))
  n_na_adj <- sum(is.na(mat_adj))
  cat(sprintf("  Matrix: %d samples x %d proteins\n", nrow(mat_raw), ncol(mat_raw)))
  cat(sprintf("  Missing cells: raw=%d, adjusted=%d\n", n_na_raw, n_na_adj))

  if (complete_cases_only) {
    complete_mask <- !apply(mat_raw, 1L, anyNA) & !apply(mat_adj, 1L, anyNA)
    if (sum(complete_mask) < 3L) stop("Fewer than 3 complete-case samples for ", contrast, call. = FALSE)
    mat_raw <- mat_raw[complete_mask, , drop = FALSE]
    mat_adj <- mat_adj[complete_mask, , drop = FALSE]
    meta_c  <- meta_c[complete_mask, ]
    cat(sprintf("  Complete-case samples: %d\n", nrow(meta_c)))
  } else {
    mat_raw <- impute_colmean(mat_raw)
    mat_adj <- impute_colmean(mat_adj)
    n_still_na <- sum(is.na(mat_raw)) + sum(is.na(mat_adj))
    if (n_still_na > 0L) {
      stop(n_still_na, " cells remain NA after imputation for ", contrast, call. = FALSE)
    }
  }

  batch_labels <- meta_c[[batch_col]]
  batch_factor <- factor(batch_labels)
  batch_vec    <- as.integer(batch_factor)
  batch_levels <- levels(batch_factor)

  if (length(batch_levels) < 2L) {
    stop("Only 1 label level found for contrast ", contrast, call. = FALSE)
  }

  batch_key <- data.frame(
    contrast = contrast,
    batch_numeric = seq_along(batch_levels),
    batch_label = batch_levels,
    stringsAsFactors = FALSE
  )
  batch_counts <- as.data.frame(table(batch = batch_labels, group = meta_c$group),
                                stringsAsFactors = FALSE)
  colnames(batch_counts) <- c("batch", "group", "n")
  batch_counts <- batch_counts[batch_counts$n > 0L, ]
  batch_counts$contrast <- contrast
  batch_counts <- batch_counts[, c("contrast", "batch", "group", "n")]

  cat("  Numeric label coding used for gPCA:\n")
  print(batch_key)
  print(batch_counts)

  write.table(batch_counts, file = out_counts, sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(batch_key, file = out_batch_key, sep = "\t", row.names = FALSE, quote = FALSE)

  # gPCA.batchdetect internally compares batch IDs to 1, 2, ..., so numeric IDs
  # are passed to gPCA; readable labels are retained for summaries and plots.
  if (!is.na(gpca_seed)) set.seed(gpca_seed)
  cat("  Pre-ComBat gPCA ...\n")
  gpca_pre <- tryCatch(
    gPCA::gPCA.batchdetect(x = mat_raw, batch = batch_vec, center = TRUE, nperm = nperm),
    error = function(e) stop("gPCA failed pre-ComBat for ", contrast, ": ", conditionMessage(e), call. = FALSE)
  )
  cat("  Post-ComBat gPCA ...\n")
  gpca_post <- tryCatch(
    gPCA::gPCA.batchdetect(x = mat_adj, batch = batch_vec, center = TRUE, nperm = nperm),
    error = function(e) stop("gPCA failed post-ComBat for ", contrast, ": ", conditionMessage(e), call. = FALSE)
  )

  if (debug_gpca) {
    cat("\nDEBUG: ", contrast, " gpca_pre names\n", sep = "")
    print(names(gpca_pre))
    cat("\nstr(gpca_pre$p.val):\n"); str(gpca_pre$p.val)
    cat("\nstr(gpca_pre$delta.p):\n"); str(gpca_pre$delta.p)
    cat("\nstr(gpca_post$p.val):\n"); str(gpca_post$p.val)
    cat("\nstr(gpca_post$delta.p):\n"); str(gpca_post$delta.p)
  }

  make_summary_row <- function(obj, stage) {
    data.frame(
      contrast     = contrast,
      stage        = stage,
      n_samples    = nrow(meta_c),
      n_proteins   = ncol(mat_raw),
      batch_col    = batch_col,
      batch_levels = paste(sort(batch_levels), collapse = "|"),
      nperm        = nperm,
      delta        = round(safe_get(obj, "delta"), 6L),
      p_value      = round(safe_pval(obj), 6L),
      varPCg1      = round(safe_get(obj, "varPCg1"), 6L),
      stringsAsFactors = FALSE
    )
  }

  summary_df <- rbind(make_summary_row(gpca_pre, "pre_combat"),
                      make_summary_row(gpca_post, "post_combat"))
  write.table(summary_df, file = out_summary, sep = "\t", row.names = FALSE, quote = FALSE)

  cat(sprintf("  Pre-ComBat  - delta=%.4f  p=%s  varPCg1=%s\n",
              summary_df$delta[1L], fmt_p(summary_df$p_value[1L]),
              ifelse(is.na(summary_df$varPCg1[1L]), "NA", sprintf("%.4f", summary_df$varPCg1[1L]))))
  cat(sprintf("  Post-ComBat - delta=%.4f  p=%s  varPCg1=%s\n",
              summary_df$delta[2L], fmt_p(summary_df$p_value[2L]),
              ifelse(is.na(summary_df$varPCg1[2L]), "NA", sprintf("%.4f", summary_df$varPCg1[2L]))))

  scores_pre  <- get_guided_scores(gpca_pre, mat_raw, meta_c, contrast, "pre_combat", batch_col, NULL)
  scores_post <- get_guided_scores(gpca_post, mat_adj, meta_c, contrast, "post_combat", batch_col, NULL)
  scores_df <- rbind(scores_pre, scores_post)
  write.table(scores_df, file = out_scores, sep = "\t", row.names = FALSE, quote = FALSE)

  plot_contrast_pdf(summary_df, scores_df, out_plots_pdf, nperm)

  pre_delta  <- summary_df$delta[summary_df$stage == "pre_combat"]
  post_delta <- summary_df$delta[summary_df$stage == "post_combat"]
  pre_pval   <- summary_df$p_value[summary_df$stage == "pre_combat"]
  post_pval  <- summary_df$p_value[summary_df$stage == "post_combat"]

  if (contrast == "neuro_vs_nonneuro") {
    interpretation_note <- "Within-LC neuro phenotype-guided check; not a technical batch test."
  } else {
    interpretation_note <- "Source/group-aligned diagnostic; source and biology are partly confounded."
  }

  cat(sprintf("  Interpretation: pre p=%s, post p=%s; delta %.4f -> %.4f. %s\n",
              fmt_p(pre_pval), fmt_p(post_pval), pre_delta, post_delta, interpretation_note))
  cat(sprintf("  Saved prefix: %s\n", prefix))

  list(summary = summary_df, counts = batch_counts, key = batch_key, scores = scores_df)
}

# -- Run all requested contrasts ----------------------------------------------
all_results <- lapply(contrasts_to_run, run_one_contrast)

combined_summary <- do.call(rbind, lapply(all_results, `[[`, "summary"))
combined_counts  <- do.call(rbind, lapply(all_results, `[[`, "counts"))
combined_key     <- do.call(rbind, lapply(all_results, `[[`, "key"))
combined_scores  <- do.call(rbind, lapply(all_results, `[[`, "scores"))

out_all_summary <- file.path(out_dir, "gpca_all_contrasts_summary.tsv")
out_all_counts  <- file.path(out_dir, "gpca_all_contrasts_batch_counts.tsv")
out_all_key     <- file.path(out_dir, "gpca_all_contrasts_batch_numeric_key.tsv")
out_all_scores  <- file.path(out_dir, "gpca_all_contrasts_scores.tsv")
out_all_pdf     <- file.path(out_dir, "gpca_all_contrasts_summary_plots.pdf")

write.table(combined_summary, file = out_all_summary, sep = "\t", row.names = FALSE, quote = FALSE)
write.table(combined_counts,  file = out_all_counts,  sep = "\t", row.names = FALSE, quote = FALSE)
write.table(combined_key,     file = out_all_key,     sep = "\t", row.names = FALSE, quote = FALSE)
write.table(combined_scores,  file = out_all_scores,  sep = "\t", row.names = FALSE, quote = FALSE)
plot_combined_summary_pdf(combined_summary, out_all_pdf, nperm)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("COMBINED gPCA SUMMARY\n")
cat(strrep("=", 60), "\n")
print(combined_summary[, c("contrast", "stage", "n_samples", "n_proteins", "delta", "p_value", "varPCg1")])
cat("\nSaved combined outputs:\n")
cat(sprintf("  %s\n", out_all_summary))
cat(sprintf("  %s\n", out_all_counts))
cat(sprintf("  %s\n", out_all_key))
cat(sprintf("  %s\n", out_all_scores))
cat(sprintf("  %s\n", out_all_pdf))
cat("\nCOMPLETE\n")
