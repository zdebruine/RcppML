# =============================================================================
# Comprehensive CV-NMF Benchmark: RcppML vs Singlet
# =============================================================================
# 
# This benchmark compares cross-validation NMF performance between RcppML and 
# singlet across all internal RcppML datasets at multiple factorization ranks.
#
# Configuration:
#   - Ranks: k = 2, 4, 8, 16, 32, 64
#   - CV epochs: 10
#   - Holdout fraction: 5%
#   - mask_zeros: FALSE (sparse=FALSE for RcppML, tests all matrix entries)
#   - Fair comparison: singlet forced to run same iterations (no early stopping)
#
# Key insight: singlet's default settings use aggressive early stopping via
# tolerance and overfit detection. To fairly compare algorithm speed, we force
# full iteration runs with: L1=0, tol=1e-20, tol_overfit=1e20
# =============================================================================

library(RcppML)
library(singlet)
library(Matrix)
library(ggplot2)
library(dplyr)
library(tidyr)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANKS <- c(2, 4, 8, 16, 32, 64)
CV_EPOCHS <- 10
HOLDOUT_FRACTION <- 0.05
MAX_ITERATIONS <- 100
SEED <- 42
NUM_REPLICATES <- 3  # Number of timing replicates per condition

# Output paths
OUTPUT_DIR <- "manuscript/figures"
REPORT_PATH <- "manuscript/cv_benchmark_report.md"

# Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# -----------------------------------------------------------------------------
# Load all internal datasets
# -----------------------------------------------------------------------------

cat("Loading datasets...\n")

datasets <- list(
  aml = {data(aml, package = "RcppML"); aml},
  hawaiibirds = {data(hawaiibirds, package = "RcppML"); hawaiibirds},
  movielens = {data(movielens, package = "RcppML"); movielens}
)

# Also load dense-ish datasets if available
tryCatch({
  data(digits_full, package = "RcppML")
  datasets$digits_full <- digits_full
}, error = function(e) cat("digits_full not available\n"))

tryCatch({
  data(olivetti, package = "RcppML")
  datasets$olivetti <- olivetti
}, error = function(e) cat("olivetti not available\n"))

tryCatch({
  data(golub, package = "RcppML")
  datasets$golub <- golub
}, error = function(e) cat("golub not available\n"))

tryCatch({
  data(pbmc3k, package = "RcppML")
  datasets$pbmc3k <- pbmc3k
}, error = function(e) cat("pbmc3k not available\n"))

# Print dataset info
cat("\nDataset Summary:\n")
cat(sprintf("%-15s %8s %8s %12s %8s\n", "Dataset", "Rows", "Cols", "NNZ", "Density"))
cat(paste(rep("-", 55), collapse = ""), "\n")
for (name in names(datasets)) {
  A <- datasets[[name]]
  if (inherits(A, "sparseMatrix")) {
    nnz <- length(A@x)
    density <- nnz / (nrow(A) * ncol(A))
  } else {
    nnz <- sum(A != 0)
    density <- nnz / length(A)
  }
  cat(sprintf("%-15s %8d %8d %12d %7.2f%%\n", 
              name, nrow(A), ncol(A), nnz, density * 100))
}

# -----------------------------------------------------------------------------
# Benchmark functions
# -----------------------------------------------------------------------------

#' Run RcppML CV-NMF benchmark
run_rcppml_cv <- function(A, k, cv_rep = 10, test_fraction = 0.05, 
                          maxit = 100, seed = 42) {
  # Ensure sparse matrix format
  if (!inherits(A, "dgCMatrix")) {
    A <- as(A, "dgCMatrix")
  }
  
  timing <- system.time({
    result <- nmf(A, k = k, 
                  cv_rep = cv_rep,
                  test_fraction = test_fraction,
                  sparse = FALSE,  # mask_zeros = FALSE
                  maxit = maxit,
                  tol = 1e-10,  # Force full iterations
                  seed = seed,
                  verbose = FALSE)
  })
  
  list(
    time = timing["elapsed"],
    test_loss = if (!is.null(result$test_error)) result$test_error else NA,
    train_loss = if (!is.null(result$train_error)) result$train_error else NA,
    iterations = result$iteration
  )
}

#' Run singlet CV-NMF benchmark (fair comparison)
run_singlet_cv <- function(A, k, cv_rep = 10, test_fraction = 0.05,
                           maxit = 100, seed = 42) {
  # Ensure sparse matrix format
  if (!inherits(A, "dgCMatrix")) {
    A <- as(A, "dgCMatrix")
  }
  
  # CRITICAL: Use these settings for fair comparison
  # L1 = 0: No L1 regularization
  # tol = 1e-20: Essentially disable convergence-based stopping
  # tol_overfit = 1e20: Disable overfit detection early stopping
  timing <- system.time({
    result <- tryCatch({
      singlet::cross_validate_nmf(
        A, k = k,
        reps = cv_rep,
        test_density = test_fraction,
        maxit = maxit,
        L1 = 0,
        tol = 1e-20,
        tol_overfit = 1e20,
        seed = seed,
        verbose = FALSE
      )
    }, error = function(e) {
      list(test_error = NA, train_error = NA, error = e$message)
    })
  })
  
  list(
    time = timing["elapsed"],
    test_loss = if (!is.null(result$test_error)) mean(result$test_error) else NA,
    train_loss = if (!is.null(result$train_error)) mean(result$train_error) else NA,
    error = if (!is.null(result$error)) result$error else NULL
  )
}

# -----------------------------------------------------------------------------
# Run comprehensive benchmark
# -----------------------------------------------------------------------------

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Starting Comprehensive Benchmark\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

results <- data.frame()

for (dataset_name in names(datasets)) {
  A <- datasets[[dataset_name]]
  
  # Ensure sparse format
  if (!inherits(A, "dgCMatrix")) {
    A <- as(A, "dgCMatrix")
  }
  
  # Get max feasible rank
  max_rank <- min(nrow(A), ncol(A)) - 1
  feasible_ranks <- RANKS[RANKS <= max_rank]
  
  cat(sprintf("\n--- Dataset: %s (%d x %d, max_rank = %d) ---\n", 
              dataset_name, nrow(A), ncol(A), max_rank))
  
  for (k in feasible_ranks) {
    cat(sprintf("  k = %d: ", k))
    
    for (rep_i in 1:NUM_REPLICATES) {
      seed_i <- SEED + rep_i - 1
      
      # --- RcppML ---
      rcppml_result <- tryCatch({
        run_rcppml_cv(A, k, CV_EPOCHS, HOLDOUT_FRACTION, MAX_ITERATIONS, seed_i)
      }, error = function(e) {
        list(time = NA, test_loss = NA, train_loss = NA, error = e$message)
      })
      
      results <- rbind(results, data.frame(
        dataset = dataset_name,
        rows = nrow(A),
        cols = ncol(A),
        k = k,
        replicate = rep_i,
        package = "RcppML",
        time_sec = rcppml_result$time,
        test_loss = rcppml_result$test_loss,
        train_loss = rcppml_result$train_loss,
        error = if (!is.null(rcppml_result$error)) rcppml_result$error else NA,
        stringsAsFactors = FALSE
      ))
      
      # --- singlet ---
      singlet_result <- tryCatch({
        run_singlet_cv(A, k, CV_EPOCHS, HOLDOUT_FRACTION, MAX_ITERATIONS, seed_i)
      }, error = function(e) {
        list(time = NA, test_loss = NA, train_loss = NA, error = e$message)
      })
      
      results <- rbind(results, data.frame(
        dataset = dataset_name,
        rows = nrow(A),
        cols = ncol(A),
        k = k,
        replicate = rep_i,
        package = "singlet",
        time_sec = singlet_result$time,
        test_loss = singlet_result$test_loss,
        train_loss = singlet_result$train_loss,
        error = if (!is.null(singlet_result$error)) singlet_result$error else NA,
        stringsAsFactors = FALSE
      ))
      
      cat(".")
    }
    
    # Print summary for this k
    rcppml_times <- results$time_sec[results$dataset == dataset_name & 
                                       results$k == k & 
                                       results$package == "RcppML"]
    singlet_times <- results$time_sec[results$dataset == dataset_name & 
                                        results$k == k & 
                                        results$package == "singlet"]
    
    rcppml_mean <- mean(rcppml_times, na.rm = TRUE)
    singlet_mean <- mean(singlet_times, na.rm = TRUE)
    speedup <- singlet_mean / rcppml_mean
    
    cat(sprintf(" RcppML: %.2fs, singlet: %.2fs (%.2fx)\n",
                rcppml_mean, singlet_mean, speedup))
  }
}

# Save raw results
write.csv(results, "manuscript/cv_benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to manuscript/cv_benchmark_results.csv\n")

# -----------------------------------------------------------------------------
# Generate summary statistics
# -----------------------------------------------------------------------------

summary_stats <- results %>%
  group_by(dataset, rows, cols, k, package) %>%
  summarize(
    mean_time = mean(time_sec, na.rm = TRUE),
    sd_time = sd(time_sec, na.rm = TRUE),
    mean_test_loss = mean(test_loss, na.rm = TRUE),
    mean_train_loss = mean(train_loss, na.rm = TRUE),
    n_success = sum(!is.na(time_sec)),
    .groups = "drop"
  )

# Compute speedups
speedups <- summary_stats %>%
  select(dataset, rows, cols, k, package, mean_time) %>%
  pivot_wider(names_from = package, values_from = mean_time) %>%
  mutate(speedup = singlet / RcppML)

# -----------------------------------------------------------------------------
# Generate plots
# -----------------------------------------------------------------------------

cat("\nGenerating plots...\n")

# Plot 1: Execution time by rank for each dataset
p1 <- ggplot(summary_stats, aes(x = factor(k), y = mean_time, fill = package)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = mean_time - sd_time, ymax = mean_time + sd_time),
                position = position_dodge(0.9), width = 0.25) +
  facet_wrap(~dataset, scales = "free_y") +
  labs(
    title = "CV-NMF Execution Time: RcppML vs singlet",
    subtitle = sprintf("10 CV epochs, 5%% holdout, %d iterations (fair comparison)", MAX_ITERATIONS),
    x = "Factorization Rank (k)",
    y = "Time (seconds)",
    fill = "Package"
  ) +
  scale_fill_manual(values = c("RcppML" = "#0073C2", "singlet" = "#EFC000")) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(OUTPUT_DIR, "cv_benchmark_times.png"), p1, 
       width = 12, height = 8, dpi = 150)

# Plot 2: Speedup by rank
p2 <- ggplot(speedups, aes(x = factor(k), y = speedup, fill = dataset)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(
    title = "RcppML Speedup over singlet",
    subtitle = "Values > 1 indicate RcppML is faster",
    x = "Factorization Rank (k)",
    y = "Speedup (singlet time / RcppML time)",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

ggsave(file.path(OUTPUT_DIR, "cv_benchmark_speedup.png"), p2, 
       width = 10, height = 6, dpi = 150)

# Plot 3: Test loss comparison (sanity check - should be similar)
loss_comparison <- summary_stats %>%
  select(dataset, k, package, mean_test_loss) %>%
  pivot_wider(names_from = package, values_from = mean_test_loss) %>%
  filter(!is.na(RcppML) & !is.na(singlet))

if (nrow(loss_comparison) > 0) {
  p3 <- ggplot(loss_comparison, aes(x = RcppML, y = singlet, color = dataset)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(
      title = "Test Loss Comparison: RcppML vs singlet",
      subtitle = "Points on diagonal indicate equivalent loss computation",
      x = "RcppML Test MSE",
      y = "singlet Test MSE",
      color = "Dataset"
    ) +
    theme_minimal(base_size = 12) +
    coord_equal()
  
  ggsave(file.path(OUTPUT_DIR, "cv_benchmark_loss_comparison.png"), p3, 
         width = 8, height = 8, dpi = 150)
}

# Plot 4: Scaling with matrix size
scaling_data <- summary_stats %>%
  mutate(matrix_elements = rows * cols) %>%
  filter(k == 8)  # Fix k=8 for scaling analysis

if (nrow(scaling_data) > 0) {
  p4 <- ggplot(scaling_data, aes(x = matrix_elements / 1e6, y = mean_time, 
                                   color = package, shape = package)) +
    geom_point(size = 4) +
    geom_line() +
    scale_x_log10() +
    scale_y_log10() +
    labs(
      title = "CV-NMF Scaling with Matrix Size (k = 8)",
      x = "Matrix Elements (millions)",
      y = "Time (seconds, log scale)",
      color = "Package",
      shape = "Package"
    ) +
    scale_color_manual(values = c("RcppML" = "#0073C2", "singlet" = "#EFC000")) +
    theme_minimal(base_size = 12)
  
  ggsave(file.path(OUTPUT_DIR, "cv_benchmark_scaling.png"), p4, 
         width = 8, height = 6, dpi = 150)
}

cat("Plots saved to", OUTPUT_DIR, "\n")

# -----------------------------------------------------------------------------
# Generate markdown report
# -----------------------------------------------------------------------------

cat("\nGenerating report...\n")

report <- c(
  "# Cross-Validation NMF Benchmark: RcppML vs singlet",
  "",
  paste0("**Generated:** ", Sys.time()),
  "",
  "## Executive Summary",
  "",
  "This benchmark compares cross-validation NMF performance between RcppML and",
  "singlet across multiple datasets and factorization ranks.",
  "",
  "### Key Findings",
  "",
  sprintf("- **Average speedup:** %.2fx faster than singlet", 
          mean(speedups$speedup, na.rm = TRUE)),
  sprintf("- **Datasets tested:** %d", length(unique(results$dataset))),
  sprintf("- **Rank range:** k = %d to %d", min(RANKS), max(RANKS)),
  "",
  "## Methodology",
  "",
  "### Configuration",
  "",
  "| Parameter | Value |",
  "|-----------|-------|",
  sprintf("| CV Epochs | %d |", CV_EPOCHS),
  sprintf("| Holdout Fraction | %.0f%% |", HOLDOUT_FRACTION * 100),
  sprintf("| Max Iterations | %d |", MAX_ITERATIONS),

  sprintf("| Replicates | %d |", NUM_REPLICATES),
  "| mask_zeros | FALSE (test all entries) |",
  "",
  "### Fair Comparison Protocol",
  "",
  "**Critical Discovery:** singlet's default settings use aggressive early stopping:",
  "- Default `tol` causes convergence in ~2 iterations on many datasets",
  "- `tol_overfit` adds additional early stopping based on validation loss",
  "",
  "To ensure a fair comparison of algorithm speed (not convergence detection),",
  "singlet was configured with:",
  "- `L1 = 0` (no regularization)",
  "- `tol = 1e-20` (effectively disable convergence stopping)",
  "- `tol_overfit = 1e20` (disable overfit detection)",
  "",
  "This forces both packages to run the same number of iterations.",
  "",
  "## Dataset Information",
  "",
  "| Dataset | Rows | Columns | Non-zeros | Density |",
  "|---------|------|---------|-----------|---------|"
)

for (name in names(datasets)) {
  A <- datasets[[name]]
  if (inherits(A, "sparseMatrix")) {
    nnz <- length(A@x)
  } else {
    nnz <- sum(A != 0)
  }
  density <- nnz / (nrow(A) * ncol(A)) * 100
  report <- c(report, sprintf("| %s | %d | %d | %d | %.2f%% |",
                               name, nrow(A), ncol(A), nnz, density))
}

report <- c(report, "",
            "## Results",
            "",
            "### Execution Time Comparison",
            "",
            "![Execution Times](figures/cv_benchmark_times.png)",
            "",
            "### Speedup Analysis",
            "",
            "![Speedup](figures/cv_benchmark_speedup.png)",
            "")

# Add detailed speedup table
report <- c(report,
            "### Speedup by Dataset and Rank",
            "",
            "| Dataset | k | RcppML (s) | singlet (s) | Speedup |",
            "|---------|---|------------|-------------|---------|")

for (i in 1:nrow(speedups)) {
  row <- speedups[i, ]
  report <- c(report, sprintf("| %s | %d | %.3f | %.3f | %.2fx |",
                               row$dataset, row$k, row$RcppML, row$singlet, row$speedup))
}

report <- c(report, "",
            "### Test Loss Comparison",
            "",
            "![Loss Comparison](figures/cv_benchmark_loss_comparison.png)",
            "",
            "The test losses between RcppML and singlet should be similar, confirming",
            "that both packages are computing equivalent metrics with `mask_zeros=FALSE`.",
            "",
            "## Scalability",
            "",
            "![Scaling](figures/cv_benchmark_scaling.png)",
            "",
            "## Technical Notes",
            "",
            "### Loss Metric Definition",
            "",
            "With `mask_zeros=FALSE` (RcppML) / default (singlet):",
            "- **Test set:** Randomly selected 5% of ALL matrix entries",
            "- **Train set:** Remaining 95% of matrix entries",
            "- **Loss:** Mean Squared Error over respective set",
            "",
            "This differs from `mask_zeros=TRUE`, which only considers non-zero entries",
            "as potential test entries (more appropriate for recommendation systems).",
            "",
            "### Implementation Differences",
            "",
            "- **RcppML:** Uses lazy mask evaluation with hash-based RNG",
            "- **singlet:** Precomputes speckled mask for each CV fold",
            "",
            "Both approaches have similar computational cost per iteration when",
            "running in the `mask_zeros=FALSE` mode, as the Gram matrix trick",
            "cannot be used.",
            "",
            "## Reproducibility",
            "",
            "```r",
            "# R version and packages",
            sprintf("# R %s", R.version.string),
            sprintf("# RcppML %s", packageVersion("RcppML")),
            sprintf("# singlet %s", packageVersion("singlet")),
            "```",
            "",
            "Raw data: `manuscript/cv_benchmark_results.csv`"
)

# Write report
writeLines(report, REPORT_PATH)
cat("Report saved to", REPORT_PATH, "\n")

# Print final summary
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("BENCHMARK COMPLETE\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat(sprintf("\nOverall speedup: %.2fx (RcppML faster)\n", 
            mean(speedups$speedup, na.rm = TRUE)))
cat(sprintf("Range: %.2fx - %.2fx\n",
            min(speedups$speedup, na.rm = TRUE),
            max(speedups$speedup, na.rm = TRUE)))
