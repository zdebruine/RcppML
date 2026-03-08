################################################################################
## RcppML: High-Performance Non-Negative Matrix Factorization for R
## Journal of Statistical Software - Replication Script
################################################################################
##
## This script generates all figures and tables for the JSS manuscript.
## Run from the manuscript/code directory.
##
################################################################################

## Required packages
library("RcppML")
library("Matrix")
library("ggplot2")
library("irlba")

## Set options for JSS style
options(prompt = "R> ", continue = "+  ", width = 70, useFancyQuotes = FALSE)

## Output directories
fig_dir <- "../figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

## Helper function for consistent plot styling
theme_jss <- function() {
  theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "gray50", linewidth = 0.5),
      strip.background = element_rect(fill = "gray95", color = NA),
      legend.position = "bottom",
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 10)
    )
}

################################################################################
## Section 1: Basic NMF Usage Demonstration
################################################################################

cat("=== Section 1: Basic NMF Demo ===\n")

## Load built-in dataset
data("hawaiibirds", package = "RcppML")
dim(hawaiibirds)

## Basic NMF factorization
set.seed(42)
model_basic <- nmf(hawaiibirds, k = 10, tol = 1e-4, maxit = 100, verbose = FALSE)

## Display model summary
cat("\nBasic NMF model:\n")
cat("  Rank:", ncol(model_basic@w), "\n")
cat("  Iterations:", model_basic@misc$iter, "\n")
cat("  Final tolerance:", format(model_basic@misc$tol, digits = 4), "\n")

################################################################################
## Section 2: Cross-Validation for Rank Selection
################################################################################

cat("\n=== Section 2: Cross-Validation ===\n")

## Generate synthetic data with known rank
generate_nmf_data <- function(m, n, k_true, noise_sd = 0.1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  W_true <- matrix(abs(rnorm(m * k_true)), m, k_true)
  H_true <- matrix(abs(rnorm(k_true * n)), k_true, n)
  A_clean <- W_true %*% H_true
  A <- A_clean + matrix(rnorm(m * n, 0, noise_sd * mean(A_clean)), m, n)
  A[A < 0] <- 0
  list(A = A, W_true = W_true, H_true = H_true, A_clean = A_clean)
}

## Create test matrix with known rank = 5
synth_data <- generate_nmf_data(200, 150, k_true = 5, noise_sd = 0.1, seed = 123)

## Cross-validation across ranks
cat("\nRunning cross-validation across ranks 2-15...\n")
cv_results <- nmf(synth_data$A, k = 2:15, reps = 3, cv_fraction = 0.1, 
                  maxit = 50, verbose = FALSE)

## Find optimal rank
optimal_k <- cv_results$k[which.min(cv_results$test_mse)]
cat("Optimal rank detected:", optimal_k, "(true rank: 5)\n")

## Plot CV results
p_cv <- ggplot(cv_results, aes(x = k, y = test_mse, group = rep)) +
  geom_line(alpha = 0.3) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", 
               color = "steelblue", linewidth = 1.2) +
  stat_summary(aes(group = 1), fun = mean, geom = "point", 
               color = "steelblue", size = 3) +
  geom_vline(xintercept = 5, linetype = "dashed", color = "red") +
  annotate("text", x = 5.5, y = max(cv_results$test_mse) * 0.95, 
           label = "True rank = 5", hjust = 0, color = "red") +
  labs(x = "Factorization rank (k)", y = "Test MSE (held-out entries)",
       title = "Cross-validation for rank selection") +
  theme_jss()

ggsave(file.path(fig_dir, "fig_cv_rank.pdf"), p_cv, width = 6, height = 4)
cat("Saved: fig_cv_rank.pdf\n")

################################################################################
## Section 3: Automatic Rank Determination (k = "auto")
################################################################################

cat("\n=== Section 3: Automatic Rank Determination ===\n")

## Use k = "auto" for automatic rank search
cat("Running automatic rank search (golden section optimization)...\n")
model_auto <- nmf(synth_data$A, k = "auto", cv_k_init = 2, cv_max_k = 20,
                  cv_tolerance = 2, maxit = 50, verbose = FALSE)

cat("Automatic rank selection result:\n")
cat("  Optimal k:", model_auto@misc$rank_search$k_optimal, "\n")
cat("  Search range: [", model_auto@misc$rank_search$k_low, ", ",
    model_auto@misc$rank_search$k_high, "]\n", sep = "")

################################################################################
## Section 4: Rank-2 NMF for Bipartitioning
################################################################################

cat("\n=== Section 4: Rank-2 Bipartitioning ===\n")

## Benchmark: rank-2 NMF vs irlba on larger sparse matrix
cat("\nBenchmarking rank-2 factorization...\n")
n_bench <- 20
set.seed(42)
A_bench <- rsparsematrix(1000, 500, density = 0.1)
A_bench <- abs(A_bench)

time_nmf <- system.time(replicate(n_bench, {
  bipartition(A_bench, tol = 1e-4, calc_dist = FALSE)
}))["elapsed"] / n_bench

time_irlba <- system.time(replicate(n_bench, {
  irlba(A_bench, nv = 2)
}))["elapsed"] / n_bench

cat("Mean time per factorization (1000x500 sparse, 10% density):\n")
cat("  RcppML bipartition:", format(time_nmf * 1000, digits = 3), "ms\n")
cat("  irlba (k=2):", format(time_irlba * 1000, digits = 3), "ms\n")
cat("  Speedup:", format(time_irlba / time_nmf, digits = 2), "x\n")

################################################################################
## Section 5: Divisive Clustering (dclust)
################################################################################

cat("\n=== Section 5: Divisive Clustering ===\n")

## Cluster a larger sparse matrix instead
set.seed(42)
A_clust <- rsparsematrix(100, 50, density = 0.3)
A_clust <- abs(A_clust)

clusters <- dclust(A_clust, min_samples = 5, min_dist = 0.01, 
                   tol = 1e-5, seed = 42)

cat("Number of clusters found:", length(clusters), "\n")

################################################################################
## Section 6: Loss Functions (MSE, MAE, Huber, KL)
################################################################################

cat("\n=== Section 6: Loss Functions ===\n")

## Create data with outliers
set.seed(42)
A_clean <- abs(matrix(rnorm(100 * 50), 100, 50))
A_outlier <- A_clean
n_outliers <- 50
outlier_idx <- sample(length(A_outlier), n_outliers)
A_outlier[outlier_idx] <- A_outlier[outlier_idx] + 10 * abs(rnorm(n_outliers))

## Fit with different loss functions
losses <- c("mse", "mae", "huber")
results_loss <- list()

for (loss in losses) {
  set.seed(42)
  model <- nmf(A_outlier, k = 5, loss = loss, maxit = 100, 
               tol = 1e-4, verbose = FALSE)
  recon <- model@w %*% diag(model@d) %*% model@h
  
  ## Compute MSE on clean data (to assess robustness)
  mse_clean <- mean((A_clean - recon)^2)
  
  results_loss[[loss]] <- data.frame(
    loss_function = loss,
    mse_on_clean = mse_clean,
    iterations = model@misc$iter
  )
  
  cat(sprintf("  %s: MSE on clean data = %.4f, iterations = %d\n",
              toupper(loss), mse_clean, model@misc$iter))
}

results_loss_df <- do.call(rbind, results_loss)

################################################################################
## Section 7: Regularization Effects
################################################################################

cat("\n=== Section 7: Regularization ===\n")

## L1 sparsity experiments
set.seed(42)
A_reg <- abs(matrix(rnorm(200 * 100), 200, 100))

L1_values <- c(0, 0.01, 0.05, 0.1, 0.2)
sparsity_results <- data.frame(
  L1 = numeric(),
  W_sparsity = numeric(),
  H_sparsity = numeric(),
  recon_mse = numeric()
)

for (L1 in L1_values) {
  model <- nmf(A_reg, k = 10, L1 = L1, maxit = 100, tol = 1e-4, verbose = FALSE)
  
  W_sparse <- mean(abs(model@w) < 0.01)
  H_sparse <- mean(abs(model@h) < 0.01)
  recon <- model@w %*% diag(model@d) %*% model@h
  mse <- mean((A_reg - recon)^2)
  
  sparsity_results <- rbind(sparsity_results, data.frame(
    L1 = L1, W_sparsity = W_sparse, H_sparsity = H_sparse, recon_mse = mse
  ))
}

cat("L1 regularization effects:\n")
print(sparsity_results)

## Plot sparsity vs reconstruction error
p_reg <- ggplot(sparsity_results, aes(x = recon_mse, y = (W_sparsity + H_sparsity) / 2 * 100)) +
  geom_path(color = "steelblue", linewidth = 1) +
  geom_point(aes(size = L1), color = "steelblue") +
  geom_text(aes(label = sprintf("L1=%.2f", L1)), vjust = -1, size = 3) +
  labs(x = "Reconstruction MSE", y = "Factor sparsity (%)",
       title = "L1 regularization trade-off") +
  scale_size_continuous(range = c(2, 6), guide = "none") +
  theme_jss()

ggsave(file.path(fig_dir, "fig_l1_tradeoff.pdf"), p_reg, width = 5, height = 4)
cat("Saved: fig_l1_tradeoff.pdf\n")

################################################################################
## Section 8: CV Overhead Benchmark
################################################################################

cat("\n=== Section 8: CV Overhead ===\n")

## Benchmark CV overhead
sizes <- c(500, 1000, 2000)
cv_fractions <- c(0, 0.05, 0.1, 0.15)

overhead_results <- data.frame(
  size = integer(),
  cv_fraction = numeric(),
  time_seconds = numeric()
)

for (sz in sizes) {
  set.seed(42)
  A_bench <- rsparsematrix(sz, sz, density = 0.3)
  A_bench <- abs(A_bench)
  
  for (cv_frac in cv_fractions) {
    t <- system.time({
      if (cv_frac == 0) {
        nmf(A_bench, k = 10, maxit = 20, verbose = FALSE)
      } else {
        nmf(A_bench, k = 10, maxit = 20, cv_fraction = cv_frac, verbose = FALSE)
      }
    })["elapsed"]
    
    overhead_results <- rbind(overhead_results, data.frame(
      size = sz, cv_fraction = cv_frac, time_seconds = t
    ))
  }
  cat(sprintf("  Size %dx%d complete\n", sz, sz))
}

## Compute overhead percentages
overhead_results$label <- paste0(overhead_results$size, "x", overhead_results$size)
baseline <- overhead_results[overhead_results$cv_fraction == 0, ]
names(baseline)[names(baseline) == "time_seconds"] <- "baseline_time"
baseline <- baseline[, c("size", "baseline_time")]

overhead_results <- merge(overhead_results, baseline, by = "size")
overhead_results$overhead_pct <- (overhead_results$time_seconds / overhead_results$baseline_time - 1) * 100

cat("\nCV overhead results:\n")
print(overhead_results[overhead_results$cv_fraction > 0, 
                       c("label", "cv_fraction", "overhead_pct")])

################################################################################
## Section 9: Scaling Benchmark
################################################################################

cat("\n=== Section 9: Scaling Benchmark ===\n")

## Compare runtime scaling
sizes_scale <- c(100, 200, 500, 1000, 2000)
k_test <- 10
density_test <- 0.1

scaling_results <- data.frame(
  size = integer(),
  time_sparse = numeric(),
  time_dense = numeric()
)

for (sz in sizes_scale) {
  set.seed(42)
  A_sp <- rsparsematrix(sz, sz, density = density_test)
  A_sp <- abs(A_sp)
  A_de <- as.matrix(A_sp)
  
  t_sparse <- system.time(nmf(A_sp, k = k_test, maxit = 20, verbose = FALSE))["elapsed"]
  t_dense <- system.time(nmf(A_de, k = k_test, maxit = 20, verbose = FALSE))["elapsed"]
  
  scaling_results <- rbind(scaling_results, data.frame(
    size = sz, time_sparse = t_sparse, time_dense = t_dense
  ))
  
  cat(sprintf("  Size %d: sparse=%.3fs, dense=%.3fs\n", sz, t_sparse, t_dense))
}

## Plot scaling
scaling_long <- data.frame(
  size = rep(scaling_results$size, 2),
  time = c(scaling_results$time_sparse, scaling_results$time_dense),
  type = rep(c("Sparse (dgCMatrix)", "Dense (matrix)"), each = nrow(scaling_results))
)

p_scale <- ggplot(scaling_long, aes(x = size, y = time, color = type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Matrix dimension (n x n)", y = "Time (seconds, log scale)",
       title = "NMF runtime scaling (k=10, 20 iterations)",
       color = "Input type") +
  theme_jss() +
  theme(legend.position = c(0.3, 0.8))

ggsave(file.path(fig_dir, "fig_scaling.pdf"), p_scale, width = 6, height = 4)
cat("Saved: fig_scaling.pdf\n")

################################################################################
## Section 10: Real Data Application - PBMC3k
################################################################################

cat("\n=== Section 10: Real Data Application ===\n")

## Use hawaiibirds for a real data demo
data("hawaiibirds", package = "RcppML")
cat("hawaiibirds dimensions:", dim(hawaiibirds)[1], "x", dim(hawaiibirds)[2], "\n")

## Run NMF
set.seed(42)
model_birds <- nmf(hawaiibirds, k = 5, maxit = 100, tol = 1e-4, 
                   L1 = c(0.01, 0), verbose = FALSE)

cat("Model converged in", model_birds@misc$iter, "iterations\n")

## Show top features per factor
W <- model_birds@w
for (i in 1:5) {
  top_features <- head(order(W[, i], decreasing = TRUE), 3)
  cat(sprintf("Factor %d top features: %s\n", i, 
              paste(rownames(hawaiibirds)[top_features], collapse = ", ")))
}

################################################################################
## Session Info
################################################################################

cat("\n=== Session Info ===\n")
sessionInfo()
