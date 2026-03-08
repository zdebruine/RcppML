#!/usr/bin/env Rscript
# =============================================================================
# Generate All Figures and Benchmark Data for RcppML JSS Manuscript
# =============================================================================
# This script generates all figures, benchmark tables, and experimental results
# needed for the Journal of Statistical Software manuscript on RcppML.
# 
# Run: Rscript generate_manuscript_figures.R
# Output directory: manuscript/jss/figures/
# =============================================================================

library(RcppML)
library(Matrix)
library(ggplot2)
library(microbenchmark)

# Create output directory
fig_dir <- "manuscript/jss/figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

cat("=== RcppML Manuscript Figure Generation ===\n\n")

# =============================================================================
# SECTION 1: Synthetic Data Cross-Validation Experiment
# Demonstrates that CV correctly identifies the true rank
# =============================================================================

cat("1. Generating synthetic CV rank identification experiment...\n")

set.seed(42)

# Generate synthetic NMF data with known rank = 5
sim_data <- simulateNMF(nrow = 200, ncol = 150, k = 5, noise = 0.3, dropout = 0)
A_synth <- sim_data$A + abs(matrix(rnorm(200 * 150, sd = 0.1), 200, 150))  # Add small noise
A_synth[A_synth < 0] <- 0

# Run CV across multiple ranks
cv_results <- nmf(A_synth, k = 2:12, reps = 3, cv_fraction = 0.1, maxit = 100, 
                  seed = 123, verbose = FALSE)

# Compute mean test MSE per rank
cv_summary <- aggregate(test_mse ~ k, data = cv_results, 
                        FUN = function(x) c(mean = mean(x), se = sd(x)/sqrt(length(x))))
cv_summary <- do.call(data.frame, cv_summary)
colnames(cv_summary) <- c("k", "mean_test_mse", "se")

# Find optimal rank
optimal_k <- cv_summary$k[which.min(cv_summary$mean_test_mse)]
cat(sprintf("  True rank: 5, CV-selected rank: %d\n", optimal_k))

# Create CV rank selection plot
p_cv <- ggplot(cv_summary, aes(x = k, y = mean_test_mse)) +
  geom_line(linewidth = 1, color = "#1f77b4") +
  geom_point(size = 3, color = "#1f77b4") +
  geom_errorbar(aes(ymin = mean_test_mse - 1.96*se, ymax = mean_test_mse + 1.96*se),
                width = 0.3, alpha = 0.5) +
  geom_vline(xintercept = 5, linetype = "dashed", color = "#d62728", linewidth = 1) +
  geom_vline(xintercept = optimal_k, linetype = "dotted", color = "#2ca02c", linewidth = 1) +
  annotate("text", x = 5.5, y = max(cv_summary$mean_test_mse), 
           label = "True rank", hjust = 0, color = "#d62728") +
  scale_x_continuous(breaks = 2:12) +
  labs(x = "Factorization Rank (k)", y = "Test MSE (held-out entries)",
       title = "Cross-Validation for Rank Selection",
       subtitle = "Synthetic data with true rank k = 5") +
  theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank())

ggsave(file.path(fig_dir, "fig_cv_rank.pdf"), p_cv, width = 7, height = 5)
cat("  Saved: fig_cv_rank.pdf\n")

# =============================================================================
# SECTION 2: Regularization Demonstration
# Shows effect of L1, L2, L21, and orthogonality penalties
# =============================================================================

cat("\n2. Generating regularization comparison figures...\n")

# Load hawaiibirds dataset
data("hawaiibirds", package = "RcppML")

# Fit models with different regularization
set.seed(123)
model_none <- nmf(hawaiibirds, k = 5, L1 = c(0, 0), maxit = 100, verbose = FALSE)
model_L1 <- nmf(hawaiibirds, k = 5, L1 = c(0.1, 0.1), maxit = 100, verbose = FALSE)
model_L2 <- nmf(hawaiibirds, k = 5, L2 = c(0.1, 0.1), maxit = 100, verbose = FALSE)
model_L21 <- nmf(hawaiibirds, k = 5, L21 = c(0.1, 0), maxit = 100, verbose = FALSE)
model_ortho <- nmf(hawaiibirds, k = 5, ortho = c(0.1, 0.1), maxit = 100, verbose = FALSE)

# Compute sparsity for each model
compute_sparsity <- function(model, name) {
  w_sparsity <- mean(model@w < 1e-10)
  h_sparsity <- mean(model@h < 1e-10)
  data.frame(
    model = name,
    matrix = c("W", "H"),
    sparsity = c(w_sparsity, h_sparsity)
  )
}

sparsity_df <- rbind(
  compute_sparsity(model_none, "No Regularization"),
  compute_sparsity(model_L1, "L1"),
  compute_sparsity(model_L2, "L2"),
  compute_sparsity(model_L21, "L21 on W"),
  compute_sparsity(model_ortho, "Orthogonality")
)
sparsity_df$model <- factor(sparsity_df$model, 
                            levels = c("No Regularization", "L1", 
                                      "L2", "L21 on W", 
                                      "Orthogonality"))

p_sparsity <- ggplot(sparsity_df, aes(x = model, y = sparsity * 100, fill = matrix)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("W" = "#1f77b4", "H" = "#ff7f0e")) +
  labs(x = "", y = "Sparsity (%)", fill = "Factor Matrix",
       title = "Effect of Regularization on Factor Sparsity") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        legend.position = "top") +
  coord_cartesian(ylim = c(0, 100))

ggsave(file.path(fig_dir, "fig_regularization_sparsity.pdf"), p_sparsity, 
       width = 8, height = 5)
cat("  Saved: fig_regularization_sparsity.pdf\n")

# =============================================================================
# SECTION 3: Cross-Validation Train vs Test Loss
# Demonstrates CV tracking capability
# =============================================================================

cat("\n3. Generating CV train/test loss comparison...\n")

# Run CV across multiple ranks to show train vs test curves
set.seed(456)
cv_full <- nmf(hawaiibirds, k = 2:15, reps = 3, cv_fraction = 0.1, maxit = 100,
               verbose = FALSE)

# Aggregate results
cv_agg <- aggregate(cbind(train_mse, test_mse) ~ k, data = cv_full, FUN = mean)

# Create train vs test plot
cv_long <- data.frame(
  k = rep(cv_agg$k, 2),
  MSE = c(cv_agg$train_mse, cv_agg$test_mse),
  Type = rep(c("Training", "Test (held-out)"), each = nrow(cv_agg))
)

p_loss <- ggplot(cv_long, aes(x = k, y = MSE, color = Type, linetype = Type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Test (held-out)" = "#ff7f0e")) +
  scale_x_continuous(breaks = seq(2, 15, 2)) +
  labs(x = "Factorization Rank (k)", y = "Mean Squared Error",
       title = "Training vs Test Loss Curves",
       subtitle = "Cross-validation with 10% held-out entries") +
  theme_bw(base_size = 12) +
  theme(legend.position = c(0.8, 0.8),
        legend.background = element_rect(fill = "white", color = "gray80"))

ggsave(file.path(fig_dir, "fig_loss_history.pdf"), p_loss, width = 7, height = 5)
cat("  Saved: fig_loss_history.pdf\n")

# Create convergence plot showing optimal rank identification
optimal_cv_k <- cv_agg$k[which.min(cv_agg$test_mse)]

p_conv <- ggplot(cv_agg, aes(x = k, y = test_mse)) +
  geom_line(linewidth = 1, color = "#1f77b4") +
  geom_point(size = 3, color = "#1f77b4") +
  geom_vline(xintercept = optimal_cv_k, linetype = "dashed", color = "#d62728") +
  annotate("text", x = optimal_cv_k + 0.5, y = max(cv_agg$test_mse) * 0.95,
           label = paste0("Optimal k = ", optimal_cv_k), hjust = 0, color = "#d62728") +
  scale_x_continuous(breaks = seq(2, 15, 2)) +
  labs(x = "Factorization Rank (k)", y = "Test MSE (held-out entries)",
       title = "Cross-Validation for Rank Selection",
       subtitle = "hawaiibirds dataset, 3 replicates, 10% holdout") +
  theme_bw(base_size = 12)

ggsave(file.path(fig_dir, "fig_convergence.pdf"), p_conv, width = 7, height = 5)
cat("  Saved: fig_convergence.pdf\n")

# =============================================================================
# SECTION 4: Consensus NMF with Heatmap
# =============================================================================

cat("\n4. Generating consensus NMF heatmap...\n")

set.seed(789)
cons <- consensus_nmf(hawaiibirds, k = 5, reps = 20, maxit = 50, verbose = FALSE)
cat(sprintf("  Cophenetic correlation: %.4f\n", cons$cophenetic))

p_consensus <- plot(cons, show_clusters = TRUE, cluster_rows = TRUE)
ggsave(file.path(fig_dir, "fig_consensus_heatmap.pdf"), p_consensus, 
       width = 7, height = 6)
cat("  Saved: fig_consensus_heatmap.pdf\n")

# =============================================================================
# SECTION 5: Benchmark: RcppML vs NMF Package
# =============================================================================

cat("\n5. Running benchmarks: RcppML vs NMF package...\n")

# Check if NMF package is available
if (!requireNamespace("NMF", quietly = TRUE)) {
  cat("  WARNING: NMF package not installed. Skipping comparison benchmarks.\n")
} else {
  
  # Create test matrices of different sizes
  benchmark_results <- list()
  
  # Dense matrix benchmark
  sizes <- list(
    c(500, 200),
    c(1000, 500),
    c(2000, 1000)
  )
  
  for (sz in sizes) {
    m <- sz[1]
    n <- sz[2]
    cat(sprintf("  Benchmarking %d x %d matrix...\n", m, n))
    
    set.seed(123)
    A_dense <- abs(matrix(rnorm(m * n), m, n))
    
    # Benchmark both packages - use explicit namespaces
    timing <- microbenchmark(
      RcppML = RcppML::nmf(A_dense, k = 10, maxit = 50, tol = 1e-5, verbose = FALSE),
      NMF_lee = NMF::nmf(A_dense, rank = 10, nrun = 1, method = "lee", 
                         maxIter = 50, .options = "v"),
      NMF_brunet = NMF::nmf(A_dense, rank = 10, nrun = 1, method = "brunet", 
                            maxIter = 50, .options = "v"),
      times = 5
    )
    
    timing_summary <- summary(timing)
    benchmark_results[[paste0(m, "x", n)]] <- data.frame(
      rows = m,
      cols = n,
      RcppML_ms = timing_summary$median[timing_summary$expr == "RcppML"],
      NMF_lee_ms = timing_summary$median[timing_summary$expr == "NMF_lee"],
      NMF_brunet_ms = timing_summary$median[timing_summary$expr == "NMF_brunet"]
    )
  }
  
  benchmark_df <- do.call(rbind, benchmark_results)
  benchmark_df$speedup_lee <- benchmark_df$NMF_lee_ms / benchmark_df$RcppML_ms
  benchmark_df$speedup_brunet <- benchmark_df$NMF_brunet_ms / benchmark_df$RcppML_ms
  
  # Save benchmark table
  write.csv(benchmark_df, file.path(fig_dir, "benchmark_nmf_comparison.csv"), 
            row.names = FALSE)
  cat("  Saved: benchmark_nmf_comparison.csv\n")
  
  # Create benchmark plot
  bench_long <- data.frame(
    size = factor(paste0(benchmark_df$rows, "×", benchmark_df$cols),
                  levels = paste0(benchmark_df$rows, "×", benchmark_df$cols)),
    Package = rep(c("RcppML", "NMF (Lee)", "NMF (Brunet)"), each = nrow(benchmark_df)),
    time_ms = c(benchmark_df$RcppML_ms, benchmark_df$NMF_lee_ms, benchmark_df$NMF_brunet_ms)
  )
  
  p_bench <- ggplot(bench_long, aes(x = size, y = time_ms / 1000, fill = Package)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("RcppML" = "#2ca02c", "NMF (Lee)" = "#1f77b4", 
                                 "NMF (Brunet)" = "#ff7f0e")) +
    labs(x = "Matrix Dimensions (rows × cols)", y = "Runtime (seconds)",
         title = "NMF Performance Comparison",
         subtitle = "Rank-10 factorization, 50 iterations") +
    theme_bw(base_size = 12) +
    theme(legend.position = "top")
  
  ggsave(file.path(fig_dir, "fig_benchmark_comparison.pdf"), p_bench, 
         width = 8, height = 5)
  cat("  Saved: fig_benchmark_comparison.pdf\n")
}

# =============================================================================
# SECTION 6: Sparse Matrix Benchmarks
# =============================================================================

cat("\n6. Running sparse matrix benchmarks...\n")

sparse_results <- list()

# Generate sparse matrices of different sizes
sparsities <- c(0.99, 0.95, 0.90)
for (sp in sparsities) {
  set.seed(123)
  A_sparse <- rsparsematrix(5000, 2000, density = 1 - sp)
  A_sparse@x <- abs(A_sparse@x)
  
  nnz <- length(A_sparse@x)
  cat(sprintf("  Benchmarking 5000x2000 sparse matrix (%.0f%% sparse, %d nnz)...\n", 
              sp * 100, nnz))
  
  timing <- microbenchmark(
    RcppML = nmf(A_sparse, k = 10, maxit = 50, tol = 1e-5, verbose = FALSE),
    times = 5
  )
  
  sparse_results[[sprintf("%.0f%%", sp * 100)]] <- data.frame(
    sparsity = sp * 100,
    nnz = nnz,
    time_ms = median(timing$time) / 1e6
  )
}

sparse_df <- do.call(rbind, sparse_results)
write.csv(sparse_df, file.path(fig_dir, "benchmark_sparse.csv"), row.names = FALSE)
cat("  Saved: benchmark_sparse.csv\n")

# =============================================================================
# SECTION 7: Cross-Validation Overhead Measurement
# =============================================================================

cat("\n7. Measuring cross-validation overhead...\n")

set.seed(123)
A_cv_test <- abs(matrix(rnorm(2000 * 1000), 2000, 1000))

cv_overhead <- microbenchmark(
  without_cv = nmf(A_cv_test, k = 10, maxit = 50, verbose = FALSE),
  with_cv = nmf(A_cv_test, k = 10, maxit = 50, cv_fraction = 0.1, verbose = FALSE),
  times = 5
)

cv_overhead_summary <- summary(cv_overhead)
overhead_pct <- (cv_overhead_summary$median[2] - cv_overhead_summary$median[1]) / 
                 cv_overhead_summary$median[1] * 100
cat(sprintf("  CV overhead: %.1f%% (median runtime increase)\n", overhead_pct))

# =============================================================================
# SECTION 8: Robust Loss Functions Comparison
# =============================================================================

cat("\n8. Generating robust loss function comparison...\n")

set.seed(123)
# Create clean synthetic data
A_clean <- abs(matrix(rnorm(500 * 200), 500, 200))

# Add outliers
A_outlier <- A_clean
outlier_idx <- sample(length(A_outlier), 500)  # 0.5% outliers
A_outlier[outlier_idx] <- A_outlier[outlier_idx] + runif(500, 5, 10)

# Fit with different loss functions
model_mse <- nmf(A_outlier, k = 5, loss = "mse", maxit = 100, verbose = FALSE)
model_mae <- nmf(A_outlier, k = 5, loss = "mae", maxit = 100, verbose = FALSE)
model_huber <- nmf(A_outlier, k = 5, loss = "huber", huber_delta = 1.0, 
                   maxit = 100, verbose = FALSE)

# Compute MSE on clean data
compute_recon_mse <- function(model, A_true) {
  recon <- model@w %*% diag(model@d) %*% model@h
  mean((A_true - recon)^2)
}

loss_comparison <- data.frame(
  Loss = c("MSE", "MAE", "Huber"),
  MSE_on_clean = c(
    compute_recon_mse(model_mse, A_clean),
    compute_recon_mse(model_mae, A_clean),
    compute_recon_mse(model_huber, A_clean)
  )
)

write.csv(loss_comparison, file.path(fig_dir, "loss_function_comparison.csv"), 
          row.names = FALSE)
cat("  Saved: loss_function_comparison.csv\n")
print(loss_comparison)

# =============================================================================
# SECTION 9: Factor Summary Statistics
# =============================================================================

cat("\n9. Generating factor summary...\n")

# Use hawaiibirds data for interpretable factors
set.seed(123)
model_hawaii <- nmf(hawaiibirds, k = 5, maxit = 100, verbose = FALSE)

# Export factor summary 
factor_summary <- data.frame(
  Factor = 1:5,
  Diagonal_value = model_hawaii@d,
  W_sparsity = apply(model_hawaii@w, 2, function(x) mean(x < 1e-10)),
  H_sparsity = apply(model_hawaii@h, 1, function(x) mean(x < 1e-10))
)
write.csv(factor_summary, file.path(fig_dir, "factor_summary.csv"), row.names = FALSE)
cat("  Saved: factor_summary.csv\n")

# =============================================================================
# SECTION 10: Runtime Scaling Analysis
# =============================================================================

cat("\n10. Generating runtime scaling analysis...\n")

scaling_results <- list()
sizes <- c(500, 1000, 2000, 3000, 5000)

for (n in sizes) {
  set.seed(123)
  A_scale <- abs(matrix(rnorm(n * n), n, n))
  
  cat(sprintf("  Timing %d x %d matrix...\n", n, n))
  
  timing <- microbenchmark(
    nmf(A_scale, k = 10, maxit = 50, tol = 1e-6, verbose = FALSE),
    times = 3
  )
  
  scaling_results[[as.character(n)]] <- data.frame(
    n = n,
    time_sec = median(timing$time) / 1e9
  )
}

scaling_df <- do.call(rbind, scaling_results)
scaling_df$elements <- scaling_df$n^2
scaling_df$per_element_ns <- scaling_df$time_sec * 1e9 / scaling_df$elements

write.csv(scaling_df, file.path(fig_dir, "scaling_analysis.csv"), row.names = FALSE)

p_scaling <- ggplot(scaling_df, aes(x = n, y = time_sec)) +
  geom_line(linewidth = 1, color = "#1f77b4") +
  geom_point(size = 3, color = "#1f77b4") +
  labs(x = "Matrix Dimension (n × n)", y = "Runtime (seconds)",
       title = "NMF Runtime Scaling",
       subtitle = "Rank-10 factorization, 50 iterations") +
  theme_bw(base_size = 12)

ggsave(file.path(fig_dir, "fig_scaling.pdf"), p_scaling, width = 7, height = 5)
cat("  Saved: fig_scaling.pdf\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n=== Figure Generation Complete ===\n")
cat(sprintf("All figures saved to: %s/\n", fig_dir))
cat("\nGenerated files:\n")
cat("  - fig_cv_rank.pdf          : Cross-validation rank selection\n")
cat("  - fig_regularization_sparsity.pdf : Regularization effects\n")
cat("  - fig_loss_history.pdf     : Training loss tracking\n")
cat("  - fig_convergence.pdf      : Convergence analysis\n")
cat("  - fig_consensus_heatmap.pdf: Consensus clustering\n")
cat("  - fig_benchmark_comparison.pdf : Package comparison\n")
cat("  - fig_scaling.pdf          : Runtime scaling\n")
cat("  - benchmark_nmf_comparison.csv : Detailed benchmark data\n")
cat("  - benchmark_sparse.csv     : Sparse matrix benchmarks\n")
cat("  - loss_function_comparison.csv : Robust loss comparison\n")
cat("  - scaling_analysis.csv     : Scaling analysis data\n")
