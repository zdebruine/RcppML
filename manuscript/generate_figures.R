#' =============================================================================
#' RcppML Manuscript: Figure and Table Generation
#' =============================================================================
#' 
#' This script generates all figures and tables for the RcppML NMF manuscript.
#' Each section corresponds to a specific experiment described in the manuscript.
#' 
#' Dependencies: RcppML, Matrix, ggplot2, dplyr, tidyr, patchwork, bench
#' =============================================================================

library(RcppML)
library(Matrix)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# Set global theme for publication-quality figures
theme_manuscript <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "gray50"),
      strip.background = element_rect(fill = "gray95"),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 14)
    )
}

# Output directory
output_dir <- "manuscript/figures"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# =============================================================================
# Experiment 1: CV Overhead Quantification
# =============================================================================

run_cv_overhead_experiment <- function() {
  cat("Running Experiment 1: CV Overhead Quantification\n")
  
  # Parameters
  sizes <- c(500, 1000, 2000)
  sparsities <- c(0.1, 0.5, 0.9)
  holdout_fractions <- c(0.05, 0.1, 0.15)
  k <- 10
  maxit <- 50
  n_reps <- 3
  
  results <- expand.grid(
    size = sizes,
    sparsity = sparsities,
    holdout = holdout_fractions,
    rep = 1:n_reps
  )
  results$time_no_cv <- NA
  results$time_cv <- NA
  results$overhead_pct <- NA
  
  for (i in seq_len(nrow(results))) {
    m <- n <- results$size[i]
    sp <- results$sparsity[i]
    hf <- results$holdout[i]
    
    set.seed(results$rep[i])
    A <- rsparsematrix(m, n, density = 1 - sp)
    A <- abs(A)
    
    # Without CV
    t1 <- system.time({
      nmf(A, k, maxit = maxit, verbose = FALSE)
    })["elapsed"]
    
    # With CV
    t2 <- system.time({
      nmf(A, k, maxit = maxit, cv_fraction = hf, verbose = FALSE)
    })["elapsed"]
    
    results$time_no_cv[i] <- t1
    results$time_cv[i] <- t2
    results$overhead_pct[i] <- 100 * (t2 - t1) / t1
    
    cat(sprintf("  Size %d, Sparsity %.0f%%, Holdout %.0f%%, Overhead: %.1f%%\n",
                m, sp * 100, hf * 100, results$overhead_pct[i]))
  }
  
  # Summarize
  results_summary <- results %>%
    group_by(size, sparsity, holdout) %>%
    summarise(
      mean_overhead = mean(overhead_pct),
      sd_overhead = sd(overhead_pct),
      .groups = "drop"
    )
  
  # Figure 1: CV Overhead
  p1 <- ggplot(results_summary, aes(x = factor(holdout * 100), y = mean_overhead,
                                     fill = factor(sparsity * 100))) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = mean_overhead - sd_overhead, 
                      ymax = mean_overhead + sd_overhead),
                  position = position_dodge(width = 0.8), width = 0.2) +
    facet_wrap(~ paste0(size, " x ", size, " matrix")) +
    labs(x = "Holdout Fraction (%)", y = "CV Overhead (%)",
         fill = "Non-zero %",
         title = "Figure 1: Cross-Validation Computational Overhead",
         subtitle = "Percentage increase in runtime when CV is enabled") +
    scale_fill_viridis_d() +
    theme_manuscript()
  
  ggsave(file.path(output_dir, "fig1_cv_overhead.pdf"), p1, width = 10, height = 6)
  ggsave(file.path(output_dir, "fig1_cv_overhead.png"), p1, width = 10, height = 6, dpi = 300)
  
  return(list(results = results, summary = results_summary, plot = p1))
}

# =============================================================================
# Experiment 2: Rank Selection Accuracy
# =============================================================================

generate_nmf_data <- function(m, n, k_true, noise_sd = 0.1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # Generate true factors
  W_true <- matrix(abs(rnorm(m * k_true)), m, k_true)
  H_true <- matrix(abs(rnorm(k_true * n)), k_true, n)
  
  # Normalize
  W_true <- scale(W_true, center = FALSE, scale = colSums(W_true))
  H_true <- t(scale(t(H_true), center = FALSE, scale = rowSums(H_true)))
  
  # Generate clean signal
  A_clean <- W_true %*% diag(runif(k_true, 0.5, 2)) %*% H_true
  
  # Add noise
  A <- A_clean + matrix(rnorm(m * n, 0, noise_sd), m, n)
  A[A < 0] <- 0  # Enforce non-negativity
  
  return(list(A = A, W_true = W_true, H_true = H_true, A_clean = A_clean))
}

run_rank_selection_experiment <- function() {
  cat("Running Experiment 2: Rank Selection Accuracy\n")
  
  # Parameters
  m <- n <- 500
  true_ranks <- c(3, 5, 8, 12)
  test_ranks <- 2:20
  noise_levels <- c(0.05, 0.1, 0.2)
  holdout_fraction <- 0.1
  n_reps <- 10
  
  results <- list()
  
  for (k_true in true_ranks) {
    for (noise_sd in noise_levels) {
      for (rep in 1:n_reps) {
        cat(sprintf("  True rank %d, noise %.2f, rep %d\n", k_true, noise_sd, rep))
        
        data <- generate_nmf_data(m, n, k_true, noise_sd, seed = rep * 1000 + k_true)
        
        # Run CV across ranks
        cv_result <- nmf(data$A, k = test_ranks, reps = 1, 
                         cv_fraction = holdout_fraction, verbose = FALSE)
        
        # Find optimal rank
        k_optimal <- cv_result$k[which.min(cv_result$test_mse)]
        
        results[[length(results) + 1]] <- data.frame(
          k_true = k_true,
          noise_sd = noise_sd,
          rep = rep,
          k_optimal = k_optimal,
          correct = (k_optimal == k_true)
        )
      }
    }
  }
  
  results_df <- bind_rows(results)
  
  # Also collect full CV curves for plotting
  cv_curves <- list()
  for (k_true in c(5, 8)) {
    for (rep in 1:3) {
      data <- generate_nmf_data(m, n, k_true, noise_sd = 0.1, seed = rep * 2000 + k_true)
      cv_result <- nmf(data$A, k = test_ranks, reps = 3, 
                       cv_fraction = holdout_fraction, verbose = FALSE)
      
      cv_curves[[length(cv_curves) + 1]] <- cv_result %>%
        mutate(k_true = k_true, seed = rep)
    }
  }
  cv_curves_df <- bind_rows(cv_curves)
  
  # Summary statistics
  accuracy_summary <- results_df %>%
    group_by(k_true, noise_sd) %>%
    summarise(
      accuracy = mean(correct),
      mean_k = mean(k_optimal),
      sd_k = sd(k_optimal),
      .groups = "drop"
    )
  
  # Figure 2: Rank Selection
  p2a <- cv_curves_df %>%
    group_by(k, k_true) %>%
    summarise(
      mean_test = mean(test_mse),
      se_test = sd(test_mse) / sqrt(n()),
      .groups = "drop"
    ) %>%
    ggplot(aes(x = k, y = mean_test, color = factor(k_true))) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean_test - 1.96 * se_test, 
                    ymax = mean_test + 1.96 * se_test,
                    fill = factor(k_true)), alpha = 0.2, color = NA) +
    geom_vline(aes(xintercept = k_true, color = factor(k_true)), 
               linetype = "dashed", linewidth = 0.8) +
    scale_color_viridis_d(name = "True Rank") +
    scale_fill_viridis_d(guide = "none") +
    labs(x = "Tested Rank (k)", y = "Test MSE",
         title = "A) CV Loss Curves by True Rank") +
    theme_manuscript()
  
  p2b <- ggplot(accuracy_summary, aes(x = factor(k_true), y = accuracy * 100,
                                       fill = factor(noise_sd))) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(x = "True Rank", y = "Correct Selection (%)",
         fill = "Noise SD",
         title = "B) Rank Selection Accuracy") +
    scale_fill_viridis_d() +
    ylim(0, 100) +
    theme_manuscript()
  
  p2 <- p2a + p2b + 
    plot_annotation(title = "Figure 2: Cross-Validation for Rank Selection")
  
  ggsave(file.path(output_dir, "fig2_rank_selection.pdf"), p2, width = 12, height = 5)
  ggsave(file.path(output_dir, "fig2_rank_selection.png"), p2, width = 12, height = 5, dpi = 300)
  
  return(list(results = results_df, accuracy = accuracy_summary, 
              cv_curves = cv_curves_df, plot = p2))
}

# =============================================================================
# Experiment 3: Loss Function Comparison
# =============================================================================

add_outliers <- function(A, outlier_frac, outlier_magnitude = 10) {
  n_outliers <- round(length(A) * outlier_frac)
  outlier_idx <- sample(length(A), n_outliers)
  A[outlier_idx] <- A[outlier_idx] + outlier_magnitude * abs(rnorm(n_outliers))
  return(A)
}

align_factors <- function(W_est, W_true) {
  # Find best permutation using Hungarian algorithm or greedy matching
  k <- ncol(W_est)
  cor_mat <- cor(W_est, W_true)
  
  # Greedy matching
  perm <- integer(k)
  used <- rep(FALSE, k)
  for (i in 1:k) {
    best_j <- which.max(abs(cor_mat[i, ]) * (!used))
    perm[i] <- best_j
    used[best_j] <- TRUE
  }
  
  return(list(
    W_aligned = W_est[, order(perm)],
    permutation = perm,
    correlations = diag(cor_mat[, perm])
  ))
}

run_loss_comparison_experiment <- function() {
  cat("Running Experiment 3: Loss Function Comparison\n")
  
  # Parameters
  m <- n <- 300
  k_true <- 5
  outlier_fracs <- c(0, 0.05, 0.1, 0.2)
  loss_types <- c("mse", "mae", "huber")
  n_reps <- 10
  
  results <- list()
  
  for (outlier_frac in outlier_fracs) {
    for (loss_type in loss_types) {
      for (rep in 1:n_reps) {
        cat(sprintf("  Outliers %.0f%%, loss %s, rep %d\n", 
                    outlier_frac * 100, loss_type, rep))
        
        # Generate data
        data <- generate_nmf_data(m, n, k_true, noise_sd = 0.1, seed = rep * 3000)
        A <- if (outlier_frac > 0) add_outliers(data$A, outlier_frac) else data$A
        
        # Fit model
        model <- nmf(A, k_true, loss = loss_type, maxit = 100, 
                     tol = 1e-5, verbose = FALSE)
        
        # Align factors and compute recovery error
        aligned <- align_factors(model$w, data$W_true)
        mean_cor <- mean(abs(aligned$correlations))
        
        # Compute reconstruction error on clean data
        recon_mse <- mean((data$A_clean - model$w %*% diag(model$d) %*% model$h)^2)
        
        results[[length(results) + 1]] <- data.frame(
          outlier_frac = outlier_frac,
          loss_type = loss_type,
          rep = rep,
          mean_correlation = mean_cor,
          recon_mse = recon_mse
        )
      }
    }
  }
  
  results_df <- bind_rows(results)
  
  # Summary
  results_summary <- results_df %>%
    group_by(outlier_frac, loss_type) %>%
    summarise(
      mean_cor = mean(mean_correlation),
      se_cor = sd(mean_correlation) / sqrt(n()),
      mean_mse = mean(recon_mse),
      se_mse = sd(recon_mse) / sqrt(n()),
      .groups = "drop"
    )
  
  # Figure 3: Loss Function Comparison
  p3a <- ggplot(results_summary, aes(x = factor(outlier_frac * 100), y = mean_cor,
                                      color = loss_type, group = loss_type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = mean_cor - 1.96 * se_cor, 
                      ymax = mean_cor + 1.96 * se_cor), width = 0.1) +
    labs(x = "Outlier Fraction (%)", y = "Factor Recovery (Correlation)",
         color = "Loss Function",
         title = "A) Factor Recovery Under Outliers") +
    scale_color_brewer(palette = "Set1") +
    theme_manuscript()
  
  p3b <- ggplot(results_summary, aes(x = factor(outlier_frac * 100), y = mean_mse,
                                      color = loss_type, group = loss_type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = mean_mse - 1.96 * se_mse, 
                      ymax = mean_mse + 1.96 * se_mse), width = 0.1) +
    labs(x = "Outlier Fraction (%)", y = "Reconstruction MSE (on clean data)",
         color = "Loss Function",
         title = "B) Reconstruction Error") +
    scale_color_brewer(palette = "Set1") +
    scale_y_log10() +
    theme_manuscript()
  
  p3 <- p3a + p3b + 
    plot_layout(guides = "collect") +
    plot_annotation(title = "Figure 3: Robust Loss Functions Under Outlier Contamination")
  
  ggsave(file.path(output_dir, "fig3_loss_comparison.pdf"), p3, width = 12, height = 5)
  ggsave(file.path(output_dir, "fig3_loss_comparison.png"), p3, width = 12, height = 5, dpi = 300)
  
  return(list(results = results_df, summary = results_summary, plot = p3))
}

# =============================================================================
# Experiment 4: Regularization Effects
# =============================================================================

compute_factor_sparsity <- function(model) {
  # Fraction of near-zero entries
  W_sparsity <- mean(abs(model$w) < 0.01)
  H_sparsity <- mean(abs(model$h) < 0.01)
  return(c(W = W_sparsity, H = H_sparsity))
}

compute_factor_orthogonality <- function(model) {
  # Mean absolute off-diagonal correlation
  W_cor <- cor(model$w)
  H_cor <- cor(t(model$h))
  W_ortho <- mean(abs(W_cor[upper.tri(W_cor)]))
  H_ortho <- mean(abs(H_cor[upper.tri(H_cor)]))
  return(c(W = W_ortho, H = H_ortho))
}

run_regularization_experiment <- function() {
  cat("Running Experiment 4: Regularization Effects\n")
  
  # Parameters
  m <- n <- 500
  k <- 10
  
  # Test L1 regularization
  L1_values <- c(0, 0.001, 0.01, 0.05, 0.1, 0.2)
  ortho_values <- c(0, 0.01, 0.1, 0.5, 1.0, 2.0)
  n_reps <- 5
  
  set.seed(42)
  A <- abs(matrix(rnorm(m * n), m, n))
  
  # L1 experiment
  l1_results <- list()
  for (L1 in L1_values) {
    for (rep in 1:n_reps) {
      model <- nmf(A, k, L1 = L1, seed = rep, maxit = 50, tol = 1e-4, verbose = FALSE)
      sparsity <- compute_factor_sparsity(model)
      mse <- mean((A - model$w %*% diag(model$d) %*% model$h)^2)
      
      l1_results[[length(l1_results) + 1]] <- data.frame(
        L1 = L1, rep = rep,
        W_sparsity = sparsity["W"], H_sparsity = sparsity["H"],
        recon_mse = mse
      )
    }
    cat(sprintf("  L1 = %.3f done\n", L1))
  }
  l1_df <- bind_rows(l1_results)
  
  # Orthogonality experiment
  ortho_results <- list()
  for (ortho in ortho_values) {
    for (rep in 1:n_reps) {
      model <- nmf(A, k, ortho = c(ortho, ortho), seed = rep, maxit = 50, 
                   tol = 1e-4, verbose = FALSE)
      orthogonality <- compute_factor_orthogonality(model)
      mse <- mean((A - model$w %*% diag(model$d) %*% model$h)^2)
      
      ortho_results[[length(ortho_results) + 1]] <- data.frame(
        ortho = ortho, rep = rep,
        W_ortho = orthogonality["W"], H_ortho = orthogonality["H"],
        recon_mse = mse
      )
    }
    cat(sprintf("  ortho = %.2f done\n", ortho))
  }
  ortho_df <- bind_rows(ortho_results)
  
  # Summaries
  l1_summary <- l1_df %>%
    group_by(L1) %>%
    summarise(
      mean_sparsity = mean((W_sparsity + H_sparsity) / 2),
      se_sparsity = sd((W_sparsity + H_sparsity) / 2) / sqrt(n()),
      mean_mse = mean(recon_mse),
      se_mse = sd(recon_mse) / sqrt(n()),
      .groups = "drop"
    )
  
  ortho_summary <- ortho_df %>%
    group_by(ortho) %>%
    summarise(
      mean_ortho = mean((W_ortho + H_ortho) / 2),
      se_ortho = sd((W_ortho + H_ortho) / 2) / sqrt(n()),
      mean_mse = mean(recon_mse),
      se_mse = sd(recon_mse) / sqrt(n()),
      .groups = "drop"
    )
  
  # Figure 4: Regularization Trade-offs
  p4a <- ggplot(l1_summary, aes(x = mean_mse, y = mean_sparsity * 100)) +
    geom_path(color = "steelblue", linewidth = 1) +
    geom_point(aes(size = L1), color = "steelblue") +
    geom_text(aes(label = sprintf("L1=%.3f", L1)), vjust = -1, size = 3) +
    labs(x = "Reconstruction MSE", y = "Factor Sparsity (%)",
         title = "A) L1 Sparsity-Error Trade-off") +
    scale_size_continuous(range = c(2, 6), guide = "none") +
    theme_manuscript()
  
  p4b <- ggplot(ortho_summary, aes(x = mean_mse, y = 1 - mean_ortho)) +
    geom_path(color = "coral", linewidth = 1) +
    geom_point(aes(size = ortho), color = "coral") +
    geom_text(aes(label = sprintf("λ=%.2f", ortho)), vjust = -1, size = 3) +
    labs(x = "Reconstruction MSE", y = "Factor Orthogonality (1 - mean |cor|)",
         title = "B) Orthogonality-Error Trade-off") +
    scale_size_continuous(range = c(2, 6), guide = "none") +
    theme_manuscript()
  
  p4 <- p4a + p4b +
    plot_annotation(title = "Figure 4: Regularization Trade-offs")
  
  ggsave(file.path(output_dir, "fig4_regularization.pdf"), p4, width = 12, height = 5)
  ggsave(file.path(output_dir, "fig4_regularization.png"), p4, width = 12, height = 5, dpi = 300)
  
  return(list(l1 = l1_df, ortho = ortho_df, 
              l1_summary = l1_summary, ortho_summary = ortho_summary, 
              plot = p4))
}

# =============================================================================
# Experiment 5: Convergence Visualization
# =============================================================================

run_convergence_experiment <- function() {
  cat("Running Experiment 5: Convergence Visualization\n")
  
  # Parameters
  m <- n <- 500
  k <- 10
  maxit <- 100
  
  set.seed(42)
  A <- abs(matrix(rnorm(m * n), m, n))
  
  # Run with loss tracking
  model_mse <- nmf(A, k, loss = "mse", maxit = maxit, track_loss = TRUE, 
                   cv_fraction = 0.1, seed = 1, verbose = FALSE)
  model_mae <- nmf(A, k, loss = "mae", maxit = maxit, track_loss = TRUE, 
                   cv_fraction = 0.1, seed = 1, verbose = FALSE)
  model_huber <- nmf(A, k, loss = "huber", maxit = maxit, track_loss = TRUE, 
                     cv_fraction = 0.1, seed = 1, verbose = FALSE)
  
  # Extract loss histories
  extract_history <- function(model, name) {
    data.frame(
      iteration = 1:length(model@misc$loss_history$train_loss),
      train_loss = model@misc$loss_history$train_loss,
      test_loss = model@misc$loss_history$test_loss,
      loss_type = name
    )
  }
  
  histories <- bind_rows(
    extract_history(model_mse, "MSE"),
    extract_history(model_mae, "MAE"),
    extract_history(model_huber, "Huber")
  )
  
  # Figure 5: Convergence
  histories_long <- histories %>%
    pivot_longer(cols = c(train_loss, test_loss), 
                 names_to = "set", values_to = "loss") %>%
    mutate(set = ifelse(set == "train_loss", "Train", "Test"))
  
  p5 <- ggplot(histories_long, aes(x = iteration, y = loss, 
                                    color = loss_type, linetype = set)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~ set, scales = "free_y") +
    labs(x = "Iteration", y = "Loss",
         color = "Loss Function", linetype = "Dataset",
         title = "Figure 5: Convergence Behavior by Loss Function",
         subtitle = "Train and test loss over NMF iterations") +
    scale_color_brewer(palette = "Set1") +
    scale_y_log10() +
    theme_manuscript()
  
  ggsave(file.path(output_dir, "fig5_convergence.pdf"), p5, width = 10, height = 5)
  ggsave(file.path(output_dir, "fig5_convergence.png"), p5, width = 10, height = 5, dpi = 300)
  
  return(list(histories = histories, plot = p5))
}

# =============================================================================
# Table Generation
# =============================================================================

generate_tables <- function() {
  cat("Generating manuscript tables\n")
  
  # Table 1: CV Parameters and Recommendations
  table1 <- data.frame(
    Parameter = c("holdout_fraction", "patience", "cv_seed"),
    Default = c("0.1", "5", "random"),
    Range = c("0.05-0.2", "3-10", "integer"),
    Notes = c("Higher for small matrices (<1000 entries)",
              "Lower for faster stopping, higher for stability",
              "Fix for reproducibility across runs")
  )
  
  # Table 2: Loss Function Properties
  table2 <- data.frame(
    Loss = c("MSE", "MAE", "Huber", "KL"),
    `Outlier Robust` = c("No", "Yes", "Yes", "No"),
    `Count Data` = c("No", "No", "No", "Yes"),
    `IRLS Required` = c("No", "Yes", "Yes", "Yes"),
    `Typical Overhead` = c("0%", "20-50%", "10-30%", "30-60%"),
    check.names = FALSE
  )
  
  # Table 3: Regularization Guide
  table3 <- data.frame(
    Regularization = c("L1 (LASSO)", "L2 (Ridge)", "L21 (Group)", "Orthogonality", "Graph Laplacian"),
    Effect = c("Element-wise sparsity", "Numerical stability", "Column/row sparsity", 
               "Non-overlapping factors", "Smooth factors over graph"),
    `Typical Range` = c("[0.001, 0.1]", "[0.001, 0.1]", "[0.01, 1.0]", "[0.1, 1.0]", "[0.1, 10]"),
    `Use Case` = c("Interpretable sparse factors", "Ill-conditioned data",
                   "Outlier samples, feature selection", "Distinct factors",
                   "Prior network information"),
    check.names = FALSE
  )
  
  # Save tables as CSV
  write.csv(table1, file.path(output_dir, "table1_cv_parameters.csv"), row.names = FALSE)
  write.csv(table2, file.path(output_dir, "table2_loss_functions.csv"), row.names = FALSE)
  write.csv(table3, file.path(output_dir, "table3_regularization.csv"), row.names = FALSE)
  
  cat("Tables saved to", output_dir, "\n")
  
  return(list(table1 = table1, table2 = table2, table3 = table3))
}

# =============================================================================
# Main Execution
# =============================================================================

run_all_experiments <- function() {
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
  cat("RcppML Manuscript: Generating Figures and Tables\n")
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")
  
  results <- list()
  
  # Tables (fast)
  results$tables <- generate_tables()
  
  # Experiments (may take some time)
  results$exp1 <- tryCatch(run_cv_overhead_experiment(), 
                           error = function(e) { cat("Exp1 error:", e$message, "\n"); NULL })
  
  results$exp2 <- tryCatch(run_rank_selection_experiment(), 
                           error = function(e) { cat("Exp2 error:", e$message, "\n"); NULL })
  
  results$exp3 <- tryCatch(run_loss_comparison_experiment(), 
                           error = function(e) { cat("Exp3 error:", e$message, "\n"); NULL })
  
  results$exp4 <- tryCatch(run_regularization_experiment(), 
                           error = function(e) { cat("Exp4 error:", e$message, "\n"); NULL })
  
  results$exp5 <- tryCatch(run_convergence_experiment(), 
                           error = function(e) { cat("Exp5 error:", e$message, "\n"); NULL })
  
  cat("\n")
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
  cat("All experiments complete. Figures saved to:", output_dir, "\n")
  cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")
  
  return(results)
}

# Run if executed directly
if (interactive()) {
  cat("Source this file and run: results <- run_all_experiments()\n")
} else {
  results <- run_all_experiments()
}
