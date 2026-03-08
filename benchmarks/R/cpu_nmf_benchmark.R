#!/usr/bin/env Rscript
#' Comprehensive NMF Benchmark: singlet vs RcppML CPU vs GPU
#'
#' This script benchmarks:
#' 1. singlet::run_nmf (CPU reference)
#' 2. RcppML::nmf (CPU optimized)
#' 3. GPU (via external CUDA benchmark, results merged later)
#'
#' Key settings for fair comparison:
#' - Fixed number of iterations (no early stopping)
#' - Same initialization across methods where possible
#' - MSE loss computed identically for all methods

library(singlet)
library(RcppML)
library(Matrix)

# ============================================================================
# Configuration
# ============================================================================

MAXIT <- 10          # Fixed iterations for all methods
TOL <- 1e-20         # Very small tolerance to disable early stopping
RANKS <- c(2, 4, 8, 16, 32, 64)
L1 <- 0              # No L1 regularization for fair comparison
L2 <- 0              # No L2 regularization for fair comparison
THREADS <- 48        # Use all available threads on bigmem

# Load data
cat("Loading pbmc3k data...\n")
load('/mnt/home/debruinz/RcppML-2/data/pbmc3k.rda')
A <- pbmc3k
m <- nrow(A)
n <- ncol(A)
nnz <- length(A@x)

cat(sprintf("Matrix: %d x %d, nnz = %d, density = %.2f%%\n", 
            m, n, nnz, 100 * nnz / (m * n)))

# ============================================================================
# MSE Loss Computation (Frobenius norm on non-zeros)
# ============================================================================

#' Compute MSE loss: ||A - W*diag(d)*H||_F^2 / nnz
#' Only counts non-zero entries in A (sparse loss)
compute_sparse_mse <- function(A, W, d, H) {
    # Reconstruct and compute residuals at non-zero positions only
    # This is memory efficient for sparse matrices
    total_sq_error <- 0
    
    # Get sparse structure
    i_idx <- A@i + 1  # R is 1-indexed
    p <- A@p + 1
    x <- A@x
    
    # For each column
    for (j in seq_len(ncol(A))) {
        col_start <- p[j]
        col_end <- p[j + 1] - 1
        
        if (col_start <= col_end) {
            rows <- i_idx[col_start:col_end]
            vals <- x[col_start:col_end]
            
            # Predicted values: W[rows,] %*% diag(d) %*% H[,j]
            pred <- as.vector(W[rows, , drop = FALSE] %*% (d * H[, j]))
            
            # Sum squared errors
            total_sq_error <- total_sq_error + sum((vals - pred)^2)
        }
    }
    
    return(total_sq_error / nnz)
}

#' Faster vectorized MSE for small matrices
compute_mse_vectorized <- function(A, W, d, H) {
    # Full reconstruction (dense) - only use for small problems
    A_hat <- W %*% diag(d) %*% H
    
    # Sparse comparison
    total_sq_error <- 0
    for (j in seq_len(ncol(A))) {
        idx <- (A@p[j] + 1):(A@p[j + 1])
        if (length(idx) > 0) {
            rows <- A@i[idx] + 1
            diff <- A@x[idx] - A_hat[rows, j]
            total_sq_error <- total_sq_error + sum(diff^2)
        }
    }
    return(total_sq_error / nnz)
}

# ============================================================================
# Benchmark Functions
# ============================================================================

#' Run singlet NMF benchmark
run_singlet_benchmark <- function(A, k, maxit, threads) {
    options(RcppML.threads = threads)
    
    # Warmup
    invisible(run_nmf(A, rank = k, maxit = 2, tol = TOL, verbose = FALSE, 
                      L1 = L1, L2 = L2, threads = threads))
    gc()
    
    # Timed run
    start_time <- Sys.time()
    result <- run_nmf(A, rank = k, maxit = maxit, tol = TOL, verbose = FALSE,
                      L1 = L1, L2 = L2, threads = threads)
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Extract factors (singlet returns list with w, d, h)
    W <- result$w
    d <- result$d
    H <- result$h
    
    # Compute loss
    mse <- compute_sparse_mse(A, W, d, H)
    
    list(
        method = "singlet",
        k = k,
        elapsed_sec = elapsed,
        time_per_iter_ms = elapsed / maxit * 1000,
        mse = mse,
        iterations = maxit
    )
}

#' Run RcppML NMF benchmark
run_rcppml_benchmark <- function(A, k, maxit, threads) {
    options(RcppML.threads = threads)
    
    # Warmup
    invisible(nmf(A, k = k, maxit = 2, tol = TOL))
    gc()
    
    # Timed run
    start_time <- Sys.time()
    result <- nmf(A, k = k, maxit = maxit, tol = TOL, L1 = c(L1, L1), L2 = c(L2, L2))
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Extract factors
    W <- result@w
    d <- result@d
    H <- result@h
    
    # Compute loss
    mse <- compute_sparse_mse(A, W, d, H)
    
    # RcppML reports iteration count in misc
    actual_iters <- result@misc$iter
    
    list(
        method = "RcppML",
        k = k,
        elapsed_sec = elapsed,
        time_per_iter_ms = elapsed / actual_iters * 1000,
        mse = mse,
        iterations = actual_iters
    )
}

# ============================================================================
# Main Benchmark Loop
# ============================================================================

cat("\n====================================================\n")
cat("NMF Benchmark: singlet vs RcppML CPU\n")
cat("====================================================\n")
cat(sprintf("Iterations: %d, Tolerance: %.1e, Threads: %d\n", MAXIT, TOL, THREADS))
cat("\n")

results <- data.frame()

for (k in RANKS) {
    cat(sprintf("--- Rank k=%d ---\n", k))
    
    # Run singlet
    cat("  Running singlet...")
    singlet_result <- run_singlet_benchmark(A, k, MAXIT, THREADS)
    cat(sprintf(" %.3f sec (%.2f ms/iter), MSE=%.6f\n", 
                singlet_result$elapsed_sec, 
                singlet_result$time_per_iter_ms,
                singlet_result$mse))
    
    # Run RcppML
    cat("  Running RcppML... ")
    rcppml_result <- run_rcppml_benchmark(A, k, MAXIT, THREADS)
    cat(sprintf(" %.3f sec (%.2f ms/iter), MSE=%.6f\n", 
                rcppml_result$elapsed_sec, 
                rcppml_result$time_per_iter_ms,
                rcppml_result$mse))
    
    # Compute speedup
    speedup <- singlet_result$elapsed_sec / rcppml_result$elapsed_sec
    mse_ratio <- rcppml_result$mse / singlet_result$mse
    cat(sprintf("  Speedup: %.2fx, MSE ratio: %.4f\n\n", speedup, mse_ratio))
    
    # Collect results
    results <- rbind(results, data.frame(
        rank = k,
        method = "singlet",
        elapsed_sec = singlet_result$elapsed_sec,
        time_per_iter_ms = singlet_result$time_per_iter_ms,
        mse = singlet_result$mse,
        iterations = singlet_result$iterations
    ))
    
    results <- rbind(results, data.frame(
        rank = k,
        method = "RcppML",
        elapsed_sec = rcppml_result$elapsed_sec,
        time_per_iter_ms = rcppml_result$time_per_iter_ms,
        mse = rcppml_result$mse,
        iterations = rcppml_result$iterations
    ))
}

# ============================================================================
# Save Results
# ============================================================================

output_file <- "/mnt/home/debruinz/RcppML-2/benchmarks/results/cpu_nmf_benchmark.csv"
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
write.csv(results, output_file, row.names = FALSE)
cat("\nResults saved to:", output_file, "\n")

# ============================================================================
# Print Summary Table
# ============================================================================

cat("\n====================================================\n")
cat("Summary Table\n")
cat("====================================================\n\n")

# Reshape for comparison
library(tidyr)
summary_wide <- results %>%
    select(rank, method, time_per_iter_ms, mse) %>%
    pivot_wider(names_from = method, values_from = c(time_per_iter_ms, mse))

print(summary_wide)

# Calculate speedups
cat("\n\nRcppML vs singlet speedup by rank:\n")
for (k in RANKS) {
    singlet_time <- results$time_per_iter_ms[results$rank == k & results$method == "singlet"]
    rcppml_time <- results$time_per_iter_ms[results$rank == k & results$method == "RcppML"]
    cat(sprintf("  k=%d: %.2fx\n", k, singlet_time / rcppml_time))
}
