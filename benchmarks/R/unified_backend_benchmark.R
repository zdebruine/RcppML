#!/usr/bin/env Rscript
#' Unified Backend Benchmark
#'
#' Validates the unified backend architecture by benchmarking:
#' 1. Standard NMF (sparse & dense) - correctness and performance
#' 2. Cross-validation NMF - test loss tracking
#' 3. NNLS solver with different bound modes
#' 4. Bipartition with thread control
#' 5. Resource parameter handling
#'
#' Run: Rscript benchmarks/R/unified_backend_benchmark.R

library(RcppML)
library(Matrix)

cat("=== RcppML Unified Backend Benchmark ===\n\n")

# ============================================================================
# Data Setup
# ============================================================================

set.seed(42)
m <- 200; n <- 150; k <- 5

# Generate ground truth NMF factors
W_true <- matrix(abs(rnorm(m * k)), m, k)
H_true <- matrix(abs(rnorm(k * n)), k, n)
A_dense <- W_true %*% H_true + abs(matrix(rnorm(m * n, sd = 0.1), m, n))

# Create sparse version (threshold small values)
A_sparse <- as(A_dense, "dgCMatrix")
A_sparse@x[A_sparse@x < quantile(A_sparse@x, 0.3)] <- 0
A_sparse <- drop0(A_sparse)

cat(sprintf("Dense:  %d x %d\n", nrow(A_dense), ncol(A_dense)))
cat(sprintf("Sparse: %d x %d, nnz=%d (%.1f%% density)\n",
            nrow(A_sparse), ncol(A_sparse), length(A_sparse@x),
            100 * length(A_sparse@x) / (m * n)))

# ============================================================================
# 1. Standard NMF - Sparse vs Dense
# ============================================================================

cat("\n--- 1. Standard NMF ---\n")

ranks <- c(2, 5, 10)
for (r in ranks) {
  t_sparse <- system.time(res_sp <- nmf(A_sparse, r, tol = 1e-6, maxit = 100, seed = 123, verbose = FALSE))
  t_dense  <- system.time(res_dn <- nmf(A_dense,  r, tol = 1e-6, maxit = 100, seed = 123, verbose = FALSE))
  
  cat(sprintf("  k=%2d  sparse: %.3fs (%d iter, loss=%.4f)  dense: %.3fs (%d iter, loss=%.4f)\n",
              r, t_sparse[3], res_sp@iterations, tail(res_sp@loss, 1),
              t_dense[3], res_dn@iterations, tail(res_dn@loss, 1)))
}

# ============================================================================
# 2. Cross-Validation NMF
# ============================================================================

cat("\n--- 2. Cross-Validation NMF ---\n")

cv_ranks <- c(2, 5, 10)
for (r in cv_ranks) {
  t_cv <- system.time(
    res_cv <- nmf(A_sparse, r, tol = 1e-6, maxit = 50, seed = 42,
                  cv = TRUE, holdout_fraction = 0.15, verbose = FALSE)
  )
  
  cat(sprintf("  k=%2d  CV: %.3fs (%d iter, train=%.4f, test=%.4f)\n",
              r, t_cv[3], res_cv@iterations,
              tail(res_cv@loss, 1), res_cv@test_loss))
}

# Dense CV
cat("  Dense CV: ")
t_dcv <- system.time(
  res_dcv <- nmf(A_dense, 5, tol = 1e-6, maxit = 50, seed = 42,
                 cv = TRUE, holdout_fraction = 0.15, verbose = FALSE)
)
cat(sprintf("%.3fs (%d iter, train=%.4f, test=%.4f)\n",
            t_dcv[3], res_dcv@iterations,
            tail(res_dcv@loss, 1), res_dcv@test_loss))

# ============================================================================
# 3. NNLS Solver Bound Modes
# ============================================================================

cat("\n--- 3. NNLS Solver ---\n")

# Standard NNLS (nonneg)
W_sub <- matrix(abs(rnorm(m * 3)), m, 3)
b <- A_dense[, 1]
x_nn <- solve(W_sub, b, nonneg = TRUE)
cat(sprintf("  Nonneg NNLS:  min=%.4f, max=%.4f (all >= 0: %s)\n",
            min(x_nn), max(x_nn), all(x_nn >= -1e-10)))

# Unconstrained solve
x_free <- solve(W_sub, b, nonneg = FALSE)
cat(sprintf("  Free solve:   min=%.4f, max=%.4f\n", min(x_free), max(x_free)))

# Upper bounded NNLS
x_ub <- solve(W_sub, b, nonneg = TRUE, upper_bound = 0.5)
cat(sprintf("  Bounded NNLS: min=%.4f, max=%.4f (all <= 0.5: %s)\n",
            min(x_ub), max(x_ub), all(x_ub <= 0.5 + 1e-10)))

# ============================================================================
# 4. Bipartition with Threads
# ============================================================================

cat("\n--- 4. Bipartition ---\n")

# Sparse bipartition
t_bp <- system.time(bp <- bipartition(A_sparse, threads = 1, verbose = FALSE))
cat(sprintf("  Sparse bipartition (1 thread): %.3fs, sizes: %d + %d\n",
            t_bp[3], bp$size1, bp$size2))

# Dense bipartition
t_bp2 <- system.time(bp2 <- bipartition(A_dense, threads = 1, verbose = FALSE))
cat(sprintf("  Dense bipartition (1 thread):  %.3fs, sizes: %d + %d\n",
            t_bp2[3], bp2$size1, bp2$size2))

# ============================================================================
# 5. Resource Parameter
# ============================================================================

cat("\n--- 5. Resource Parameter ---\n")

# CPU explicit
t_cpu <- system.time(
  res_cpu <- nmf(A_sparse, 5, tol = 1e-6, maxit = 50, seed = 99,
                 resource = "cpu", verbose = FALSE)
)
cat(sprintf("  resource='cpu':  %.3fs (%d iter)\n", t_cpu[3], res_cpu@iterations))

# Auto (default)
t_auto <- system.time(
  res_auto <- nmf(A_sparse, 5, tol = 1e-6, maxit = 50, seed = 99,
                  resource = "auto", verbose = FALSE)
)
cat(sprintf("  resource='auto': %.3fs (%d iter)\n", t_auto[3], res_auto@iterations))

# Results should be identical (same seed, both CPU)
w_diff <- max(abs(res_cpu@w - res_auto@w))
cat(sprintf("  CPU vs auto W diff: %.2e (should be ~0)\n", w_diff))

# ============================================================================
# 6. Regularization
# ============================================================================

cat("\n--- 6. Regularization ---\n")

# L1 regularization
res_l1 <- nmf(A_sparse, 5, tol = 1e-6, maxit = 100, seed = 42, L1 = c(0.1, 0.1), verbose = FALSE)
cat(sprintf("  L1 (0.1): sparsity W=%.2f%%, H=%.2f%%\n",
            100 * mean(res_l1@w == 0), 100 * mean(res_l1@h == 0)))

# L2 regularization
res_l2 <- nmf(A_sparse, 5, tol = 1e-6, maxit = 100, seed = 42, L2 = c(0.5, 0.5), verbose = FALSE)
cat(sprintf("  L2 (0.5): W norm=%.2f, H norm=%.2f\n",
            norm(res_l1@w, "F"), norm(res_l1@h, "F")))

# No regularization baseline
res_none <- nmf(A_sparse, 5, tol = 1e-6, maxit = 100, seed = 42, verbose = FALSE)
cat(sprintf("  None:     sparsity W=%.2f%%, H=%.2f%%\n",
            100 * mean(res_none@w == 0), 100 * mean(res_none@h == 0)))

# ============================================================================
# 7. Scaling Benchmark
# ============================================================================

cat("\n--- 7. Scaling (time vs rank) ---\n")

scaling_ranks <- c(2, 5, 10, 20)
for (r in scaling_ranks) {
  t <- system.time(nmf(A_sparse, r, tol = 1e-6, maxit = 50, seed = 1, verbose = FALSE))
  cat(sprintf("  k=%2d: %.3fs\n", r, t[3]))
}

cat("\n=== Benchmark Complete ===\n")
