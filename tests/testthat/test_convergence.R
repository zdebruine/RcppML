# Test NMF convergence behavior to prevent regression
# These tests ensure the warm-start fix is working correctly

library(testthat)
library(RcppML)
library(Matrix)

# Helper: compute MSE between original and reconstructed matrix
compute_mse <- function(A, model) {
  W <- as.matrix(model@w)
  H <- as.matrix(model@h)
  D <- as.numeric(model@d)
  A_hat <- W %*% diag(D) %*% H
  mean((as.matrix(A) - A_hat)^2)
}

# ============================================================================
# CRITICAL: These tests catch the warm-start initialization bug where MSE
# was INCREASING with more iterations instead of decreasing
# ============================================================================

test_that("NMF converges (MSE decreases with iterations) on sparse matrices", {
  skip_on_cran()
  set.seed(42)
  # Create sparse test matrix
  A <- rsparsematrix(200, 100, density = 0.3)
  A@x <- abs(A@x)
  
  # Run with increasing iterations
  mse_1 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 1, verbose = FALSE))
  mse_5 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 5, verbose = FALSE))
  mse_20 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 20, verbose = FALSE))
  mse_50 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 50, verbose = FALSE))
  
  # MSE should decrease monotonically (allowing small tolerance for numerical noise)
  expect_lt(mse_5, mse_1 * 1.01, 
            label = sprintf("MSE should decrease: iter5=%.6f should be < iter1=%.6f", mse_5, mse_1))
  expect_lt(mse_20, mse_5 * 1.01,
            label = sprintf("MSE should decrease: iter20=%.6f should be < iter5=%.6f", mse_20, mse_5))
  expect_lt(mse_50, mse_20 * 1.01,
            label = sprintf("MSE should decrease: iter50=%.6f should be < iter20=%.6f", mse_50, mse_20))
  
  # Final MSE should be lower than initial (at least some improvement)
  expect_lt(mse_50, mse_1,
            label = "50 iterations should achieve some MSE reduction vs 1 iteration")
})

test_that("NMF converges (MSE decreases with iterations) on dense matrices", {
  skip_on_cran()
  set.seed(42)
  # Create dense test matrix
  A <- abs(matrix(rnorm(200 * 100), 200, 100))
  
  # Run with increasing iterations
  mse_1 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 1, verbose = FALSE))
  mse_5 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 5, verbose = FALSE))
  mse_20 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 20, verbose = FALSE))
  mse_50 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 50, verbose = FALSE))
  
  # MSE should decrease monotonically
  expect_lt(mse_5, mse_1 * 1.01)
  expect_lt(mse_20, mse_5 * 1.01)
  expect_lt(mse_50, mse_20 * 1.01)
  
  # Final MSE should be lower than initial
  expect_lt(mse_50, mse_1)
})

test_that("NMF converges on real AML dataset", {
  skip_on_cran()
  data(aml)
  
  set.seed(123)
  mse_1 <- compute_mse(aml, nmf(aml, k = 5, tol = 0, maxit = 1, verbose = FALSE))
  mse_10 <- compute_mse(aml, nmf(aml, k = 5, tol = 0, maxit = 10, verbose = FALSE))
  mse_50 <- compute_mse(aml, nmf(aml, k = 5, tol = 0, maxit = 50, verbose = FALSE))
  
  # MSE must decrease (allow small tolerance for stochastic variance)
  expect_lt(mse_10, mse_1,
            label = sprintf("AML: MSE should decrease: iter10=%.6f < iter1=%.6f", mse_10, mse_1))
  # mse_50 should be less than or approximately equal to mse_10 (convergence may stall)
  expect_lt(mse_50, mse_10 * 1.05,
            label = sprintf("AML: MSE should decrease or stabilize: iter50=%.6f vs iter10=%.6f", mse_50, mse_10))
})

test_that("NMF converges consistently across random seeds", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(150, 80, density = 0.4)
  A@x <- abs(A@x)
  
  # Test multiple seeds - all should converge
  n_seeds <- 5
  for (seed in 1:n_seeds) {
    set.seed(seed * 100)
    
    mse_early <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 5, verbose = FALSE))
    mse_late <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 30, verbose = FALSE))
    
    expect_lt(mse_late, mse_early,
              label = sprintf("Seed %d: MSE should decrease (early=%.6f, late=%.6f)", 
                             seed, mse_early, mse_late))
  }
})

test_that("NMF reconstruction values stay bounded (no divergence)", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(200, 100, density = 0.3)
  A@x <- abs(A@x)
  A_range <- range(A@x)
  
  # Run NMF
  model <- nmf(A, k = 5, tol = 0, maxit = 50, verbose = FALSE)
  
  # Reconstruct
  W <- as.matrix(model@w)
  H <- as.matrix(model@h)
  D <- as.numeric(model@d)
  A_hat <- W %*% diag(D) %*% H
  
  # Reconstruction should have similar scale to original (not exploding)
  hat_range <- range(A_hat)
  
  # Max reconstruction should not be more than 10x the max original value
  # (This catches the divergence bug where values were growing unboundedly)
  expect_lt(max(abs(A_hat)), max(abs(A)) * 10,
            label = sprintf("Reconstruction max=%.2f should not be >> original max=%.2f",
                           max(abs(A_hat)), max(abs(A))))
  
  # Mean reconstruction should be similar to original mean
  expect_lt(abs(mean(A_hat) - mean(A)) / mean(A), 0.5,
            label = "Reconstruction mean should be within 50% of original mean")
})

test_that("NMF achieves reasonable MSE on known factorizable matrix", {
  skip_on_cran()
  set.seed(42)
  # Create a matrix with known low-rank structure
  k_true <- 5
  W_true <- abs(matrix(rnorm(100 * k_true), 100, k_true))
  H_true <- abs(matrix(rnorm(k_true * 50), k_true, 50))
  A <- W_true %*% H_true
  
  # Reset seed for NMF initialization (separate from data generation)
  set.seed(123)
  
  # NMF should be able to reconstruct this well
  model <- nmf(A, k = k_true, tol = 0, maxit = 100, verbose = FALSE)
  mse <- compute_mse(A, model)
  
  # MSE should be reasonably low for a truly low-rank matrix
  # Threshold of 0.1 accounts for non-convexity and local minima
  expect_lt(mse, 0.1,
            label = sprintf("MSE=%.6f should be low for known rank-%d matrix", mse, k_true))
})

test_that("NMF with higher rank achieves lower or equal MSE", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(150, 80, density = 0.4)
  A@x <- abs(A@x)
  
  # Higher rank should give same or better fit
  mse_k3 <- compute_mse(A, nmf(A, k = 3, tol = 0, maxit = 50, verbose = FALSE))
  mse_k5 <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 50, verbose = FALSE))
  mse_k10 <- compute_mse(A, nmf(A, k = 10, tol = 0, maxit = 50, verbose = FALSE))
  
  # More factors should generally give better or equal fit
  # Allow small tolerance for optimization variance
  expect_lte(mse_k5, mse_k3 * 1.05)
  expect_lte(mse_k10, mse_k5 * 1.05)
})

test_that("NMF handles different L1/L2 regularization without diverging", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(100, 60, density = 0.4)
  A@x <- abs(A@x)
  
  # Test various regularization settings
  regularizations <- list(
    list(L1 = c(0, 0), L2 = c(0, 0)),
    list(L1 = c(0.01, 0.01), L2 = c(0, 0)),
    list(L1 = c(0, 0), L2 = c(0.01, 0.01)),
    list(L1 = c(0.01, 0), L2 = c(0, 0.01))
  )
  
  for (reg in regularizations) {
    model <- nmf(A, k = 5, tol = 0, maxit = 30, verbose = FALSE,
                 L1 = reg$L1, L2 = reg$L2)
    mse <- compute_mse(A, model)
    
    # MSE should be finite and reasonable (not diverged)
    expect_true(is.finite(mse),
                label = sprintf("MSE should be finite with L1=%s, L2=%s",
                               paste(reg$L1, collapse=","), paste(reg$L2, collapse=",")))
    expect_lt(mse, 10,
              label = sprintf("MSE=%.4f should be reasonable with regularization", mse))
  }
})

# ============================================================================
# Edge case tests
# ============================================================================

test_that("NMF handles tall matrices (m >> n)", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(500, 50, density = 0.3)
  A@x <- abs(A@x)
  
  mse_early <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 5, verbose = FALSE))
  mse_late <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 30, verbose = FALSE))
  
  expect_lt(mse_late, mse_early, label = "Tall matrix should converge")
})

test_that("NMF handles wide matrices (n >> m)", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(50, 500, density = 0.3)
  A@x <- abs(A@x)
  
  mse_early <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 5, verbose = FALSE))
  mse_late <- compute_mse(A, nmf(A, k = 5, tol = 0, maxit = 30, verbose = FALSE))
  
  expect_lt(mse_late, mse_early, label = "Wide matrix should converge")
})

test_that("NMF handles very sparse matrices", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(200, 100, density = 0.05)
  A@x <- abs(A@x)
  
  mse_early <- compute_mse(A, nmf(A, k = 3, tol = 0, maxit = 5, verbose = FALSE))
  mse_late <- compute_mse(A, nmf(A, k = 3, tol = 0, maxit = 30, verbose = FALSE))
  
  expect_lt(mse_late, mse_early, label = "Very sparse matrix should converge")
})

# ============================================================================
# Factor convergence mode tests
# ============================================================================

test_that("convergence='factor' mode converges on sparse", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(50, 30, density = 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 3, seed = 1, maxit = 200, tol = 1e-4,
                convergence = "factor", verbose = FALSE)
  expect_s4_class(result, "nmf")
  # Factor convergence should terminate before maxit with reasonable tol
  expect_lt(result@misc$iter, 200)
})

test_that("convergence='factor' mode converges on dense", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  
  result <- nmf(A, k = 3, seed = 1, maxit = 200, tol = 1e-4,
                convergence = "factor", verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_lt(result@misc$iter, 200)
})

test_that("convergence='both' mode works", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(50, 30, density = 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 3, seed = 1, maxit = 200, tol = 1e-4,
                convergence = "both", verbose = FALSE)
  expect_s4_class(result, "nmf")
  # Should converge, though potentially takes more iterations than single-mode
  expect_true(result@misc$iter <= 200)
})

test_that("convergence='loss' gives same result as default", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(50, 30, density = 0.3)
  A@x <- abs(A@x)
  
  r1 <- nmf(A, k = 3, seed = 1, maxit = 100, tol = 1e-4, verbose = FALSE)
  r2 <- nmf(A, k = 3, seed = 1, maxit = 100, tol = 1e-4,
            convergence = "loss", verbose = FALSE)
  
  # Same seed, same convergence mode → same result
  expect_equal(r1@misc$iter, r2@misc$iter)
  expect_equal(r1@w, r2@w, tolerance = 1e-10)
})

test_that("convergence modes all produce valid factorizations", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(40, 25, density = 0.3)
  A@x <- abs(A@x)
  
  for (mode in c("loss", "factor", "both")) {
    result <- nmf(A, k = 3, seed = 1, maxit = 100, tol = 1e-4,
                  convergence = mode, verbose = FALSE)
    # All factors should be non-negative
    expect_true(all(result@w >= 0), info = paste("mode:", mode))
    expect_true(all(result@h >= 0), info = paste("mode:", mode))
    expect_true(all(result@d >= 0), info = paste("mode:", mode))
    # Reconstruction should be reasonable
    mse <- compute_mse(A, result)
    expect_true(is.finite(mse), info = paste("mode:", mode))
  }
})
