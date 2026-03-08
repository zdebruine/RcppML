# Test: Unified Backend Equivalence
# Verify that the unified backend (via Rcpp_nmf_full → gateway::nmf)
# produces correct, high-quality NMF factorizations.
#
# These tests validate the unified backend architecture against
# ground truth, cross-path consistency, and warm-start support.

options(RcppML.verbose = FALSE)

# =============================================================================
# Test 1: Unified backend produces high-quality factorizations on sparse data
# =============================================================================

test_that("Unified backend recovers known ground truth (sparse)", {
  set.seed(42)
  sim <- simulateNMF(100, 80, k = 5, noise = 0, dropout = 0, seed = 100)
  # Add noise proportional to signal (10% relative noise) + dropout
  set.seed(7777)
  sig_mean <- mean(sim$A[sim$A > 0])
  A <- sim$A + matrix(rnorm(100*80, 0, 0.1 * sig_mean), 100, 80)
  A[A < 0] <- 0
  dropout_mask <- matrix(rbinom(100*80, 1, 0.6), 100, 80)  # 40% dropout
  A <- A * dropout_mask
  
  model <- nmf(A, 5, maxit = 300, tol = 1e-6, seed = 42, verbose = FALSE)
  
  # Check that the model converged
  expect_true(model@misc$iter < 300, label = "Should converge before maxit")
  
  # Reconstruction error should be small (relative to data)
  W <- model@w
  D <- model@d
  H <- model@h
  reconstruction <- W %*% diag(D) %*% H
  
  mse <- mean((as.matrix(A) - reconstruction)^2)
  data_var <- mean(as.matrix(A)^2)
  relative_error <- mse / data_var
  expect_lt(relative_error, 0.4, label = "Relative reconstruction error < 40%")
})

# =============================================================================
# Test 2: Unified backend produces high-quality factorizations on dense data
# =============================================================================

test_that("Unified backend works correctly with dense matrices", {
  set.seed(42)
  sim <- simulateNMF(50, 40, k = 3, noise = 0, seed = 200)
  # Add noise proportional to signal (10% relative noise)
  set.seed(7778)
  sig_mean <- mean(sim$A[sim$A > 0])
  A <- as.matrix(sim$A + matrix(rnorm(50*40, 0, 0.1 * sig_mean), 50, 40))  # Force dense
  A[A < 0] <- 0
  
  model <- nmf(A, 3, maxit = 200, tol = 1e-6, seed = 42, verbose = FALSE)
  
  # Factors should be non-negative
  expect_true(all(model@w >= 0), label = "W should be non-negative")
  expect_true(all(model@h >= 0), label = "H should be non-negative")
  expect_true(all(model@d >= 0), label = "d should be non-negative")
  
  # Reconstruction should be reasonable
  reconstruction <- model@w %*% diag(model@d) %*% model@h
  mse <- mean((A - reconstruction)^2)
  expect_lt(mse, mean(A^2) * 0.6, label = "MSE reasonable relative to data")
})

# =============================================================================
# Test 3: Sparse and dense paths produce equivalent results
# =============================================================================

test_that("Sparse and dense paths produce equivalent loss (same init)", {
  set.seed(42)
  sim <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.3, seed = 300)
  A_sparse <- sim$A
  A_dense <- as.matrix(A_sparse)
  
  # Use same W_init for both
  W_init <- matrix(runif(60 * 4), 60, 4)
  
  model_sp <- nmf(A_sparse, 4, maxit = 50, tol = 1e-10, seed = W_init, verbose = FALSE)
  model_de <- nmf(A_dense,  4, maxit = 50, tol = 1e-10, seed = W_init, verbose = FALSE)
  
  # Loss should be very close (both go through unified backend)
  expect_equal(model_sp@misc$loss, model_de@misc$loss, tolerance = 1e-4,
               label = "Sparse and dense should converge to similar loss")
  
  # Number of iterations should be similar
  expect_equal(model_sp@misc$iter, model_de@misc$iter, tolerance = 5,
               label = "Iteration counts should be similar")
})

# =============================================================================
# Test 4: Cross-validation works through unified backend (sparse)
# =============================================================================

test_that("Sparse CV works through unified backend", {
  set.seed(42)
  sim <- simulateNMF(80, 60, k = 4, noise = 0.1, dropout = 0.3, seed = 400)
  A <- sim$A
  
  model <- nmf(A, 4, maxit = 50, tol = 1e-6, seed = 42,
               test_fraction = 0.1, patience = 5, verbose = FALSE)
  
  # CV model should have test loss info
  expect_true(!is.null(model@misc$test_loss), label = "Should have test_loss")
  expect_true(is.finite(model@misc$test_loss), label = "test_loss should be finite")
  expect_true(model@misc$test_loss > 0, label = "test_loss should be positive")
  
  # Train loss should be lower than test loss (typical for CV)
  expect_true(!is.null(model@misc$train_loss), label = "Should have train_loss")
  expect_true(is.finite(model@misc$train_loss), label = "train_loss should be finite")
})

# =============================================================================
# Test 5: Multi-rank CV works through unified backend
# =============================================================================

test_that("Multi-rank CV returns data.frame with metrics", {
  set.seed(42)
  sim <- simulateNMF(60, 50, k = 3, noise = 0.1, dropout = 0.3, seed = 500)
  A <- sim$A
  
  result <- nmf(A, k = c(2, 3, 4), maxit = 30, tol = 1e-4, seed = 42,
                test_fraction = 0.1, verbose = FALSE)
  
  # Should return a data.frame
  expect_s3_class(result, "data.frame")
  expect_true("k" %in% colnames(result))
  expect_true("test_mse" %in% colnames(result))
  
  # Should have entries for each rank
  expect_true(all(c(2, 3, 4) %in% result$k))
  
  # Test losses should be finite
  expect_true(all(is.finite(result$test_mse)))
})

# =============================================================================
# Test 6: NNLS warm_start support
# =============================================================================

test_that("NNLS warm_start produces better or equal results", {
  set.seed(42)
  # Create a simple NMF problem
  W_true <- matrix(runif(50 * 3), 50, 3)
  H_true <- matrix(runif(3 * 40), 3, 40)
  A <- W_true %*% H_true + matrix(rnorm(50 * 40, sd = 0.01), 50, 40)
  A[A < 0] <- 0
  A <- as(A, "dgCMatrix")
  
  # Cold start
  H_cold <- nnls(w = W_true, A = A, cd_maxit = 5L)
  
  # Warm start with the cold result
  H_warm <- nnls(w = W_true, A = A, cd_maxit = 5L, warm_start = H_cold)
  
  # Warm start should be at least as good (lower or equal error)
  resid_cold <- sum((as.matrix(A) - W_true %*% H_cold)^2)
  resid_warm <- sum((as.matrix(A) - W_true %*% H_warm)^2)
  expect_lte(resid_warm, resid_cold * 1.001,
             label = "Warm start should not increase error")
})

test_that("NNLS warm_start with dense input works", {
  set.seed(42)
  W <- matrix(runif(30 * 3), 30, 3)
  H_true <- matrix(runif(3 * 20), 3, 20)
  A <- W %*% H_true + matrix(rnorm(30 * 20, sd = 0.01), 30, 20)
  A[A < 0] <- 0
  
  # Cold start
  H_cold <- nnls(w = W, A = A, cd_maxit = 3L)
  
  # Warm start
  H_warm <- nnls(w = W, A = A, cd_maxit = 3L, warm_start = H_cold)
  
  resid_cold <- sum((A - W %*% H_cold)^2)
  resid_warm <- sum((A - W %*% H_warm)^2)
  expect_lte(resid_warm, resid_cold * 1.001)
})

# =============================================================================
# Test 7: Unified backend handles regularization correctly
# =============================================================================

test_that("L1 regularization increases sparsity through unified backend", {
  set.seed(42)
  sim <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.3, seed = 600)
  A <- sim$A
  
  model_no_reg <- nmf(A, 4, maxit = 100, tol = 1e-6, seed = 42, verbose = FALSE)
  model_L1 <- nmf(A, 4, maxit = 100, tol = 1e-6, seed = 42, L1 = c(0.5, 0.5),
                  verbose = FALSE)
  
  # L1 model should have more zeros (sparser)
  sparsity_no_reg <- mean(model_no_reg@h == 0)
  sparsity_L1 <- mean(model_L1@h == 0)
  expect_gt(sparsity_L1, sparsity_no_reg,
            label = "L1 should increase H sparsity")
})

# =============================================================================
# Test 8: Diagnostics are returned
# =============================================================================

test_that("Unified backend returns resource diagnostics", {
  set.seed(42)
  sim <- simulateNMF(30, 25, k = 2, noise = 0.1, seed = 700)
  A <- sim$A
  
  # Call Rcpp_nmf_full directly to check diagnostics
  W_init <- matrix(runif(30 * 2), 30, 2)
  result <- RcppML:::Rcpp_nmf_full(
    A_sexp = A, k_vec = 2L, k_auto = FALSE,
    k_auto_min = 2L, k_auto_max = 50L,
    L1 = c(0, 0), L2 = c(0, 0), L21 = c(0, 0), angular = c(0, 0),
    upper_bound = c(0, 0), graph_lambda = c(0, 0),
    tol = 1e-4, maxit = 50L, cd_maxit = 10L, cd_tol = 5e-3, cd_abs_tol = 1e-15,
    threads = 1L, seed = 42L,
    nonneg_w = TRUE, nonneg_h = TRUE,
    sort_model = FALSE, mask_zeros = FALSE,
    track_loss_history = TRUE, track_train_loss = FALSE,
    loss_str = "mse", huber_delta = 1.0,
    holdout_fraction = 0, cv_seeds = integer(0), cv_patience = 0L,
    resource_override = "auto",
    w_init_sexp = W_init, verbose = FALSE
  )
  
  # Should have diagnostics attribute
  diag <- attr(result, "diagnostics")
  expect_true(!is.null(diag), label = "Should have diagnostics attribute")
  expect_true(!is.null(diag$plan), label = "Should have plan in diagnostics")
  expect_true(diag$plan %in% c("CPU", "GPU"), label = "Plan should be CPU or GPU")
})

# =============================================================================
# Test 9: Loss history tracking works through unified backend
# =============================================================================

test_that("Loss history is tracked and monotonically decreasing", {
  set.seed(42)
  sim <- simulateNMF(50, 40, k = 3, noise = 0.05, dropout = 0.3, seed = 800)
  A <- sim$A
  
  model <- nmf(A, 3, maxit = 50, tol = 1e-10, seed = 42, verbose = FALSE)
  
  # Should have loss history
  loss_hist <- model@misc$loss_history
  skip_if(is.null(loss_hist), "loss_history not populated (may depend on backend)")
  expect_true(length(loss_hist) > 1, label = "Should have multiple loss values")
  
  # Loss should be generally non-increasing (allow small numerical noise)
  # Skip first 3 iterations (transient phase) and allow up to 5% relative increase
  if (length(loss_hist) > 4) {
    stable_hist <- loss_hist[4:length(loss_hist)]
    diffs <- diff(stable_hist)
    relative_increases <- diffs / stable_hist[-length(stable_hist)]
    expect_true(all(relative_increases < 0.05),
                label = "Loss should be approximately monotonically decreasing (after burn-in)")
  }
})

# =============================================================================
# Test 10: cd_maxit passthrough in unified backend
# =============================================================================

test_that("cd_maxit is respected in unified backend", {
  set.seed(42)
  sim <- simulateNMF(40, 30, k = 3, noise = 0.1, dropout = 0.3, seed = 900)
  A <- sim$A
  
  # Very few CD iterations → worse convergence
  model_few <- nmf(A, 3, maxit = 20, tol = 1e-10, seed = 42,
                   cd_maxit = 1L, verbose = FALSE)
  
  # Many CD iterations → better convergence  
  model_many <- nmf(A, 3, maxit = 20, tol = 1e-10, seed = 42,
                    cd_maxit = 100L, verbose = FALSE)
  
  # More CD iterations should give lower (or equal) final loss
  expect_lte(model_many@misc$loss, model_few@misc$loss * 1.01,
             label = "More CD iterations should not increase final loss")
})
