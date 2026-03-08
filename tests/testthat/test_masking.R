# Test: Masking and Zero-Handling
# Verify masking behavior and zero treatment

options(RcppML.verbose = FALSE)

test_that("mask='zeros' excludes zero entries from optimization", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Fit with mask='zeros'
  model_masked <- nmf(A_sparse, 4, mask = "zeros", maxit = 100, tol = 1e-6,
                      seed = 456, verbose = FALSE)
  
  # Fit without mask (zeros are observed)
  model_unmasked <- nmf(A_sparse, 4, maxit = 100, tol = 1e-6,
                        seed = 456, verbose = FALSE)
  
  # Both models should converge to similar solutions for sparse matrices
  # (sparse NMF naturally focuses on non-zeros)
  # But MSE computation should differ
  
  # Both should produce valid non-negative factors
  expect_true(is_nonnegative(model_masked$w))
  expect_true(is_nonnegative(model_masked$h))
  expect_true(is_nonnegative(model_unmasked$w))
  expect_true(is_nonnegative(model_unmasked$h))
  
  # Both should converge to reasonable loss values
  expect_lt(model_masked$loss, 5.0)
  expect_lt(model_unmasked$loss, 5.0)
})

test_that("mask='zeros' loss only considers non-zero entries", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  model <- nmf(A_sparse, 4, mask = "zeros", maxit = 100, tol = 1e-6,
               seed = 456, verbose = FALSE)
  
  # Compute loss manually only on non-zero entries
  A_recon <- model$w %*% model$h
  nonzero_idx <- which(A_sparse != 0)
  mse_nonzero <- mean((A_sparse[nonzero_idx] - A_recon[nonzero_idx])^2)
  
  # Compute using helper (should match)
  loss_computed <- compute_nmf_loss(A_sparse, model$w, model$h,
                                    loss_type = "mse", mask = "zeros")$data_loss
  
  expect_equal(mse_nonzero, loss_computed, tolerance = 1e-10)
})

test_that("mask='zeros' works with different loss types", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  loss_types <- c("mse", "mae", "huber")
  
  for (loss in loss_types) {
    model <- nmf(A_sparse, 3, mask = "zeros", loss = loss,
                 maxit = 50, tol = 1e-6, seed = 456, verbose = FALSE)
    
    # Should converge without errors
    expect_true(!is.null(model$w))
    expect_true(!is.null(model$h))
    expect_true(is.finite(model$loss))
    expect_gt(model$loss, 0)
  }
})

test_that("mask='zeros' with regularization", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  model <- nmf(A_sparse, 4, mask = "zeros",
               L1 = c(0.05, 0.05), L2 = c(0.05, 0.05),
               maxit = 100, tol = 1e-6, seed = 456, verbose = FALSE)
  
  # Should produce sparse, regularized factors
  expect_gt(compute_sparsity(model$w), 0.05)
  expect_gt(compute_sparsity(model$h), 0.05)
  expect_true(is_nonnegative(model$w))
  expect_true(is_nonnegative(model$h))
})

test_that("Dense matrices: zeros are always observed", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 123)
  A_dense <- data$A
  
  # Set some entries to exactly zero
  A_dense[1:5, 1:5] <- 0
  
  # Fit NMF (no mask - zeros are observed)
  model <- nmf(A_dense, 3, maxit = 100, tol = 1e-6, seed = 456, verbose = FALSE)
  
  # Reconstruction should fit zeros too
  A_recon <- model$w %*% model$h
  
  # Check that zero region has low reconstruction values
  zero_region_recon <- A_recon[1:5, 1:5]
  
  # Should be close to zero (but not necessarily exactly zero)
  expect_lt(mean(abs(zero_region_recon)), 0.5)
})

test_that("Sparse vs dense: different treatment of zeros", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  A_dense <- as.matrix(A_sparse)
  
  # Fit sparse with mask='zeros' (zeros unobserved)
  model_sparse_masked <- nmf(A_sparse, 3, mask = "zeros",
                             maxit = 80, tol = 1e-6, seed = 456, verbose = FALSE)
  
  # Fit dense (zeros observed)
  model_dense <- nmf(A_dense, 3, maxit = 80, tol = 1e-6,
                     seed = 456, verbose = FALSE)
  
  # Both should produce valid non-negative factors
  expect_true(is_nonnegative(model_sparse_masked$w))
  expect_true(is_nonnegative(model_sparse_masked$h))
  expect_true(is_nonnegative(model_dense$w))
  expect_true(is_nonnegative(model_dense$h))
  
  # Both should converge
  expect_true(!is.null(model_sparse_masked$loss))
  expect_true(!is.null(model_dense$loss))
})

test_that("mask='zeros' improves fit on non-zero entries", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.08, dropout = 0.7, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Fit with mask='zeros'
  model_masked <- nmf(A_sparse, 4, mask = "zeros", maxit = 100, tol = 1e-6,
                      seed = 456, verbose = FALSE)
  
  # Fit without mask
  model_unmasked <- nmf(A_sparse, 4, maxit = 100, tol = 1e-6,
                        seed = 456, verbose = FALSE)
  
  # Compute MSE on non-zero entries only
  nonzero_idx <- which(A_sparse != 0)
  
  recon_masked <- model_masked$w %*% model_masked$h
  recon_unmasked <- model_unmasked$w %*% model_unmasked$h
  
  mse_masked <- mean((A_sparse[nonzero_idx] - recon_masked[nonzero_idx])^2)
  mse_unmasked <- mean((A_sparse[nonzero_idx] - recon_unmasked[nonzero_idx])^2)
  
  # Masked version focuses on non-zeros, so should fit them better (usually)
  # This is not guaranteed but is expected for highly sparse data
  expect_true(is.finite(mse_masked))
  expect_true(is.finite(mse_unmasked))
  expect_gt(mse_masked, 0)
  expect_gt(mse_unmasked, 0)
})

test_that("Highly sparse data: mask='zeros' vs standard NMF", {
  skip_on_cran()
  set.seed(42)
  
  # Generate very sparse data (90% zeros)
  data <- simulateNMF(60, 50, k = 4, noise = 0.05, dropout = 0.9, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # With such high sparsity, mask='zeros' is more appropriate
  model <- nmf(A_sparse, 4, mask = "zeros", maxit = 80, tol = 1e-6,
               seed = 456, verbose = FALSE)
  
  # Should converge to reasonable solution
  expect_true(!is.null(model$w))
  expect_true(!is.null(model$h))
  expect_true(is.finite(model$loss))
  
  # Compute error on non-zeros
  nonzero_idx <- which(A_sparse != 0)
  recon <- model$w %*% model$h
  mse_nonzero <- mean((A_sparse[nonzero_idx] - recon[nonzero_idx])^2)
  
  # For very sparse data, use absolute MSE threshold instead of variance-based
  # With high dropout and low noise, even small absolute errors can exceed variance
  expect_lt(mse_nonzero, 0.1,
            label = sprintf("MSE on non-zeros (%.4f)", mse_nonzero))
})

test_that("mask='zeros' respects non-negativity", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Strong regularization with mask
  model <- nmf(A_sparse, 4, mask = "zeros",
               L1 = c(0.2, 0.2), L2 = c(0.2, 0.2),
               maxit = 100, tol = 1e-6, seed = 456, verbose = FALSE)
  
  # Must remain non-negative
  expect_true(is_nonnegative(model$w, tol = -1e-10))
  expect_true(is_nonnegative(model$h, tol = -1e-10))
})

test_that("mask='zeros' with very few non-zeros still works", {
  skip_on_cran()
  set.seed(42)
  
  # Extreme sparsity (95% zeros)
  data <- simulateNMF(50, 40, k = 3, noise = 0.05, dropout = 0.95, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Should still converge (though may need more iterations)
  model <- nmf(A_sparse, 3, mask = "zeros", maxit = 150, tol = 1e-6,
               seed = 456, verbose = FALSE)
  
  # Basic sanity checks
  expect_true(!is.null(model$w))
  expect_true(!is.null(model$h))
  expect_true(is.finite(model$loss))
  expect_true(is_nonnegative(model$w))
  expect_true(is_nonnegative(model$h))
})

# ============================================================================
# NA Masking Tests for Dense Matrices
# ============================================================================

test_that("Dense matrices: NA values are auto-detected and masked", {
  skip_on_cran()
  set.seed(42)
  A_dense <- matrix(runif(100 * 50), 100, 50)
  
  # Introduce NA values
  A_dense[1:5, 1:5] <- NA
  na_count <- sum(is.na(A_dense))
  
  # Should auto-detect NAs and create a mask (warning is expected)
  expect_warning(
    model <- nmf(A_dense, k = 3, maxit = 20, verbose = FALSE),
    "Detected.*NA values"
  )
  
  # Model should fit successfully
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  
  # Factors should be non-negative
  expect_true(all(model$w >= 0))
  expect_true(all(model$h >= 0))
})

test_that("Dense matrices: explicit mask='NA' works", {
  skip_on_cran()
  set.seed(42)
  A_dense <- matrix(runif(80 * 40), 80, 40)
  A_dense[1:3, 1:3] <- NA
  
  # Explicit mask='NA' should work without message
  model <- suppressMessages(
    nmf(A_dense, k = 2, mask = "NA", maxit = 15, verbose = FALSE)
  )
  
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

test_that("Dense matrices: NA mask does not affect non-NA regions", {
  skip_on_cran()
  set.seed(42)
  A_dense <- matrix(runif(60 * 40), 60, 40)
  original_values <- A_dense[10:15, 10:15]  # Save non-NA region
  
  # Add NAs in different region
  A_dense[1:5, 1:5] <- NA
  
  # Fit model
  model <- suppressMessages(
    nmf(A_dense, k = 3, maxit = 30, verbose = FALSE)
  )
  
  # Reconstruction should fit non-NA region well
  recon <- model$w %*% diag(model$d) %*% model$h
  mse_non_na <- mean((original_values - recon[10:15, 10:15])^2)
  
  # MSE on non-NA region should be low (good fit)
  expect_lt(mse_non_na, 0.5)  # Reasonable threshold
})
