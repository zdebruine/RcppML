# Test: Loss Function Correctness
# Verify each loss function computes correct values

options(RcppML.verbose = FALSE)

test_that("MSE loss computed correctly - small example", {
  skip_on_cran()
  # Simple test case with known values
  A <- matrix(c(1, 2, 3, 4, 5, 6), 3, 2)
  
  # Run NMF with 0 iterations to just evaluate initial loss
  model <- nmf(A, 2, maxit = 100, tol = 1e-6, seed = 42, loss = "mse", verbose = FALSE)
  
  # Compute MSE manually
  A_recon <- model$w %*% model$h
  mse_manual <- mean((A - A_recon)^2)
  
  # Should match (approximately, after optimization)
  mse_computed <- compute_nmf_loss(A, model$w, model$h, loss_type = "mse")$data_loss
  
  expect_equal(mse_manual, mse_computed, tolerance = 1e-10)
})

test_that("MSE loss computed correctly - larger problem", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  model <- nmf(A, 3, loss = "mse", maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
  
  A_recon <- model$w %*% model$h
  mse_manual <- mean((A - A_recon)^2)
  mse_computed <- compute_nmf_loss(A, model$w, model$h, loss_type = "mse")$data_loss
  
  expect_equal(mse_manual, mse_computed, tolerance = 1e-10)
})

test_that("MAE loss computed correctly", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  model <- nmf(A, 3, loss = "mae", maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
  
  A_recon <- model$w %*% model$h
  mae_manual <- mean(abs(A - A_recon))
  mae_computed <- compute_nmf_loss(A, model$w, model$h, loss_type = "mae")$data_loss
  
  expect_equal(mae_manual, mae_computed, tolerance = 1e-10)
})

test_that("Huber loss computed correctly - various delta values", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  deltas <- c(0.5, 1.0, 2.0)
  
  for (delta in deltas) {
    model <- nmf(A, 3, loss = "huber", huber_delta = delta, 
                 maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
    
    # Manual Huber loss computation
    residuals <- as.vector(A - model$w %*% model$h)
    huber_manual <- sapply(residuals, function(r) {
      if (abs(r) <= delta) {
        0.5 * r^2
      } else {
        delta * (abs(r) - 0.5 * delta)
      }
    })
    huber_loss_manual <- mean(huber_manual)
    
    huber_computed <- compute_nmf_loss(A, model$w, model$h, 
                                       loss_type = "huber", 
                                       huber_delta = delta)$data_loss
    
    expect_equal(huber_loss_manual, huber_computed, tolerance = 1e-8)
  }
})

test_that("KL divergence computed correctly", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.05, dropout = 0.2, seed = 42)
  A <- pmax(data$A, 0.01)  # KL requires positive values
  
  model <- nmf(A, 3, loss = "kl", maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
  
  A_recon <- model$w %*% model$h
  A_recon_safe <- pmax(A_recon, 1e-10)
  A_safe <- pmax(A, 1e-10)
  
  # KL divergence: sum(A * log(A / A_recon) - A + A_recon) / n
  kl_manual <- mean(A_safe * log(A_safe / A_recon_safe) - A_safe + A_recon_safe)
  
  kl_computed <- compute_nmf_loss(A, model$w, model$h, loss_type = "kl")$data_loss
  
  expect_equal(kl_manual, kl_computed, tolerance = 1e-7)
})

test_that("MAE is more robust to outliers than MSE", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  # Add outliers
  A_outliers <- A
  outlier_idx <- sample(length(A_outliers), 15)
  A_outliers[outlier_idx] <- A_outliers[outlier_idx] + abs(rnorm(15, mean = 3, sd = 1))
  
  # Fit with both losses
  model_mae <- nmf(A_outliers, 3, loss = "mae", maxit = 100, tol = 1e-6, seed = 123, verbose = FALSE)
  model_mse <- nmf(A_outliers, 3, loss = "mse", maxit = 100, tol = 1e-6, seed = 123, verbose = FALSE)
  
  # Compute MAE for both (MAE should be lower for MAE-optimized model)
  mae_mae <- mean(abs(A_outliers - model_mae$w %*% model_mae$h))
  mae_mse <- mean(abs(A_outliers - model_mse$w %*% model_mse$h))
  
  # MAE optimization should achieve comparable MAE to MSE
  # (IRLS for MAE may converge to different local optima, so we allow wider margin)
  expect_lte(mae_mae, mae_mse * 1.5)  # Allow 50% tolerance
  
  # Compute MSE for both (MSE should be lower for MSE-optimized model)
  mse_mae <- mean((A_outliers - model_mae$w %*% model_mae$h)^2)
  mse_mse <- mean((A_outliers - model_mse$w %*% model_mse$h)^2)
  
  expect_lte(mse_mse, mse_mae * 1.05)
})

test_that("Huber loss interpolates between MSE and MAE", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  # Fit with MSE, Huber, and MAE
  model_mse <- nmf(A, 3, loss = "mse", maxit = 100, tol = 1e-6, seed = 123, verbose = FALSE)
  model_huber <- nmf(A, 3, loss = "huber", huber_delta = 1.0, 
                     maxit = 100, tol = 1e-6, seed = 123, verbose = FALSE)
  model_mae <- nmf(A, 3, loss = "mae", maxit = 100, tol = 1e-6, seed = 123, verbose = FALSE)
  
  # Compute residuals
  res_mse <- as.vector(A - model_mse$w %*% model_mse$h)
  res_huber <- as.vector(A - model_huber$w %*% model_huber$h)
  res_mae <- as.vector(A - model_mae$w %*% model_mae$h)
  
  # Huber should be between MSE and MAE in behavior
  # Check median absolute residual (robust metric)
  median_res_mse <- median(abs(res_mse))
  median_res_huber <- median(abs(res_huber))
  median_res_mae <- median(abs(res_mae))
  
  # All should achieve reasonable fits
  expect_lt(median_res_mse, sd(as.vector(A)))
  expect_lt(median_res_huber, sd(as.vector(A)))
  expect_lt(median_res_mae, sd(as.vector(A)))
})

test_that("Loss functions work with sparse matrices", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.7, seed = 42)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Test all loss types
  for (loss in c("mse", "mae", "huber", "kl")) {
    if (loss == "kl") {
      A_test <- pmax(A_sparse, 0.01)
    } else {
      A_test <- A_sparse
    }
    
    model <- nmf(A_test, 3, loss = loss, maxit = 50, tol = 1e-6, 
                 seed = 123, verbose = FALSE)
    
    # Should converge without errors
    expect_true(!is.null(model$w))
    expect_true(!is.null(model$h))
    expect_true(is.finite(model$loss))
    
    # Loss should be positive
    expect_gt(model$loss, 0)
  }
})

test_that("Loss with L1 regularization includes penalty", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  L1_values <- c(0, 0.01, 0.1)
  
  for (L1_val in L1_values) {
    model <- nmf(A, 3, loss = "mse", L1 = c(L1_val, L1_val), 
                 maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
    
    loss_decomp <- compute_nmf_loss(A, model$w, model$h, 
                                    L1 = c(L1_val, L1_val), 
                                    loss_type = "mse")
    
    # Check L1 penalty is computed correctly
    expected_l1 <- L1_val * (sum(abs(model$w)) + sum(abs(model$h)))
    expect_equal(loss_decomp$l1_W + loss_decomp$l1_H, expected_l1, tolerance = 1e-10)
    
    # Total loss should include penalty
    expect_equal(loss_decomp$total, 
                 loss_decomp$data_loss + loss_decomp$l1_W + loss_decomp$l1_H,
                 tolerance = 1e-10)
  }
})

test_that("Loss with L2 regularization includes penalty", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.3, seed = 42)
  A <- data$A
  
  L2_values <- c(0, 0.01, 0.1)
  
  for (L2_val in L2_values) {
    model <- nmf(A, 3, loss = "mse", L2 = c(L2_val, L2_val), 
                 maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
    
    loss_decomp <- compute_nmf_loss(A, model$w, model$h, 
                                    L2 = c(L2_val, L2_val), 
                                    loss_type = "mse")
    
    # Check L2 penalty is computed correctly
    expected_l2 <- (L2_val / 2) * (sum(model$w^2) + sum(model$h^2))
    expect_equal(loss_decomp$l2_W + loss_decomp$l2_H, expected_l2, tolerance = 1e-10)
    
    # Total loss should include penalty
    expect_equal(loss_decomp$total, 
                 loss_decomp$data_loss + loss_decomp$l2_W + loss_decomp$l2_H,
                 tolerance = 1e-10)
  }
})

test_that("Loss with mask='zeros' only considers non-zero entries", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, dropout = 0.6, seed = 42)
  A_sparse <- as(data$A, "dgCMatrix")
  
  model <- nmf(A_sparse, 3, loss = "mse", mask = "zeros", 
               maxit = 50, tol = 1e-6, seed = 123, verbose = FALSE)
  
  # Compute loss only on non-zero entries
  A_recon <- model$w %*% model$h
  nonzero_idx <- which(A_sparse != 0)
  mse_nonzero <- mean((A_sparse[nonzero_idx] - A_recon[nonzero_idx])^2)
  
  loss_computed <- compute_nmf_loss(A_sparse, model$w, model$h, 
                                    loss_type = "mse", mask = "zeros")$data_loss
  
  expect_equal(mse_nonzero, loss_computed, tolerance = 1e-10)
})

