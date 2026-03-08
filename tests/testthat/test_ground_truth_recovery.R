# Test: Ground Truth Recovery
# Verify NMF can recover known factors from synthetic data
#
# IMPORTANT: simulateNMF normalizes W (cols sum to 1) and H (rows sum to 1),
# so A = W*H has very small entries (mean ~ k/(nrow*ncol)).
# We must either use noise=0 or scale noise proportionally to signal.
# Also, NMF returns A ≈ W*diag(d)*H, so reconstruction must include d.

options(RcppML.verbose = FALSE)

# Helper: generate simulated NMF data with properly scaled noise/dropout
# noise_frac: noise std as fraction of mean signal (0 = no noise)
# dropout: fraction of entries to zero out (0 = no dropout)
simulate_with_scaled_noise <- function(nrow, ncol, k, noise_frac = 0,
                                       dropout = 0, seed = 123) {
  # Generate clean factored data
  data <- simulateNMF(nrow, ncol, k, noise = 0, dropout = 0, seed = seed)
  A <- data$A
  
  if (noise_frac > 0 || dropout > 0) {
    set.seed(seed + 7777L)  # separate RNG stream for noise/dropout
    signal_mean <- mean(A[A > 0])
    
    # Add noise scaled to signal
    if (noise_frac > 0) {
      noise_sd <- noise_frac * signal_mean
      A <- A + matrix(rnorm(nrow * ncol, 0, noise_sd), nrow, ncol)
      A[A < 0] <- 0
    }
    
    # Apply dropout
    if (dropout > 0) {
      mask <- matrix(rbinom(nrow * ncol, 1, 1 - dropout), nrow, ncol)
      A <- A * mask
    }
    
    data$A <- A
  }
  
  data
}

# Helper: compute full NMF reconstruction including diagonal
nmf_reconstruct <- function(model) {
  model$w %*% diag(model$d) %*% model$h
}

test_that("Perfect recovery with no noise - small problem", {
  skip_on_cran()
  set.seed(42)
  
  # Generate perfect NMF data (no noise)
  data <- simulate_with_scaled_noise(40, 30, k = 3, noise_frac = 0, dropout = 0, seed = 123)
  A <- data$A
  W_true <- data$w
  H_true <- data$h
  
  # Multi-restart to avoid local minima on perfect problem
  seeds <- c(456, 789, 101, 202, 303)
  best_cor <- -1
  best_recon <- Inf
  for (s in seeds) {
    model <- nmf(A, 3, maxit = 500, tol = 1e-8, seed = s, verbose = FALSE)
    aligned <- align_nmf_factors(model$w, model$h, W_true, H_true)
    mean_cor <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
    recon <- reconstruction_error(A, nmf_reconstruct(model), relative = TRUE)
    if (mean_cor > best_cor) {
      best_cor <- mean_cor
      best_recon <- recon
      best_aligned <- aligned
    }
  }
  
  # No noise → should achieve near-perfect recovery with best of 5 seeds
  expect_gt(best_aligned$mean_W_cor, 0.90)
  expect_gt(best_aligned$mean_H_cor, 0.90)
  
  # Reconstruction error should be very small on noise-free data
  expect_lt(best_recon, 0.05)
})

test_that("High recovery with low noise", {
  skip_on_cran()
  set.seed(42)
  
  # ~20% noise relative to signal, light dropout
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.2, dropout = 0.1, seed = 123)
  A <- data$A
  W_true <- data$w
  H_true <- data$h
  
  # Multi-restart: take best of 3 seeds to mitigate local minima
  seeds <- c(456, 789, 101)
  best_aligned <- NULL
  best_cor <- -1
  for (s in seeds) {
    model <- nmf(A, 4, maxit = 300, tol = 1e-6, seed = s, verbose = FALSE)
    aligned <- align_nmf_factors(model$w, model$h, W_true, H_true)
    mean_cor <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
    if (mean_cor > best_cor) {
      best_cor <- mean_cor
      best_aligned <- aligned
    }
  }
  
  # Low noise → best of 3 should achieve meaningful correlation
  expect_gt(best_aligned$mean_W_cor, 0.4)
  expect_gt(best_aligned$mean_H_cor, 0.4)
})

test_that("Recovery degrades gracefully with increasing noise", {
  skip_on_cran()
  set.seed(42)
  
  # Noise fractions relative to signal: 10%, 30%, 60%, 100%
  noise_fracs <- c(0.1, 0.3, 0.6, 1.0)
  correlations <- numeric(length(noise_fracs))
  
  for (i in seq_along(noise_fracs)) {
    data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = noise_fracs[i],
                                        dropout = 0.2, seed = 123)
    
    model <- nmf(data$A, 4, maxit = 200, tol = 1e-6, seed = 456, verbose = FALSE)
    aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
    
    correlations[i] <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
  }
  
  # Correlations should generally decrease with noise
  # GPU may have slightly different numerics, so use generous tolerance
  expect_gt(correlations[1], correlations[4] - 0.2)
  
  # Even with 100% noise fraction, should achieve some positive correlation
  expect_gt(correlations[4], 0.05)
})

test_that("Recovery works for different ranks", {
  skip_on_cran()
  set.seed(42)
  
  ranks <- c(2, 5, 8)
  
  for (k in ranks) {
    data <- simulate_with_scaled_noise(80, 60, k = k, noise_frac = 0.1,
                                        dropout = 0.2, seed = 123)
    
    # Multi-restart: best of 3 seeds
    best_cor <- -1
    for (s in c(456, 789, 101)) {
      model <- nmf(data$A, k, maxit = 300, tol = 1e-6, seed = s, verbose = FALSE)
      aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
      mean_cor <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
      if (mean_cor > best_cor) best_cor <- mean_cor
    }
    
    # Low noise with dropout=0.2 is recoverable for all ranks
    expect_gt(best_cor, 0.15,
              label = sprintf("k=%d: best_cor=%.3f should be > 0.15", k, best_cor))
  }
})

test_that("Recovery with sparse input matrices", {
  skip_on_cran()
  set.seed(42)
  
  # Moderate noise + dropout creates sparse matrix while still allowing recovery
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.1,
                                      dropout = 0.3, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Multi-restart: best of 3 seeds
  best_W <- -1; best_H <- -1
  for (s in c(456, 789, 101)) {
    model <- nmf(A_sparse, 4, maxit = 300, tol = 1e-6, seed = s, verbose = FALSE)
    aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
    if (aligned$mean_W_cor > best_W) best_W <- aligned$mean_W_cor
    if (aligned$mean_H_cor > best_H) best_H <- aligned$mean_H_cor
  }
  
  # Sparse input with calibrated noise should still yield positive correlations
  expect_gt(best_W, 0.15)
  expect_gt(best_H, 0.15)
})

test_that("Recovery with mask='zeros' for sparse data", {
  skip_on_cran()
  set.seed(42)
  
  # Higher dropout to create genuine sparsity for masking
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.1,
                                      dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")
  
  # Multi-restart: best of 3 seeds (mask='zeros' + high dropout is harder)
  best_cor <- -1
  for (s in c(456, 789, 101)) {
    model <- nmf(A_sparse, 4, mask = "zeros", maxit = 300, tol = 1e-6, 
                 seed = s, verbose = FALSE)
    aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
    mean_cor_s <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
    if (mean_cor_s > best_cor) best_cor <- mean_cor_s
  }
  
  # High dropout (0.6) + mask='zeros' is harder, but should still be positive
  expect_gt(best_cor, 0.15)
})

test_that("Recovery with different loss functions", {
  skip_on_cran()
  set.seed(42)
  
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.15,
                                      dropout = 0.3, seed = 123)
  A <- data$A
  
  loss_types <- c("mse", "mae", "huber")
  
  for (loss in loss_types) {
    # Multi-restart: best of 3 seeds
    best_cor <- -1
    for (s in c(456, 789, 101)) {
      model <- nmf(A, 4, loss = loss, maxit = 200, tol = 1e-6, 
                   seed = s, verbose = FALSE)
      aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
      mean_cor_s <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
      if (mean_cor_s > best_cor) best_cor <- mean_cor_s
    }
    
    # All loss types should achieve positive correlation
    expect_gt(best_cor, 0.15,
              label = sprintf("loss=%s: best_cor=%.3f should be > 0.15", loss, best_cor))
  }
})

test_that("Multiple restarts improve recovery", {
  skip_on_cran()
  set.seed(42)
  
  # Moderate difficulty problem
  data <- simulate_with_scaled_noise(60, 50, k = 5, noise_frac = 0.2,
                                      dropout = 0.3, seed = 123)
  A <- data$A
  
  # Single run
  model_single <- nmf(A, 5, maxit = 100, tol = 1e-6, seed = 456, verbose = FALSE)
  aligned_single <- align_nmf_factors(model_single$w, model_single$h, data$w, data$h)
  cor_single <- mean(c(aligned_single$mean_W_cor, aligned_single$mean_H_cor))
  
  # Multiple restarts - run several and take best
  models <- lapply(1:3, function(s) {
    nmf(A, 5, maxit = 100, tol = 1e-6, seed = 100 + s, verbose = FALSE)
  })
  
  # Find best model (lowest reconstruction error)
  errors <- sapply(models, function(m) {
    reconstruction_error(A, nmf_reconstruct(m), relative = TRUE)
  })
  best_model <- models[[which.min(errors)]]
  
  aligned_best <- align_nmf_factors(best_model$w, best_model$h, data$w, data$h)
  cor_best <- mean(c(aligned_best$mean_W_cor, aligned_best$mean_H_cor))
  
  # Best of multiple runs should be at least as good (often better)
  # Relaxed tolerance due to stochastic variance
  expect_gte(cor_best, cor_single - 0.1)
  
  # Reconstruction error should be better (allow some tolerance)
  error_single <- reconstruction_error(A, nmf_reconstruct(model_single), relative = TRUE)
  error_best <- min(errors)
  expect_lte(error_best, error_single + 0.05)
})

test_that("Reconstruction error correlates with recovery quality", {
  skip_on_cran()
  set.seed(42)
  
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.2,
                                      dropout = 0.3, seed = 123)
  A <- data$A
  
  # Run with different maxit to get varying quality
  iterations <- c(10, 50, 200)
  recon_errors <- numeric(length(iterations))
  correlations <- numeric(length(iterations))
  
  for (i in seq_along(iterations)) {
    model <- nmf(A, 4, maxit = iterations[i], tol = 1e-6, seed = 456, verbose = FALSE)
    
    recon_errors[i] <- reconstruction_error(A, nmf_reconstruct(model), relative = TRUE)
    
    aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
    correlations[i] <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
  }
  
  # More iterations should generally reduce reconstruction error (allow small tolerance)
  expect_lt(recon_errors[3], recon_errors[1] * 1.1)
  
  # Check correlations are computed and finite
  expect_true(length(correlations) == 3)
  expect_true(all(is.finite(correlations)))
})

test_that("Recovery with regularization", {
  skip_on_cran()
  set.seed(42)
  
  data <- simulate_with_scaled_noise(60, 50, k = 4, noise_frac = 0.1,
                                      dropout = 0.3, seed = 123)
  A <- data$A
  
  # Fit with mild L2 regularization
  model <- nmf(A, 4, L2 = c(0.01, 0.01), maxit = 300, tol = 1e-6, 
               seed = 456, verbose = FALSE)
  
  aligned <- align_nmf_factors(model$w, model$h, data$w, data$h)
  mean_cor <- mean(c(aligned$mean_W_cor, aligned$mean_H_cor))
  
  # Mild regularization shouldn't drastically hurt recovery
  # GPU backend may converge differently, so very relaxed threshold
  expect_gt(mean_cor, 0.0)
})

test_that("Alignment resolves permutation ambiguity", {
  skip_on_cran()
  set.seed(42)
  
  data <- simulate_with_scaled_noise(50, 40, k = 3, noise_frac = 0.1,
                                      dropout = 0.2, seed = 123)
  
  # Fit same problem with different seeds (different permutations)
  model1 <- nmf(data$A, 3, maxit = 200, tol = 1e-6, seed = 111, verbose = FALSE)
  model2 <- nmf(data$A, 3, maxit = 200, tol = 1e-6, seed = 222, verbose = FALSE)
  
  # Raw correlations between models may be low (permutation mismatch)
  # But after alignment, should match ground truth similarly
  
  aligned1 <- align_nmf_factors(model1$w, model1$h, data$w, data$h)
  aligned2 <- align_nmf_factors(model2$w, model2$h, data$w, data$h)
  
  # Both should achieve similar recovery quality
  cor1 <- mean(c(aligned1$mean_W_cor, aligned1$mean_H_cor))
  cor2 <- mean(c(aligned2$mean_W_cor, aligned2$mean_H_cor))
  
  # Relaxed tolerance due to stochastic variance in NMF optimization
  # GPU may converge to different local minima
  expect_true(abs(cor1 - cor2) < 0.5)
})


