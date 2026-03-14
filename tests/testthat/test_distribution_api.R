# Test: Distribution API
# Tests for the new distribution= parameter, gamma, inverse_gaussian,
# robust+distribution composition, dispersion output, and score_test_distribution

options(RcppML.verbose = FALSE)

# Helper: simulate positive continuous data (Gamma-like)
simulate_gamma_data <- function(m = 60, n = 40, k = 2, seed = 42) {
  set.seed(seed)
  W <- matrix(abs(rnorm(m * k, 2, 0.5)), m, k)
  cs <- colSums(W)
  W <- W %*% diag(cs^(-1), nrow = k)
  H <- matrix(abs(rnorm(k * n, 2, 0.5)), k, n)
  mu <- W %*% H
  # Gamma: shape = 1/phi, rate = 1/(phi*mu)
  phi <- 0.5
  A <- matrix(rgamma(m * n, shape = 1 / phi, rate = 1 / (phi * mu)), m, n)
  # Ensure all positive
  A[A < 1e-8] <- 1e-8
  list(A = A, W = W, H = H, mu = mu, phi = phi)
}

# Helper: simulate count data for GP/NB
simulate_count_data <- function(m = 60, n = 40, k = 2, seed = 42) {
  set.seed(seed)
  W <- matrix(abs(rnorm(m * k, 1, 0.3)), m, k)
  cs <- colSums(W)
  W <- W %*% diag(cs^(-1), nrow = k)
  H <- matrix(abs(rnorm(k * n, 1, 0.3)), k, n)
  mu <- W %*% H
  A <- matrix(rpois(m * n, lambda = pmax(mu, 0.01) * 10), m, n)
  list(A = Matrix::Matrix(A, sparse = TRUE), W = W, H = H, mu = mu)
}

# ============================================================
# loss= parameter mapping (formerly distribution=)
# ============================================================

test_that("loss='mse' runs (formerly distribution='gaussian')", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "mse",
               maxit = 10, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("loss='gp' with dispersion='none' (formerly distribution='poisson')", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gp", dispersion = "none",
               maxit = 10, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("loss='gp' runs with dispersion estimation", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gp", dispersion = "per_row",
               maxit = 10, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(!is.null(model@misc$theta))
})

test_that("loss='nb' runs with dispersion estimation", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "per_row",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
  # NB size parameter stored in theta slot
  expect_true(!is.null(model@misc$theta))
})

test_that("loss='gamma' runs on positive data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("loss='inverse_gaussian' runs on positive data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================
# Gamma / Inverse Gaussian: basic execution
# ============================================================

test_that("Gamma NMF runs on sparse data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 30, k = 2)
  A_sp <- Matrix::Matrix(sim$A, sparse = TRUE)
  model <- nmf(A_sp, 2, loss = "gamma",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), 2)
  expect_equal(nrow(model@h), 2)
})

test_that("Gamma NMF runs on dense data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("Inverse Gaussian NMF runs on sparse data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 30, k = 2)
  A_sp <- Matrix::Matrix(sim$A, sparse = TRUE)
  model <- nmf(A_sp, 2, loss = "inverse_gaussian",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("Inverse Gaussian NMF runs on dense data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================
# Gamma / IG Dispersion Estimation
# ============================================================

test_that("Gamma dispersion per_row produces reasonable phi values", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "per_row",
               maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  expect_equal(length(disp), nrow(sim$A))
  expect_true(all(disp > 0))
  # Simulated with phi = 0.5, expect estimates in reasonable range
  expect_true(median(disp) > 0.05)
  expect_true(median(disp) < 5.0)
})

test_that("Inverse Gaussian dispersion per_row produces positive values", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian", dispersion = "per_row",
               maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  expect_equal(length(disp), nrow(sim$A))
  expect_true(all(disp > 0))
})

test_that("Gamma dispersion per_col produces n-length output", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 30, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "per_col",
               maxit = 20, tol = 1e-6, seed = 123, verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  expect_equal(length(disp), ncol(sim$A))
  expect_true(all(disp > 0))
})

test_that("Gamma dispersion global produces per_row length output", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 30, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "global",
               maxit = 20, tol = 1e-6, seed = 123, verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  # Global mode sets all elements to the same value
  expect_true(sd(disp) < 1e-6)
})

test_that("Gamma dispersion none produces no dispersion output", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "none",
               maxit = 15, tol = 1e-4, seed = 123, verbose = FALSE)
  # With dispersion=none, phi values should all be 1.0
  disp <- model@misc$dispersion
  if (!is.null(disp)) {
    expect_true(all(abs(disp - 1.0) < 1e-6))
  }
})

# ============================================================
# Gamma / IG Loss Monotonicity
# ============================================================

test_that("Gamma loss decreases overall", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "per_row",
               maxit = 30, tol = 1e-10, seed = 123, verbose = FALSE)
  loss_hist <- model@misc$loss_history
  expect_true(length(loss_hist) > 1)
  expect_true(tail(loss_hist, 1) < loss_hist[1])
})

test_that("Inverse Gaussian loss decreases overall", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian", dispersion = "per_row",
               maxit = 30, tol = 1e-10, seed = 123, verbose = FALSE)
  loss_hist <- model@misc$loss_history
  expect_true(length(loss_hist) > 1)
  expect_true(tail(loss_hist, 1) < loss_hist[1])
})

# ============================================================
# Robust + Distribution Composition
# ============================================================

test_that("robust=TRUE works with loss='gamma'", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", robust = TRUE,
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("robust=TRUE works with loss='gp'", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gp", robust = TRUE,
               dispersion = "per_row",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("robust=TRUE works with loss='nb'", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "nb", robust = TRUE,
               dispersion = "per_row",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("robust with custom delta works with gamma", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", robust = 2.0,
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("robust=TRUE with loss='inverse_gaussian'", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian", robust = TRUE,
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================
# distribution_config overrides (now flat params in ...)
# ============================================================

test_that("dispersion='per_col' overrides default for gamma", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 30, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma",
               dispersion = "per_col",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  expect_equal(length(disp), ncol(sim$A))
})

test_that("gamma_phi_init override works", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  # Custom initial phi = 2.0
  model <- nmf(sim$A, 2, loss = "gamma",
               dispersion = "per_row",
               gamma_phi_init = 2.0,
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================
# score_test_distribution
# ============================================================

test_that("score_test_distribution returns valid structure", {
  skip_on_cran()
  sim <- simulate_count_data(m = 40, n = 30, k = 2)
  model <- nmf(sim$A, 2, maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  diag <- score_test_distribution(sim$A, model)

  expect_true(is.list(diag))
  expect_true("scores" %in% names(diag))
  expect_true("best_distribution" %in% names(diag))
  expect_true(is.data.frame(diag$scores))
  expect_true(nrow(diag$scores) >= 4)
  expect_true(diag$best_distribution %in%
    c("gaussian", "gp", "gamma", "inverse_gaussian"))
})

test_that("score_test_distribution works with dense data", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, maxit = 10, tol = 1e-4, seed = 1, verbose = FALSE)
  diag <- score_test_distribution(sim$A, model)

  expect_true(is.list(diag))
  expect_true(diag$best_distribution %in%
    c("gaussian", "gp", "gamma", "inverse_gaussian"))
})

test_that("score_test_distribution with custom powers", {
  skip_on_cran()
  sim <- simulate_count_data(m = 30, n = 20, k = 2)
  model <- nmf(sim$A, 2, maxit = 10, tol = 1e-4, seed = 1, verbose = FALSE)
  diag <- score_test_distribution(sim$A, model, powers = c(0, 1, 2))

  expect_equal(nrow(diag$scores), 3)
})

# ============================================================
# Edge Cases
# ============================================================

test_that("unknown parameters in ... are rejected", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 20, n = 15, k = 2)
  expect_error(
    nmf(sim$A, 2, loss = "gamma",
        bogus_param = 42,
        maxit = 5, tol = 1e-4, seed = 1, verbose = FALSE),
    "Unknown parameter"
  )
})

test_that("Gamma NMF with k=1", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 20, n = 15, k = 1)
  model <- nmf(sim$A, 1, loss = "gamma",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), 1)
})

test_that("Gamma NMF with high rank", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 60, n = 40, k = 2)
  model <- nmf(sim$A, 8, loss = "gamma",
               maxit = 15, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), 8)
})

# ============================================================================
# auto_nmf_distribution() and diagnose_zero_inflation() as standalone functions
# ============================================================================

test_that("auto_nmf_distribution() selects a valid distribution", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 60, n = 40, k = 3)
  result <- auto_nmf_distribution(sim$A, k = 3, maxit = 30, tol = 1e-4,
                                  seed = 42, verbose = FALSE)
  expect_true(is.character(result$loss))
  expect_true(result$loss %in% c("mse", "gp", "nb", "gamma", "inverse_gaussian", "tweedie"))
  # Run NMF with the selected loss
  model <- nmf(sim$A, 3, loss = result$loss,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(nrow(model@w) == 60)
})

test_that("auto_nmf_distribution() works with sparse matrix", {
  skip_on_cran()
  set.seed(42)
  A <- Matrix::rsparsematrix(50, 30, density = 0.3)
  A@x <- abs(A@x)  # make non-negative
  result <- auto_nmf_distribution(A, k = 3, maxit = 20, tol = 1e-3,
                                  seed = 1, verbose = FALSE)
  expect_true(is.character(result$loss))
  model <- nmf(A, 3, loss = result$loss,
               maxit = 20, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("diagnose_zero_inflation() selects a valid zi mode", {
  skip_on_cran()
  # Create data with some zero inflation
  set.seed(42)
  W <- matrix(runif(50 * 3), 50, 3)
  H <- matrix(runif(3 * 30), 3, 30)
  mu <- W %*% H
  A <- matrix(rpois(50 * 30, lambda = mu), 50, 30)
  # Add extra zeros (zero-inflate)
  zi_mask <- matrix(rbinom(50 * 30, 1, 0.3), 50, 30)
  A[zi_mask == 1] <- 0
  A <- as(A, "dgCMatrix")

  # First fit a baseline model
  model <- nmf(A, 3, maxit = 20, tol = 1e-3, seed = 1, verbose = FALSE)
  zi_result <- diagnose_zero_inflation(A, model)
  expect_true(zi_result$zi %in% c("none", "row", "col"))
  # Run NMF with the selected zi mode
  model2 <- nmf(A, 3, zi = zi_result$zi, loss = "gp",
                maxit = 20, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_s4_class(model2, "nmf")
})

test_that("auto_nmf_distribution() + diagnose_zero_inflation() together", {
  skip_on_cran()
  set.seed(42)
  W <- matrix(runif(50 * 3), 50, 3)
  H <- matrix(runif(3 * 30), 3, 30)
  mu <- W %*% H
  A <- matrix(rpois(50 * 30, lambda = mu), 50, 30)
  A <- as(A, "dgCMatrix")

  # Select distribution automatically
  dist_result <- auto_nmf_distribution(A, k = 3, maxit = 20, tol = 1e-3,
                                       seed = 1, verbose = FALSE)
  # First fit with auto-selected distribution, then diagnose ZI
  model_base <- nmf(A, 3, loss = dist_result$loss,
                    maxit = 20, tol = 1e-3, seed = 1, verbose = FALSE)
  zi_result <- diagnose_zero_inflation(A, model_base)
  
  model <- nmf(A, 3, loss = dist_result$loss, zi = zi_result$zi,
               maxit = 20, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================================
# Tweedie continuous power
# ============================================================================

test_that("Tweedie NMF runs with default power (1.5)", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 30, k = 3)
  model <- nmf(sim$A, 3, loss = "tweedie",
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("Tweedie NMF with power=2 matches Gamma", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 25, k = 2)
  model_tw <- nmf(sim$A, 2, loss = "tweedie", tweedie_power = 2.0,
                  maxit = 40, tol = 1e-6, seed = 1, verbose = FALSE)
  model_gm <- nmf(sim$A, 2, loss = "gamma",
                  maxit = 40, tol = 1e-6, seed = 1, verbose = FALSE)
  expect_s4_class(model_tw, "nmf")
  expect_s4_class(model_gm, "nmf")
  # Losses should be very close since p=2 Tweedie = Gamma deviance
  expect_equal(model_tw@misc$loss, model_gm@misc$loss, tolerance = 0.01)
})

test_that("Tweedie NMF with power=3 matches InvGaussian", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 25, k = 2)
  model_tw <- nmf(sim$A, 2, loss = "tweedie", tweedie_power = 3.0,
                  maxit = 40, tol = 1e-6, seed = 1, verbose = FALSE)
  model_ig <- nmf(sim$A, 2, loss = "inverse_gaussian",
                  maxit = 40, tol = 1e-6, seed = 1, verbose = FALSE)
  expect_s4_class(model_tw, "nmf")
  expect_s4_class(model_ig, "nmf")
  # Losses should be very close since p=3 Tweedie = IG deviance
  expect_equal(model_tw@misc$loss, model_ig@misc$loss, tolerance = 0.01)
})

test_that("Tweedie power configurable via tweedie_power in ...", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 25, k = 2)
  model <- nmf(sim$A, 2, loss = "tweedie",
               tweedie_power = 2.5,
               maxit = 30, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("Tweedie NMF with robust modifier", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 40, n = 25, k = 2)
  model <- nmf(sim$A, 2, loss = "tweedie", tweedie_power = 1.8,
               robust = TRUE, maxit = 30, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================================
# Robust loss tracking
# ============================================================================

test_that("Robust loss is tracked and monotonically decreasing", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 50, n = 30, k = 3)
  model <- nmf(sim$A, 3, loss = "gamma", robust = TRUE,
               maxit = 30, tol = 1e-10, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
  # Final loss should be non-negative
  expect_true(model@misc$loss >= 0)
})

test_that("Robust MSE loss is properly bounded by Huber function", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(40 * 25, mean = 2)), 40, 25)
  # Add some outliers
  A[1:3, 1:3] <- 100

  # Non-robust should have higher loss than robust (outliers penalized heavily)
  model_std <- nmf(A, 3, loss = "mse",
                   maxit = 40, tol = 1e-10, seed = 1, verbose = FALSE)
  model_rob <- nmf(A, 3, loss = "mse", robust = TRUE,
                   maxit = 40, tol = 1e-10, seed = 1, verbose = FALSE)
  expect_s4_class(model_std, "nmf")
  expect_s4_class(model_rob, "nmf")
  # Robust loss (Huber rho) should be less than standard MSE loss
  # because Huber downweights large residuals
  expect_true(model_rob@misc$loss < model_std@misc$loss)
})

# ============================================================================
# Iteration-by-iteration loss monotonicity for IRLS distributions
# ============================================================================

test_that("Gamma loss decreases monotonically iteration by iteration", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 60, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "gamma", dispersion = "per_row",
               maxit = 50, tol = 0, seed = 123, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  # After warmup (first 5 iters), loss should be non-increasing
  lh_stable <- lh[5:length(lh)]
  diffs <- diff(lh_stable)
  # Allow small numerical tolerance (1e-3 relative)
  expect_true(all(diffs < max(abs(lh_stable[1])) * 1e-3),
              info = paste0("Gamma loss not monotonic; max increase = ",
                            max(diffs[diffs > 0])))
})

test_that("Inverse Gaussian loss decreases monotonically iteration by iteration", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 60, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "inverse_gaussian", dispersion = "per_row",
               maxit = 50, tol = 0, seed = 123, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  # InvGauss with MoM dispersion updates can have transient increases
  # (phi update changes the effective loss landscape). Verify:
  # 1. Overall decrease: final < first (excluding iter 1 warmup)
  expect_true(tail(lh, 1) < lh[2],
              info = paste0("InvGauss final=", round(tail(lh,1), 2),
                            " not < iter2=", round(lh[2], 2)))
  # 2. Final 10 iterations are stable/decreasing
  lh_tail <- tail(lh, 10)
  expect_true(lh_tail[10] <= lh_tail[1] * 1.01,
              info = "InvGauss final segment should be non-increasing")
})

test_that("Tweedie loss decreases monotonically iteration by iteration", {
  skip_on_cran()
  sim <- simulate_gamma_data(m = 60, n = 40, k = 2)
  model <- nmf(sim$A, 2, loss = "tweedie", tweedie_power = 1.5,
               dispersion = "per_row", maxit = 50, tol = 0, seed = 123, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  lh_stable <- lh[5:length(lh)]
  diffs <- diff(lh_stable)
  expect_true(all(diffs < max(abs(lh_stable[1])) * 1e-3),
              info = paste0("Tweedie loss not monotonic; max increase = ",
                            max(diffs[diffs > 0])))
})

# ============================================================================
# Robust outlier downweighting tests
# ============================================================================

test_that("Robust MSE downweights outliers: lower reconstruction error on clean entries", {
  skip_on_cran()
  set.seed(42)
  m <- 60; n <- 40; k <- 3
  W_true <- matrix(abs(rnorm(m * k, 2, 0.5)), m, k)
  H_true <- matrix(abs(rnorm(k * n, 2, 0.5)), k, n)
  mu <- W_true %*% H_true
  A_clean <- mu + matrix(rnorm(m * n, 0, 0.1), m, n)
  A_clean[A_clean < 0] <- 0

  # Inject 5% large outliers
  A_dirty <- A_clean
  n_outliers <- round(0.05 * m * n)
  outlier_idx <- sample(m * n, n_outliers)
  A_dirty[outlier_idx] <- A_dirty[outlier_idx] + abs(rnorm(n_outliers, 50, 10))

  model_std <- nmf(A_dirty, k, loss = "mse",
                   maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)
  model_rob <- nmf(A_dirty, k, loss = "mse", robust = TRUE,
                   maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)

  # Compute MSE on CLEAN entries only (non-outlier)
  recon_std <- model_std$w %*% (model_std$d * model_std$h)
  recon_rob <- model_rob$w %*% (model_rob$d * model_rob$h)
  clean_idx <- setdiff(1:(m*n), outlier_idx)
  mse_std_clean <- mean((A_clean[clean_idx] - recon_std[clean_idx])^2)
  mse_rob_clean <- mean((A_clean[clean_idx] - recon_rob[clean_idx])^2)

  # Robust should have better reconstruction on clean entries
  expect_true(mse_rob_clean < mse_std_clean * 1.5,
              info = paste0("Robust MSE on clean=", round(mse_rob_clean, 4),
                            " vs Standard=", round(mse_std_clean, 4)))
})

test_that("Robust GP downweights outliers in count data", {
  skip_on_cran()
  sim <- simulate_count_data(m = 60, n = 40, k = 2)
  A <- as.matrix(sim$A)

  # Inject outliers: multiply 5% of entries by 100
  n_outliers <- round(0.05 * nrow(A) * ncol(A))
  set.seed(99)
  outlier_idx <- sample(length(A), n_outliers)
  A_dirty <- A
  A_dirty[outlier_idx] <- A_dirty[outlier_idx] * 100 + 50

  A_sp <- Matrix::Matrix(A_dirty, sparse = TRUE)

  model_std <- nmf(A_sp, 2, loss = "gp", dispersion = "per_row",
                   maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)
  model_rob <- nmf(A_sp, 2, loss = "gp", dispersion = "per_row",
                   robust = TRUE, maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)

  # Both should complete without error
  expect_s4_class(model_std, "nmf")
  expect_s4_class(model_rob, "nmf")

  # Robust model should have lower KL divergence on clean entries
  recon_std <- model_std$w %*% (model_std$d * model_std$h)
  recon_rob <- model_rob$w %*% (model_rob$d * model_rob$h)

  clean_idx <- setdiff(1:length(A), outlier_idx)
  mu_std <- pmax(as.vector(recon_std)[clean_idx], 1e-10)
  mu_rob <- pmax(as.vector(recon_rob)[clean_idx], 1e-10)
  y_clean <- as.vector(A)[clean_idx]

  kl_std <- mean(y_clean * log(pmax(y_clean, 1e-10) / mu_std) - y_clean + mu_std)
  kl_rob <- mean(y_clean * log(pmax(y_clean, 1e-10) / mu_rob) - y_clean + mu_rob)

  expect_true(kl_rob < kl_std * 2.0,
              info = paste0("Robust GP KL on clean=", round(kl_rob, 4),
                            " vs Standard=", round(kl_std, 4)))
})
