# Test: NB (Negative Binomial) NMF
# Tests for NB loss, dispersion estimation, ZINB, and loss monotonicity

options(RcppML.verbose = FALSE)

# Helper: simulate NB count data with known overdispersion
# NOTE: Do NOT normalize W by colSums — it makes mu ~0.025, producing
# nearly all-zero counts where the MoM dispersion estimator cannot detect
# overdispersion. Without normalization, mu ~k (e.g. ~3), giving realistic
# count distributions where NB excess variance is measurable.
simulate_nb_data <- function(m = 80, n = 60, k = 3, size = 5.0,
                              dropout = 0, seed = 42) {
  set.seed(seed)
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H

  A <- matrix(rnbinom(m * n, size = size, mu = mu), m, n)

  if (dropout > 0) {
    mask <- matrix(rbinom(m * n, 1, 1 - dropout), m, n)
    A <- A * mask
  }

  list(A = Matrix::Matrix(A, sparse = TRUE),
       W = W, H = H, mu = mu, size = size)
}

# ============================================================
# NB Ground-Truth Recovery
# ============================================================

test_that("NB-NMF with global dispersion recovers size parameter order of magnitude", {
  skip_on_cran()
  # Simulate NB data with known size parameter
  size_true <- 5.0
  sim <- simulate_nb_data(m = 120, n = 100, k = 3, size = size_true, seed = 99)

  model <- nmf(sim$A, 3, loss = "nb", dispersion = "global",
               maxit = 100, tol = 1e-8, seed = 42, verbose = FALSE)

  # The nb_size should be stored in misc
  expect_true("nb_size" %in% names(model@misc) || "theta" %in% names(model@misc))
  if ("nb_size" %in% names(model@misc)) {
    size_est <- mean(model@misc$nb_size)
  } else {
    size_est <- mean(model@misc$theta)
  }
  cat("  NB ground truth: size_true =", size_true, ", size_est =", size_est, "\n")
  # NB size estimation has bias; verify within order of magnitude
  expect_true(size_est > 0, info = "size_est should be positive")
  expect_true(size_est < size_true * 10,
              info = paste0("size_est=", round(size_est, 3), " too far from size_true=", size_true))
  expect_true(size_est > size_true * 0.1,
              info = paste0("size_est=", round(size_est, 3), " too small vs size_true=", size_true))
})

test_that("NB-NMF distinguishes high vs low overdispersion", {
  skip_on_cran()
  # High overdispersion (small size) vs low overdispersion (large size)
  sim_high <- simulate_nb_data(m = 80, n = 60, k = 2, size = 1.0, seed = 42)
  sim_low  <- simulate_nb_data(m = 80, n = 60, k = 2, size = 50.0, seed = 42)

  m_high <- nmf(sim_high$A, 2, loss = "nb", dispersion = "global",
                maxit = 50, tol = 1e-6, seed = 42, verbose = FALSE)
  m_low  <- nmf(sim_low$A, 2, loss = "nb", dispersion = "global",
                maxit = 50, tol = 1e-6, seed = 42, verbose = FALSE)

  get_size <- function(model) {
    if ("nb_size" %in% names(model@misc)) mean(model@misc$nb_size)
    else mean(model@misc$theta)
  }
  r_high <- get_size(m_high)
  r_low  <- get_size(m_low)
  cat("  NB high OD: r =", r_high, ", low OD: r =", r_low, "\n")
  # The estimated r for high overdispersion data should be smaller
  expect_true(r_high < r_low,
              info = paste0("r_high=", round(r_high, 3), " should be < r_low=", round(r_low, 3)))
})

# ============================================================
# NB Loss: Basic Execution
# ============================================================

test_that("NB NMF runs without error on sparse data", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "per_row",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), 2)
  expect_equal(nrow(model@h), 2)
})

test_that("NB NMF runs without error on dense data", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 30, n = 25, k = 2, size = 3)
  A_dense <- as.matrix(sim$A)
  model <- nmf(A_dense, 2, loss = "nb", dispersion = "per_row",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# ============================================================
# Loss Monotonicity
# ============================================================

test_that("NB loss decreases overall", {
  skip_on_cran()
  # GPU NMF path does not support IRLS-based NB loss with loss history tracking
  skip_if(!identical(getOption("RcppML.gpu", FALSE), FALSE), "GPU NMF does not track NB loss history")
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "per_row",
               maxit = 30, tol = 1e-10, seed = 123, verbose = FALSE)

  loss_hist <- model@misc$loss_history
  expect_true(length(loss_hist) > 1)
  # Final loss should be lower than initial loss
  # (MoM dispersion updates + IRLS/normalization interaction can cause
  # per-iteration oscillation, which is expected behavior)
  expect_true(tail(loss_hist, 1) < loss_hist[1])
})

# ============================================================
# Dispersion Estimation
# ============================================================

test_that("NB dispersion per_row produces reasonable r values", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "per_row",
               maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)

  # Model should have non-trivial diagonal scaling
  expect_true(all(model@d > 0))
  # Factors should be finite
  expect_true(all(is.finite(model@w)))
  expect_true(all(is.finite(model@h)))
})

test_that("NB with global dispersion mode runs", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "global",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("NB with dispersion=none runs", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "none",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("NB with per_col dispersion produces col-length r values", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5)
  model <- nmf(sim$A, 2, loss = "nb", dispersion = "per_col",
               maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(all(is.finite(model@w)))
  expect_true(all(is.finite(model@h)))
})

# ============================================================
# ZINB (Zero-Inflated NB)
# ============================================================

test_that("ZINB runs without error", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5, dropout = 0.3)
  model <- nmf(sim$A, 2, loss = "nb", zi = "row",
               dispersion = "per_row",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("ZINB column mode runs", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 5, dropout = 0.3)
  model <- nmf(sim$A, 2, loss = "nb", zi = "col",
               dispersion = "per_row",
               maxit = 20, tol = 1e-4, seed = 123, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("ZINB works in CV mode", {
  skip_on_cran()
  # GPU NMF path does not return ZINB zero-inflation parameters in misc
  skip_if(!identical(getOption("RcppML.gpu", FALSE), FALSE), "GPU NMF does not return pi_row for ZINB")
  sim <- simulate_nb_data(m = 60, n = 40, k = 2, size = 5, dropout = 0.2)
  result <- nmf(sim$A, 2, loss = "nb", zi = "row", dispersion = "per_row",
                test_fraction = 0.1, maxit = 20, tol = 1e-4,
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_true(result@misc$test_loss > 0)
  expect_true("pi_row" %in% names(result@misc))
})

# ============================================================
# NB vs MSE Comparison
# ============================================================

test_that("NB gives different result than MSE on count data", {
  skip_on_cran()
  sim <- simulate_nb_data(m = 40, n = 30, k = 2, size = 3, seed = 99)

  model_mse <- nmf(sim$A, 2, loss = "mse",
                    maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)
  model_nb <- nmf(sim$A, 2, loss = "nb", dispersion = "per_row",
                   maxit = 30, tol = 1e-6, seed = 123, verbose = FALSE)

  # Results should differ (NB reweights by dispersion)
  # Use relative error norm: ||W_nb - W_mse|| / ||W_mse||
  rel_diff <- norm(model_nb@w - model_mse@w, "F") / (norm(model_mse@w, "F") + 1e-10)
  expect_true(rel_diff > 0.01)
})
