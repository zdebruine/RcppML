# Test: Non-MSE Distribution Losses (Gamma, InverseGaussian, Tweedie)
# Tests correctness of NMF fitting under distribution-based losses

options(RcppML.verbose = FALSE)

# Shared positive data generator
make_positive_data <- function(m = 60, n = 40, k = 3, seed = 42) {
  set.seed(seed)
  W <- matrix(abs(rnorm(m * k, 2, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 2, 0.5)), k, n)
  mu <- W %*% H
  # Gamma-distributed observations
  phi <- 0.3
  A <- matrix(rgamma(m * n, shape = 1 / phi, rate = 1 / (phi * mu)), m, n)
  A[A < 1e-8] <- 1e-8
  A
}

# ============================================================================
# Gamma Loss
# ============================================================================

test_that("Gamma NMF converges and produces nonneg factors", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  model <- nmf(A, 2, loss = "gamma", dispersion = "per_row",
               maxit = 50, tol = 1e-5, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(all(model$w >= 0))
  expect_true(all(model$h >= 0))
  expect_true(is.finite(model@misc$loss))
})

test_that("Gamma NMF loss decreases over iterations", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  # Fit with limited iterations at different stopping points
  m5  <- nmf(A, 2, loss = "gamma", maxit = 5,  tol = 1e-10, seed = 1, verbose = FALSE)
  m20 <- nmf(A, 2, loss = "gamma", maxit = 20, tol = 1e-10, seed = 1, verbose = FALSE)
  m50 <- nmf(A, 2, loss = "gamma", maxit = 50, tol = 1e-10, seed = 1, verbose = FALSE)

  # Loss should decrease or stay flat
  expect_lte(m20@misc$loss, m5@misc$loss + 1e-3)
  expect_lte(m50@misc$loss, m20@misc$loss + 1e-3)
})

test_that("Gamma NMF with global dispersion", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  model <- nmf(A, 2, loss = "gamma", dispersion = "global",
               maxit = 30, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(!is.null(model@misc$dispersion))
})

# ============================================================================
# Inverse Gaussian Loss
# ============================================================================

test_that("InverseGaussian NMF converges and produces nonneg factors", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  model <- nmf(A, 2, loss = "inverse_gaussian", dispersion = "per_row",
               maxit = 50, tol = 1e-5, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(all(model$w >= 0))
  expect_true(all(model$h >= 0))
  expect_true(is.finite(model@misc$loss))
})

test_that("InverseGaussian NMF loss decreases over iterations", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  m5  <- nmf(A, 2, loss = "inverse_gaussian", maxit = 5,  tol = 1e-10, seed = 1, verbose = FALSE)
  m50 <- nmf(A, 2, loss = "inverse_gaussian", maxit = 50, tol = 1e-10, seed = 1, verbose = FALSE)

  expect_lte(m50@misc$loss, m5@misc$loss + 1e-3)
})

# ============================================================================
# Tweedie Loss
# ============================================================================

test_that("Tweedie NMF converges with power=1.5", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  model <- nmf(A, 2, loss = "tweedie", tweedie_power = 1.5,
               maxit = 50, tol = 1e-5, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(all(model$w >= 0))
  expect_true(all(model$h >= 0))
  expect_true(is.finite(model@misc$loss))
})

test_that("Tweedie NMF loss decreases over iterations", {
  skip_on_cran()
  A <- make_positive_data(m = 50, n = 30, k = 2)

  m5  <- nmf(A, 2, loss = "tweedie", tweedie_power = 1.5,
             maxit = 5,  tol = 1e-10, seed = 1, verbose = FALSE)
  m50 <- nmf(A, 2, loss = "tweedie", tweedie_power = 1.5,
             maxit = 50, tol = 1e-10, seed = 1, verbose = FALSE)

  expect_lte(m50@misc$loss, m5@misc$loss + 1e-3)
})

# ============================================================================
# Sparse Input with Non-MSE Losses
# ============================================================================

test_that("GP loss works on sparse (dgCMatrix) input", {
  skip_on_cran()
  set.seed(42)
  A <- abs(rsparsematrix(60, 40, density = 0.3))

  model <- nmf(A, 3, loss = "gp", dispersion = "per_row",
               maxit = 30, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(all(model$w >= 0))
})

test_that("NB loss works on sparse input", {
  skip_on_cran()
  set.seed(42)
  A <- abs(rsparsematrix(60, 40, density = 0.3))

  model <- nmf(A, 3, loss = "nb", dispersion = "per_row",
               maxit = 30, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(!is.null(model@misc$theta))
})

test_that("Gamma loss works on dense input", {
  skip_on_cran()
  A <- make_positive_data(m = 40, n = 30, k = 2)

  model <- nmf(A, 2, loss = "gamma", dispersion = "global",
               maxit = 30, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
})

# ============================================================================
# Reproducibility Across Losses
# ============================================================================

test_that("GP NMF is reproducible with same seed", {
  skip_on_cran()
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3))

  m1 <- nmf(A, 2, loss = "gp", maxit = 20, seed = 99, verbose = FALSE)
  m2 <- nmf(A, 2, loss = "gp", maxit = 20, seed = 99, verbose = FALSE)

  expect_equal(m1$w, m2$w)
  expect_equal(m1$h, m2$h)
})

test_that("Gamma NMF is reproducible with same seed", {
  skip_on_cran()
  A <- make_positive_data(m = 40, n = 30, k = 2)

  m1 <- nmf(A, 2, loss = "gamma", maxit = 20, seed = 99, verbose = FALSE)
  m2 <- nmf(A, 2, loss = "gamma", maxit = 20, seed = 99, verbose = FALSE)

  expect_equal(m1$w, m2$w)
  expect_equal(m1$h, m2$h)
})
