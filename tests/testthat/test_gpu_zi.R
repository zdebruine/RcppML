# Test: GPU Zero-Inflated NMF (ZIGP and ZINB)
#
# GPU ZI is currently a no-op: the GPU backend accepts the zi parameter but
# ignores it (produces identical results to non-ZI, and does not return
# pi_row/pi_col in misc).  ZI twoway GPU has been fixed with EMA damping
# (FIX-ZI-TWOWAY-GPU) — it no longer produces NaN on high-sparsity data.
#
# These tests verify:
#  1. GPU accepts zi= parameter without error for all combinations
#  2. GPU produces valid (finite, non-negative) factors
#  3. CPU ZI works correctly (produces pi vectors, different loss from non-ZI)
# Skipped automatically if GPU is not available.

options(RcppML.verbose = FALSE)

# Helper: simulate count data with dropout (zero inflation)
simulate_zi_data <- function(m = 80, n = 60, k = 3, theta = 0.5,
                             dropout = 0.2, seed = 42) {
  set.seed(seed)
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  W <- W %*% diag(1 / colSums(W))
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  size <- pmax(mu / max(theta, 0.01), 0.1)
  A <- matrix(rnbinom(m * n, size = size, mu = mu), m, n)
  if (dropout > 0) {
    mask <- matrix(rbinom(m * n, 1, 1 - dropout), m, n)
    A <- A * mask
  }
  Matrix::Matrix(A, sparse = TRUE)
}

skip_if_no_gpu <- function() {
  skip_if_not(
    exists("gpu_available", where = asNamespace("RcppML")) &&
      RcppML:::gpu_available(),
    "GPU not available"
  )
}

# ============================================================
# ZIGP: GPU smoke tests (GPU accepts zi= without crashing)
# ============================================================

test_that("GPU ZIGP-row accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 60, n = 40, k = 3, dropout = 0.2)
  model <- nmf(A, 3, loss = "gp", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(is.finite(model@w)))
  expect_true(all(model@w >= 0))
  expect_true(all(is.finite(model@h)))
  expect_true(all(model@h >= 0))
})

test_that("GPU ZIGP-col accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 50, n = 35, k = 2, dropout = 0.15)
  model <- nmf(A, 2, loss = "gp", dispersion = "per_row",
               zi = "col", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("GPU ZIGP-twoway accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 60, n = 40, k = 3, dropout = 0.2)
  model <- nmf(A, 3, loss = "gp", dispersion = "per_row",
               zi = "twoway", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

# ============================================================
# ZIGP: CPU ZI correctness reference
# ============================================================

test_that("CPU ZIGP-row returns pi_row and has different loss than non-ZI", {
  skip_on_cran()

  A <- simulate_zi_data(m = 60, n = 40, k = 3, dropout = 0.2)

  cpu_zi <- nmf(A, 3, loss = "gp", dispersion = "per_row",
                zi = "row", maxit = 20, tol = 1e-10,
                seed = 42, resource = "cpu", verbose = FALSE)
  cpu_no_zi <- nmf(A, 3, loss = "gp", dispersion = "per_row",
                   maxit = 20, tol = 1e-10,
                   seed = 42, resource = "cpu", verbose = FALSE)

  expect_s4_class(cpu_zi, "nmf")
  expect_true("pi_row" %in% names(cpu_zi@misc))
  pi_vals <- cpu_zi@misc$pi_row
  expect_length(pi_vals, nrow(A))
  expect_true(all(pi_vals >= 0 & pi_vals <= 1))
  # ZI should produce different loss than non-ZI
  expect_false(isTRUE(all.equal(cpu_zi@misc$loss, cpu_no_zi@misc$loss)))
})

# ============================================================
# ZINB: GPU smoke tests
# ============================================================

test_that("GPU ZINB-row accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 60, n = 40, k = 3, dropout = 0.2)
  model <- nmf(A, 3, loss = "nb", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(is.finite(model@w)))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("GPU ZINB-col accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 50, n = 35, k = 2, dropout = 0.15)
  model <- nmf(A, 2, loss = "nb", dispersion = "per_row",
               zi = "col", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("GPU ZINB-twoway accepts zi param and produces valid factors", {
  skip_on_cran()
  skip_if_no_gpu()

  A <- simulate_zi_data(m = 50, n = 35, k = 2, dropout = 0.15)
  model <- nmf(A, 2, loss = "nb", dispersion = "per_row",
               zi = "twoway", maxit = 20, tol = 1e-4,
               seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

# ============================================================
# ZINB: CPU ZI correctness reference
# ============================================================

test_that("CPU ZINB-row returns pi_row and has different loss than non-ZI", {
  skip_on_cran()

  A <- simulate_zi_data(m = 60, n = 40, k = 3, dropout = 0.2)

  cpu_zi <- nmf(A, 3, loss = "nb", dispersion = "per_row",
                zi = "row", maxit = 20, tol = 1e-10,
                seed = 42, resource = "cpu", verbose = FALSE)
  cpu_no_zi <- nmf(A, 3, loss = "nb", dispersion = "per_row",
                   maxit = 20, tol = 1e-10,
                   seed = 42, resource = "cpu", verbose = FALSE)

  expect_s4_class(cpu_zi, "nmf")
  expect_true("pi_row" %in% names(cpu_zi@misc))
  pi_vals <- cpu_zi@misc$pi_row
  expect_length(pi_vals, nrow(A))
  expect_true(all(pi_vals >= 0 & pi_vals <= 1))
  expect_false(isTRUE(all.equal(cpu_zi@misc$loss, cpu_no_zi@misc$loss)))
})

# ============================================================
# GPU ZI Twoway Tests (FIX-ZI-TWOWAY-GPU — EMA damping)
# ============================================================

test_that("GPU ZI twoway GP doesn't produce NaN on 95%-sparse matrix", {
  skip_if_no_gpu()

  set.seed(42)
  m <- 300; n <- 150; density <- 0.05
  A <- Matrix::rsparsematrix(m, n, density = density)
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result <- nmf(A, k = 5, loss = "gp", zi = "twoway", seed = 42,
                resource = "gpu", maxit = 30, tol = 1e-10, verbose = FALSE)

  expect_true(is.finite(result@misc$loss))
  expect_true(all(is.finite(result@w)))
  expect_true(all(is.finite(result@h)))
  expect_true(all(result@w >= 0))
  expect_true(all(result@h >= 0))
})

test_that("GPU ZI twoway NB doesn't produce NaN on 95%-sparse matrix", {
  skip_if_no_gpu()

  set.seed(42)
  m <- 300; n <- 150; density <- 0.05
  A <- Matrix::rsparsematrix(m, n, density = density)
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result <- nmf(A, k = 5, loss = "nb", zi = "twoway", seed = 42,
                resource = "gpu", maxit = 30, tol = 1e-10, verbose = FALSE)

  expect_true(is.finite(result@misc$loss))
  expect_true(all(is.finite(result@w)))
  expect_true(all(is.finite(result@h)))
  expect_true(all(result@w >= 0))
  expect_true(all(result@h >= 0))
})
