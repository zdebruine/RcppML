# Test: Cross-validation with IRLS distributions
# Tests CV (speckled + full mask) for GP, NB, Gamma, InvGauss, Tweedie, and dense CV

options(RcppML.verbose = FALSE)

# Helper: positive continuous data
make_pos_data <- function(m = 50, n = 35, k = 2, seed = 42) {
  set.seed(seed)
  A <- abs(matrix(rnorm(m * n, 2, 0.5), m, n))
  A[A < 1e-8] <- 1e-8
  A
}

# ============================================================
# Dense CV with MSE (speckled + full)
# ============================================================

test_that("Dense MSE CV speckled returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)
  model <- nmf(A, 3, loss = "mse", test_fraction = 0.1,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(model@misc$test_loss > 0)
})

test_that("Dense MSE CV full mask returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)
  model <- nmf(A, 3, loss = "mse", test_fraction = 0.1, mask_zeros = FALSE,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(model@misc$test_loss > 0)
})

# ============================================================
# GP CV full mask
# ============================================================

test_that("GP CV full mask returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.3)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.3)), k, n)
  mu <- W %*% H
  A <- Matrix::Matrix(matrix(rpois(m * n, lambda = pmax(mu, 0.01) * 5), m, n), sparse = TRUE)

  model <- nmf(A, k, loss = "gp", dispersion = "per_row",
               test_fraction = 0.1, mask_zeros = FALSE,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# NB CV (speckled + full)
# ============================================================

test_that("NB CV speckled returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- Matrix::Matrix(matrix(rnbinom(m * n, size = 5, mu = mu), m, n), sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("NB CV full mask returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- Matrix::Matrix(matrix(rnbinom(m * n, size = 5, mu = mu), m, n), sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               test_fraction = 0.1, mask_zeros = FALSE,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# Gamma CV
# ============================================================

test_that("Gamma CV speckled returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(50, 35, seed = 42)
  A_sp <- Matrix::Matrix(A, sparse = TRUE)

  model <- nmf(A_sp, 2, loss = "gamma", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# InvGauss CV
# ============================================================

test_that("InvGauss CV speckled returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(50, 35, seed = 42)
  A_sp <- Matrix::Matrix(A, sparse = TRUE)

  model <- nmf(A_sp, 2, loss = "inverse_gaussian", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# Tweedie CV
# ============================================================

test_that("Tweedie CV speckled returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(50, 35, seed = 42)
  A_sp <- Matrix::Matrix(A, sparse = TRUE)

  model <- nmf(A_sp, 2, distribution = "tweedie", tweedie_power = 1.5,
               dispersion = "per_row", test_fraction = 0.1,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# Dense CV with IRLS distributions
# ============================================================

test_that("Dense GP CV returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)
  # Make it count-like by rounding
  A <- round(A * 3)
  A[A < 0] <- 0

  model <- nmf(A, 2, loss = "gp", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Dense NB CV returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)
  A <- round(A * 3)
  A[A < 0] <- 0

  model <- nmf(A, 2, loss = "nb", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Dense Gamma CV returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)

  model <- nmf(A, 2, loss = "gamma", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Dense InvGauss CV returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)

  model <- nmf(A, 2, loss = "inverse_gaussian", dispersion = "per_row",
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Dense Tweedie CV returns valid test loss", {
  skip_on_cran()
  A <- make_pos_data(40, 30, seed = 42)

  model <- nmf(A, 2, distribution = "tweedie", tweedie_power = 1.5,
               dispersion = "per_row", test_fraction = 0.1,
               maxit = 30, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

# ============================================================
# Projective + Symmetric CV
# ============================================================

test_that("Projective NMF CV sparse returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  A <- Matrix::Matrix(abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40)), sparse = TRUE)
  model <- nmf(A, 3, loss = "mse", projective = TRUE,
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Projective NMF CV dense returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30))
  model <- nmf(A, 3, loss = "mse", projective = TRUE,
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Symmetric NMF CV sparse returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  m <- 40
  A_raw <- matrix(abs(rnorm(m * m, 1, 0.3)), m, m)
  A_sym <- (A_raw + t(A_raw)) / 2
  A_sp <- Matrix::Matrix(A_sym, sparse = TRUE)

  model <- nmf(A_sp, 3, loss = "mse", symmetric = TRUE,
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})

test_that("Symmetric NMF CV dense returns valid test loss", {
  skip_on_cran()
  set.seed(42)
  m <- 40
  A_raw <- matrix(abs(rnorm(m * m, 1, 0.3)), m, m)
  A_sym <- (A_raw + t(A_raw)) / 2

  model <- nmf(A_sym, 3, loss = "mse", symmetric = TRUE,
               test_fraction = 0.1, maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$test_loss))
})
