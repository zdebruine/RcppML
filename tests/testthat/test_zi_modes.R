# Test: Zero-Inflation modes across distributions
# Tests ZI row/col for GP and NB on CPU

options(RcppML.verbose = FALSE)

# ============================================================
# ZI_COL GP tests
# ============================================================

test_that("ZIGP col returns pi_col with correct length and range", {
  skip_on_cran()
  set.seed(42)
  m <- 60; n <- 40; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.3)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.3)), k, n)
  mu <- W %*% H
  A <- matrix(rpois(m * n, lambda = pmax(mu, 0.01) * 5), m, n)
  # Add structural zeros (20% dropout)
  mask <- matrix(rbinom(m * n, 1, 0.8), m, n)
  A <- A * mask
  A <- Matrix::Matrix(A, sparse = TRUE)

  model <- nmf(A, k, loss = "gp", dispersion = "per_row",
               zi = "col", maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true("pi_col" %in% names(model@misc))
  pi_col <- model@misc$pi_col
  expect_length(pi_col, n)
  expect_true(all(pi_col >= 0 & pi_col <= 1))
  # With dropout, pi should be non-trivial (not all 0 or all 1)
  expect_true(mean(pi_col) > 0.01, info = "pi_col should detect some zero inflation")
})

test_that("ZIGP col loss converges", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.3)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.3)), k, n)
  mu <- W %*% H
  A <- matrix(rpois(m * n, lambda = pmax(mu, 0.01) * 5), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.7), m, n)
  A <- Matrix::Matrix(A * mask, sparse = TRUE)

  model <- nmf(A, k, loss = "gp", dispersion = "per_row",
               zi = "col", maxit = 40, tol = 0, seed = 42, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  # Loss should improve overall (final < initial)
  expect_true(tail(lh, 1) < lh[1],
              info = paste0("ZIGP col: final=", tail(lh, 1), " not < initial=", lh[1]))
})

# ============================================================
# ZI_ROW NB tests
# ============================================================

test_that("ZINB row returns pi_row with correct length and range", {
  skip_on_cran()
  set.seed(42)
  m <- 60; n <- 40; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- matrix(rnbinom(m * n, size = 5, mu = mu), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.8), m, n)
  A <- Matrix::Matrix(A * mask, sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "row", maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true("pi_row" %in% names(model@misc))
  pi_row <- model@misc$pi_row
  expect_length(pi_row, m)
  expect_true(all(pi_row >= 0 & pi_row <= 1))
  expect_true(mean(pi_row) > 0.01, info = "pi_row should detect some zero inflation")
})

test_that("ZINB row loss converges", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- matrix(rnbinom(m * n, size = 5, mu = mu), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.7), m, n)
  A <- Matrix::Matrix(A * mask, sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "row", maxit = 40, tol = 0, seed = 42, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  expect_true(tail(lh, 1) < lh[1])
})

# ============================================================
# ZI_COL NB tests
# ============================================================

test_that("ZINB col returns pi_col with correct length and range", {
  skip_on_cran()
  set.seed(42)
  m <- 60; n <- 40; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- matrix(rnbinom(m * n, size = 5, mu = mu), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.8), m, n)
  A <- Matrix::Matrix(A * mask, sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "col", maxit = 30, tol = 1e-4,
               seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true("pi_col" %in% names(model@misc))
  pi_col <- model@misc$pi_col
  expect_length(pi_col, n)
  expect_true(all(pi_col >= 0 & pi_col <= 1))
  expect_true(mean(pi_col) > 0.01, info = "pi_col should detect some zero inflation")
})

test_that("ZINB col loss converges", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 35; k <- 2
  W <- matrix(abs(rnorm(m * k, 1, 0.5)), m, k)
  H <- matrix(abs(rnorm(k * n, 1, 0.5)), k, n)
  mu <- W %*% H
  A <- matrix(rnbinom(m * n, size = 5, mu = mu), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.7), m, n)
  A <- Matrix::Matrix(A * mask, sparse = TRUE)

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "col", maxit = 40, tol = 0, seed = 42, verbose = FALSE)
  lh <- model@misc$loss_history
  expect_true(length(lh) >= 10)
  expect_true(tail(lh, 1) < lh[1])
})
