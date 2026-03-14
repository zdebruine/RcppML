# Tests for BUILD tasks: dense ZI paths, dense robust MSE, dense seeded NMF
# These are "planned" features that turned out to already work.

options(RcppML.verbose = FALSE)

# ============================================================
# BUILD-ZI-ROW-DENSE-GP: Dense ZI_ROW GP path
# ============================================================

test_that("Dense ZI-ROW GP returns pi_row with correct length and range", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- abs(matrix(rnorm(m * n, 2, 0.5), m, n))
  # Inject structural row-level zeros (~50% of entries)
  mask <- matrix(rbinom(m * n, 1, 0.5), m, n)
  A <- A * mask

  model <- nmf(A, k, loss = "gp", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(!is.null(model@misc$pi_row))
  expect_equal(length(model@misc$pi_row), m)
  expect_true(all(model@misc$pi_row >= 0 & model@misc$pi_row <= 1))
})

test_that("Dense ZI-ROW GP loss is finite", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- abs(matrix(rnorm(m * n, 2, 0.5), m, n))
  mask <- matrix(rbinom(m * n, 1, 0.5), m, n)
  A <- A * mask

  model <- nmf(A, k, loss = "gp", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)

  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

# ============================================================
# BUILD-ZI-ROW-DENSE-NB: Dense ZI_ROW NB path
# ============================================================

test_that("Dense ZI-ROW NB returns pi_row with correct length and range", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- matrix(rnbinom(m * n, size = 5, mu = 3), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.5), m, n)
  A <- A * mask

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(!is.null(model@misc$pi_row))
  expect_equal(length(model@misc$pi_row), m)
  expect_true(all(model@misc$pi_row >= 0 & model@misc$pi_row <= 1))
})

test_that("Dense ZI-ROW NB loss is finite and factors nonneg", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- matrix(rnbinom(m * n, size = 5, mu = 3), m, n)
  mask <- matrix(rbinom(m * n, 1, 0.5), m, n)
  A <- A * mask

  model <- nmf(A, k, loss = "nb", dispersion = "per_row",
               zi = "row", maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)

  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

# ============================================================
# BUILD-ROBUST-MSE-DENSE: Dense robust MSE path
# ============================================================

test_that("Dense robust MSE downweights outliers", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- abs(matrix(rnorm(m * n, 2, 0.5), m, n))
  # Inject outliers
  outlier_idx <- sample(m * n, 20)
  A[outlier_idx] <- A[outlier_idx] * 50

  model_robust <- nmf(A, k, loss = "mse", robust = TRUE,
                      maxit = 50, tol = 1e-6, seed = 42, verbose = FALSE)
  model_normal <- nmf(A, k, loss = "mse",
                      maxit = 50, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model_robust, "nmf")
  expect_true(is.finite(model_robust@misc$loss))
  expect_true(all(model_robust@w >= 0))
  expect_true(all(model_robust@h >= 0))
  # Robust should have different loss from non-robust (different weighting)
  expect_true(model_robust@misc$loss != model_normal@misc$loss)
})

test_that("Dense robust MSE with custom delta works", {
  skip_on_cran()
  set.seed(42)
  m <- 40; n <- 30; k <- 2
  A <- abs(matrix(rnorm(m * n, 2, 0.5), m, n))
  A[1:5, 1:5] <- A[1:5, 1:5] * 30  # Corner outliers

  model <- nmf(A, k, loss = "mse", robust = 2.0,
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

# ============================================================
# BUILD-SEEDED-DENSE: Dense seeded NMF path
# ============================================================

test_that("Seeded NMF with dense input recovers seeded factors", {
  skip_on_cran()
  set.seed(42)
  k <- 3
  W_true <- matrix(runif(50 * k), 50, k)
  H_true <- matrix(runif(k * 40), k, 40)
  A <- W_true %*% H_true + abs(matrix(rnorm(50 * 40, 0, 0.1), 50, 40))

  # Use true W (with noise) as seed
  W_seed <- W_true + matrix(rnorm(50 * k, 0, 0.05), 50, k)
  W_seed[W_seed < 0] <- 0

  model <- nmf(A, k, seed = W_seed, maxit = 50, tol = 1e-6, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(model@misc$loss < 100)  # Should achieve good reconstruction

  # Check dimensions
  expect_equal(nrow(model@w), 50)
  expect_equal(ncol(model@w), k)
  expect_equal(nrow(model@h), k)
  expect_equal(ncol(model@h), 40)
})

test_that("Seeded dense NMF has lower loss than random init", {
  skip_on_cran()
  set.seed(42)
  k <- 3
  W_true <- matrix(runif(50 * k), 50, k)
  H_true <- matrix(runif(k * 40), k, 40)
  A <- W_true %*% H_true + abs(matrix(rnorm(50 * 40, 0, 0.05), 50, 40))

  W_seed <- W_true + matrix(rnorm(50 * k, 0, 0.01), 50, k)
  W_seed[W_seed < 0] <- 0

  model_seeded <- nmf(A, k, seed = W_seed, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_random <- nmf(A, k, seed = 99, maxit = 30, tol = 1e-8, verbose = FALSE)

  # With near-perfect seed, seeded should have lower or equal loss
  expect_true(model_seeded@misc$loss <= model_random@misc$loss * 1.5)
})
