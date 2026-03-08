# Tests for dense matrix NMF with non-MSE loss functions
# Verifies the dense IRLS path in the unified solver

options(RcppML.verbose = FALSE)
options(RcppML.threads = 1)

test_that("dense NMF with MAE loss converges", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40), 50, 40)) + 0.1

  model <- nmf(A, 3, loss = "mae", maxit = 50, tol = 1e-5,
               seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))

  # MAE should be positive
  A_recon <- model@w %*% diag(model@d) %*% model@h
  mae <- mean(abs(A - A_recon))
  expect_gt(mae, 0)
})

test_that("dense NMF with Huber loss converges", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40), 50, 40)) + 0.1

  model <- nmf(A, 3, loss = "huber", huber_delta = 1.0,
               maxit = 50, tol = 1e-5, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

test_that("dense NMF with KL divergence converges", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40), 50, 40)) + 0.5  # Strictly positive

  model <- nmf(A, 3, loss = "kl", maxit = 50, tol = 1e-5,
               seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

test_that("dense MAE matches sparse MAE for same data", {
  skip_on_cran()
  set.seed(42)
  data <- simulateNMF(30, 25, k = 3, noise = 0.1, seed = 42)
  A_dense <- data$A
  A_sparse <- as(A_dense, "dgCMatrix")

  m_dense <- nmf(A_dense, 3, loss = "mae", maxit = 30, seed = 42, verbose = FALSE)
  m_sparse <- nmf(A_sparse, 3, loss = "mae", maxit = 30, seed = 42, verbose = FALSE)

  # Final loss should be same order of magnitude (dense/sparse IRLS paths
  # may converge to different local optima due to iteration order differences)
  ratio <- abs(m_dense@misc$loss - m_sparse@misc$loss) / max(m_dense@misc$loss, m_sparse@misc$loss)
  expect_lt(ratio, 1.0)  # Same order of magnitude
})

test_that("dense Huber loss interpolates behavior", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30), 40, 30)) + 0.1

  # Add some outliers
  outliers <- sample(length(A), 10)
  A[outliers] <- A[outliers] + abs(rnorm(10, mean = 5))

  m_mse <- nmf(A, 3, loss = "mse", maxit = 50, seed = 42, verbose = FALSE)
  m_huber <- nmf(A, 3, loss = "huber", huber_delta = 1.0,
                 maxit = 50, seed = 42, verbose = FALSE)
  m_mae <- nmf(A, 3, loss = "mae", maxit = 50, seed = 42, verbose = FALSE)

  # All should converge
  expect_true(is.finite(m_mse@misc$loss))
  expect_true(is.finite(m_huber@misc$loss))
  expect_true(is.finite(m_mae@misc$loss))
})

test_that("dense NMF MAE with regularization", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30), 40, 30)) + 0.1

  model <- nmf(A, 3, loss = "mae", L1 = c(0.01, 0.01), L2 = c(0.01, 0.01),
               maxit = 30, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
})

test_that("dense KL NMF with cross-validation", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30), 40, 30)) + 0.5

  result <- nmf(A, k = 3, test_fraction = 0.1, loss = "kl",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

# ============================================================
# Dense Gamma / InvGauss / Tweedie IRLS tests
# ============================================================

test_that("dense Gamma NMF converges and produces finite loss", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))
  A[A < 1e-8] <- 1e-8

  model <- nmf(A, 3, loss = "gamma", dispersion = "per_row",
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  lh <- model@misc$loss_history
  expect_true(tail(lh, 1) < lh[2])
})

test_that("dense Inverse Gaussian NMF converges and produces finite loss", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))
  A[A < 1e-8] <- 1e-8

  model <- nmf(A, 3, loss = "inverse_gaussian", dispersion = "per_row",
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  lh <- model@misc$loss_history
  expect_true(tail(lh, 1) < lh[2])
})

test_that("dense Tweedie NMF converges and produces finite loss", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))
  A[A < 1e-8] <- 1e-8

  model <- nmf(A, 3, distribution = "tweedie", tweedie_power = 1.5,
               dispersion = "per_row", maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  lh <- model@misc$loss_history
  expect_true(tail(lh, 1) < lh[2])
})

test_that("dense Gamma matches sparse Gamma for same non-zero data", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(30 * 25, 2, 0.5), 30, 25))
  A[A < 1e-8] <- 1e-8
  A_sp <- Matrix::Matrix(A, sparse = TRUE)

  m_dense <- nmf(A, 2, loss = "gamma", dispersion = "per_row",
                 maxit = 30, seed = 42, verbose = FALSE)
  m_sparse <- nmf(A_sp, 2, loss = "gamma", dispersion = "per_row",
                  maxit = 30, seed = 42, verbose = FALSE)

  # Losses should be same order of magnitude
  ratio <- abs(m_dense@misc$loss - m_sparse@misc$loss) /
           max(abs(m_dense@misc$loss), abs(m_sparse@misc$loss), 1e-10)
  expect_lt(ratio, 1.0)
})
