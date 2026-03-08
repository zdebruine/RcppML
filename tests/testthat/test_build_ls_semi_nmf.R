# Tests for BUILD tasks: Unconstrained LS and Semi-NMF
# These features were already implemented via the nonneg flag in the NNLS solver.

options(RcppML.verbose = FALSE)

# ============================================================
# BUILD-LS-CPU-SPARSE: Unconstrained LS with sparse input
# ============================================================

test_that("Unconstrained LS achieves lower loss than NNLS on mixed-sign data (sparse)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(rnorm(k * n), k, n)
  A_dense <- W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n)
  A <- Matrix::Matrix(A_dense, sparse = TRUE)

  model_ls <- nmf(A, k, nonneg = c(FALSE, FALSE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  model_nnls <- nmf(A, k, nonneg = c(TRUE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_true(model_ls@misc$loss < model_nnls@misc$loss)
})

test_that("Unconstrained LS factors have negative entries (sparse)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(rnorm(k * n), k, n)
  A <- Matrix::Matrix(W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n), sparse = TRUE)

  model <- nmf(A, k, nonneg = c(FALSE, FALSE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(any(model@w < 0), info = "W should have negative entries for unconstrained LS")
  expect_true(any(model@h < 0), info = "H should have negative entries for unconstrained LS")
  expect_true(is.finite(model@misc$loss))
})

# ============================================================
# BUILD-LS-CPU-DENSE: Unconstrained LS with dense input
# ============================================================

test_that("Unconstrained LS achieves lower loss than NNLS on mixed-sign data (dense)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(rnorm(k * n), k, n)
  A <- W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n)

  model_ls <- nmf(A, k, nonneg = c(FALSE, FALSE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  model_nnls <- nmf(A, k, nonneg = c(TRUE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_true(model_ls@misc$loss < model_nnls@misc$loss)
})

test_that("Unconstrained LS factors have negative entries (dense)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(rnorm(k * n), k, n)
  A <- W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n)

  model <- nmf(A, k, nonneg = c(FALSE, FALSE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(any(model@w < 0))
  expect_true(any(model@h < 0))
  expect_true(is.finite(model@misc$loss))
})

test_that("Unconstrained LS loss decreases over iterations (dense)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  A <- matrix(rnorm(m * n), m, n)

  model <- nmf(A, k, nonneg = c(FALSE, FALSE), maxit = 20, tol = 1e-10, seed = 42, verbose = FALSE)

  lh <- model@misc$loss_history
  expect_true(length(lh) > 1)
  # Overall loss should decrease
  expect_true(tail(lh, 1) < lh[1])
})

# ============================================================
# BUILD-SEMI-NMF-CPU-SPARSE: Semi-NMF with sparse input
# ============================================================

test_that("Semi-NMF produces unconstrained W and nonneg H (sparse)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- Matrix::Matrix(W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n), sparse = TRUE)

  model <- nmf(A, k, nonneg = c(FALSE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(any(model@w < 0), info = "W should have negative entries in Semi-NMF")
  expect_true(all(model@h >= 0), info = "H should be all non-negative in Semi-NMF")
  expect_true(is.finite(model@misc$loss))
})

test_that("Semi-NMF loss converges (sparse)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  A <- Matrix::Matrix(matrix(rnorm(m * n, 2, 1), m, n), sparse = TRUE)

  model <- nmf(A, k, nonneg = c(FALSE, TRUE), maxit = 30, tol = 1e-10, seed = 42, verbose = FALSE)

  lh <- model@misc$loss_history
  expect_true(length(lh) > 1)
  expect_true(tail(lh, 1) < lh[1])
})

# ============================================================
# BUILD-SEMI-NMF-CPU-DENSE: Semi-NMF with dense input
# ============================================================

test_that("Semi-NMF produces unconstrained W and nonneg H (dense)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n)

  model <- nmf(A, k, nonneg = c(FALSE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(any(model@w < 0))
  expect_true(all(model@h >= 0))
  expect_true(is.finite(model@misc$loss))
})

test_that("Semi-NMF loss converges (dense)", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  A <- matrix(rnorm(m * n, 2, 1), m, n)

  model <- nmf(A, k, nonneg = c(FALSE, TRUE), maxit = 30, tol = 1e-10, seed = 42, verbose = FALSE)

  lh <- model@misc$loss_history
  expect_true(length(lh) > 1)
  expect_true(tail(lh, 1) < lh[1])
})

test_that("Semi-NMF has lower loss than full NNLS on mixed-sign W data", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(rnorm(m * k), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true + matrix(rnorm(m * n, 0, 0.1), m, n)

  model_semi <- nmf(A, k, nonneg = c(FALSE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  model_nnls <- nmf(A, k, nonneg = c(TRUE, TRUE), maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)

  expect_true(model_semi@misc$loss < model_nnls@misc$loss)
})
