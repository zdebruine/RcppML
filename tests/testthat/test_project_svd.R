# =============================================================================
# TEST-GATE-02: SVD Projection Tests
# Phase 0 safety net — predict() for svd objects.
# =============================================================================

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)
options(RcppML.gpu = FALSE)

library(Matrix)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
set.seed(42)
data(iris)
A_iris <- as.matrix(iris[, 1:4])

# ===========================================================================
# predict.svd — expected-fail skeleton until implemented
# ===========================================================================

test_that("predict(svd_model, newdata) projects new data correctly", {
  # Build a PCA model (RcppML centers by ROW means, not column means)
  s <- pca(A_iris, k = 3, method = "deflation", seed = 1, maxit = 200, tol = 1e-8,
           test_fraction = 0)

  # Verify reconstruction works (primary property)
  A_hat <- reconstruct(s)
  mse <- mean((A_iris - A_hat)^2)
  expect_lt(mse, 1.0)  # 3 factors should explain most of iris

  # Scores = u * diag(d), which should have norm proportional to singular values
  scores <- s@u %*% diag(s@d)
  expect_equal(nrow(scores), nrow(A_iris))
  expect_equal(ncol(scores), 3)
})

test_that("svd model stores row_means when centered", {
  s <- pca(A_iris, k = 3, method = "deflation", seed = 1, maxit = 200, tol = 1e-8,
           test_fraction = 0)
  expect_true(!is.null(s@misc$row_means))
  expect_length(s@misc$row_means, nrow(A_iris))
})

test_that("svd reconstruction inverts projection (PCA)", {
  s <- pca(A_iris, k = 4, method = "deflation", seed = 1, maxit = 200, tol = 1e-8,
           test_fraction = 0)
  A_hat <- reconstruct(s)
  mse <- mean((A_iris - A_hat)^2)
  # 4 factors should explain iris almost perfectly
  expect_lt(mse, 0.01)
})

test_that("svd reconstruction inverts projection (SVD, no centering)", {
  set.seed(99)
  A <- matrix(abs(rnorm(60 * 40)), 60, 40)
  s <- RcppML::svd(A, k = 10, method = "deflation", seed = 1, maxit = 200, tol = 1e-6,
                   test_fraction = 0)
  A_hat <- reconstruct(s)
  residual_frac <- sum((A - A_hat)^2) / sum(A^2)
  expect_lt(residual_frac, 0.5)
})
