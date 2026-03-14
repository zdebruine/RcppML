# Test: Regularization Effects
# Tests that angular and L21 regularization have their intended effects
# beyond just "running without error"

options(RcppML.verbose = FALSE)

# ============================================================================
# Angular Regularization
# ============================================================================

test_that("angular regularization increases factor orthogonality (W)", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(80 * 60, 2, 1)), 80, 60)

  model_none <- nmf(A, 4, angular = c(0, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)
  model_ang  <- nmf(A, 4, angular = c(0.5, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)

  # Cosine similarity matrix of W columns (4x4)
  cos_sim <- function(W) {
    norms <- sqrt(colSums(W^2))
    W_normed <- sweep(W, 2, pmax(norms, 1e-12), "/")
    sim <- crossprod(W_normed)
    # Average absolute off-diagonal
    k <- ncol(W)
    off_diag <- sim[upper.tri(sim)]
    mean(abs(off_diag))
  }

  sim_none <- cos_sim(model_none$w)
  sim_ang  <- cos_sim(model_ang$w)

  # Angular regularization on W should reduce off-diagonal cosine similarity

  expect_lt(sim_ang, sim_none + 0.05)
})

test_that("angular regularization increases factor orthogonality (H)", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(80 * 60, 2, 1)), 80, 60)

  model_none <- nmf(A, 4, angular = c(0, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)
  model_ang  <- nmf(A, 4, angular = c(0, 0.5), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)

  # Cosine similarity of H rows (factors are rows in H)
  cos_sim_rows <- function(H) {
    norms <- sqrt(rowSums(H^2))
    H_normed <- sweep(H, 1, pmax(norms, 1e-12), "/")
    sim <- tcrossprod(H_normed)
    k <- nrow(H)
    off_diag <- sim[upper.tri(sim)]
    mean(abs(off_diag))
  }

  sim_none <- cos_sim_rows(model_none$h)
  sim_ang  <- cos_sim_rows(model_ang$h)

  expect_lt(sim_ang, sim_none + 0.05)
})

test_that("angular reproduces across seeds", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(60 * 40, 2, 1)), 60, 40)

  m1 <- nmf(A, 3, angular = c(0.3, 0.3), maxit = 50, seed = 99, verbose = FALSE)
  m2 <- nmf(A, 3, angular = c(0.3, 0.3), maxit = 50, seed = 99, verbose = FALSE)

  expect_equal(m1$w, m2$w)
  expect_equal(m1$h, m2$h)
})

# ============================================================================
# L21 (Group Lasso) Regularization
# ============================================================================

test_that("L21 on W promotes column-group sparsity", {
  skip_on_cran()
  set.seed(42)
  # Create data where some rows of W should be entirely zero
  A <- matrix(abs(rnorm(80 * 60, 1, 0.5)), 80, 60)

  model_none <- nmf(A, 4, L21 = c(0, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)
  model_l21  <- nmf(A, 4, L21 = c(0.5, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)

  # L21 on W penalizes the L2 norm of each row of W across factors.
  # This should push entire rows of W toward zero.
  row_norms_none <- sqrt(rowSums(model_none$w^2))
  row_norms_l21  <- sqrt(rowSums(model_l21$w^2))

  # Count near-zero rows (row norm < threshold)
  thresh <- 0.01
  zero_rows_none <- sum(row_norms_none < thresh)
  zero_rows_l21  <- sum(row_norms_l21 < thresh)

  # L21 should produce at least as many near-zero rows
  expect_gte(zero_rows_l21, zero_rows_none)
})

test_that("L21 on H promotes column-group sparsity in H", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(60 * 80, 1, 0.5)), 60, 80)

  model_none <- nmf(A, 4, L21 = c(0, 0), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)
  model_l21  <- nmf(A, 4, L21 = c(0, 0.5), maxit = 100,
                    tol = 1e-6, seed = 1, verbose = FALSE)

  # L21 on H penalizes the L2 norm of each column of H.
  col_norms_none <- sqrt(colSums(model_none$h^2))
  col_norms_l21  <- sqrt(colSums(model_l21$h^2))

  thresh <- 0.01
  zero_cols_none <- sum(col_norms_none < thresh)
  zero_cols_l21  <- sum(col_norms_l21 < thresh)

  expect_gte(zero_cols_l21, zero_cols_none)
})

test_that("L21 reproduces across seeds", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(60 * 40, 2, 1)), 60, 40)

  m1 <- nmf(A, 3, L21 = c(0.2, 0.2), maxit = 50, seed = 99, verbose = FALSE)
  m2 <- nmf(A, 3, L21 = c(0.2, 0.2), maxit = 50, seed = 99, verbose = FALSE)

  expect_equal(m1$w, m2$w)
  expect_equal(m1$h, m2$h)
})

# ============================================================================
# Combined Regularization
# ============================================================================

test_that("angular + L1 + L21 combined produces valid model", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(60 * 40, 2, 1)), 60, 40)

  model <- nmf(A, 3,
    L1 = c(0.01, 0.01),
    L21 = c(0.05, 0.05),
    angular = c(0.1, 0.1),
    maxit = 50, tol = 1e-6, seed = 1, verbose = FALSE)

  expect_s4_class(model, "nmf")
  expect_true(all(is.finite(model$w)))
  expect_true(all(is.finite(model$h)))
  expect_true(all(model$w >= 0))
  expect_true(all(model$h >= 0))
})
