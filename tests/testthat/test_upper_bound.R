library(testthat)
library(RcppML)
library(Matrix)

# =============================================================================
# S3: Test upper_bound parameter
# =============================================================================

test_that("upper_bound clamps factor values for W", {
  skip_on_cran()
  set.seed(42)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  # Without upper bound
  result_no_bound <- nmf(A, k = 3, tol = 1e-4, maxit = 100, verbose = FALSE,
                         upper_bound = c(0, 0))

  # With upper bound on W
  result_bounded <- nmf(A, k = 3, tol = 1e-4, maxit = 100, verbose = FALSE,
                        upper_bound = c(1.0, 0))

  # W values should be <= 1.0 when bounded
  expect_true(all(result_bounded@w <= 1.0 + 1e-10),
              label = "All W values should be <= upper_bound")

  # Bounded max should be <= unbounded max (bounding restricts values)
  expect_lte(max(result_bounded@w), max(result_no_bound@w) + 1e-10)
})

test_that("upper_bound clamps factor values for H", {
  skip_on_cran()
  set.seed(99)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  result_bounded <- nmf(A, k = 3, tol = 1e-4, maxit = 100, verbose = FALSE,
                        upper_bound = c(0, 0.5))

  # H values should be <= 0.5 when bounded
  expect_true(all(result_bounded@h <= 0.5 + 1e-10),
              label = "All H values should be <= upper_bound")
})

test_that("upper_bound = 0 means no bound (default behavior)", {
  skip_on_cran()
  set.seed(123)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  result1 <- nmf(A, k = 3, seed = 42, tol = 1e-4, maxit = 50, verbose = FALSE,
                 upper_bound = c(0, 0))
  result2 <- nmf(A, k = 3, seed = 42, tol = 1e-4, maxit = 50, verbose = FALSE)

  # Should give identical results (0 = no bound, which is the default)
  expect_equal(result1@w, result2@w, tolerance = 1e-10)
  expect_equal(result1@h, result2@h, tolerance = 1e-10)
})

test_that("upper_bound on both W and H simultaneously", {
  skip_on_cran()
  set.seed(456)
  A <- rsparsematrix(40, 30, 0.3)
  A <- abs(A)

  result <- nmf(A, k = 3, tol = 1e-4, maxit = 100, verbose = FALSE,
                upper_bound = c(2.0, 2.0))

  expect_true(all(result@w <= 2.0 + 1e-10))
  expect_true(all(result@h <= 2.0 + 1e-10))
})

test_that("upper_bound affects convergence and loss", {
  skip_on_cran()
  set.seed(789)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  result_free <- nmf(A, k = 3, seed = 42, tol = 1e-8, maxit = 200, verbose = FALSE)
  result_tight <- nmf(A, k = 3, seed = 42, tol = 1e-8, maxit = 200, verbose = FALSE,
                      upper_bound = c(0.1, 0.1))

  # Tight bounds should give higher (worse) loss than unconstrained
  loss_free <- evaluate(result_free, A, loss = "mse")
  loss_tight <- evaluate(result_tight, A, loss = "mse")
  expect_gt(loss_tight, loss_free)
})
