# Test predict method for nmf objects
# predict() projects w onto data, returning a new nmf object with projected h
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("predict works on sparse data", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- predict(model, A)
  
  expect_s4_class(result, "nmf")
  expect_true(is.matrix(result@h))
  expect_equal(ncol(result@h), ncol(A))
})

test_that("predict works on dense data", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- predict(model, A)
  
  expect_s4_class(result, "nmf")
  expect_true(is.matrix(result@h))
  expect_equal(ncol(result@h), ncol(A))
})

test_that("predict with L1 penalty induces sparsity", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result_no_penalty <- predict(model, A, L1 = 0)
  result_with_penalty <- predict(model, A, L1 = 0.5)
  
  expect_s4_class(result_with_penalty, "nmf")
  # Guard against NaN from numerical instability in edge cases
  expect_false(any(is.nan(result_no_penalty@h)))
  expect_false(any(is.nan(result_with_penalty@h)))
  sparsity_no <- sum(result_no_penalty@h == 0)
  sparsity_with <- sum(result_with_penalty@h == 0)
  expect_true(sparsity_with >= sparsity_no)
})

test_that("predict with L2 penalty", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- predict(model, A, L2 = 0.1)
  
  expect_s4_class(result, "nmf")
  expect_true(is.matrix(result@h))
  expect_equal(ncol(result@h), ncol(A))
})

test_that("predict errors with invalid L1", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  expect_error(predict(model, A, L1 = 1), "L1 penalty must be strictly")
  expect_error(predict(model, A, L1 = -0.1), "L1 penalty must be strictly")
  expect_error(predict(model, A, L1 = c(0.1, 0.2)), "must be a single value")
})

test_that("predict errors with invalid L2", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  expect_error(predict(model, A, L2 = -1), "L2 penalty must be strictly")
})

test_that("predict preserves column count", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(100, 50, 0.2))
  model <- nmf(A, 5, seed = 1, maxit = 30, tol = 1e-4, verbose = FALSE)
  
  result <- predict(model, A)
  expect_equal(ncol(result@h), 50)
})

test_that("predict result has non-negative values", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- predict(model, A)
  expect_true(all(result@h >= 0))
})
