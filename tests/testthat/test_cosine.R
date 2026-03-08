# Test cosine similarity
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("cosine of matrix with itself gives identity diagonal", {
  library(Matrix)
  set.seed(42)
  A <- matrix(runif(30), nrow = 5, ncol = 6)
  
  result <- cosine(A)
  
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(6, 6))
  # Self-cosine should be 1 on diagonal

  expect_equal(diag(result), rep(1, 6), tolerance = 1e-10)
  # Should be symmetric
  expect_equal(result, t(result), tolerance = 1e-10)
})

test_that("cosine between two matrices works", {
  library(Matrix)
  set.seed(42)
  A <- matrix(runif(30), nrow = 5, ncol = 6)
  B <- matrix(runif(20), nrow = 5, ncol = 4)
  
  result <- cosine(A, B)
  
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(6, 4))
  # All values should be in [-1, 1] range (approximately)
  expect_true(all(result >= -1.01 & result <= 1.01))
})

test_that("cosine between matrix and vector works", {
  library(Matrix)
  set.seed(42)
  A <- matrix(runif(30), nrow = 5, ncol = 6)
  v <- runif(5)
  
  result <- cosine(A, v)
  
  expect_true(is.numeric(result))
  expect_length(result, 6)
})

test_that("cosine between vector and matrix works", {
  library(Matrix)
  set.seed(42)
  A <- matrix(runif(30), nrow = 5, ncol = 6)
  v <- runif(5)
  
  result <- cosine(v, A)
  
  expect_true(is.numeric(result))
  expect_length(result, 6)
})

test_that("cosine between two vectors works", {
  v1 <- c(1, 0, 0)
  v2 <- c(0, 1, 0)
  
  # Orthogonal vectors should have cosine = 0
  result <- cosine(v1, v2)
  expect_equal(as.numeric(result), 0, tolerance = 1e-10)
  
  # Identical vectors should have cosine = 1
  result2 <- cosine(v1, v1)
  expect_equal(as.numeric(result2), 1, tolerance = 1e-10)
})

test_that("cosine works with sparse matrices", {
  library(Matrix)
  set.seed(42)
  A <- rsparsematrix(10, 5, 0.5)
  
  result <- cosine(A)
  
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(5, 5))
  expect_equal(result, t(result), tolerance = 1e-10)
})

test_that("cosine errors when x is vector and y is NULL", {
  expect_error(cosine(c(1, 2, 3)), "x is a vector and y is NULL")
})
