# Tests for dimnames preservation through nmf()

set.seed(123)

test_that("nmf() preserves rownames on W and colnames on H for dense input", {
  A <- abs(matrix(rnorm(30 * 10), nrow = 30, ncol = 10))
  rownames(A) <- paste0("gene", 1:30)
  colnames(A) <- paste0("sample", 1:10)
  
  result <- nmf(A, 3, maxit = 10, tol = 1e-10, seed = 1)
  
  expect_equal(rownames(result@w), paste0("gene", 1:30))
  expect_equal(colnames(result@h), paste0("sample", 1:10))
})

test_that("nmf() preserves dimnames for dgCMatrix input", {
  A <- abs(matrix(rnorm(30 * 10), nrow = 30, ncol = 10))
  A[A < 0.5] <- 0  # make sparse
  rownames(A) <- paste0("feature", 1:30)
  colnames(A) <- paste0("obs", 1:10)
  A_sp <- as(A, "dgCMatrix")
  
  result <- nmf(A_sp, 3, maxit = 10, tol = 1e-10, seed = 1)
  
  expect_equal(rownames(result@w), paste0("feature", 1:30))
  expect_equal(colnames(result@h), paste0("obs", 1:10))
})

test_that("nmf() works when input has no dimnames", {
  A <- abs(matrix(rnorm(30 * 10), nrow = 30, ncol = 10))
  
  result <- nmf(A, 3, maxit = 10, tol = 1e-10, seed = 1)
  
  expect_null(rownames(result@w))
  expect_null(colnames(result@h))
})

test_that("nmf() preserves only rownames when colnames are absent", {
  A <- abs(matrix(rnorm(30 * 10), nrow = 30, ncol = 10))
  rownames(A) <- paste0("r", 1:30)
  
  result <- nmf(A, 3, maxit = 10, tol = 1e-10, seed = 1)
  
  expect_equal(rownames(result@w), paste0("r", 1:30))
  expect_null(colnames(result@h))
})

test_that("nmf() preserves only colnames when rownames are absent", {
  A <- abs(matrix(rnorm(30 * 10), nrow = 30, ncol = 10))
  colnames(A) <- paste0("c", 1:10)
  
  result <- nmf(A, 3, maxit = 10, tol = 1e-10, seed = 1)
  
  expect_null(rownames(result@w))
  expect_equal(colnames(result@h), paste0("c", 1:10))
})
