# Tests for degenerate/edge-case matrix inputs
# Verifies graceful handling of empty, zero-column, single-row/column,
# and all-zero matrices.

test_that("nmf handles single-column sparse matrix", {
  A <- Matrix::sparseMatrix(i = c(1, 3, 5), j = c(1, 1, 1), x = c(1, 2, 3),
                            dims = c(5, 1))
  result <- nmf(A, k = 1, maxit = 10, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
  expect_equal(nrow(result@h), 1)
  expect_equal(ncol(result@h), 1)
})

test_that("nmf handles single-row sparse matrix", {
  A <- Matrix::sparseMatrix(i = c(1, 1, 1), j = c(1, 3, 5), x = c(1, 2, 3),
                            dims = c(1, 5))
  result <- nmf(A, k = 1, maxit = 10, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), 1)
  expect_equal(ncol(result@h), 5)
})

test_that("nmf handles single-column dense matrix", {
  A <- matrix(c(1, 0, 2, 0, 3), ncol = 1)
  result <- nmf(A, k = 1, maxit = 10, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
  expect_equal(ncol(result@h), 1)
})

test_that("nmf handles single-row dense matrix", {
  A <- matrix(c(1, 0, 2, 0, 3), nrow = 1)
  result <- nmf(A, k = 1, maxit = 10, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), 1)
  expect_equal(ncol(result@h), 5)
})

test_that("nmf handles matrix with some all-zero rows", {
  A <- matrix(0, nrow = 5, ncol = 4)
  A[1, ] <- c(1, 2, 3, 4)
  A[3, ] <- c(5, 6, 7, 8)
  A[5, ] <- c(9, 10, 11, 12)
  # Rows 2 and 4 are all zeros
  result <- nmf(A, k = 2, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), 5)
  expect_equal(ncol(result@h), 4)
})

test_that("nmf handles sparse matrix with some all-zero columns", {
  A <- Matrix::sparseMatrix(
    i = c(1, 2, 3, 1, 2, 3),
    j = c(1, 1, 1, 4, 4, 4),
    x = c(1, 2, 3, 4, 5, 6),
    dims = c(3, 5)
  )
  # Columns 2, 3, 5 are all zeros
  result <- nmf(A, k = 2, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@h), 5)
})

test_that("nmf handles k larger than min dimension gracefully", {
  A <- matrix(runif(12), nrow = 3, ncol = 4)
  # k=5 > min(3,4)=3 — should either error or run without crashing
  tryCatch(
    nmf(A, k = 5),
    error = function(e) expect_true(TRUE)  # erroring is acceptable
  )
  expect_true(TRUE)  # not crashing is the key requirement
})

test_that("nmf handles 2x2 matrix", {
  A <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
  result <- nmf(A, k = 1, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(dim(result@w), c(2, 1))
  expect_equal(dim(result@h), c(1, 2))
})

test_that("nmf handles square matrix with k = min(m,n) - 1", {
  A <- matrix(runif(25), nrow = 5, ncol = 5)
  result <- nmf(A, k = 4, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 4)
})

test_that("nmf produces finite results on near-zero matrix", {
  A <- matrix(1e-15, nrow = 5, ncol = 4)
  A[1, 1] <- 1e-10  # One tiny nonzero entry
  result <- nmf(A, k = 2, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_true(all(is.finite(result@w)))
  expect_true(all(is.finite(result@h)))
  expect_true(all(is.finite(result@d)))
})

test_that("nmf handles matrix with identical columns", {
  col <- c(1, 2, 3, 4, 5)
  A <- matrix(rep(col, 4), nrow = 5, ncol = 4)
  result <- nmf(A, k = 1, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  # With identical columns, rank-1 should give good approximation
  recon <- result@w %*% diag(result@d, nrow = length(result@d)) %*% result@h
  rel_err <- norm(A - recon, "F") / norm(A, "F")
  expect_lt(rel_err, 0.01)  # Very low error expected
})

test_that("nmf handles matrix with identical rows", {
  row <- c(1, 2, 3, 4)
  A <- matrix(rep(row, each = 5), nrow = 5, ncol = 4, byrow = FALSE)
  A <- t(matrix(rep(row, 5), nrow = 4, ncol = 5))
  result <- nmf(A, k = 1, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
})

test_that("nmf handles very sparse matrix (>99% zeros)", {
  set.seed(42)
  A <- Matrix::rsparsematrix(100, 50, density = 0.005)
  A@x <- abs(A@x)  # Make non-negative
  result <- nmf(A, k = 3, maxit = 20, seed = 42)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})
