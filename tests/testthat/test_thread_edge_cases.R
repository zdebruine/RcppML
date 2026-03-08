# Tests for thread count edge cases
# Verifies NMF handles unusual thread count values gracefully.

test_that("nmf works with threads = 1", {
  A <- Matrix::rsparsematrix(20, 10, 0.3)
  A@x <- abs(A@x)
  result <- nmf(A, k = 3, maxit = 10, seed = 42, threads = 1)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})

test_that("nmf works with threads = 0 (auto-detect)", {
  A <- Matrix::rsparsematrix(20, 10, 0.3)
  A@x <- abs(A@x)
  result <- nmf(A, k = 3, maxit = 10, seed = 42, threads = 0)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})

test_that("nmf works with threads = 2", {
  A <- Matrix::rsparsematrix(20, 10, 0.3)
  A@x <- abs(A@x)
  result <- nmf(A, k = 3, maxit = 10, seed = 42, threads = 2)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})

test_that("nmf gives consistent results regardless of thread count", {
  A <- Matrix::rsparsematrix(30, 15, 0.3)
  A@x <- abs(A@x)
  
  result_1 <- nmf(A, k = 3, maxit = 20, seed = 42, threads = 1)
  result_2 <- nmf(A, k = 3, maxit = 20, seed = 42, threads = 2)
  
  # Results should be very similar (not necessarily identical due to
  # floating-point thread-order effects in parallel reductions)
  expect_equal(result_1@d, result_2@d, tolerance = 1e-4)
})

test_that("nmf handles very large thread count gracefully", {
  # Setting threads higher than available cores should not crash
  A <- Matrix::rsparsematrix(20, 10, 0.3)
  A@x <- abs(A@x)
  result <- nmf(A, k = 3, maxit = 10, seed = 42, threads = 9999)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})

test_that("bipartition works with threads = 1", {
  A <- Matrix::rsparsematrix(20, 10, 0.3)
  A@x <- abs(A@x)
  result <- bipartition(A, seed = 42, threads = 1)
  expect_true(!is.null(result))
})

test_that("nnls works with threads = 1", {
  set.seed(42)
  w <- matrix(runif(20), nrow = 5, ncol = 4)
  A <- matrix(runif(15), nrow = 5, ncol = 3)
  result <- nnls(w = w, A = A, threads = 1)
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(4, 3))
})
