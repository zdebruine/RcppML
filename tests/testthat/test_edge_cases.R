library(testthat)
library(RcppML)
library(Matrix)

# =============================================================================
# S2: Edge case tests — k=1, extreme ranks, zero rows/cols, mixed penalties
# =============================================================================

test_that("NMF works with k=1 (rank-1 factorization)", {
  set.seed(42)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)  # Make non-negative

  result <- nmf(A, k = 1, tol = 1e-4, maxit = 100, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
  expect_equal(nrow(result@h), 1)
  expect_equal(length(result@d), 1)

  # Reconstruction should be valid
  recon <- result@w %*% diag(result@d, nrow = 1) %*% result@h
  expect_equal(dim(recon), dim(A))
})

test_that("NMF works with k=1 on dense matrix", {
  set.seed(99)
  A <- matrix(abs(rnorm(200)), 20, 10)

  result <- nmf(A, k = 1, tol = 1e-4, maxit = 50, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
  expect_equal(nrow(result@h), 1)
})

test_that("NMF works with k equal to min dimension minus 1", {
  set.seed(123)
  A <- rsparsematrix(15, 10, 0.5)
  A <- abs(A)

  # k = min(15, 10) - 1 = 9
  result <- nmf(A, k = 9, tol = 1e-4, maxit = 50, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 9)
  expect_equal(nrow(result@h), 9)
})

test_that("NMF with mixed penalty combinations (L1 + L21 + graph)", {
  set.seed(456)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  # Construct graph adjacency
  n <- ncol(A)
  graph_H <- Matrix::sparseMatrix(
    i = c(1:n, 1:(n-1)),
    j = c(1:n, 2:n),
    x = c(rep(1, n), rep(0.1, n-1)),
    dims = c(n, n)
  )

  result <- nmf(A, k = 3, tol = 1e-4, maxit = 100, verbose = FALSE,
                L1 = c(0.1, 0.1), L21 = c(0.05, 0),
                graph_H = graph_H, graph_lambda = c(0, 0.1))

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)

  # Result should have non-trivial sparsity from L1
  w_sparsity <- mean(result@w < 1e-10)
  expect_gt(w_sparsity, 0)
})

test_that("NMF handles matrix with some all-zero columns gracefully", {
  set.seed(789)
  A <- rsparsematrix(30, 20, 0.3)
  A <- abs(A)
  # Zero out a few columns
  A[, c(5, 15)] <- 0

  result <- nmf(A, k = 3, tol = 1e-4, maxit = 50, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(dim(result@h)[2], 20)
})

test_that("NMF handles custom w_init", {
  set.seed(321)
  A <- rsparsematrix(40, 30, 0.3)
  A <- abs(A)

  # Provide initial W
  W_init <- matrix(runif(40 * 3), 40, 3)
  result <- nmf(A, k = 3, tol = 1e-4, maxit = 50, verbose = FALSE, w_init = W_init)

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 3)
})

test_that("NMF k=1 CV mode works", {
  set.seed(555)
  A <- rsparsematrix(50, 40, 0.3)
  A <- abs(A)

  result <- nmf(A, k = 1, tol = 1e-4, maxit = 50, 
                test_fraction = 0.1, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
})
