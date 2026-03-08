options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

testConvergence <- function(A, k, mask = NULL, ...){
  m0 <- nmf(A, k, tol = 1e-10, maxit = 1, mask = mask, ...)
  m1 <- nmf(A, k, tol = 1e-10, maxit = 20, mask = mask, ...)
  # test convergence
  expect_gt(evaluate(m0, A, mask), evaluate(m1, A, mask))
  
  # test that non-negativity constraints are enforced
  p <- list(...)
  if(k > 1) { # k == 1 of non-negative inputs gives non-negative results regardless of constraints
    expect_equal(min(m0$w) >= 0, TRUE)
    expect_equal(min(m0$h) >= 0, TRUE)
    expect_equal(min(m1$w) >= 0, TRUE)
    expect_equal(min(m1$h) >= 0, TRUE)
  }
}

does_nmf_converge <- function(A, k){
    testConvergence(A, k)
    testConvergence(A, k, mask = "zeros")
    testConvergence(A, k, L1 = c(0.1, 0))
    testConvergence(A, k, L1 = c(0, 0.1))
    testConvergence(A, k, L1 = c(0.1, 0.1))
}

library(Matrix)
A <- simulateNMF(nrow = 50, ncol = 50, k = 5, noise = 0.5, dropout = 0.5, seed = 123)
A <- as(A$A, "dgCMatrix")
test_that("nmf converges over several iterations for sparse asymmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as.matrix(A)
test_that("nmf converges over several iterations for dense asymmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as(crossprod(A), "dgCMatrix")
test_that("nmf converges over several iterations for sparse symmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as.matrix(A)
test_that("nmf converges over several iterations for dense symmetric inputs (k = 5)", { does_nmf_converge(A, 5)})

A <- abs(Matrix::rsparsematrix(100, 100, 0.1))
test_that("L1 regularization increases factor sparsity", {
  # Use more iterations and stronger L1 to see effect
  # Force CPU to avoid GPU L1 implementation differences
  eps <- 1e-6  # near-zero threshold for fp32 compatibility
  L1_ <- nmf(A, 5, maxit = 50, L1 = c(0, 0), seed = 123, v = F, resource = "cpu")
  L1_w <- nmf(A, 5, maxit = 50, L1 = c(0.5, 0), seed = 123, v = F, resource = "cpu")
  L1_h <- nmf(A, 5, maxit = 50, L1 = c(0, 0.5), seed = 123, v = F, resource = "cpu")
  L1_wh <- nmf(A, 5, maxit = 50, L1 = c(0.5, 0.5), seed = 123, v = F, resource = "cpu")
  expect_gt(sum(abs(L1_w$w) < eps), sum(abs(L1_$w) < eps))
  expect_gt(sum(abs(L1_w$w) < eps), sum(abs(L1_w$h) < eps))
  expect_gt(sum(abs(L1_h$h) < eps), sum(abs(L1_$h) < eps))
  expect_gt(sum(abs(L1_h$h) < eps), sum(abs(L1_h$w) < eps))
  expect_gt(sum(abs(L1_wh$h) < eps), sum(abs(L1_$h) < eps))
  expect_gt(sum(abs(L1_wh$w) < eps), sum(abs(L1_$w) < eps))
})

set.seed(123)
w_ <- matrix(runif(500), 5, 100)
test_that("setting the random seed gives identical models (sparse)", {
  # Use threads = 1 for exact reproducibility (multi-threaded has FP non-determinism)
  m1 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  m2 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  expect_equal(m1@w, m2@w)
  
  m3 <- nmf(A, 5, maxit = 3, seed = w_, threads = 1, v = F)
  m4 <- nmf(A, 5, maxit = 3, seed = w_, threads = 1, v = F)
  expect_equal(m3@w, m4@w)
  
  # Different seeds should produce different results
  m5 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  m6 <- nmf(A, 5, maxit = 3, seed = 234, threads = 1, v = F)
  expect_false(all(m5@w == m6@w))
})
A <- as.matrix(A)
test_that("setting the random seed gives identical models (dense)", {
  # Use threads = 1 for exact reproducibility (multi-threaded has FP non-determinism)
  m1 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  m2 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  expect_equal(m1@w, m2@w)
  
  m3 <- nmf(A, 5, maxit = 3, seed = w_, threads = 1, v = F)
  m4 <- nmf(A, 5, maxit = 3, seed = w_, threads = 1, v = F)
  expect_equal(m3@w, m4@w)
  
  # Different seeds should produce different results
  m5 <- nmf(A, 5, maxit = 3, seed = 123, threads = 1, v = F)
  m6 <- nmf(A, 5, maxit = 3, seed = 234, threads = 1, v = F)
  expect_false(all(m5@w == m6@w))
})

