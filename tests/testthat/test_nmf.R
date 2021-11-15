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
  L1_ <- nmf(A, 5, maxit = 5, L1 = c(0, 0), seed = 123, v = F)
  L1_w <- nmf(A, 5, maxit = 5, L1 = c(0.1, 0), seed = 123, v = F)
  L1_h <- nmf(A, 5, maxit = 5, L1 = c(0, 0.1), seed = 123, v = F)
  L1_wh <- nmf(A, 5, maxit = 5, L1 = c(0.1, 0.1), seed = 123, v = F)
  expect_gt(sum(L1_w$w == 0), sum(L1_$w == 0))
  expect_gt(sum(L1_w$w == 0), sum(L1_w$h == 0))
  expect_gt(sum(L1_h$h == 0), sum(L1_$h == 0))
  expect_gt(sum(L1_h$h == 0), sum(L1_h$w == 0))
  expect_gt(sum(L1_wh$h == 0), sum(L1_$h == 0))
  expect_gt(sum(L1_wh$w == 0), sum(L1_$w == 0))
})

set.seed(123)
w_ <- matrix(runif(500), 5, 100)
test_that("setting the random seed gives identical models (sparse)", {
  expect_equal(nmf(A, 5, maxit = 3, seed = 123, v = F)$w, nmf(A, 5, maxit = 3, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, maxit = 3, seed = list(w_), v = F)$w, nmf(A, 5, maxit = 3, seed = w_, v = F)$w)
  expect_equal(all(nmf(A, 5, maxit = 3, seed = 123, v = F)$w == nmf(A, 5, maxit = 3, seed = 234, v = F)$w), FALSE)
})
A <- as.matrix(A)
test_that("setting the random seed gives identical models (dense)", {
  expect_equal(nmf(A, 5, maxit = 3, seed = 123, v = F)$w, nmf(A, 5, maxit = 3, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, maxit = 3, seed = w_, v = F)$w, nmf(A, 5, maxit = 3, seed = w_, v = F)$w)
  expect_equal(all(nmf(A, 5, maxit = 3, seed = 123, v = F)$w == nmf(A, 5, maxit = 3, seed = 234, v = F)$w), FALSE)
})