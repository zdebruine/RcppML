testConvergence <- function(A, k, ...){
  m0 <- nmf(A, k, v = F, tol = 1e-10, maxit = 1, ...)
  m1 <- nmf(A, k, v = F, tol = 1e-10, maxit = 10, ...)
  # test convergence
  expect_gt(mse(A, m0$w, m0$d, m0$h, ...), mse(A, m1$w, m1$d, m1$h, ...))
  
  # test that non-negativity constraints are enforced
  p <- list(...)
  if(k > 1) { # k == 1 of non-negative inputs gives non-negative results regardless of constraints
    is_nonneg <- is.null(p$nonneg)
    expect_equal(min(m0$w) >= 0, is_nonneg)
    expect_equal(min(m0$h) >= 0, is_nonneg)
    expect_equal(min(m1$w) >= 0, is_nonneg)
    expect_equal(min(m1$h) >= 0, is_nonneg)
  }
}

does_nmf_converge <- function(A, k){
    testConvergence(A, k)
    if(is(A, "sparseMatrix") && k > 2) # mask zeros (only supported for sparse matrices and k > 2)
      testConvergence(A, k, mask_zeros = T)
    testConvergence(A, k, update_in_place = T)
    testConvergence(A, k, L1 = c(0.1, 0))
    testConvergence(A, k, L1 = c(0, 0.1))
    testConvergence(A, k, L1 = c(0.1, 0.1))
    testConvergence(A, k, nonneg = F)
    testConvergence(A, k, diag = F)
    testConvergence(A, k, update_in_place = T, nonneg = F, diag = F)
}

set.seed(123)
library(Matrix)
setRcppMLthreads(1)
A <- abs(Matrix::rsparsematrix(100, 100, 0.1))
test_that("nmf converges over several iterations for sparse asymmetric inputs (k = 1)", { does_nmf_converge(A, 1) })
test_that("nmf converges over several iterations for sparse asymmetric inputs (k = 2)", { does_nmf_converge(A, 2)})
test_that("nmf converges over several iterations for sparse asymmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as.matrix(A)
test_that("nmf converges over several iterations for dense asymmetric inputs (k = 1)", { does_nmf_converge(A, 1)})
test_that("nmf converges over several iterations for dense asymmetric inputs (k = 2)", { does_nmf_converge(A, 2)})
test_that("nmf converges over several iterations for dense asymmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as(crossprod(A), "dgCMatrix")
test_that("nmf converges over several iterations for sparse symmetric inputs (k = 1)", { does_nmf_converge(A, 1)})
test_that("nmf converges over several iterations for sparse symmetric inputs (k = 2)", { does_nmf_converge(A, 2)})
test_that("nmf converges over several iterations for sparse symmetric inputs (k = 5)", { does_nmf_converge(A, 5)})
A <- as.matrix(A)
test_that("nmf converges over several iterations for dense symmetric inputs (k = 1)", { does_nmf_converge(A, 1)})
test_that("nmf converges over several iterations for dense symmetric inputs (k = 2)", { does_nmf_converge(A, 2)})
test_that("nmf converges over several iterations for dense symmetric inputs (k = 5)", { does_nmf_converge(A, 5)})

test_that("diagonal scaling enforces symmetry in symmetric factorization", {
  sim <- as(A, "dgCMatrix")
  model <- nmf(sim, 5, diag = F, seed = 123, tol = 1e-3, v = F)
  model2 <- nmf(sim, 5, diag = T, seed = 123, tol = 1e-3, v = F)
  cor_model <- cor(as.vector(model$w), as.vector(t(model$h)))
  cor_model2 <- cor(as.vector(model2$w), as.vector(t(model2$h)))
  expect_lt(cor_model, cor_model2)
  expect_equal(mean(model$d), 1)
})

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
  expect_equal(nmf(A, 5, maxit = 3, w_init = w_, v = F)$w, nmf(A, 5, maxit = 3, w_init = w_, v = F)$w)
  expect_equal(nmf(A, 5, maxit = 3, seed = 123, v = F)$w, nmf(A, 5, maxit = 3, w_init = w_, v = F)$w)
  expect_equal(all(nmf(A, 5, maxit = 3, seed = 123, v = F)$w == nmf(A, 5, maxit = 3, seed = 234, v = F)$w), FALSE)
})
A <- as.matrix(A)
test_that("setting the random seed gives identical models (dense)", {
  expect_equal(nmf(A, 5, maxit = 3, seed = 123, v = F)$w, nmf(A, 5, maxit = 3, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, maxit = 3, w_init = w_, v = F)$w, nmf(A, 5, maxit = 3, w_init = w_, v = F)$w)
  expect_equal(nmf(A, 5, maxit = 3, seed = 123, v = F)$w, nmf(A, 5, maxit = 3, w_init = w_, v = F)$w)
  expect_equal(all(nmf(A, 5, maxit = 3, seed = 123, v = F)$w == nmf(A, 5, maxit = 3, seed = 234, v = F)$w), FALSE)
})

test_that("factorizing for specific samples or features acts as expected (sparse)", {
  s <- sample(1:100, 20)
  f <- sample(1:100, 30)
  expect_equal(nmf(A, 5, samples = s, seed = 123, v = F)$w, nmf(A[, sort(s)], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, samples = s, seed = 123, v = F, update_in_place = TRUE)$w, nmf(A, 5, samples = s, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = s, seed = 123, v = F, update_in_place = TRUE)$w, nmf(A[sort(s), ], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = s, seed = 123, v = F)$w, nmf(A[sort(s), ], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = f, samples = s, seed = 123, v = F)$w, nmf(A[sort(f), sort(s)], 5, seed = 123, v = F)$w)
})
A <- as.matrix(A)
test_that("factorizing for specific samples or features acts as expected (dense)", {
  s <- sample(1:100, 20)
  f <- sample(1:100, 30)
  expect_equal(nmf(A, 5, samples = s, seed = 123, v = F)$w, nmf(A[, sort(s)], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, samples = s, seed = 123, v = F, update_in_place = TRUE)$w, nmf(A, 5, samples = s, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = s, seed = 123, v = F, update_in_place = TRUE)$w, nmf(A[sort(s), ], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = s, seed = 123, v = F)$w, nmf(A[sort(s), ], 5, seed = 123, v = F)$w)
  expect_equal(nmf(A, 5, features = f, samples = s, seed = 123, v = F)$w, nmf(A[sort(f), sort(s)], 5, seed = 123, v = F)$w)
})