options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

library(Matrix)
A <- simulateNMF(nrow = 50, ncol = 50, k = 5, noise = 0.5, dropout = 0.5, seed = 123)
A <- as(A$A, "dgCMatrix")

test_that("solver='auto' selects Cholesky for small k MSE no-L1", {
  res <- nmf(A, k = 8, solver = "auto", loss = "mse", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 1L)
})

test_that("solver='auto' selects CD when L1 > 0", {
  res <- nmf(A, k = 8, solver = "auto", loss = "mse", L1 = 0.1, seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 0L)
})

test_that("solver='auto' selects CD for large k", {
  res <- nmf(A, k = 35, solver = "auto", loss = "mse", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 0L)
})

test_that("solver='auto' selects CD for non-MSE distributions", {
  res <- nmf(A, k = 4, solver = "auto", loss = "gp", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 0L)
})
