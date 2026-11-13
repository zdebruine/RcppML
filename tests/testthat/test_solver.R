# Not part of the CRAN subset (CRAN allows about ten minutes and two cores
# for the whole check). Runs when NOT_CRAN=true, as in CI and devtools::test().
testthat::skip_on_cran()

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

library(Matrix)
A <- simulateNMF(nrow = 50, ncol = 50, k = 5, noise = 0.5, dropout = 0.5, seed = 123)
A <- as(A$A, "dgCMatrix")

test_that("solver='auto' selects Cholesky for small k MSE no-L1", {
  res <- nmf(A, k = 8, solver = "auto", loss = "mse", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 1L)
})

test_that("solver='auto' selects exact Cholesky when L1 > 0", {
  # the Cholesky path is exact for any L1, and faster than coordinate descent
  res <- nmf(A, k = 8, solver = "auto", loss = "mse", L1 = 0.1, seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 1L)
})

test_that("solver='auto' selects exact Cholesky for large k", {
  res <- nmf(A, k = 35, solver = "auto", loss = "mse", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 1L)
})

test_that("solver='auto' selects CD for non-MSE distributions", {
  res <- nmf(A, k = 4, solver = "auto", loss = "gp", seed = 42, maxit = 3)
  expect_equal(res@misc$solver_mode, 0L)
})
