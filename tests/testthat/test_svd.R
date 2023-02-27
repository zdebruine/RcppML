options(RcppML.threads = 1)
options(RcppML.verbose = TRUE)

testConvergence <- function(A, k, mask = NULL, ...){
  m0 <- svd(A, k)
  # test convergence
  dbg <- m0$dbg
  
  expect_gt(dbg[0], dbg[1])
}

does_svd_converge <- function(A, k){
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
}

library(Matrix)
A <- simulateNMF(nrow = 50, ncol = 50, k = 5, noise = 0.5, dropout = 0.5, seed = 123)
A <- as(A$A, "dgCMatrix")
#test_that("svd converges over several iterations for sparse asymmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as.matrix(A)
test_that("svd converges over several iterations for dense asymmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as(crossprod(A), "dgCMatrix")
#test_that("svd converges over several iterations for sparse symmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as.matrix(A)
test_that("svd converges over several iterations for dense symmetric inputs (k = 5)", { does_svd_converge(A, 5)})
