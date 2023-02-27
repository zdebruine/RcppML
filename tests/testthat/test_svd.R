options(RcppML.threads = 1)
options(RcppML.verbose = TRUE)

testConvergence <- function(A, k, mask = NULL, ...){
  m <- svd(A, k)
 
  # Test that reconstructed array is equal to input
  A_reconstruct = m$u %*% m$v
  expect_equal(A_reconstruct, A)
}

does_svd_converge <- function(A, k){
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
    testConvergence(A, k)
}


library(Matrix)
U_ <- matrix(rnorm(100), ncol=5)
V_ <- matrix(rnorm(100), nrow=5)
A <- as(U_ %*% V_, "dgCMatrix")
#test_that("svd converges over several iterations for sparse asymmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as.matrix(A)
test_that("svd converges over several iterations for dense asymmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as(crossprod(A), "dgCMatrix")
#test_that("svd converges over several iterations for sparse symmetric inputs (k = 5)", { does_svd_converge(A, 5)})
A <- as.matrix(A)
test_that("svd converges over several iterations for dense symmetric inputs (k = 5)", { does_svd_converge(A, 5)})
