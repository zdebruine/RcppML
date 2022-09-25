library(Matrix)
A <- rsparsematrix(100, 1000, density = 0.5,
                   rand.x = function(n) { sample(1:10, n, replace = T)})
v1 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "CSC")))
v2 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "CSC_P")))
v3 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "CSC_NP")))
v4 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "TCSC_P")))
v5 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "TCSC_NP")))
v6 <- all.equal(as.vector(colSums(A)), as.vector(colsums(A, encoding = "TRCSC_NP")))

test_that("column sums are giving the same values across multiple encodings", {
  expect_equal(sum(v1, v2, v3, v4, v5, v6), 6)
})


x <- rsparsematrix(100, 1000, density = 0.5,
                   rand.x = function(n){sample(1 : 10, n, replace = T)})
y <- matrix(runif(10 * 1000), 10, 1000)

z <- as.matrix(t(Matrix::tcrossprod(x, y)))
attr(z, "dimnames") <- NULL

v1 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "CSC"))
v2 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "CSC_P"))
v3 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "CSC_NP"))
v4 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "TCSC_P"))
v5 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "TCSC_NP"))
v6 <- all.equal(z, VSE::tcrossprod(x, y, encoding = "TRCSC_NP"))

test_that("tcrossprod is giving results as expected", {
  expect_equal(sum(v1, v2, v3, v4, v5, v6), 6)
})