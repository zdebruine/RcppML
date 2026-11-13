# Not part of the CRAN subset (CRAN allows about ten minutes and two cores
# for the whole check). Runs when NOT_CRAN=true, as in CI and devtools::test().
testthat::skip_on_cran()

test_that("Testing RcppML::bipartition", {
  library(Matrix)
  options(RcppML.threads = 1)
  
  # Create test matrix with known structure
  A <- sparseMatrix(
    i = c(1,1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10),
    j = c(1,2,5,6,3,4,9,2,5,8,7,9,1,4,2,8,4,6,3,7,5,10,6,10),
    x = c(4,2,3,2,5,4,3,3,5,3,4,3,2,3,3,4,5,4,3,5,4,3,5,4),
    dims = c(10,10)
  )
  
  # Test with seed=1 for reproducible results
  # With correct orientation (features x samples), this produces a 5-5 bipartition with dist ≈ 0.137
  model <- bipartition(A, seed = 1, calc_dist = TRUE)
  
  # Firm expectations based on seed=1
  # dist is the mean over samples of cos(x, own center) - cos(x, other center),
  # computed here from the returned partition (samples are 0-based)
  Ad <- as.matrix(A)
  s1 <- model$samples1 + 1
  s2 <- model$samples2 + 1
  c1 <- rowMeans(Ad[, s1, drop = FALSE])
  c2 <- rowMeans(Ad[, s2, drop = FALSE])
  cosv <- function(x, y) sum(x * y) / sqrt(sum(x^2) * sum(y^2))
  dist_ref <- mean(c(
    vapply(s1, function(j) cosv(Ad[, j], c1) - cosv(Ad[, j], c2), numeric(1)),
    vapply(s2, function(j) cosv(Ad[, j], c2) - cosv(Ad[, j], c1), numeric(1))))
  expect_equal(model$dist, dist_ref, tolerance = 1e-8)
  expect_true(model$dist > 0 && model$dist <= 1)
  expect_equal(model$size1, 5)
  expect_equal(model$size2, 5)
  
  # Verify partition is complete (all samples assigned)
  expect_equal(length(model$samples1) + length(model$samples2), 10)
  
  # Verify no overlap between partitions
  expect_equal(length(intersect(model$samples1, model$samples2)), 0)
})
