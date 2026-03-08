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
  expect_equal(model$dist, 0.137, tolerance = 0.01)
  expect_equal(model$size1, 5)
  expect_equal(model$size2, 5)
  
  # Verify partition is complete (all samples assigned)
  expect_equal(length(model$samples1) + length(model$samples2), 10)
  
  # Verify no overlap between partitions
  expect_equal(length(intersect(model$samples1, model$samples2)), 0)
})
