test_that("Testing RcppML::nmf", {
  setRcppMLthreads(1)
  
  A <- rsparsematrix(100, 1000, 0.1)
  m <- dclust(A, min_samples = 100, min_dist = 0)
  
  expect_equal(sum(unlist(lapply(m, function(x) length(x$samples)))) == 1000, TRUE)

})