test_that("Testing RcppML::nmf", {
  
  A <- rsparsematrix(100, 1000, 0.1)
  m <- dclust(A)
  
  expect_equal(sum(unlist(lapply(m, function(x) length(x$samples)))) == 1000, TRUE)
  
  # need additional tests
})