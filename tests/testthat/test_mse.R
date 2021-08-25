test_that("Testing RcppML::mse", {
  setRcppMLthreads(1)
  library(Matrix)
  # test that mean squared error is equivalent to computing it with R (SPARSE)
  A <- abs(rsparsematrix(100, 100, 0.1))
  model <- nmf(A, 10, maxit = 5, v = F)
  actual_mse <- mean((A - model$w %*% Diagonal(x = model$d) %*% model$h)^2)
  calculated_mse <- mse(A, model$w, model$d, model$h)
  expect_equal(calculated_mse, actual_mse, tolerance = 1e-6)

})