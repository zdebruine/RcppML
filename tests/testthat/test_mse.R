test_that("Testing RcppML::mse", {
  
  # test that mean squared error is equivalent to computing it with R
  A <- rsparsematrix(100, 100, 0.1)
  model <- nmf(A, 10, maxit = 5)
  actual_mse <- mean((A - model$w %*% Diagonal(x = model$d) %*% model$h)^2)
  calculated_mse <- mse(A, model$w, model$d, model$h)
  expect_equal(calculated_mse, actual_mse, tolerance = 1e-6)

  # repeat for non-diagonalized factorization
  model <- nmf(A, 10, diag = FALSE, maxit = 5)
  actual_mse <- mean((A - model$w %*% model$h)^2)
  calculated_mse <- mse(A, w = model$w, h = model$h)
  expect_equal(calculated_mse, actual_mse, tolerance = 1e-6)
})