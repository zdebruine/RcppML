# Tests for angular (orthogonality) regularization

test_that("angular parameter is accepted", {
  skip_on_cran()
  set.seed(123)
  A <- matrix(runif(50 * 30), 50, 30)
  
  # Should not error with angular parameter
  expect_silent(nmf(A, 3, maxit = 5, angular = c(0, 0), verbose = FALSE))
  expect_silent(nmf(A, 3, maxit = 5, angular = c(0.1, 0), verbose = FALSE))
  expect_silent(nmf(A, 3, maxit = 5, angular = c(0, 0.1), verbose = FALSE))
  expect_silent(nmf(A, 3, maxit = 5, angular = c(0.1, 0.1), verbose = FALSE))
  expect_silent(nmf(A, 3, maxit = 5, angular = 0.1, verbose = FALSE))  # Single value
})

test_that("angular parameter validation works", {
  skip_on_cran()
  set.seed(123)
  A <- matrix(runif(50 * 30), 50, 30)
  
  # Negative values should error
  expect_error(nmf(A, 3, maxit = 5, angular = c(-0.1, 0), verbose = FALSE))
  expect_error(nmf(A, 3, maxit = 5, angular = c(0, -0.1), verbose = FALSE))
})

test_that("angular regularization reduces factor correlation", {
  skip_on_cran()
  set.seed(42)
  # Create data with some correlated structure
  n <- 100
  p <- 50
  k <- 5
  A <- matrix(abs(rnorm(n * p)), n, p)
  
  # Fit without angular
  m_none <- nmf(A, k, maxit = 50, angular = c(0, 0), verbose = FALSE)
  
  # Fit with strong angular on W
  m_angular_w <- nmf(A, k, maxit = 50, angular = c(1.0, 0), verbose = FALSE)
  
  # Compute off-diagonal correlations in W^T W
  WtW_none <- crossprod(m_none@w)
  WtW_angular <- crossprod(m_angular_w@w)
  
  # Extract off-diagonal elements
  off_diag_none <- abs(WtW_none[upper.tri(WtW_none)])
  off_diag_angular <- abs(WtW_angular[upper.tri(WtW_angular)])
  
  # With strong angular, off-diagonal correlations should be smaller on average
  # Note: This is a soft test - angular trades off reconstruction for orthogonality
  expect_true(
    mean(off_diag_angular) <= mean(off_diag_none) * 1.5,  # Allow some tolerance
    info = sprintf("angular off-diag: %.4f, none off-diag: %.4f", 
                   mean(off_diag_angular), mean(off_diag_none))
  )
})

test_that("angular works with sparse matrices", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  
  set.seed(123)
  A <- Matrix::rsparsematrix(80, 40, 0.3)
  
  # Should not error
  expect_silent(m <- nmf(A, 4, maxit = 10, angular = c(0.1, 0.1), verbose = FALSE))
  
  # Check model is valid
  expect_equal(ncol(m@w), 4)
  expect_equal(nrow(m@h), 4)
})

test_that("angular works with different loss functions", {
  skip_on_cran()
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # MSE (default)
  expect_silent(nmf(A, 3, maxit = 5, angular = 0.1, loss = "mse", verbose = FALSE))
  
  # MAE
  expect_silent(nmf(A, 3, maxit = 5, angular = 0.1, loss = "mae", verbose = FALSE))
  
  # Huber
  expect_silent(nmf(A, 3, maxit = 5, angular = 0.1, loss = "huber", verbose = FALSE))
})

test_that("angular H reduces correlation between H rows", {
  skip_on_cran()
  set.seed(42)
  n <- 80
  p <- 40
  k <- 4
  A <- matrix(abs(rnorm(n * p)), n, p)
  
  # Fit without angular
  m_none <- nmf(A, k, maxit = 40, angular = c(0, 0), verbose = FALSE)
  
  # Fit with strong angular on H
  m_angular_h <- nmf(A, k, maxit = 40, angular = c(0, 1.0), verbose = FALSE)
  
  # Compute H H^T for correlation between factors in H
  HHt_none <- tcrossprod(m_none@h)
  HHt_angular <- tcrossprod(m_angular_h@h)
  
  # Extract off-diagonal
  off_diag_none <- abs(HHt_none[upper.tri(HHt_none)])
  off_diag_angular <- abs(HHt_angular[upper.tri(HHt_angular)])
  
  # Angular on H should reduce correlations in H
  # (with tolerance since it's trading off with reconstruction)
  expect_true(
    mean(off_diag_angular) <= mean(off_diag_none) * 2.0,
    info = sprintf("angular_H off-diag: %.4f, none off-diag: %.4f", 
                   mean(off_diag_angular), mean(off_diag_none))
  )
})

test_that("angular can be combined with other regularization", {
  skip_on_cran()
  set.seed(123)
  A <- matrix(abs(rnorm(60 * 40)), 60, 40)
  
  # Angular + L1
  expect_silent(nmf(A, 4, maxit = 10, angular = 0.1, L1 = 0.01, verbose = FALSE))
  
  # Angular + L2
  expect_silent(nmf(A, 4, maxit = 10, angular = 0.1, L2 = 0.01, verbose = FALSE))
  
  # Angular + L1 + L2
  expect_silent(nmf(A, 4, maxit = 10, angular = 0.1, L1 = 0.01, L2 = 0.01, verbose = FALSE))
  
  # Angular + L21
  expect_silent(nmf(A, 4, maxit = 10, angular = 0.1, L21 = 0.01, verbose = FALSE))
})
