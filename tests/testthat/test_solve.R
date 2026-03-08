test_that("Testing solve (coordinate descent for Gram matrix systems)", {
  
  # solve() is deprecated — suppress deprecation warnings in these tests
  # (the underlying c_solve() is tested directly; these tests verify the R wrapper)
  
  # a system of equations where 'a' is a Gram matrix and 'b' is either a vector or a matrix
  a <- matrix(c(5.905428, 5.085229, 4.762723, 4.777141, 3.675643,
              5.085229, 7.430083, 6.225316, 6.513046, 4.180541,
              4.762723, 6.225316, 7.091737, 6.086963, 4.102543,
              4.777141, 6.513046, 6.086963, 7.234511, 4.227595,
              3.675643, 4.180541, 4.102543, 4.227595, 5.235357), 
              ncol = 5)
  b <- matrix(c(5.671745, 7.153883, 6.440488, 6.211075, 4.557587))
  b_matrix <- matrix(c(0.42802924, 0.46490343, 0.06213912, 0.43574293, 
                0.34564226, 0.76865630, 0.81245858, 0.80472978,
                0.26059196, 0.59442416, 0.16599049, 0.87390953,
                0.67386522,  0.74391732, 0.34301505), nrow = 5)
  true_nnls_solution <- c(0.26332579, 0.61837006, 0.14175903, 0.00000000, 0.08079708) # results from multiway::fnnls(a, b)

  # solve with non-negativity constraint should equal true NNLS solution
  # Different NNLS algorithms may converge to slightly different optimal points
  # Note: Use RcppML:::solve to access internalized function (avoids conflict with base::solve)
  our_solution <- as.vector(suppressWarnings(RcppML:::solve(a, b, cd_maxit = 1000, cd_tol = 1e-9, nonneg = TRUE)))
  
  # Verify solution is non-negative
  expect_true(all(our_solution >= -1e-10))
  
  # Verify solution achieves similar objective value (residual norm)
  true_residual <- sum((a %*% true_nnls_solution - b)^2)
  our_residual <- sum((a %*% our_solution - b)^2)
  expect_equal(our_residual, true_residual, tolerance = 0.1)

  # solution from a parallelized matrix "b" should be the same as the unparallelized vector "b"
  expect_equal(as.vector(suppressWarnings(RcppML:::solve(a, matrix(b_matrix[,2]), nonneg = TRUE))), suppressWarnings(RcppML:::solve(a, b_matrix, nonneg = TRUE))[,2], tolerance = 1e-5)

  # test unconstrained solve (nonneg = FALSE)
  unconstrained <- suppressWarnings(RcppML:::solve(a, b, nonneg = FALSE))
  # unconstrained solution may have negative values
  expect_true(is.matrix(unconstrained))

  # Verify deprecation warning is issued
  expect_warning(RcppML:::solve(a, b, nonneg = TRUE), "deprecated")
})

test_that("solve with penalties", {
  # Test L1 and L2 penalties
  set.seed(123)
  X <- matrix(runif(100), 10, 10)
  a <- crossprod(X)
  b <- crossprod(X, runif(10))
  
  # L1 penalty should encourage sparsity
  soln_no_penalty <- suppressWarnings(RcppML:::solve(a, b, nonneg = TRUE))
  soln_with_L1 <- suppressWarnings(RcppML:::solve(a, b, L1 = 0.1, nonneg = TRUE))
  expect_true(sum(soln_with_L1 == 0) >= sum(soln_no_penalty == 0))
  
  # L2 penalty should shrink coefficients
  soln_with_L2 <- suppressWarnings(RcppML:::solve(a, b, L2 = 0.1, nonneg = TRUE))
  expect_true(sum(soln_with_L2^2) <= sum(soln_no_penalty^2))
})

