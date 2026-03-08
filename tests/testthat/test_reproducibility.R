# Test: Reproducibility and Determinism
#
# NOTE: Basic "same seed => same result" behavior is covered combinatorially
# in test_nmf.R and test_parameters.R. This file tests the three cases that
# are NOT covered elsewhere:
#   1. Representative single-run reproducibility check (sparse + mask)
#   2. CV reproducibility across ranks with a fixed cv_seed
#   3. Multi-thread determinism (parallel execution)

options(RcppML.verbose = FALSE)

test_that("Same seed produces identical results (sparse, mask='zeros')", {
  # Masked NMF reproducibility depends on thread scheduling order for

  # floating-point reductions — skip on CRAN where thread count varies
  skip_on_cran()

  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.1, dropout = 0.6, seed = 123)
  A_sparse <- as(data$A, "dgCMatrix")

  model1 <- nmf(A_sparse, 4, mask = "zeros", maxit = 100, tol = 1e-6,
                seed = 456, verbose = FALSE)
  model2 <- nmf(A_sparse, 4, mask = "zeros", maxit = 100, tol = 1e-6,
                seed = 456, verbose = FALSE)

  expect_equal(model1@w, model2@w, tolerance = 1e-6)
  expect_equal(model1@h, model2@h, tolerance = 1e-6)
})

test_that("CV reproducibility with fixed cv_seed", {
  set.seed(42)
  data <- simulateNMF(60, 50, k = 4, noise = 0.05, dropout = 0.4, seed = 123)
  A <- data$A

  cv1 <- nmf(A, k = 3:5, cv_seed = c(456, 457), test_fraction = 0.1,
             verbose = FALSE, maxit = 50, tol = 1e-5)
  cv2 <- nmf(A, k = 3:5, cv_seed = c(456, 457), test_fraction = 0.1,
             verbose = FALSE, maxit = 50, tol = 1e-5)

  expect_equal(cv1$train_mse, cv2$train_mse, tolerance = 1e-10)
  expect_equal(cv1$test_mse,  cv2$test_mse,  tolerance = 1e-10)
})

test_that("Parallel execution is deterministic", {
  skip_if(getOption("RcppML.threads", 1L) < 2L, "Need multiple threads for parallel test")

  set.seed(42)
  data <- simulateNMF(100, 80, k = 5, noise = 0.1, dropout = 0.3, seed = 123)
  A <- data$A

  model1 <- nmf(A, 5, maxit = 50, tol = 1e-6, seed = 456, verbose = FALSE)
  model2 <- nmf(A, 5, maxit = 50, tol = 1e-6, seed = 456, verbose = FALSE)

  expect_equal(model1@w, model2@w, tolerance = 1e-11)
  expect_equal(model1@h, model2@h, tolerance = 1e-11)
})
