test_that("Testing RcppML::project solutions", {
  skip("skip")
  library(Matrix)

  # simulate a random sparse matrix
  A <- rsparsematrix(100, 10, 0.5)
  
  # expect that two projections of H give approximately equal results
  H <- matrix(runif(10 * 3), 3, 10)
  expect_equal(RcppML::project(A, H = H), RcppML::project(A, H = H))
  
  # expect that two projections of W give approximately equal results
  W <- matrix(runif(100 * 3), 100, 3)
  expect_equal(RcppML::project(A, W = W), RcppML::project(A, W = W))
  
  # project W and test that error of solved model is better than error of a random model
  W <- matrix(runif(100 * 3), 100, 3)
  H <- RcppML::project(A, W = W)
  H_random <- matrix(runif(10 * 3), 3, 10)
  err_random_model_H <- sum((A - W %*% H_random)^2)
  err_projected_model_H <- sum((A - W %*% H)^2)
  expect_lt(err_projected_model_H, err_random_model_H)

  # project H and test that error of solved model is better than error of a random model
  H <- matrix(runif(10 * 3), 3, 10)
  W <- t(RcppML::project(A, H = H))
  W_random <- matrix(runif(100 * 3), 100, 3)
  err_random_model_W <- sum((A - W_random %*% H)^2)
  err_projected_model_W <- sum((A - W %*% H)^2)
  expect_lt(err_projected_model_W, err_random_model_W)

  # test that nonneg successfully enforces non-negativity constraints
  H <- project(A, W = W, nonneg = TRUE)
  expect_equal(sum(H[H < 0]), 0)
    
  # test that error gets better after several alternating non-negative iterations
  W_initial <- W
  H_initial <- H
  W <- project(A, H = H, nonneg = TRUE)
  H <- project(A, W = W, nonneg = TRUE)
  W <- project(A, H = H, nonneg = TRUE)
  H <- project(A, W = W, nonneg = TRUE)
  W <- project(A, H = H, nonneg = TRUE)
  H <- project(A, W = W, nonneg = TRUE)
  err_random_model <- sum((A - t(W_initial) %*% H_initial)^2)
  err_refined_model <- sum((A - t(W) %*% H)^2)
  expect_lt(err_refined_model, err_random_model)
  
  # test that L1 model is sparser than a model without L1  
  model_without_L1 <- project(A, H = H, nonneg = TRUE, L1 = 0)
  model_with_L1 <- project(A, H = H, nonneg = TRUE, L1 = max(H) * 0.5)
  sparsity_without_L1 <- sum(model_without_L1 == 0) / prod(dim(W))
  sparsity_with_L1 <- sum(model_with_L1 == 0) / prod(dim(W))
  expect_lt(sparsity_without_L1, sparsity_with_L1)

})