test_that("Testing RcppML::project", {

  # example taken from NNLM::nnlm R function documentation examples
  x <- matrix(runif(50*20), 50, 20);
  beta <- matrix(rexp(20*2), 20, 2);
  y <- x %*% beta + 0.1*matrix(runif(50*2), 50, 2);
  x <- as(x, "dgCMatrix")
  
  # expect that two projections of y give approximately equal results
  expect_equal(project(x, y), project(x, y))
  expect_equal(project(x, y, nonneg = TRUE), project(x, y, nonneg = TRUE))
  
  # project and test that error of solved model is better than error of a random model
  beta_hat <- project(x, y)
  beta_rand <- beta_hat[sample(nrow(beta_hat)), sample(ncol(beta_hat))]
  err_random_model <- sum((as.matrix(x) - y %*% beta_rand)^2)
  err_projected_model <- sum((as.matrix(x) - y %*% beta_hat)^2)
  expect_lt(err_projected_model, err_random_model)

  # test that nonneg successfully enforces non-negativity constraints
  beta_hat <- project(x, y, nonneg = TRUE)
  expect_equal(sum(beta_hat[beta_hat < 0]), 0)
  
  # test that error gets better after several alternating non-negative iterations
  A <- rsparsematrix(100, 100, 0.5)
  H_initial <- matrix(runif(100 * 5), 5, 100)
  W_initial <- t(project(t(A), t(H_initial)))
  H <- project(A, W_initial)
  for(i in 1:10){
    W <- t(project(t(A), t(H)))
    H <- project(A, W)
  }
  err_random_model <- sum((A - W_initial %*% H_initial)^2)
  err_refined_model <- sum((A - W %*% H)^2)
  expect_lt(err_refined_model, err_random_model)
})