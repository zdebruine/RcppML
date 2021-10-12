test_that("Testing RcppML::project", {
  setRcppMLthreads(1)
  
  # example taken from NNLM::nnlm R function documentation examples
  x <- matrix(runif(50*20), 50, 20);
  beta <- matrix(rexp(20*2), 20, 2);
  y <- x %*% beta + 0.1*matrix(runif(50*2), 50, 2);
  x <- as.matrix(x)
  
  x <- as(x, "dgCMatrix")
  # expect that two projections of y give approximately equal results (SPARSE)
  expect_equal(project(y, x), project(y, x))
  expect_equal(project(y, x, nonneg = FALSE), project(y, x, nonneg = FALSE))
  
  # project and test that error of solved model is better than error of a random model (SPARSE)
  beta_hat <- project(y, x)
  beta_rand <- beta_hat[sample(nrow(beta_hat)), sample(ncol(beta_hat))]
  err_random_model <- sum((as.matrix(x) - y %*% beta_rand)^2)
  err_projected_model <- sum((as.matrix(x) - y %*% beta_hat)^2)
  expect_lt(err_projected_model, err_random_model)

  # test that nonneg successfully enforces non-negativity constraints (SPARSE)
  beta_hat <- project(y, x)
  expect_equal(sum(beta_hat[beta_hat < 0]), 0)
  
})