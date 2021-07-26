test_that("Testing RcppML::nmf", {
  
  set.seed(123)
  
  # test that matrix factorization converges over time
  A <- abs(rsparsematrix(100, 100, 0.1))
  model_bad <- RcppML::nmf(A, 5, maxit = 2, tol = 1e-10, threads = 1)
  model_good <- RcppML::nmf(A, 5, maxit = 20, tol = 1e-10, threads = 1)
  model_bad_mse <- RcppML::mse(A, model_bad$w, model_bad$d, model_bad$h, threads = 1)
  model_good_mse <- RcppML::mse(A, model_good$w, model_good$d, model_good$h, threads = 1)
  expect_lt(model_good_mse, model_bad_mse)

  # test that non-negativity constraints are applied
  model <- model_bad
  expect_equal(min(model$w) >= 0, TRUE)
  expect_equal(min(model$h) >= 0, TRUE)
  
  # test that non-negativity constraints are not applied when nonneg = FALSE
  model <- RcppML::nmf(A, 5, nonneg = FALSE, threads = 1)
  expect_equal(min(model$w) < 0, TRUE)
  expect_equal(min(model$h) < 0, TRUE)
  
  # test that diagonalization enforces symmetry
  sim <- as(crossprod(A), "dgCMatrix")
  model <- RcppML::nmf(sim, 5, diag = F, seed = 123, tol = 1e-6, threads = 1)
  model2 <- RcppML::nmf(sim, 5, diag = T, seed = 123, tol = 1e-6, threads = 1)
  cor_model <- cor(as.vector(model$w), as.vector(t(model$h)))
  cor_model2 <- cor(as.vector(model2$w), as.vector(t(model2$h)))
  expect_lt(cor_model, cor_model2)
  expect_equal(mean(model$d), 1)
  
})