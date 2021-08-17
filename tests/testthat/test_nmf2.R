test_that("Testing nmf", {
  
  set.seed(123)
  
  # test that matrix factorization converges over time (SPARSE)
  A <- abs(rsparsematrix(100, 100, 0.1))
  model_bad <- nmf2(A, maxit = 1, tol = 1e-10, seed = 123)
  model_good <- nmf2(A, maxit = 20, tol = 1e-10, seed = 123)
  model_bad_mse <- mse(A, model_bad$w, model_bad$d, model_bad$h)
  model_good_mse <- mse(A, model_good$w, model_good$d, model_good$h)
  expect_lt(model_good_mse, model_bad_mse)
  
  # test that non-negativity constraints are applied (SPARSE)
  model <- model_bad
  expect_equal(min(model$w) >= 0, TRUE)
  expect_equal(min(model$h) >= 0, TRUE)
  
  # test that non-negativity constraints are not applied when nonneg = FALSE (SPARSE)
  model <- nmf2(A, nonneg = FALSE)
  expect_equal(min(model$w) < 0, TRUE)
  expect_equal(min(model$h) < 0, TRUE)
  
  # test that diagonalization enforces symmetry (SPARSE)
  sim <- as(crossprod(A), "dgCMatrix")
  model <- nmf2(sim, diag = F, seed = 123, tol = 1e-6)
  model2 <- nmf2(sim, diag = T, seed = 123, tol = 1e-6)
  cor_model <- cor(as.vector(model$w), as.vector(t(model$h)))
  cor_model2 <- cor(as.vector(model2$w), as.vector(t(model2$h)))
  expect_lt(cor_model, cor_model2)
  expect_equal(mean(model$d), 1)
  
  # test that matrix factorization converges over time (DENSE)
  A <- as.matrix(A)
  model_bad <- nmf2(A, maxit = 2, tol = 1e-10)
  model_good <- nmf2(A, maxit = 20, tol = 1e-10)
  model_bad_mse <- mse(A, model_bad$w, model_bad$d, model_bad$h)
  model_good_mse <- mse(A, model_good$w, model_good$d, model_good$h)
  expect_lt(model_good_mse, model_bad_mse)
  
  # test that non-negativity constraints are applied (DENSE)
  model <- model_bad
  expect_equal(min(model$w) >= 0, TRUE)
  expect_equal(min(model$h) >= 0, TRUE)
  
  # test that non-negativity constraints are not applied when nonneg = FALSE (DENSE)
  model <- nmf2(A, nonneg = FALSE)
  expect_equal(min(model$w) < 0, TRUE)
  expect_equal(min(model$h) < 0, TRUE)
  
  # test that diagonalization enforces symmetry (DENSE)
  sim <- as(crossprod(A), "dgCMatrix")
  model <- nmf2(sim, diag = F, seed = 123, tol = 1e-6)
  model2 <- nmf2(sim, diag = T, seed = 123, tol = 1e-6)
  cor_model <- cor(as.vector(model$w), as.vector(t(model$h)))
  cor_model2 <- cor(as.vector(model2$w), as.vector(t(model2$h)))
  expect_lt(cor_model, cor_model2)
  expect_equal(mean(model$d), 1)
  
  # test that setting the random seed gives identical models (DENSE)
  model1 <- nmf2(A, maxit = 5, seed = 123)
  model1_repeat <- nmf2(A, maxit = 5, seed = 123)
  model2 <- nmf2(A, maxit = 5, seed = 234)
  expect_equal(all(model1$w == model1_repeat$w), TRUE)
  expect_equal(all(model1$w == model2$w), FALSE)
  
  # test that setting the random seed gives identical models (SPARSE)
  A <- as(A, "dgCMatrix")
  model1 <- nmf2(A, maxit = 5, seed = 123)
  model1_repeat <- nmf2(A, maxit = 5, seed = 123)
  model2 <- nmf2(A, maxit = 5, seed = 234)
  expect_equal(all(model1$w == model1_repeat$w), TRUE)
  expect_equal(all(model1$w == model2$w), FALSE)
  
})