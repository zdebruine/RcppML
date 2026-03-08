# Test evaluate() and mse()
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("evaluate with MSE loss on sparse data", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- evaluate(model, A, loss = "mse")
  
  expect_true(is.numeric(result))
  expect_length(result, 1)
  expect_true(result >= 0)
  expect_true(is.finite(result))
})

test_that("evaluate with MSE loss on dense data", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result <- evaluate(model, A, loss = "mse")
  
  expect_true(is.numeric(result))
  expect_true(result >= 0)
})

test_that("evaluate supports all loss functions", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  mse_val <- evaluate(model, A, loss = "mse")
  mae_val <- evaluate(model, A, loss = "mae")
  huber_val <- evaluate(model, A, loss = "huber")
  kl_val <- evaluate(model, A, loss = "kl")
  
  expect_true(is.numeric(mse_val) && mse_val >= 0)
  expect_true(is.numeric(mae_val) && mae_val >= 0)
  expect_true(is.numeric(huber_val) && huber_val >= 0)
  expect_true(is.numeric(kl_val) && kl_val >= 0)
  
  expect_false(is.na(mse_val))
  expect_false(is.na(mae_val))
})

test_that("evaluate with mask_zeros option", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  result_masked <- evaluate(model, A, mask = "zeros", loss = "mse")
  expect_true(is.numeric(result_masked))
  expect_true(result_masked >= 0)
})

test_that("evaluate with explicit mask matrix", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  mask <- rsparsematrix(50, 30, 0.1, rand.x = NULL)
  mask <- as(as(mask, "dMatrix"), "dgCMatrix")
  
  result <- evaluate(model, A, mask = mask, missing_only = TRUE, loss = "mse")
  
  expect_true(is.numeric(result))
  expect_true(result >= 0)
})

test_that("evaluate errors when missing_only but no mask", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  expect_error(evaluate(model, A, missing_only = TRUE), "mask matrix must be specified")
})

test_that("mse function works as wrapper", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  mse_result <- mse(model@w, model@d, model@h, A)
  eval_result <- evaluate(model, A, loss = "mse")
  
  expect_equal(mse_result, eval_result, tolerance = 1e-10)
})

test_that("mse works without d vector", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  w_scaled <- model@w %*% diag(model@d)
  h <- model@h
  
  result <- RcppML:::mse(w_scaled, h = h, data = A)
  
  expect_true(is.numeric(result))
  expect_true(result >= 0)
})

test_that("huber_delta parameter affects huber loss", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  
  huber_small <- evaluate(model, A, loss = "huber", huber_delta = 0.01)
  huber_large <- evaluate(model, A, loss = "huber", huber_delta = 100)
  
  expect_true(is.numeric(huber_small))
  expect_true(is.numeric(huber_large))
  expect_false(isTRUE(all.equal(huber_small, huber_large)))
})

test_that("better model has lower MSE", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, 0.3))
  model_few <- nmf(A, 3, seed = 1, maxit = 5, tol = 1e-10, verbose = FALSE)
  model_many <- nmf(A, 3, seed = 1, maxit = 100, tol = 1e-10, verbose = FALSE)
  
  mse_few <- evaluate(model_few, A, loss = "mse")
  mse_many <- evaluate(model_many, A, loss = "mse")
  
  expect_true(mse_many <= mse_few)
})
