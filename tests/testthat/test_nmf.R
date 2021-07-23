test_that("Testing RcppML::nmf", {
  
  # test that matrix factorization converges over time
  data(moca7k)
  A <- moca7k[1:500,1:500]
  model <- RcppML::nmf(A, 5, maxit = 20, tol = 1e-10)
  expect_lt(model$tol[10], model$tol[1])
  expect_lt(model$tol[20], model$tol[10])

  # test that non-negativity constraints are applied 
  expect_equal(min(model$w) >= 0, TRUE)
  expect_equal(min(model$h) >= 0, TRUE)
  
  # test that non-negativity constraints are not applied when nonneg = FALSE
  model <- RcppML::nmf(A, 5, nonneg = FALSE)
  expect_equal(min(model$w) < 0, TRUE)
  expect_equal(min(model$h) < 0, TRUE)
  
  # test that diagonalization enforces symmetry
  data(moca7k)
  sim <- as(cor(as.matrix(moca7k[,1:100])), "dgCMatrix")
  model <- RcppML::nmf(sim, 5, diag = F, symmetric = F, seed = 123, tol = 1e-6)
  model2 <- RcppML::nmf(sim, 5, diag = T, symmetric = F, seed = 123, tol = 1e-6)
  cor_model <- cor(as.vector(model$w), as.vector(t(model$h)))
  cor_model2 <- cor(as.vector(model2$w), as.vector(t(model2$h)))
  expect_lt(cor_model, cor_model2)
  
  # test that symmetric = TRUE enforces symmetry and tolerance is lower
  model3 <- RcppML::nmf(sim, 5, symmetric = T, diag = F, seed = 123, tol = 1e-6)
  expect_gt(model3$tol[10], model2$tol[10])
  expect_gt(cor(as.vector(model3$w), as.vector(t(model3$h))), 0.99)
  expect_gt(mean(model3$d), 1)
  
  # test that fast_maxit = 0 and fast_maxit = 10 gives same result
  model_cdonly <- RcppML::nmf(A, 5, fast_maxit = 0, seed = 123, tol = 1e-5)
  model_fastcd <- RcppML::nmf(A, 5, fast_maxit = 10, seed = 123, tol = 1e-5)
  expect_gt(cor(as.vector(model_cdonly$w), as.vector(model_fastcd$w)), 0.99)
  
})