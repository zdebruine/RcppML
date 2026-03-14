# Tests for cross-validation with robust fitting (IRLS + CV interaction)
# Exercises the CV IRLS helpers in fit_cv.hpp
#
# NOTE: Robust CV is slower than MSE CV due to per-column IRLS.
# Tests here use maxit=20 and small matrices (40x30) to keep them fast.

options(RcppML.verbose = FALSE)
options(RcppML.threads = 1)

test_that("single-rank CV with robust='mae' produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 3, test_fraction = 0.1, robust = "mae",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
  expect_true(all(result@w >= 0))
  expect_true(all(result@h >= 0))
})

test_that("single-rank CV with robust=TRUE (Huber) produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 3, test_fraction = 0.1, robust = TRUE,
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("single-rank CV with GP loss produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.05, seed = 42)
  A <- pmax(data$A, 0.01)  # GP requires positive values

  result <- nmf(A, k = 3, test_fraction = 0.1, loss = "gp",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("multi-rank CV with robust='mae' returns valid data.frame", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 2:4, test_fraction = 0.1, robust = "mae",
                cv_seed = 1, maxit = 20, seed = 42)

  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 3)
  expect_true(all(result$test_mse > 0))
  expect_true(all(is.finite(result$test_mse)))
})

test_that("multi-rank CV with robust=TRUE returns valid data.frame", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 2:4, test_fraction = 0.1, robust = TRUE,
                cv_seed = 1, maxit = 20, seed = 42)

  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 3)
  expect_true(all(result$test_mse > 0))
})

test_that("CV with sparse matrix and robust='mae'", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, dropout = 0.5, seed = 42)
  A_sparse <- as(data$A, "dgCMatrix")

  result <- nmf(A_sparse, k = 3, test_fraction = 0.1, robust = "mae",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("CV with robust fitting is reproducible", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  r1 <- nmf(A, k = 2:3, test_fraction = 0.1, robust = "mae",
            cv_seed = 42, maxit = 15, seed = 1)
  r2 <- nmf(A, k = 2:3, test_fraction = 0.1, robust = "mae",
            cv_seed = 42, maxit = 15, seed = 1)

  expect_equal(r1$test_mse, r2$test_mse)
})

test_that("CV robust and MSE produce valid but potentially different results", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  r_mse <- nmf(A, k = 3, test_fraction = 0.1, loss = "mse",
               cv_seed = 1, maxit = 30, seed = 42, verbose = FALSE)
  r_robust <- nmf(A, k = 3, test_fraction = 0.1, robust = "mae",
               cv_seed = 1, maxit = 30, seed = 42, verbose = FALSE)

  # Both should produce valid numeric loss values
  expect_true(is.numeric(r_mse@misc$loss))
  expect_true(is.numeric(r_robust@misc$loss))
  expect_true(r_mse@misc$loss > 0)
  expect_true(r_robust@misc$loss > 0)

  # Both models should produce valid factorizations
  expect_equal(dim(r_mse), dim(r_robust))
})

# ============================================================
# High-rank CV regression test
# ============================================================

test_that("CV with high rank (k=16) produces valid results", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(100, 80, k = 10, noise = 0.2, seed = 123)
  A <- data$A

  result <- nmf(A, k = 16, test_fraction = 0.1, loss = "mse",
                cv_seed = 1, maxit = 30, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 16)
  expect_equal(nrow(result@h), 16)
  expect_true(result@misc$test_loss > 0)
  expect_true(is.finite(result@misc$test_loss))
})

test_that("multi-rank CV sweep to k=16 produces valid data.frame", {
  skip_on_cran()
  set.seed(123)
  data <- simulateNMF(100, 80, k = 10, noise = 0.2, seed = 123)
  A <- data$A

  result <- nmf(A, k = c(2, 4, 8, 16), test_fraction = 0.1,
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)
  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 4)
  expect_true(all(result$test_mse > 0))
  expect_true(all(is.finite(result$test_mse)))
})
