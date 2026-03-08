# Tests for cross-validation with non-MSE loss functions (IRLS + CV interaction)
# Exercises the CV IRLS helpers in fit_cv.hpp
#
# NOTE: Non-MSE CV (MAE/Huber/KL) is slower than MSE CV due to per-column IRLS.
# Tests here use maxit=20 and small matrices (40x30) to keep them fast.
# The prior hang was traced to an older code path; current code has bounded loops.

options(RcppML.verbose = FALSE)
options(RcppML.threads = 1)

test_that("single-rank CV with MAE loss produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 3, test_fraction = 0.1, loss = "mae",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
  expect_true(all(result@w >= 0))
  expect_true(all(result@h >= 0))
})

test_that("single-rank CV with Huber loss produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 3, test_fraction = 0.1, loss = "huber",
                huber_delta = 1.0,
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("single-rank CV with KL loss produces valid result", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.05, seed = 42)
  A <- pmax(data$A, 0.01)  # KL requires positive values

  result <- nmf(A, k = 3, test_fraction = 0.1, loss = "kl",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("multi-rank CV with MAE loss returns valid data.frame", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 2:4, test_fraction = 0.1, loss = "mae",
                cv_seed = 1, maxit = 20, seed = 42)

  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 3)
  expect_true(all(result$test_mse > 0))
  expect_true(all(is.finite(result$test_mse)))
})

test_that("multi-rank CV with Huber loss returns valid data.frame", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  result <- nmf(A, k = 2:4, test_fraction = 0.1, loss = "huber",
                huber_delta = 0.5,
                cv_seed = 1, maxit = 20, seed = 42)

  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 3)
  expect_true(all(result$test_mse > 0))
})

test_that("CV with sparse matrix and MAE loss", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, dropout = 0.5, seed = 42)
  A_sparse <- as(data$A, "dgCMatrix")

  result <- nmf(A_sparse, k = 3, test_fraction = 0.1, loss = "mae",
                cv_seed = 1, maxit = 20, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
})

test_that("CV with non-MSE loss is reproducible", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  r1 <- nmf(A, k = 2:3, test_fraction = 0.1, loss = "mae",
            cv_seed = 42, maxit = 15, seed = 1)
  r2 <- nmf(A, k = 2:3, test_fraction = 0.1, loss = "mae",
            cv_seed = 42, maxit = 15, seed = 1)

  expect_equal(r1$test_mse, r2$test_mse)
})

test_that("CV MAE and MSE produce valid but potentially different results", {
  set.seed(42)
  data <- simulateNMF(40, 30, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  r_mse <- nmf(A, k = 3, test_fraction = 0.1, loss = "mse",
               cv_seed = 1, maxit = 30, seed = 42, verbose = FALSE)
  r_mae <- nmf(A, k = 3, test_fraction = 0.1, loss = "mae",
               cv_seed = 1, maxit = 30, seed = 42, verbose = FALSE)

  # Both should produce valid numeric loss values
  expect_true(is.numeric(r_mse@misc$loss))
  expect_true(is.numeric(r_mae@misc$loss))
  expect_true(r_mse@misc$loss > 0)
  expect_true(r_mae@misc$loss > 0)

  # Both models should produce valid factorizations
  expect_equal(dim(r_mse), dim(r_mae))
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
