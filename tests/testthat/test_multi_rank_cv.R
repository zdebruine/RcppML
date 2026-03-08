# Tests for multi-rank cross-validation
# Verifies nmf(A, k = c(2, 3, 4, 5), test_fraction = 0.1) returns a proper
# nmfCrossValidate data.frame.

test_that("multi-rank CV returns nmfCrossValidate data.frame", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 2:4, test_fraction = 0.1,
                cv_seed = 1, maxit = 10, seed = 42)
  
  expect_s3_class(result, "nmfCrossValidate")
  expect_s3_class(result, "data.frame")
  expect_true("k" %in% names(result))
  expect_true("test_mse" %in% names(result))
  expect_true("train_mse" %in% names(result))
  expect_true("rep" %in% names(result))
  
  # Should have one row per rank
  expect_equal(nrow(result), 3)  # k = 2, 3, 4
  expect_equal(sort(result$k), 2:4)
})

test_that("multi-rank CV with multiple replicates", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 2:3, test_fraction = 0.1,
                cv_seed = c(1, 2), maxit = 10, seed = 42)
  
  expect_s3_class(result, "nmfCrossValidate")
  # 2 ranks × 2 replicates = 4 rows
  expect_equal(nrow(result), 4)
})

test_that("multi-rank CV test_mse values are positive", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 2:4, test_fraction = 0.1,
                cv_seed = 1, maxit = 10, seed = 42)
  
  expect_true(all(result$test_mse > 0))
  expect_true(all(is.finite(result$test_mse)))
})

test_that("multi-rank CV train_mse values are positive when tracked", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  result <- nmf(A, k = 2:4, test_fraction = 0.1,
                cv_seed = 1, maxit = 10, seed = 42,
                track_train_loss = TRUE)
  
  non_na_train <- result$train_mse[!is.na(result$train_mse)]
  if (length(non_na_train) > 0) {
    expect_true(all(non_na_train > 0))
  }
})

test_that("multi-rank CV is reproducible with same cv_seed", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  result1 <- nmf(A, k = 2:4, test_fraction = 0.1,
                 cv_seed = 42, maxit = 10, seed = 1)
  result2 <- nmf(A, k = 2:4, test_fraction = 0.1,
                 cv_seed = 42, maxit = 10, seed = 1)
  
  expect_equal(result1$test_mse, result2$test_mse)
})

test_that("multi-rank CV with dense matrix", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(runif(500), nrow = 25, ncol = 20))
  
  result <- nmf(A, k = 2:4, test_fraction = 0.1,
                cv_seed = 1, maxit = 10, seed = 42)
  
  expect_s3_class(result, "nmfCrossValidate")
  expect_equal(nrow(result), 3)
})

test_that("single-rank CV returns nmf object, not data.frame", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(50, 30, 0.3)
  A@x <- abs(A@x)
  
  # Single k with test_fraction should still return nmf object
  result <- nmf(A, k = 3, test_fraction = 0.1,
                cv_seed = 1, maxit = 10, seed = 42)
  
  # For single-rank CV, result should be an nmf S4 object
  # (the data.frame format is only for multi-rank)
  expect_s4_class(result, "nmf")
})
