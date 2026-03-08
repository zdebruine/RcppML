# Tests for plot.nmfCrossValidate

test_that("plot.nmfCrossValidate returns ggplot", {
  skip_if_not_installed("ggplot2")
  library(Matrix)
  A <- abs(rsparsematrix(50, 30, 0.3))

  cv_result <- nmf(A, k = 2:4, test_fraction = 0.1,
                   cv_seed = 1, maxit = 10, seed = 42)

  p <- plot(cv_result)
  expect_s3_class(p, "gg")
})

test_that("plot.nmfCrossValidate with multiple replicates", {
  skip_if_not_installed("ggplot2")
  library(Matrix)
  A <- abs(rsparsematrix(50, 30, 0.3))

  cv_result <- nmf(A, k = 2:3, test_fraction = 0.1,
                   cv_seed = c(1, 2), maxit = 10, seed = 42)

  p <- plot(cv_result)
  expect_s3_class(p, "gg")
})

test_that("plot.nmfCrossValidate with show_train=FALSE", {
  skip_if_not_installed("ggplot2")
  library(Matrix)
  A <- abs(rsparsematrix(50, 30, 0.3))

  cv_result <- nmf(A, k = 2:4, test_fraction = 0.1,
                   cv_seed = 1, maxit = 10, seed = 42,
                   track_train_loss = TRUE)

  p <- plot(cv_result, show_train = FALSE)
  expect_s3_class(p, "gg")
})

test_that("plot.nmfCrossValidate with show_train=TRUE", {
  skip_if_not_installed("ggplot2")
  library(Matrix)
  A <- abs(rsparsematrix(50, 30, 0.3))

  cv_result <- nmf(A, k = 2:4, test_fraction = 0.1,
                   cv_seed = 1, maxit = 10, seed = 42,
                   track_train_loss = TRUE)

  p <- plot(cv_result, show_train = TRUE)
  expect_s3_class(p, "gg")
})
