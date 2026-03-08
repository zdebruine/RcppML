# Tests for plot.consensus_nmf

test_that("plot.consensus_nmf returns a ggplot", {
  skip_if_not_installed("ggplot2")
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))

  result <- consensus_nmf(A, k = 2, reps = 3, method = "hard",
                          seed = 123, verbose = FALSE)

  p <- plot(result)
  expect_s3_class(p, "gg")
})

test_that("plot.consensus_nmf with cluster_rows=FALSE", {
  skip_if_not_installed("ggplot2")
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))

  result <- consensus_nmf(A, k = 2, reps = 3, method = "hard",
                          seed = 123, verbose = FALSE)

  p <- plot(result, cluster_rows = FALSE)
  expect_s3_class(p, "gg")
})

test_that("plot.consensus_nmf with show_clusters=FALSE", {
  skip_if_not_installed("ggplot2")
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))

  result <- consensus_nmf(A, k = 3, reps = 3, method = "hard",
                          seed = 123, verbose = FALSE)

  p <- plot(result, show_clusters = FALSE)
  expect_s3_class(p, "gg")
})
