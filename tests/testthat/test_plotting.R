test_that("plot.nmf returns ggplot for loss type", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  # GPU backend doesn't track loss history; force CPU for this test
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE, resource = "cpu")
  p <- suppressWarnings(plot(m, type = "loss"))
  expect_s3_class(p, "gg")
})

test_that("plot.nmf returns ggplot for convergence type", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  # GPU backend doesn't track loss history; force CPU for this test
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE, resource = "cpu")
  p <- plot(m, type = "convergence")
  expect_s3_class(p, "gg")
})

test_that("plot.nmf returns ggplot for sparsity type", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE)
  p <- plot(m, type = "sparsity")
  expect_s3_class(p, "gg")
})

test_that("plot.nmf returns ggplot for regularization type", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  # GPU backend doesn't track loss history; force CPU for this test
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE, resource = "cpu")
  p <- plot(m, type = "regularization")
  expect_s3_class(p, "gg")
})

test_that("biplot.nmf returns ggplot", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE)
  p <- biplot(m, factors = c(1, 2), matrix = "w")
  expect_s3_class(p, "gg")
})

test_that("compare_nmf returns ggplot", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  # GPU backend doesn't track loss history; force CPU for this test
  m1 <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE, resource = "cpu")
  m2 <- nmf(A, k = 5, maxit = 10, seed = 42, verbose = FALSE, resource = "cpu")
  p <- compare_nmf(m1, m2, labels = c("k=3", "k=5"))
  expect_s3_class(p, "gg")
})

test_that("summary and plot.nmfSummary work", {
  skip_if_not_installed("ggplot2")
  data(hawaiibirds, package = "RcppML")
  A <- hawaiibirds
  m <- nmf(A, k = 3, maxit = 10, seed = 42, verbose = FALSE)
  s <- summary(m, group = rep(letters[1:3], length.out = nrow(m@w)))
  expect_true(!is.null(s))
  p <- plot(s)
  expect_s3_class(p, "gg")
})
