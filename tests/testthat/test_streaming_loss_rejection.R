# Tests that streaming SPZ NMF correctly handles non-MSE losses.
# Streaming IRLS uses weight_zeros=true so that zero entries get correct
# distribution-derived weights instead of unit weights.

options(RcppML.verbose = FALSE)
options(RcppML.threads = 1)

test_that("streaming SPZ NMF runs with GP loss", {
  skip(".st_dispatch not yet re-implemented")
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(movielens, tmp, include_transpose = TRUE)

  old_opt <- getOption("RcppML.streaming")
  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = old_opt), add = TRUE)

  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, loss = "gp",
                dispersion = "none", seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
  expect_true(is.finite(result@misc$loss))
  expect_true(result@misc$loss >= 0)
})

test_that("streaming SPZ NMF runs with NB loss", {
  skip(".st_dispatch not yet re-implemented")
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(movielens, tmp, include_transpose = TRUE)

  old_opt <- getOption("RcppML.streaming")
  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = old_opt), add = TRUE)

  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, loss = "nb",
                seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
  expect_true(result@misc$loss >= 0)
})

test_that("streaming SPZ NMF runs with robust MSE", {
  skip(".st_dispatch not yet re-implemented")
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(movielens, tmp, include_transpose = TRUE)

  old_opt <- getOption("RcppML.streaming")
  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = old_opt), add = TRUE)

  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, loss = "mse",
                robust = TRUE, seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_true(is.finite(result@misc$loss))
  expect_true(result@misc$loss >= 0)
})

test_that("streaming SPZ NMF with MSE loss still works", {
  skip(".st_dispatch not yet re-implemented")
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(movielens, tmp, include_transpose = TRUE)

  old_opt <- getOption("RcppML.streaming")
  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = old_opt), add = TRUE)

  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, loss = "mse",
                seed = 42, verbose = FALSE)

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
  expect_true(is.finite(result@misc$loss))
  expect_true(result@misc$loss >= 0)
})
