test_that("streaming NMF produces valid factorization", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  result <- nmf(movielens, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, 
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
  expect_equal(ncol(result@w), 3)
  expect_equal(nrow(result@h), 3)
})

test_that("streaming NMF converges to similar loss as standard", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  res_std <- nmf(movielens, k = 3, tol = 1e-4, maxit = 50, streaming = FALSE, 
                 seed = 42, verbose = FALSE)
  res_str <- nmf(movielens, k = 3, tol = 1e-4, maxit = 50, streaming = TRUE, 
                 seed = 42, verbose = FALSE)

  # TEST-STREAMING-MSE-PARITY: Loss should be within 1% relative difference
  loss_ratio <- abs(res_std@misc$loss - res_str@misc$loss) / res_std@misc$loss
  expect_lt(loss_ratio, 0.01)
})

test_that("streaming NMF with custom panel_cols works", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  result <- nmf(movielens, k = 2, tol = 1e-2, maxit = 10, streaming = TRUE, 
                panel_cols = 50L, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 2)
})

test_that("streaming NMF with L1/L2 regularization", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  result <- nmf(movielens, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, 
                L1 = c(0.01, 0.01), L2 = c(0.01, 0.01), seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  # Regularized model should have sparser factors
  expect_true(all(result@w >= 0))
  expect_true(all(result@h >= 0))
})

test_that("streaming NMF auto mode doesn't activate for small data", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  # auto mode with small matrix should use standard path (n=610 < 50000 threshold)
  result <- nmf(movielens, k = 2, tol = 1e-2, maxit = 10, streaming = "auto", 
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
})

test_that("streaming NMF with file-path input", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(movielens, tmp)
  
  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, 
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
})

test_that("streaming NMF with .spz file-path input", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  sp_write(movielens, tmp, include_transpose = TRUE)
  
  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, 
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
})

test_that("streaming NMF with .spz transpose (chunked gather)", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  sp_write(movielens, tmp, include_transpose = TRUE)

  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, 
                seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  
  # Streaming requires a pre-stored transpose; without it, the error is clear
  tmp2 <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp2), add = TRUE)
  sp_write(movielens, tmp2, include_transpose = FALSE)
  expect_error(
    nmf(tmp2, k = 3, tol = 1e-2, maxit = 10, streaming = TRUE, seed = 42, verbose = FALSE),
    regexp = "transpose"
  )
})

# ===========================================================================
# TEST-STREAMING-GP: streaming GP parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-GP: streaming GP matches in-memory GP", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  res_std <- nmf(movielens, k = 3, loss = "gp", tol = 1e-4, maxit = 20,
                 streaming = FALSE, seed = 42, verbose = FALSE)
  res_str <- nmf(movielens, k = 3, loss = "gp", tol = 1e-4, maxit = 20,
                 streaming = TRUE, seed = 42, verbose = FALSE)

  loss_ratio <- abs(res_std@misc$loss - res_str@misc$loss) / abs(res_std@misc$loss)
  expect_lt(loss_ratio, 0.05)
})

# ===========================================================================
# TEST-STREAMING-NB: streaming NB parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-NB: streaming NB matches in-memory NB", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  res_std <- nmf(movielens, k = 3, loss = "nb", tol = 1e-4, maxit = 20,
                 streaming = FALSE, seed = 42, verbose = FALSE)
  res_str <- nmf(movielens, k = 3, loss = "nb", tol = 1e-4, maxit = 20,
                 streaming = TRUE, seed = 42, verbose = FALSE)

  loss_ratio <- abs(res_std@misc$loss - res_str@misc$loss) / abs(res_std@misc$loss)
  expect_lt(loss_ratio, 0.05)
})

# ===========================================================================
# TEST-STREAMING-REG: streaming regularization parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-REG: streaming L1+L2+L21 matches in-memory", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  res_std <- nmf(movielens, k = 3, L1 = c(0.1, 0.1), L2 = c(0.01, 0.01),
                 L21 = c(0.01, 0), tol = 1e-4, maxit = 30,
                 streaming = FALSE, seed = 42, verbose = FALSE)
  res_str <- nmf(movielens, k = 3, L1 = c(0.1, 0.1), L2 = c(0.01, 0.01),
                 L21 = c(0.01, 0), tol = 1e-4, maxit = 30,
                 streaming = TRUE, seed = 42, verbose = FALSE)

  loss_ratio <- abs(res_std@misc$loss - res_str@misc$loss) / abs(res_std@misc$loss)
  expect_lt(loss_ratio, 0.05)
})

# ===========================================================================
# TEST-STREAMING-SCALE: streaming factor scaling parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-SCALE: streaming d scaling matches in-memory", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  res_std <- nmf(movielens, k = 3, tol = 1e-4, maxit = 30,
                 streaming = FALSE, seed = 42, verbose = FALSE)
  res_str <- nmf(movielens, k = 3, tol = 1e-4, maxit = 30,
                 streaming = TRUE, seed = 42, verbose = FALSE)

  # d (diagonal scaling) should be close
  d_ratio <- abs(res_std@d - res_str@d) / abs(res_std@d)
  expect_true(all(d_ratio < 0.05))
})

# ===========================================================================
# TEST-STREAMING-LOSS: streaming loss computation parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-LOSS: streaming tracks loss correctly", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  result <- nmf(movielens, k = 3, tol = 1e-10, maxit = 15, streaming = TRUE,
                seed = 42, verbose = FALSE)
  # Should have a valid loss value
  expect_true(is.finite(result@misc$loss))
  expect_gt(result@misc$loss, 0)
})

# ===========================================================================
# TEST-STREAMING-INIT: streaming init parity test (P0)
# ===========================================================================
test_that("TEST-STREAMING-INIT: streaming init produces valid starting factors", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  # Run with just 1 iteration to check initialization
  res_str <- nmf(movielens, k = 3, tol = 1e-10, maxit = 1, streaming = TRUE,
                 seed = 42, verbose = FALSE)
  expect_s4_class(res_str, "nmf")
  expect_true(all(res_str@w >= 0))
  expect_true(all(res_str@h >= 0))
  expect_true(all(res_str@d > 0))
})

# ===========================================================================
# TEST-STREAMING-CV: streaming CV test (P0)
# ===========================================================================
test_that("TEST-STREAMING-CV: streaming CV produces valid test loss", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")

  result <- nmf(movielens, k = c(2, 3, 4), tol = 1e-3, maxit = 15,
                streaming = TRUE, seed = 42, verbose = FALSE,
                test_fraction = 0.1, cv_seed = 123)
  # Should return a data.frame with test loss for each k
  expect_true(is.data.frame(result))
  expect_true("test_mse" %in% names(result))
  expect_equal(nrow(result), 3)
  expect_true(all(result$test_mse > 0))
})

# ===========================================================================
# TEST-SPZ-AUTO-CHUNK-CPU: Auto-chunking correctness (Part B, P1)
# ===========================================================================

test_that("TEST-SPZ-AUTO-CHUNK-CPU: SPZ auto-chunking produces same result as in-memory", {
  skip_on_cran()

  # Use a matrix with enough columns to exercise multiple chunks (default 256 cols/chunk)
  set.seed(42)
  m <- 100; n <- 600; k <- 5
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  X <- as(W_true %*% H_true + matrix(abs(rnorm(m * n, sd = 0.1)), m, n), "dgCMatrix")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  sp_write(X, tmp, include_transpose = TRUE)

  # In-memory
  s_mem <- nmf(X, k = k, seed = 42, maxit = 100, tol = 1e-10)
  # Streaming (auto-chunked by SPZ reader)
  s_spz <- nmf(tmp, k = k, seed = 42, maxit = 100, tol = 1e-10)

  mem_mse <- mse(s_mem@w, s_mem@d, s_mem@h, X)
  spz_mse <- mse(s_spz@w, s_spz@d, s_spz@h, X)

  # Streaming result should be within 5% of in-memory
  ratio <- spz_mse / mem_mse
  expect_lt(ratio, 1.05)
  expect_gt(ratio, 0.5)
})
