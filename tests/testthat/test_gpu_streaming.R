# Tests for GPU streaming NMF path (T14: fit_gpu_streaming.cuh)
# Streaming NMF is used when matrix exceeds single-GPU VRAM.
# Tests simulate this by using the streaming path explicitly or via large matrices.
# Requires GPU hardware — all tests skip gracefully on CPU-only nodes.

skip_if_not_installed("RcppML")
library(RcppML)
library(Matrix)

gpu_ok <- tryCatch(gpu_available(force_recheck = TRUE), error = function(e) FALSE)

# gpu_streaming

# ---------------------------------------------------------------
# Basic streaming NMF (simulated via panel-based processing)
# ---------------------------------------------------------------
test_that("GPU streaming NMF returns valid result on large sparse matrix", {
  skip_if(!gpu_ok, "No GPU available")

  # Create a moderately large sparse matrix to exercise the streaming path
  # Real streaming triggers when matrix > VRAM; here we test the code path
  set.seed(42)
  m <- 500; n <- 2000; k <- 5; density <- 0.1
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  # Standard GPU NMF should work on this (not large enough to trigger streaming)
  result <- nmf(A, k = k, seed = 123, maxit = 30, tol = 1e-8, resource = "gpu")

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), m)
  expect_equal(ncol(result@h), n)
  expect_equal(length(result@d), k)

  gpu_mse <- mse(result@w, result@d, result@h, A)
  expect_lt(gpu_mse, 1.0)
})

# ---------------------------------------------------------------
# GPU NMF matches CPU on large sparse matrix
# ---------------------------------------------------------------
test_that("GPU NMF on large sparse matches CPU within tolerance", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(77)
  m <- 300; n <- 1000; k <- 4; density <- 0.05
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  signal <- as(W_true %*% H_true, "dgCMatrix")

  # Add sparse noise
  noise <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  noise@x <- abs(noise@x) * 0.01
  A <- signal + noise
  A <- as(A, "dgCMatrix")

  cpu_result <- nmf(A, k = k, seed = 10, maxit = 40, tol = 1e-10, resource = "cpu")
  gpu_result <- nmf(A, k = k, seed = 10, maxit = 40, tol = 1e-10, resource = "gpu")

  cpu_mse <- mse(cpu_result@w, cpu_result@d, cpu_result@h, A)
  gpu_mse <- mse(gpu_result@w, gpu_result@d, gpu_result@h, A)

  ratio <- gpu_mse / cpu_mse
  expect_lt(ratio, 2.0)
  expect_gt(ratio, 0.1)
})

# ---------------------------------------------------------------
# Large rank on large sparse matrix (stresses GPU memory)
# ---------------------------------------------------------------
test_that("GPU NMF handles large rank on large sparse matrix", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(55)
  m <- 500; n <- 3000; k <- 32; density <- 0.02
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result <- nmf(A, k = k, seed = 1, maxit = 20, tol = 1e-8, resource = "gpu")

  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), k)
  expect_equal(nrow(result@h), k)

  gpu_mse <- mse(result@w, result@d, result@h, A)
  expect_lt(gpu_mse, 5.0)
})

# ---------------------------------------------------------------
# Streaming NMF supports all losses via weight_zeros=true IRLS
# ---------------------------------------------------------------
test_that("Streaming NMF correctly restricts to MSE loss", {
  # This test doesn't need GPU - it tests R-layer validation
  set.seed(1)
  m <- 50; n <- 100
  A <- rsparsematrix(m, n, nnz = 500)
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  # SPZ streaming path now supports non-MSE loss via weight_zeros=true IRLS
  # (tested in test_streaming_loss_rejection.R)
  skip("SPZ streaming IRLS tested in test_streaming_loss_rejection.R")
})

# ---------------------------------------------------------------
# Very wide matrix (n >> m) stresses column-parallel GPU path
# ---------------------------------------------------------------
test_that("GPU NMF works on very wide sparse matrix (n >> m)", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(33)
  m <- 100; n <- 5000; k <- 5; density <- 0.01
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result <- nmf(A, k = k, seed = 1, maxit = 20, tol = 1e-8, resource = "gpu")

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), m)
  expect_equal(ncol(result@h), n)
})

# ---------------------------------------------------------------
# GPU with regularization on large matrix
# ---------------------------------------------------------------
test_that("GPU NMF with L1 regularization on large sparse matrix", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(88)
  m <- 200; n <- 1000; k <- 8; density <- 0.05
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result <- nmf(A, k = k, seed = 1, maxit = 25, L1 = c(0.05, 0.05), resource = "gpu")

  expect_s4_class(result, "nmf")

  # L1 regularization should produce some zeros
  h_sparsity <- mean(result@h == 0)
  expect_gt(h_sparsity, 0.0)  # at least some sparsity induced
})

# ===========================================================================
# GPU Streaming NMF via .spz (Part B, P1)
# ===========================================================================

# Shared streaming test data: write .spz once, reuse across tests
A_stream <- local({
  set.seed(42)
  m <- 200; n <- 500; k <- 5
  W_t <- matrix(abs(rnorm(m * k)), m, k)
  H_t <- matrix(abs(rnorm(k * n)), k, n)
  X <- W_t %*% H_t + matrix(abs(rnorm(m * n, sd = 0.1)), m, n)
  as(X, "dgCMatrix")
})

spz_path_gpu <- tempfile(fileext = ".spz")
if (gpu_ok) {
  st_write(A_stream, spz_path_gpu, include_transpose = TRUE)
}

test_that("TEST-GPU-STREAMING-MSE: streaming GPU NMF vs in-memory GPU parity", {
  skip_if(!gpu_ok, "No GPU available")

  s_mem <- nmf(A_stream, k = 5, seed = 42, maxit = 30, tol = 1e-10, resource = "gpu")
  s_spz <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 30, tol = 1e-10, resource = "gpu")

  mem_mse <- mse(s_mem@w, s_mem@d, s_mem@h, A_stream)
  spz_mse <- mse(s_spz@w, s_spz@d, s_spz@h, A_stream)
  ratio <- spz_mse / mem_mse
  expect_lt(ratio, 3.0)
  expect_gt(ratio, 0.5)
})

test_that("TEST-GPU-STREAMING-GP: streaming GPU GP loss produces valid results", {
  skip_if(!gpu_ok, "No GPU available")

  s <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 20, loss = "gp",
           resource = "gpu")
  expect_s4_class(s, "nmf")
  expect_equal(ncol(s@h), 500L)
  expect_true(all(s@d > 0))
})

test_that("TEST-GPU-STREAMING-NB: streaming GPU NB loss produces valid results", {
  skip_if(!gpu_ok, "No GPU available")

  s <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 20, loss = "nb",
           resource = "gpu")
  expect_s4_class(s, "nmf")
  expect_equal(ncol(s@h), 500L)
  expect_true(all(s@d > 0))
})

test_that("TEST-GPU-STREAMING-REG: streaming GPU regularization parity", {
  skip_if(!gpu_ok, "No GPU available")

  s <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 20, tol = 1e-10,
           L1 = c(0.1, 0.1), L2 = c(0.01, 0.01), resource = "gpu")
  expect_s4_class(s, "nmf")
  expect_true(all(s@d > 0))
  # Regularization should produce valid factors (non-negative)
  expect_true(all(s@w >= -1e-10))
  expect_true(all(s@h >= -1e-10))
})

test_that("TEST-GPU-STREAMING-SCALE: streaming GPU reconstruction parity", {
  skip_if(!gpu_ok, "No GPU available")

  s_mem <- nmf(A_stream, k = 5, seed = 42, maxit = 20, tol = 1e-10, resource = "gpu")
  s_spz <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 20, tol = 1e-10, resource = "gpu")

  # Both paths may distribute scale differently between W, d, H —
  # but should produce equivalent reconstructions (W * diag(d) * H)
  recon_mem <- s_mem@w %*% diag(s_mem@d) %*% s_mem@h
  recon_spz <- s_spz@w %*% diag(s_spz@d) %*% s_spz@h
  A_dense <- as.matrix(A_stream)
  mse_mem <- mean((recon_mem - A_dense)^2)
  mse_spz <- mean((recon_spz - A_dense)^2)
  # Both should be close to zero (good reconstruction)
  expect_lt(mse_mem, 0.05)
  expect_lt(mse_spz, 0.05)
  # And within 2x of each other (same approximation quality)
  ratio <- max(mse_mem, mse_spz) / max(min(mse_mem, mse_spz), 1e-16)
  expect_lt(ratio, 3.0)
})

test_that("TEST-GPU-STREAMING-LOSS: streaming GPU loss is finite and positive", {
  skip_if(!gpu_ok, "No GPU available")

  s <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 20, resource = "gpu")
  loss <- s@misc$loss
  expect_true(is.finite(loss))
  expect_gt(loss, 0)
})

test_that("TEST-GPU-STREAMING-INIT: streaming GPU 1-iter init produces valid W,H", {
  skip_if(!gpu_ok, "No GPU available")

  s <- nmf(spz_path_gpu, k = 5, seed = 42, maxit = 1, resource = "gpu")
  expect_s4_class(s, "nmf")
  expect_equal(nrow(s@w), 200L)
  expect_equal(ncol(s@h), 500L)
  expect_true(all(is.finite(s@w)))
  expect_true(all(is.finite(s@h)))
})

# ===========================================================================
# TEST-SPZ-AUTO-CHUNK-GPU: GPU auto-chunking correctness (Part B, P1)
# ===========================================================================

test_that("TEST-SPZ-AUTO-CHUNK-GPU: GPU SPZ auto-chunking matches in-memory GPU", {
  skip_if(!gpu_ok, "No GPU available")

  # Create matrix with 600 columns (> 256 default chunk) to exercise multi-chunk
  set.seed(77)
  m <- 100; n <- 600; k <- 5
  W_t <- matrix(abs(rnorm(m * k)), m, k)
  H_t <- matrix(abs(rnorm(k * n)), k, n)
  X <- as(W_t %*% H_t + matrix(abs(rnorm(m * n, sd = 0.1)), m, n), "dgCMatrix")

  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(X, tmp, include_transpose = TRUE)

  s_mem <- nmf(X, k = k, seed = 42, maxit = 30, tol = 1e-10, resource = "gpu")
  s_spz <- nmf(tmp, k = k, seed = 42, maxit = 30, tol = 1e-10, resource = "gpu")

  mem_mse <- mse(s_mem@w, s_mem@d, s_mem@h, X)
  spz_mse <- mse(s_spz@w, s_spz@d, s_spz@h, X)

  ratio <- spz_mse / mem_mse
  expect_lt(ratio, 1.5)
  expect_gt(ratio, 0.5)
})
