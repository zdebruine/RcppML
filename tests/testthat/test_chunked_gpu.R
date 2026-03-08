# Tests for GPU chunked NMF path (fit_chunked_gpu.cuh)
# Exercises the DataLoader-based chunked streaming path that keeps W on GPU
# and streams A in panels to handle matrices larger than GPU VRAM.
#
# All tests skip gracefully on CPU-only nodes or when GPU is unavailable.
# The chunked path is triggered by specifying a small panel_cols value.

skip_if_not_installed("RcppML")
library(RcppML)
library(Matrix)

gpu_ok <- tryCatch(gpu_available(force_recheck = TRUE), error = function(e) FALSE)

# gpu_chunked_nmf

# -----------------------------------------------------------------------
# Basic functionality: chunked GPU NMF returns valid nmf object
# -----------------------------------------------------------------------
test_that("GPU chunked NMF returns valid result on sparse matrix", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(42)
  m <- 200; n <- 500; k <- 5
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.1))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  # Use a small panel_cols to force the chunked streaming path
  result <- nmf(A, k = k, seed = 123, maxit = 30, tol = 1e-8,
                resource = "gpu", panel_cols = 50)

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), m)
  expect_equal(ncol(result@h), n)
  expect_equal(length(result@d), k)
  expect_true(all(result@w >= 0), info = "W must be non-negative")
  expect_true(all(result@h >= 0), info = "H must be non-negative")
  expect_true(all(result@d > 0),  info = "d must be positive")
})

# -----------------------------------------------------------------------
# Loss decreases: chunked NMF converges (MSE decreases with more iters)
# -----------------------------------------------------------------------
test_that("GPU chunked NMF achieves lower loss at higher maxit", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(1)
  m <- 150; n <- 300; k <- 4
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.15))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  result_short <- nmf(A, k = k, seed = 77, maxit = 5,  tol = 1e-12,
                      resource = "gpu", panel_cols = 60)
  result_long  <- nmf(A, k = k, seed = 77, maxit = 50, tol = 1e-12,
                      resource = "gpu", panel_cols = 60)

  mse_short <- evaluate(result_short, A)
  mse_long  <- evaluate(result_long, A)
  expect_lt(mse_long, mse_short, label = "More iterations → lower MSE")
})

# -----------------------------------------------------------------------
# Reproducibility: same seed → identical chunked results
# -----------------------------------------------------------------------
test_that("GPU chunked NMF is reproducible with same seed", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(99)
  m <- 100; n <- 200; k <- 3
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.2))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  r1 <- nmf(A, k = k, seed = 42, maxit = 20, tol = 1e-10,
            resource = "gpu", panel_cols = 40)
  r2 <- nmf(A, k = k, seed = 42, maxit = 20, tol = 1e-10,
            resource = "gpu", panel_cols = 40)

  expect_equal(r1@w, r2@w, tolerance = 1e-8, info = "W not reproducible")
  expect_equal(r1@h, r2@h, tolerance = 1e-8, info = "H not reproducible")
  expect_equal(r1@d, r2@d, tolerance = 1e-8, info = "d not reproducible")
})

# -----------------------------------------------------------------------
# Consistency: chunked and non-chunked paths give similar results
# -----------------------------------------------------------------------
test_that("GPU chunked and non-chunked paths give similar loss", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(5)
  m <- 120; n <- 250; k <- 4
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.1))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  # Standard GPU NMF (panel_cols = 0 → auto, usually no chunking for small A)
  r_full <- nmf(A, k = k, seed = 10, maxit = 40, tol = 1e-8,
                resource = "gpu", panel_cols = 0)

  # Forced chunked (small panel)
  r_chunk <- nmf(A, k = k, seed = 10, maxit = 40, tol = 1e-8,
                 resource = "gpu", panel_cols = 50)

  mse_full  <- evaluate(r_full, A)
  mse_chunk <- evaluate(r_chunk, A)

  # Both should achieve reasonable reconstruction; chunked may differ slightly
  # due to floating-point ordering in panel accumulation, but MSE should be
  # within a factor of 2 of each other
  expect_lt(mse_chunk / mse_full, 2.0,
            label = "Chunked MSE should not be much worse than full GPU MSE")
  expect_lt(mse_full / mse_chunk, 2.0,
            label = "Full GPU MSE should not be much worse than chunked MSE")
})

# -----------------------------------------------------------------------
# Different panel sizes produce valid (not necessarily identical) results
# -----------------------------------------------------------------------
test_that("GPU chunked NMF works across different panel sizes", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(7)
  m <- 80; n <- 300; k <- 3
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.2))
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  for (panel in c(30, 75, 150, 300)) {
    result <- nmf(A, k = k, seed = 1, maxit = 20, tol = 1e-8,
                  resource = "gpu", panel_cols = panel)
    expect_s4_class(result, "nmf")
    expect_true(all(result@d > 0),
                info = sprintf("panel_cols = %d: d not positive", panel))
  }
})
