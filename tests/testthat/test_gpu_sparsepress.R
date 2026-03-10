# Test GPU StreamPress Integration
#
# Tests st_read_gpu() and st_free_gpu() with actual GPU hardware.
# Skipped automatically if GPU is not available.
# skip_if_no_gpu defined in helper-test-utils.R

test_that("st_read_gpu reads .spz v2 to GPU memory", {
  skip_if_no_sp_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:200, 1:100]
  A <- as(A, "dgCMatrix")

  # Write v2 format (required for GPU decode)
  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path), add = TRUE)
  st_write(A, path, include_transpose = TRUE)  # v2

  # Read directly to GPU
  gpu_mat <- st_read_gpu(path)

  expect_s3_class(gpu_mat, "gpu_sparse_matrix")
  expect_equal(gpu_mat$m, nrow(A))
  expect_equal(gpu_mat$n, ncol(A))
  expect_equal(gpu_mat$nnz, length(A@x))
  expect_equal(gpu_mat$device, 0L)

  # Pointers should be non-zero

  expect_true(gpu_mat$.col_ptr != 0)
  expect_true(gpu_mat$.row_idx != 0)
  expect_true(gpu_mat$.values != 0)

  # Clean up GPU memory
  st_free_gpu(gpu_mat)
})


test_that("st_free_gpu frees GPU memory", {
  skip_if_no_sp_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:100, 1:50]
  A <- as(A, "dgCMatrix")

  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path), add = TRUE)
  st_write(A, path, include_transpose = TRUE)

  gpu_mat <- st_read_gpu(path)
  expect_true(gpu_mat$.col_ptr != 0)

  # Free explicitly
  result <- st_free_gpu(gpu_mat)
  expect_null(result)

  # Pointers should be zeroed
  expect_equal(gpu_mat$.col_ptr, 0)
  expect_equal(gpu_mat$.row_idx, 0)
  expect_equal(gpu_mat$.values, 0)
})


test_that("st_free_gpu errors on non-gpu_sparse_matrix input", {
  skip_if_no_sp_gpu()

  expect_error(st_free_gpu(list(a = 1)), "gpu_sparse_matrix")
  expect_error(st_free_gpu("not_a_matrix"), "gpu_sparse_matrix")
})


test_that("st_read_gpu errors on missing file", {
  skip_if_no_sp_gpu()

  expect_error(st_read_gpu("/nonexistent/path.spz"))
})


test_that("gpu_sparse_matrix has correct print/dim methods", {
  skip_if_no_sp_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:100, 1:50]
  A <- as(A, "dgCMatrix")

  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path), add = TRUE)
  st_write(A, path, include_transpose = TRUE)

  gpu_mat <- st_read_gpu(path)
  on.exit(st_free_gpu(gpu_mat), add = TRUE)

  # dim methods
  expect_equal(dim(gpu_mat), c(100L, 50L))
  expect_equal(nrow(gpu_mat), 100L)
  expect_equal(ncol(gpu_mat), 50L)

  # print method should not error
  expect_output(print(gpu_mat), "GPU Sparse Matrix")
})


test_that("NMF with gpu_sparse_matrix input works (zero-copy)", {
  skip_if_no_sp_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path), add = TRUE)
  st_write(A, path, include_transpose = TRUE)

  gpu_mat <- st_read_gpu(path)
  on.exit(st_free_gpu(gpu_mat), add = TRUE)

  # Run NMF on GPU-resident data
  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- nmf(gpu_mat, k = 5, maxit = 10, tol = 1e-10, verbose = FALSE)

  # Should produce valid output
  expect_true(is(result, "nmf") || is.list(result))

  if (is(result, "nmf")) {
    W <- result@w; d <- result@d; H <- result@h
  } else {
    W <- result$w; d <- result$d; H <- result$h
  }

  expect_equal(ncol(W), 5)
  expect_equal(nrow(W), 300)
  expect_equal(length(d), 5)
  expect_equal(nrow(H), 5)
  expect_equal(ncol(H), 150)

  # Factors should be finite and non-negative
  expect_true(all(is.finite(W)))
  expect_true(all(is.finite(H)))
  expect_true(all(W >= 0))
  expect_true(all(H >= 0))
})


test_that("GPU NMF from .spz matches CPU NMF from dgCMatrix", {
  skip_if_no_sp_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path), add = TRUE)
  st_write(A, path, include_transpose = TRUE)

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  # CPU reference from dgCMatrix
  options(RcppML.gpu = FALSE)
  cpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  # GPU from .spz
  gpu_mat <- st_read_gpu(path)
  on.exit(st_free_gpu(gpu_mat), add = TRUE)
  options(RcppML.gpu = TRUE)
  gpu_result <- nmf(gpu_mat, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto")

  # Compare MSE
  cpu_mse <- compute_mse(A, cpu_result)
  gpu_mse <- compute_mse(A, gpu_result)

  # Both should achieve similar MSE (within 15% due to GPU precision + path differences)
  expect_true(abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16) < 0.15,
              label = sprintf("CPU MSE=%.6e, GPU .spz MSE=%.6e", cpu_mse, gpu_mse))
})
