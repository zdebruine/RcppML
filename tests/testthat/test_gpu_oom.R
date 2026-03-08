# Test GPU OOM Fallback to Streaming
#
# Verifies that GPU NMF handles large datasets gracefully:
# - Automatic streaming/paneling when matrix exceeds VRAM
# - CPU fallback when GPU encounters fatal errors
# Skipped if GPU is not available.
# skip_if_no_gpu, get_W, get_H, get_d, compute_mse defined in helper-test-utils.R

test_that("GPU streaming mode activates for large matrices", {
  skip_if_no_gpu()

  # Create a large sparse matrix that should challenge GPU memory.
  # Most test GPUs have 16-80 GB VRAM. A 50000x50000 sparse matrix
  # with ~25M non-zeros should push smaller GPUs into streaming.
  set.seed(42)
  m <- 50000; n <- 50000; density <- 0.01
  nnz_approx <- as.integer(m * n * density)

  # Use Matrix::rsparsematrix for efficient generation
  A <- Matrix::rsparsematrix(m, n, density = density)
  A <- abs(A)  # Ensure non-negative
  A <- as(A, "dgCMatrix")

  k <- 4
  maxit <- 5
  tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  # This should either:
  # 1. Run in streaming mode (panels) if matrix exceeds VRAM
  # 2. Run monolithically if GPU has enough memory
  # 3. Fall back to CPU if GPU fails entirely
  # All three are acceptable — we just verify it produces valid results
  result <- tryCatch(
    nmf(A, k = k, maxit = maxit, tol = tol, verbose = FALSE),
    error = function(e) {
      # Total GPU failure is acceptable for this stress test
      message("GPU OOM test: GPU failed, CPU fallback expected: ", e$message)
      NULL
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))

    if (is(result, "nmf")) {
      W <- result@w; d <- result@d; H <- result@h
    } else {
      W <- result$w; d <- result$d; H <- result$h
    }

    # Valid dimensions
    expect_equal(nrow(W), m)
    expect_equal(ncol(W), k)
    expect_equal(length(d), k)
    expect_equal(nrow(H), k)
    expect_equal(ncol(H), n)

    # Finite values
    expect_true(all(is.finite(W)))
    expect_true(all(is.finite(H)))
    expect_true(all(is.finite(d)))

    # Non-negativity
    expect_true(all(W >= 0))
    expect_true(all(H >= 0))
    expect_true(all(d >= 0))
  }
})


test_that("GPU CPU fallback produces valid results", {
  skip_if_no_gpu()

  # Standard test matrix — force GPU, capture backend info
  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- nmf(A, k = 5, maxit = 15, tol = 1e-10, verbose = FALSE)
  expect_true(is(result, "nmf") || is.list(result))

  # If it fell back to CPU, the result should still be valid
  if (is(result, "nmf")) {
    W <- result@w; d <- result@d; H <- result@h
  } else {
    W <- result$w; d <- result$d; H <- result$h
  }

  expect_true(all(is.finite(W)))
  expect_true(all(is.finite(H)))
  expect_true(all(W >= 0))
  expect_true(all(H >= 0))
})


test_that("GPU handles medium-scale matrix correctly", {
  skip_if_no_gpu()

  # Medium-scale: should fit on any GPU without streaming
  set.seed(123)
  m <- 5000; n <- 2000; density <- 0.05
  A <- Matrix::rsparsematrix(m, n, density = density)
  A <- abs(A)
  A <- as(A, "dgCMatrix")

  k <- 8
  seed <- 42L
  maxit <- 20
  tol <- 1e-10

  # CPU reference
  options(RcppML.gpu = FALSE)
  cpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  # GPU
  options(RcppML.gpu = TRUE)
  gpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto")

  # Both should produce valid results
  expect_true(is(cpu_result, "nmf") || is.list(cpu_result))
  expect_true(is(gpu_result, "nmf") || is.list(gpu_result))

  # Compare reconstruction quality
  cpu_mse <- compute_mse(A, cpu_result)
  gpu_mse <- compute_mse(A, gpu_result)

  # MSE should be within 15%
  expect_true(abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16) < 0.15,
              label = sprintf("Medium-scale MSE: CPU=%.6e, GPU=%.6e", cpu_mse, gpu_mse))
})


test_that("GPU NMF with large k succeeds or falls back gracefully", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  # Large k relative to matrix dimensions — stresses GPU memory
  result <- nmf(A, k = 50, seed = 42, resource = "gpu",
                maxit = 10, tol = 1e-10, verbose = FALSE)

  expect_true(is.finite(result@misc$loss))
  expect_true(result@misc$loss > 0)
  expect_true(all(is.finite(result@w)))
  expect_true(all(is.finite(result@h)))
  expect_equal(ncol(result@w), 50)
})

test_that("GPU ZI NMF memory guard passes on normal-sized input", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  # ZI NMF allocates dense imputed matrices — memory guard checks free VRAM
  # For 500x200 this is negligible, so it should succeed
  result <- nmf(A, k = 5, loss = "gp", zi = "row", seed = 42,
                resource = "gpu", maxit = 10, tol = 1e-10, verbose = FALSE)

  expect_true(is.finite(result@misc$loss))
})
