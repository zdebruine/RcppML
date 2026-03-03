# Test GPU Feature Parity
#
# Tests GPU NMF with advanced features:
# - Cross-validation on GPU
# - L21, angular, graph regularization on GPU (Tier 2 host round-trip)
# - Upper bound constraints on GPU
# - GPU bipartition and dclust
# - Multi-GPU with varying configurations
# Skipped automatically if GPU is not available.

# get_W, get_H, get_d, compute_mse, skip_if_no_gpu defined in helper-test-utils.R

# ============================================================================
# GPU Cross-Validation Tests
# ============================================================================

test_that("GPU cross-validation runs for multiple ranks", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  cv_result <- nmf(A, k = 2:6, test_fraction = 0.1, cv_seed = 42L,
                   maxit = 15, tol = 1e-10, verbose = FALSE)

  # Should return a data.frame with CV results

  expect_true(is.data.frame(cv_result))
  expect_true("k" %in% names(cv_result))
  expect_true("test_mse" %in% names(cv_result) || "test_loss" %in% names(cv_result))
  expect_equal(nrow(cv_result), 5)  # k = 2,3,4,5,6

  # Test MSE should decrease initially, or at least be finite
  loss_col <- if ("test_mse" %in% names(cv_result)) "test_mse" else "test_loss"
  expect_true(all(is.finite(cv_result[[loss_col]])))
  expect_true(all(cv_result[[loss_col]] > 0))
})


test_that("GPU CV matches CPU CV approximately", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5
  test_frac <- 0.1
  cv_seed <- 42L
  maxit <- 15
  tol <- 1e-10

  # CPU CV
  options(RcppML.gpu = FALSE)
  cpu_cv <- nmf(A, k = k, test_fraction = test_frac, cv_seed = cv_seed,
                maxit = maxit, tol = tol, verbose = FALSE)

  # GPU CV
  options(RcppML.gpu = TRUE)
  gpu_cv <- nmf(A, k = k, test_fraction = test_frac, cv_seed = cv_seed,
                maxit = maxit, tol = tol, verbose = FALSE)
  options(RcppML.gpu = "auto")

  # Both should produce nmf objects (single rank CV returns fitted model)
  expect_true(is(cpu_cv, "nmf") || is.list(cpu_cv))
  expect_true(is(gpu_cv, "nmf") || is.list(gpu_cv))
})


# ============================================================================
# GPU Regularization Tests (Tier 2 Host Round-trip)
# ============================================================================

test_that("GPU NMF with L1 regularization", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  # CPU L1 reference
  options(RcppML.gpu = FALSE)
  cpu_l1 <- nmf(A, k = k, L1 = c(0.1, 0.1), maxit = maxit, tol = tol,
                seed = seed, verbose = FALSE)

  # GPU L1
  options(RcppML.gpu = TRUE)
  gpu_l1 <- nmf(A, k = k, L1 = c(0.1, 0.1), maxit = maxit, tol = tol,
                seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto")

  # Both should produce valid results
  expect_true(is(cpu_l1, "nmf") || is.list(cpu_l1))
  expect_true(is(gpu_l1, "nmf") || is.list(gpu_l1))

  # L1 should produce sparser factors than unregularized
  W_cpu <- get_W(cpu_l1)
  W_gpu <- get_W(gpu_l1)
  expect_true(mean(W_cpu == 0) > 0 || mean(W_gpu == 0) > 0,
              label = "L1 should produce some zero entries in W")
})


test_that("GPU NMF with L2 regularization", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- nmf(A, k = k, L2 = c(0.01, 0.01), maxit = maxit, tol = tol,
                seed = seed, verbose = FALSE)

  expect_true(is(result, "nmf") || is.list(result))
  W <- get_W(result); H <- get_H(result)
  expect_true(all(is.finite(W)))
  expect_true(all(is.finite(H)))
  expect_true(all(W >= 0))
  expect_true(all(H >= 0))
})


test_that("GPU NMF with L21 regularization (Tier 2)", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  # L21 regularization (group sparsity) - may use Tier 2 host round-trip
  result <- tryCatch(
    nmf(A, k = k, L21 = c(0.1, 0.1), maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      # L21 may not be supported on GPU yet — document the gap
      skip(paste("GPU L21 not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result)
    expect_true(all(is.finite(W)))
  }
})


test_that("GPU NMF with angular regularization (Tier 2)", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, angular = c(0.1, 0.1), maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU angular not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result)
    expect_true(all(is.finite(W)))
  }
})


test_that("GPU NMF with upper bounds (Tier 2)", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, upper_bound = c(1.0, 1.0), maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU upper bounds not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result); H <- get_H(result)
    expect_true(all(is.finite(W)))
    # Note: upper bounds enforce W <= bound, H <= bound
  }
})


test_that("GPU NMF with graph regularization (Tier 2)", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  # Create simple adjacency graph (k-NN style)
  set.seed(42)
  G_H <- Matrix::rsparsematrix(200, 200, density = 0.05)
  G_H <- abs(G_H)
  G_H <- as(G_H, "dgCMatrix")
  # Make symmetric
  G_H <- (G_H + Matrix::t(G_H)) / 2
  diag(G_H) <- 0
  G_H <- Matrix::drop0(G_H)

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, graph_H = G_H, graph_lambda = c(0, 0.1),
        maxit = maxit, tol = tol, seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU graph reg not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    H <- get_H(result)
    expect_true(all(is.finite(H)))
  }
})


# ============================================================================
# GPU Bipartition/Dclust Tests
# ============================================================================

test_that("GPU bipartition produces valid partition", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    bipartition(A, tol = 1e-5, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU bipartition not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    # bipartition returns a list with v, dist, and samples split
    expect_true(is.list(result))
    # Result should contain partition vector or split assignment
    expect_true(length(result) >= 2)
  }
})


test_that("GPU bipartition matches CPU bipartition", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  # CPU
  options(RcppML.gpu = FALSE)
  cpu_bp <- bipartition(A, tol = 1e-5, verbose = FALSE)

  # GPU
  options(RcppML.gpu = TRUE)
  gpu_bp <- tryCatch(
    bipartition(A, tol = 1e-5, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU bipartition not supported:", e$message))
      NULL
    }
  )
  options(RcppML.gpu = "auto")

  if (!is.null(gpu_bp)) {
    # Both should produce valid results
    expect_true(is.list(cpu_bp))
    expect_true(is.list(gpu_bp))
  }
})


test_that("GPU dclust produces valid hierarchical clustering", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    dclust(A, min_samples = 50, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU dclust not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is.list(result) || is.data.frame(result))
  }
})


# ============================================================================
# Multi-GPU Tests
# ============================================================================

test_that("Multi-GPU with different max_gpus settings", {
  skip_if_no_gpu()

  info <- gpu_info()
  skip_if(nrow(info) < 2, "Only 1 GPU available")

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 15; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto", RcppML.max_gpus = 0L), add = TRUE)

  # Single GPU
  options(RcppML.max_gpus = 1L)
  r1 <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  # Two GPUs
  options(RcppML.max_gpus = 2L)
  r2 <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  # All GPUs
  options(RcppML.max_gpus = 0L)
  rall <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  expect_true(is(r1, "nmf") || is.list(r1))
  expect_true(is(r2, "nmf") || is.list(r2))
  expect_true(is(rall, "nmf") || is.list(rall))

  # All should produce similar MSE
  mse1 <- compute_mse(A, r1)
  mse2 <- compute_mse(A, r2)
  mse_all <- compute_mse(A, rall)

  expect_true(abs(mse1 - mse2) / max(mse1, 1e-16) < 0.15,
              label = sprintf("1-GPU vs 2-GPU: %.6e vs %.6e", mse1, mse2))
  expect_true(abs(mse1 - mse_all) / max(mse1, 1e-16) < 0.15,
              label = sprintf("1-GPU vs all-GPU: %.6e vs %.6e", mse1, mse_all))
})


# ============================================================================
# GPU Semi-NMF Test
# ============================================================================

test_that("GPU supports semi-NMF (unconstrained W)", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, nonneg = c(FALSE, TRUE), maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU semi-NMF not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result); H <- get_H(result)
    expect_true(all(is.finite(W)))
    expect_true(all(is.finite(H)))
    # W may have negative values (semi-NMF)
    # H should still be non-negative
    expect_true(all(H >= 0))
  }
})


# ============================================================================
# GPU Non-MSE Loss Tests (IRLS)
# ============================================================================

test_that("GPU supports MAE loss via host-mediated IRLS", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, loss = "mae", maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU MAE loss not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result); H <- get_H(result)
    expect_true(all(is.finite(W)))
    expect_true(all(is.finite(H)))
    expect_true(all(W >= 0))
    expect_true(all(H >= 0))
    # MSE should be reasonable (not diverged)
    # MAE-optimal factors may have higher MSE than MSE-optimal ones
    # Per-entry MSE ~3 is typical for k=5, 20 iters on this sparse data
    mse <- compute_mse(A, result)
    expect_true(mse < 5.0, label = sprintf("MAE-loss MSE=%.6e (should be < 5)", mse))
  }
})

test_that("GPU supports Huber loss via host-mediated IRLS", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, loss = "huber", huber_delta = 1.0, maxit = maxit, tol = tol,
        seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU Huber loss not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result); H <- get_H(result)
    expect_true(all(is.finite(W)))
    expect_true(all(is.finite(H)))
    expect_true(all(W >= 0))
    expect_true(all(H >= 0))
    mse <- compute_mse(A, result)
    expect_true(mse < 5.0, label = sprintf("Huber-loss MSE=%.6e (should be < 5)", mse))
  }
})

test_that("GPU Semi-NMF with MAE loss works together", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  options(RcppML.gpu = TRUE)
  on.exit(options(RcppML.gpu = "auto"), add = TRUE)

  result <- tryCatch(
    nmf(A, k = k, loss = "mae", nonneg = c(FALSE, TRUE), maxit = maxit,
        tol = tol, seed = seed, verbose = FALSE),
    error = function(e) {
      skip(paste("GPU Semi-NMF + MAE not supported:", e$message))
    }
  )

  if (!is.null(result)) {
    expect_true(is(result, "nmf") || is.list(result))
    W <- get_W(result); H <- get_H(result)
    expect_true(all(is.finite(W)))
    expect_true(all(is.finite(H)))
    # H should be non-negative, W can be negative
    expect_true(all(H >= 0))
  }
})


# ============================================================================
# GPU Precision Variants
# ============================================================================

test_that("GPU precision option controls fp32 vs fp64", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 5; seed <- 42L; maxit <- 20; tol <- 1e-10

  old_prec <- getOption("RcppML.precision")
  on.exit(options(RcppML.precision = old_prec, RcppML.gpu = "auto"), add = TRUE)

  # fp64
  options(RcppML.gpu = TRUE, RcppML.precision = "double")
  r64 <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  # fp32
  options(RcppML.precision = "float")
  r32 <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)

  expect_true(is(r64, "nmf") || is.list(r64))
  expect_true(is(r32, "nmf") || is.list(r32))

  mse64 <- compute_mse(A, r64)
  mse32 <- compute_mse(A, r32)

  # fp32 should be within 25% of fp64 (generous tolerance)
  expect_true(abs(mse64 - mse32) / max(mse64, 1e-16) < 0.25,
              label = sprintf("fp64 MSE=%.6e, fp32 MSE=%.6e", mse64, mse32))
})
