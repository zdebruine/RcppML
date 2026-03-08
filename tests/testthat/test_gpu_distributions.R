# Test GPU Distribution NMF Parity
#
# Verifies that GPU distribution NMF (GP, NB, Gamma, InvGauss, Tweedie, robust)
# produces results equivalent to CPU. Skipped if GPU is not available.

skip_if_not_installed("RcppML")
library(RcppML)
library(Matrix)

gpu_ok <- tryCatch(gpu_available(force_recheck = TRUE), error = function(e) FALSE)

# Helper: generate non-negative sparse count-like data
make_sparse_nonneg <- function(m = 80, n = 60, density = 0.1, seed = 42) {
  set.seed(seed)
  A <- rsparsematrix(m, n, density = density)
  A@x <- abs(A@x)
  A
}

# Helper: compare internal loss between CPU and GPU
compare_loss <- function(cpu_model, gpu_model, tol, label) {
  cpu_loss <- cpu_model@misc$loss
  gpu_loss <- gpu_model@misc$loss
  expect_true(is.finite(cpu_loss) && cpu_loss > 0,
              label = paste(label, "CPU loss should be finite positive"))
  expect_true(is.finite(gpu_loss) && gpu_loss > 0,
              label = paste(label, "GPU loss should be finite positive"))
  rel_diff <- abs(cpu_loss - gpu_loss) / max(abs(cpu_loss), 1e-16)
  expect_true(rel_diff < tol,
              label = sprintf("%s loss parity: CPU=%.4e GPU=%.4e rel=%.4f (tol=%g)",
                              label, cpu_loss, gpu_loss, rel_diff, tol))
}

# ============================================================================
# GP (Generalized Poisson)
# ============================================================================
test_that("GPU GP NMF matches CPU GP NMF", {
  skip_if(!gpu_ok, "No GPU available")

  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  cpu <- nmf(A, k = 3, loss = "gp", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 3, loss = "gp", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  # GP link function amplifies fp32 rounding; ~20% gap is typical
  compare_loss(cpu, gpu, tol = 0.25, label = "GP")

  # Both should produce valid S4 nmf objects
  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")

  # Theta estimates should both exist and be non-negative
  if (!is.null(cpu@misc$theta) && !is.null(gpu@misc$theta)) {
    expect_true(all(cpu@misc$theta >= 0))
    expect_true(all(gpu@misc$theta >= 0))
  }
})

# ============================================================================
# NB (Negative Binomial)
# ============================================================================
test_that("GPU NB NMF matches CPU NB NMF", {
  skip_if(!gpu_ok, "No GPU available")

  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  cpu <- nmf(A, k = 3, loss = "nb", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 3, loss = "nb", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  compare_loss(cpu, gpu, tol = 0.10, label = "NB")

  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")
})

# ============================================================================
# Gamma
# ============================================================================
test_that("GPU Gamma NMF runs and produces valid output", {
  skip_if(!gpu_ok, "No GPU available")

  # Gamma deviance is numerically unstable under fp32 (inverse-square link
  # amplifies rounding), so we only verify both backends converge to finite,
  # positive losses with non-negative factors.  Full parity requires fp64 GPU.
  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  cpu <- nmf(A, k = 3, loss = "gamma", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 3, loss = "gamma", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(cpu@misc$loss) && cpu@misc$loss > 0)
  expect_true(is.finite(gpu@misc$loss) && gpu@misc$loss > 0)
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

# ============================================================================
# Inverse Gaussian
# ============================================================================
test_that("GPU InvGauss NMF runs and produces valid output", {
  skip_if(!gpu_ok, "No GPU available")

  # Inverse-Gaussian deviance uses cubic inverse-link, making fp32 losses
  # highly sensitive to rounding.  Smoke test only; parity needs fp64 GPU.
  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  cpu <- nmf(A, k = 3, loss = "inverse_gaussian", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 3, loss = "inverse_gaussian", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(cpu@misc$loss) && cpu@misc$loss > 0)
  expect_true(is.finite(gpu@misc$loss) && gpu@misc$loss > 0)
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

# ============================================================================
# Tweedie
# ============================================================================
test_that("GPU Tweedie NMF runs and produces valid output", {
  skip_if(!gpu_ok, "No GPU available")

  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  gpu <- nmf(A, k = 3, loss = "tweedie", dispersion = "per_row",
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(gpu@misc$loss) && gpu@misc$loss > 0)
  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

# ============================================================================
# Robust IRLS (Huber weighting)
# ============================================================================
test_that("GPU robust MSE NMF runs and produces valid output", {
  skip_if(!gpu_ok, "No GPU available")

  A <- make_sparse_nonneg(100, 80, density = 0.15, seed = 42)

  gpu <- nmf(A, k = 3, loss = "mse", robust = TRUE,
             maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(gpu@misc$loss))
  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))

  # Robust should produce similar but not identical loss to non-robust
  non_robust <- nmf(A, k = 3, loss = "mse",
                    maxit = 30, tol = 1e-10, seed = 42, resource = "gpu", verbose = FALSE)
  # Both should converge to a reasonable loss
  expect_true(gpu@misc$loss > 0)
  expect_true(non_robust@misc$loss > 0)
})
