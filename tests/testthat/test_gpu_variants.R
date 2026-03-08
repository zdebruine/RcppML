# Test GPU NMF Variant Parity (Projective and Symmetric)
#
# Verifies GPU projective and symmetric NMF produce results close to CPU.

skip_if_not_installed("RcppML")
library(RcppML)
library(Matrix)

gpu_ok <- tryCatch(gpu_available(force_recheck = TRUE), error = function(e) FALSE)

# ============================================================================
# Projective NMF
# ============================================================================
test_that("GPU projective NMF loss within 15% of CPU", {
  skip_if(!gpu_ok, "No GPU available")

  data(pbmc3k)
  A <- pbmc3k[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  cpu <- nmf(A, k = 5, projective = TRUE, maxit = 20, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, projective = TRUE, maxit = 20, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(cpu@misc$loss) && cpu@misc$loss > 0)
  expect_true(is.finite(gpu@misc$loss) && gpu@misc$loss > 0)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.15,
    label = sprintf("Projective parity: rel=%.4f (tol=0.15)", rel))

  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

# ============================================================================
# Symmetric NMF
# ============================================================================
test_that("GPU symmetric NMF loss within 15% of CPU", {
  skip_if(!gpu_ok, "No GPU available")

  data(pbmc3k)
  A <- pbmc3k[1:200, 1:100]
  A <- as(A, "dgCMatrix")
  B <- crossprod(A)  # symmetric matrix

  cpu <- nmf(B, k = 3, symmetric = TRUE, maxit = 20, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(B, k = 3, symmetric = TRUE, maxit = 20, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  expect_s4_class(cpu, "nmf")
  expect_s4_class(gpu, "nmf")
  expect_true(is.finite(cpu@misc$loss) && cpu@misc$loss > 0)
  expect_true(is.finite(gpu@misc$loss) && gpu@misc$loss > 0)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.15,
    label = sprintf("Symmetric parity: rel=%.4f (tol=0.15)", rel))

  expect_true(all(gpu@w >= 0))
})
