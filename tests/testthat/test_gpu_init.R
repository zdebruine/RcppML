# GPU Initialization Mode Parity Tests
# Tests: lanczos, random, user-supplied, and dense init modes

# ── Lanczos init parity ─────────────────────────────────────────────

test_that("GPU lanczos init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  cpu <- nmf(A, k = 5, init = "lanczos", maxit = 20, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, init = "lanczos", maxit = 20, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("Lanczos init parity: rel=%.4f (tol=0.05)", rel))
})

test_that("GPU lanczos init produces valid factors", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  gpu <- nmf(A, k = 5, init = "lanczos", maxit = 20, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
  expect_true(is.finite(gpu@misc$loss))
})

# ── Random init parity ──────────────────────────────────────────────

test_that("GPU random init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  cpu <- nmf(A, k = 5, init = "random", maxit = 20, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, init = "random", maxit = 20, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("Random init parity: rel=%.4f (tol=0.05)", rel))
})

# ── User-supplied init parity ───────────────────────────────────────

test_that("GPU user-supplied init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  set.seed(42)
  W_init <- matrix(abs(rnorm(300 * 5)), 300, 5)

  cpu <- nmf(A, k = 5, seed = W_init, maxit = 20, tol = 1e-10,
             resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, seed = W_init, maxit = 20, tol = 1e-10,
             resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("User init parity: rel=%.4f (tol=0.05)", rel))
})

test_that("GPU user init produces non-negative factors", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  set.seed(42)
  W_init <- matrix(abs(rnorm(300 * 5)), 300, 5)

  gpu <- nmf(A, k = 5, seed = W_init, maxit = 20, tol = 1e-10,
             resource = "gpu", verbose = FALSE)

  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

# ── Dense input init parity ─────────────────────────────────────────

test_that("GPU dense random init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- as.matrix(pbmc3k[1:300, 1:150])

  cpu <- nmf(A, k = 5, init = "random", maxit = 50, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, init = "random", maxit = 50, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("Dense random init parity: rel=%.4f (tol=0.05)", rel))
})

test_that("GPU dense lanczos init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- as.matrix(pbmc3k[1:300, 1:150])

  cpu <- nmf(A, k = 5, init = "lanczos", maxit = 50, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, init = "lanczos", maxit = 50, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("Dense lanczos init parity: rel=%.4f (tol=0.05)", rel))
})

test_that("GPU dense user init loss within 5% of CPU", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- as.matrix(pbmc3k[1:300, 1:150])

  set.seed(42)
  W_init <- matrix(abs(rnorm(300 * 5)), 300, 5)

  cpu <- nmf(A, k = 5, seed = W_init, maxit = 50, tol = 1e-10,
             resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = 5, seed = W_init, maxit = 50, tol = 1e-10,
             resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("Dense user init parity: rel=%.4f (tol=0.05)", rel))
})
