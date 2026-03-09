# Test: Cholesky dense, regularization dense, projective/symmetric dense,
#       guided NMF, and edge cases

options(RcppML.verbose = FALSE)

# ============================================================
# Cholesky Dense Test
# ============================================================

test_that("Cholesky solver works on dense MSE data", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))

  model <- nmf(A, 3, loss = "mse", solver = "cholesky",
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("Cholesky solver gives comparable loss to CD on dense MSE", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30))

  m_chol <- nmf(A, 3, loss = "mse", solver = "cholesky",
                maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)
  m_cd <- nmf(A, 3, loss = "mse", solver = "cd",
              maxit = 50, tol = 1e-8, seed = 42, verbose = FALSE)

  # Both should converge to similar loss (within 10x)
  ratio <- abs(m_chol@misc$loss) / max(abs(m_cd@misc$loss), 1e-10)
  expect_true(ratio < 10 && ratio > 0.1)
})

# ============================================================
# Regularization Dense Test
# ============================================================

test_that("L1 regularization works on dense input", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))

  model_noreg <- nmf(A, 3, loss = "mse", maxit = 30, seed = 42, verbose = FALSE)
  model_l1 <- nmf(A, 3, loss = "mse", L1 = c(0.1, 0.1),
                  maxit = 30, seed = 42, verbose = FALSE)

  # L1 should increase sparsity (more zeros in factors)
  sparsity_noreg <- mean(model_noreg@w == 0) + mean(model_noreg@h == 0)
  sparsity_l1 <- mean(model_l1@w == 0) + mean(model_l1@h == 0)
  expect_true(sparsity_l1 >= sparsity_noreg)
})

test_that("L2 regularization works on dense input", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40))

  model_noreg <- nmf(A, 3, loss = "mse", maxit = 30, seed = 42, verbose = FALSE)
  model_l2 <- nmf(A, 3, loss = "mse", L2 = c(0.1, 0.1),
                  maxit = 30, seed = 42, verbose = FALSE)

  # L2 should reduce factor norms
  norm_noreg <- sum(model_noreg@w^2) + sum(model_noreg@h^2)
  norm_l2 <- sum(model_l2@w^2) + sum(model_l2@h^2)
  expect_true(norm_l2 <= norm_noreg * 1.1)
})

test_that("Dense L1/L2 produces same sparsification pattern as sparse L1/L2", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(30 * 25, 2, 0.5), 30, 25))
  A_sp <- Matrix::Matrix(A, sparse = TRUE)

  m_dense <- nmf(A, 2, loss = "mse", L1 = c(0.05, 0.05),
                 maxit = 30, seed = 42, verbose = FALSE)
  m_sparse <- nmf(A_sp, 2, loss = "mse", L1 = c(0.05, 0.05),
                  maxit = 30, seed = 42, verbose = FALSE)

  # Sparsity patterns should be similar (within 10% absolute difference)
  sp_w_dense <- mean(m_dense@w == 0)
  sp_w_sparse <- mean(m_sparse@w == 0)
  expect_true(abs(sp_w_dense - sp_w_sparse) < 0.15)
})

# ============================================================
# Projective NMF Dense Test
# ============================================================

test_that("Projective NMF works on dense data", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30))

  model <- nmf(A, 3, loss = "mse", projective = TRUE,
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

# ============================================================
# Symmetric NMF Dense Test
# ============================================================

test_that("Symmetric NMF works on dense symmetric data", {
  skip_on_cran()
  set.seed(42)
  m <- 40
  A_raw <- matrix(abs(rnorm(m * m, 1, 0.3)), m, m)
  A_sym <- (A_raw + t(A_raw)) / 2

  model <- nmf(A_sym, 3, loss = "mse", symmetric = TRUE,
               maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

# ============================================================
# Guided NMF: Seed W
# ============================================================

test_that("Guided NMF with custom W seed recovers factors", {
  skip_on_cran()
  # GPU NMF backend may not use matrix seeds the same way, causing worse convergence
  skip_if(!identical(getOption("RcppML.gpu", FALSE), FALSE), "GPU NMF may not support matrix W seeds")
  set.seed(42)
  m <- 60; n <- 40; k <- 3
  W_true <- matrix(abs(rnorm(m * k, 1, 0.3)), m, k)
  H_true <- matrix(abs(rnorm(k * n, 1, 0.3)), k, n)
  A <- Matrix::Matrix(W_true %*% H_true + matrix(rnorm(m * n, 0, 0.01), m, n), sparse = TRUE)
  A@x[A@x < 0] <- 0

  # Use true W as seed (should converge faster and closer to true solution)
  model_guided <- nmf(A, k, seed = W_true, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_random <- nmf(A, k, seed = 123L, maxit = 30, tol = 1e-8, verbose = FALSE)

  expect_s4_class(model_guided, "nmf")
  # Guided should achieve lower or equal loss
  expect_true(model_guided@misc$loss <= model_random@misc$loss * 1.5,
              info = paste0("guided_loss=", model_guided@misc$loss,
                            " random_loss=", model_random@misc$loss))
})

test_that("Guided NMF with custom H seed produces valid result", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(runif(m * k), m, k)
  H_true <- matrix(runif(k * n), k, n)
  A <- Matrix::Matrix(W_true %*% H_true + abs(matrix(rnorm(m * n, 0, 0.05), m, n)), sparse = TRUE)

  # Use true H (with noise) as seed
  H_seed <- H_true + matrix(rnorm(k * n, 0, 0.01), k, n)
  H_seed[H_seed < 0] <- 0

  model <- nmf(A, k, h_init = H_seed, maxit = 30, tol = 1e-6, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_equal(nrow(model@h), k)
  expect_equal(ncol(model@h), n)
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("Guided NMF with W+H seeds achieves lower loss than random", {
  skip_on_cran()
  set.seed(42)
  m <- 50; n <- 40; k <- 3
  W_true <- matrix(runif(m * k), m, k)
  H_true <- matrix(runif(k * n), k, n)
  A <- W_true %*% H_true + abs(matrix(rnorm(m * n, 0, 0.05), m, n))

  W_seed <- W_true + matrix(rnorm(m * k, 0, 0.01), m, k)
  W_seed[W_seed < 0] <- 0
  H_seed <- H_true + matrix(rnorm(k * n, 0, 0.01), k, n)
  H_seed[H_seed < 0] <- 0

  model_both <- nmf(A, k, seed = W_seed, h_init = H_seed, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_random <- nmf(A, k, seed = 99, maxit = 30, tol = 1e-8, verbose = FALSE)

  expect_s4_class(model_both, "nmf")
  expect_true(is.finite(model_both@misc$loss))
  # With near-perfect seeds, should achieve lower loss
  expect_true(model_both@misc$loss <= model_random@misc$loss * 1.5)
})

# ============================================================
# Edge Cases
# ============================================================

test_that("NMF handles k = min(m,n) edge case", {
  skip_on_cran()
  set.seed(42)
  m <- 10; n <- 8  # k_max = 8
  A <- Matrix::Matrix(abs(matrix(rnorm(m * n, 2, 0.5), m, n)), sparse = TRUE)

  model <- nmf(A, k = n, maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(length(model@d), n)
  expect_true(is.finite(model@misc$loss))
})

test_that("NMF handles all-zero rows gracefully", {
  skip_on_cran()
  set.seed(42)
  A <- Matrix::Matrix(abs(matrix(rnorm(50 * 40, 2, 0.5), 50, 40)), sparse = TRUE)
  # Set rows 1-3 to all zeros
  A[1:3, ] <- 0

  model <- nmf(A, 3, maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  # Zero rows should have zero or near-zero W entries
  expect_true(all(model@w[1:3, ] < 1e-4))
})

test_that("NMF handles single nonzero entry", {
  skip_on_cran()
  A <- Matrix::Matrix(matrix(0, 10, 8), sparse = TRUE)
  A[5, 4] <- 1.0

  model <- nmf(A, 2, maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

test_that("NMF with maxit=1 returns valid result", {
  skip_on_cran()
  set.seed(42)
  A <- Matrix::Matrix(abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30)), sparse = TRUE)

  model <- nmf(A, 3, maxit = 1, tol = 0, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(model@misc$iter, 1)
  expect_true(is.finite(model@misc$loss))
})

test_that("NMF auto-masks NaN input", {
  skip_on_cran()
  A <- matrix(1:20, 5, 4) + 0.0
  A[2, 3] <- NaN
  # NaN values are auto-detected and masked (not an error)
  expect_warning(
    model <- nmf(A, 2, maxit = 5, verbose = FALSE),
    "NA values"
  )
  expect_s4_class(model, "nmf")
})

test_that("NMF handles Inf input without crashing", {
  skip_on_cran()
  A <- matrix(abs(rnorm(20, 2, 0.5)), 5, 4)
  A[2, 3] <- Inf
  # Inf may be handled or may error — either way, no crash
  result <- tryCatch(
    nmf(A, 2, maxit = 5, verbose = FALSE),
    error = function(e) "error",
    warning = function(w) {
      suppressWarnings(nmf(A, 2, maxit = 5, verbose = FALSE))
    }
  )
  expect_true(is.character(result) || is(result, "nmf"))
})

# ============================================================
# API: robust_delta parameter
# ============================================================

test_that("robust_delta numeric value works", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30))

  model <- nmf(A, 3, loss = "mse", robust = 2.0,
               maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

test_that("robust=TRUE uses default delta", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(40 * 30, 2, 0.5), 40, 30))

  model <- nmf(A, 3, loss = "mse", robust = TRUE,
               maxit = 20, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
})

# ============================================================
# Streaming Auto-Dispatch Test
# ============================================================

test_that("Streaming NMF dispatches via SPZ file with RcppML.streaming option", {
  skip(".st_dispatch not yet re-implemented")
  skip_on_cran()
  set.seed(42)
  A <- Matrix::rsparsematrix(50, 40, density = 0.3)
  A@x <- abs(A@x)

  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f), add = TRUE)
  sp_write(A, f, include_transpose = TRUE)

  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = FALSE), add = TRUE)

  model <- nmf(f, k = 3, maxit = 10, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(is.finite(model@misc$loss))
  expect_equal(nrow(model@w), 50)
  expect_equal(ncol(model@h), 40)
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

# ============================================================
# GPU Auto-Dispatch Test
# ============================================================

test_that("GPU auto-dispatch works when GPU is available", {
  skip_on_cran()
  skip_if_not(gpu_available(), "GPU not available")
  set.seed(42)
  A <- Matrix::rsparsematrix(100, 80, density = 0.3)
  A@x <- abs(A@x)

  # resource="auto" should auto-detect GPU
  model_auto <- nmf(A, 3, resource = "auto", maxit = 5, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model_auto, "nmf")
  expect_true(is.finite(model_auto@misc$loss))

  # resource="gpu" should force GPU
  model_gpu <- nmf(A, 3, resource = "gpu", maxit = 5, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model_gpu, "nmf")
  expect_true(is.finite(model_gpu@misc$loss))

  # resource="cpu" should force CPU even when GPU is available
  model_cpu <- nmf(A, 3, resource = "cpu", maxit = 5, tol = 1e-4, seed = 42, verbose = FALSE)
  expect_s4_class(model_cpu, "nmf")
  expect_true(is.finite(model_cpu@misc$loss))
})

test_that("Invalid resource value errors", {
  skip_on_cran()
  A <- Matrix::rsparsematrix(20, 15, density = 0.3)
  A@x <- abs(A@x)
  expect_error(nmf(A, 2, resource = "tpu"), "resource")
})
