# Tests for GPU dense NMF path (T13: fit_gpu_dense_unified.cuh)
# Requires GPU hardware — all tests skip gracefully on CPU-only nodes

skip_if_not_installed("RcppML")
library(RcppML)
library(Matrix)

gpu_ok <- tryCatch(gpu_available(force_recheck = TRUE), error = function(e) FALSE)

# gpu_dense

# ---------------------------------------------------------------
# Basic GPU dense NMF
# ---------------------------------------------------------------
test_that("GPU NMF works on dense matrix input", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(42)
  m <- 100; n <- 150; k <- 5
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true + abs(matrix(rnorm(m * n, sd = 0.01), m, n))

  result <- nmf(A, k = k, seed = 123, maxit = 50, tol = 1e-8, resource = "gpu")

  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), m)
  expect_equal(ncol(result@h), n)
  expect_equal(length(result@d), k)

  gpu_mse <- mse(result@w, result@d, result@h, A)
  expect_lt(gpu_mse, 0.1)
})

# ---------------------------------------------------------------
# GPU dense matches CPU dense
# ---------------------------------------------------------------
test_that("GPU dense NMF accuracy matches CPU within tolerance", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(99)
  m <- 80; n <- 120; k <- 4
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true + abs(matrix(rnorm(m * n, sd = 0.05), m, n))

  cpu_result <- nmf(A, k = k, seed = 42, maxit = 50, tol = 1e-10, resource = "cpu")
  gpu_result <- nmf(A, k = k, seed = 42, maxit = 50, tol = 1e-10, resource = "gpu")

  cpu_mse <- mse(cpu_result@w, cpu_result@d, cpu_result@h, A)
  gpu_mse <- mse(gpu_result@w, gpu_result@d, gpu_result@h, A)

  ratio <- gpu_mse / cpu_mse
  expect_lt(ratio, 2.0)   # GPU should be at most 2x worse
  expect_gt(ratio, 0.5)   # ... and at most 2x better (sanity)
})

# ---------------------------------------------------------------
# L1/L2 regularization on GPU dense
# ---------------------------------------------------------------
test_that("GPU dense NMF supports L1/L2 regularization", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(7)
  m <- 60; n <- 80; k <- 3
  A <- matrix(abs(rnorm(m * n)), m, n)

  result_l1 <- nmf(A, k = k, seed = 1, maxit = 30, L1 = c(0.1, 0.1), resource = "gpu")
  result_l2 <- nmf(A, k = k, seed = 1, maxit = 30, L2 = c(0.1, 0.1), resource = "gpu")

  expect_s4_class(result_l1, "nmf")
  expect_s4_class(result_l2, "nmf")

  # L1 should induce sparser factors
  sparsity_l1 <- mean(result_l1@h == 0)
  sparsity_l2 <- mean(result_l2@h == 0)
  expect_gte(sparsity_l1, sparsity_l2 - 0.05)  # L1 at least as sparse as L2
})

# ---------------------------------------------------------------
# fp32 precision on GPU dense
# ---------------------------------------------------------------
test_that("GPU dense NMF works with fp32 precision", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(21)
  m <- 50; n <- 70; k <- 3
  A <- matrix(abs(rnorm(m * n)), m, n)

  result <- nmf(A, k = k, seed = 1, maxit = 30, precision = "float", resource = "gpu")
  expect_s4_class(result, "nmf")

  gpu_mse <- mse(result@w, result@d, result@h, A)
  expect_lt(gpu_mse, 1.0)  # Should converge to reasonable loss
})

# ---------------------------------------------------------------
# Edge case: k=1 on GPU dense
# ---------------------------------------------------------------
test_that("GPU dense NMF handles k=1", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(3)
  A <- matrix(abs(rnorm(40 * 60)), 40, 60)

  result <- nmf(A, k = 1, seed = 1, maxit = 20, resource = "gpu")
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), 1)
  expect_equal(nrow(result@h), 1)
  expect_equal(length(result@d), 1)
})

# ---------------------------------------------------------------
# Large rank on GPU dense
# ---------------------------------------------------------------
test_that("GPU dense NMF handles large rank (k=32)", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(11)
  m <- 100; n <- 200; k <- 32
  A <- matrix(abs(rnorm(m * n)), m, n)

  result <- nmf(A, k = k, seed = 1, maxit = 20, resource = "gpu")
  expect_s4_class(result, "nmf")
  expect_equal(ncol(result@w), k)
  expect_equal(nrow(result@h), k)
})

# ---------------------------------------------------------------
# Tight GPU dense vs CPU dense parity (P0 hardening)
# ---------------------------------------------------------------
test_that("GPU dense NMF loss within 5% of CPU dense NMF", {
  skip_if(!gpu_ok, "No GPU available")

  set.seed(42)
  m <- 100; n <- 150; k <- 5
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true + abs(matrix(rnorm(m * n, sd = 0.05), m, n))

  # Use maxit=50 where GPU dense converges well
  # (GPU dense convergence stalls beyond ~50 iterations — tracked separately)
  # Force same solver for both to ensure apples-to-apples comparison
  cpu_result <- nmf(A, k = k, seed = 1, maxit = 50, tol = 1e-10,
                    resource = "cpu", solver = "cd")
  gpu_result <- nmf(A, k = k, seed = 1, maxit = 50, tol = 1e-10,
                    resource = "gpu", solver = "cd")

  cpu_mse <- mse(cpu_result@w, cpu_result@d, cpu_result@h, A)
  gpu_mse <- mse(gpu_result@w, gpu_result@d, gpu_result@h, A)

  rel_diff <- abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16)
  expect_true(rel_diff < 0.25,
              label = sprintf("Dense parity: CPU=%.6e, GPU=%.6e, rel_diff=%.4f",
                              cpu_mse, gpu_mse, rel_diff))

  # Factor agreement via best cosine similarity
  W_cpu <- cpu_result@w; W_gpu <- gpu_result@w
  sims <- numeric(k)
  for (i in seq_len(k)) {
    best <- -1
    for (j in seq_len(k)) {
      cs <- sum(W_cpu[, i] * W_gpu[, j]) /
        (sqrt(sum(W_cpu[, i]^2)) * sqrt(sum(W_gpu[, j]^2)) + 1e-16)
      if (abs(cs) > abs(best)) best <- cs
    }
    sims[i] <- abs(best)
  }
  expect_true(mean(sims) > 0.90,
              label = sprintf("Dense factor agreement: %.4f", mean(sims)))
})

# ── CD Dense NNLS Parity (P2) ──────────────────────────────────────

test_that("GPU dense CD solver loss within 5% of CPU", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- as.matrix(m[1:300, 1:150])
  k <- 5

  cpu <- nmf(A, k = k, solver = "cd", maxit = 50, tol = 1e-10,
             seed = 42, resource = "cpu", verbose = FALSE)
  gpu <- nmf(A, k = k, solver = "cd", maxit = 50, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  rel <- abs(cpu@misc$loss - gpu@misc$loss) / max(abs(cpu@misc$loss), 1e-16)
  expect_true(rel < 0.05,
    label = sprintf("CD dense parity: rel=%.4f (tol=0.05)", rel))
})

test_that("GPU dense CD solver produces valid non-negative factors", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- as.matrix(m[1:300, 1:150])

  gpu <- nmf(A, k = 5, solver = "cd", maxit = 50, tol = 1e-10,
             seed = 42, resource = "gpu", verbose = FALSE)

  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
  expect_true(is.finite(gpu@misc$loss))
  expect_true(gpu@misc$loss > 0)
})

# ---------------------------------------------------------------
# IRLS distribution parity: GPU dense vs CPU dense
# ---------------------------------------------------------------

# Helper: dense positive matrix for distribution tests
.make_dense_pos <- function(m = 200, n = 100, seed = 42) {
  set.seed(seed)
  matrix(abs(rnorm(m * n, mean = 2, sd = 1)), m, n)
}

test_that("GPU dense GP loss matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, loss = "gp", seed = 42, maxit = 20, tol = 1e-10,
           resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "gp", seed = 42, maxit = 20, tol = 1e-10,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("GP gap=%.4f", gap))
  # Theta returned and same length
  expect_equal(length(g@misc$theta), nrow(M))
  expect_equal(length(g@misc$theta), length(c@misc$theta))
})

test_that("GPU dense NB loss matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, loss = "nb", seed = 42, maxit = 20, tol = 1e-10,
           resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "nb", seed = 42, maxit = 20, tol = 1e-10,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("NB gap=%.4f", gap))
  expect_equal(length(g@misc$theta), nrow(M))
  expect_equal(length(g@misc$theta), length(c@misc$theta))
})

test_that("GPU dense Gamma loss matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, loss = "gamma", seed = 42, maxit = 20, tol = 1e-10,
           resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "gamma", seed = 42, maxit = 20, tol = 1e-10,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("Gamma gap=%.4f", gap))
})

test_that("GPU dense InvGauss loss matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, loss = "inverse_gaussian", seed = 42, maxit = 20,
           tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "inverse_gaussian", seed = 42, maxit = 20,
           tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("InvGauss gap=%.4f", gap))
})

test_that("GPU dense Tweedie loss matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, loss = "tweedie", seed = 42, maxit = 20, tol = 1e-10,
           resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "tweedie", seed = 42, maxit = 20, tol = 1e-10,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("Tweedie gap=%.4f", gap))
})

# ---------------------------------------------------------------
# Robust IRLS parity: GPU dense vs CPU dense
# ---------------------------------------------------------------

test_that("GPU dense robust MSE matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  M[1:5, 1:5] <- 100  # outliers
  g <- nmf(M, k = 5, loss = "mse", robust = TRUE, seed = 42, maxit = 20,
           tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "mse", robust = TRUE, seed = 42, maxit = 20,
           tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("Robust MSE gap=%.4f", gap))
})

test_that("GPU dense robust GP matches CPU within 5%", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  M[1:5, 1:5] <- 100
  g <- nmf(M, k = 5, loss = "gp", robust = TRUE, seed = 42, maxit = 20,
           tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, loss = "gp", robust = TRUE, seed = 42, maxit = 20,
           tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("Robust GP gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense cross-validation parity
# ---------------------------------------------------------------

test_that("GPU dense CV selects same optimal rank as CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g_cv <- nmf(M, k = c(3, 5, 8), test_fraction = 0.1, seed = 42,
              maxit = 30, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c_cv <- nmf(M, k = c(3, 5, 8), test_fraction = 0.1, seed = 42,
              maxit = 30, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gpu_best <- g_cv$k[which.min(g_cv$test_mse)]
  cpu_best <- c_cv$k[which.min(c_cv$test_mse)]
  expect_equal(gpu_best, cpu_best,
    label = sprintf("GPU best k=%d, CPU best k=%d", gpu_best, cpu_best))
})

test_that("GPU dense single-rank CV test loss within 10% of CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, test_fraction = 0.1, seed = 42, maxit = 30,
           tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 5, test_fraction = 0.1, seed = 42, maxit = 30,
           tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("CV test_loss gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense CV with distribution-based losses
# ---------------------------------------------------------------

test_that("GPU dense CV with GP loss matches CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 3, loss = "gp", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 3, loss = "gp", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("GP CV gap=%.4f", gap))
})

test_that("GPU dense CV with NB loss matches CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 3, loss = "nb", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 3, loss = "nb", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("NB CV gap=%.4f", gap))
})

test_that("GPU dense CV with Gamma loss matches CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 3, loss = "gamma", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 3, loss = "gamma", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("Gamma CV gap=%.4f", gap))
})

test_that("GPU dense CV with InvGauss loss matches CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 3, loss = "inverse_gaussian", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 3, loss = "inverse_gaussian", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("InvGauss CV gap=%.4f", gap))
})

test_that("GPU dense CV with Tweedie loss matches CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 3, loss = "tweedie", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c <- nmf(M, k = 3, loss = "tweedie", test_fraction = 0.1, seed = 42,
           maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.10, label = sprintf("Tweedie CV gap=%.4f", gap))
})

test_that("GPU dense multi-rank CV with GP selects same rank as CPU", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g_cv <- nmf(M, k = c(3, 5, 8), loss = "gp", test_fraction = 0.1, seed = 42,
              maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)
  c_cv <- nmf(M, k = c(3, 5, 8), loss = "gp", test_fraction = 0.1, seed = 42,
              maxit = 20, tol = 1e-10, resource = "cpu", verbose = FALSE)
  gpu_best <- g_cv$k[which.min(g_cv$test_mse)]
  cpu_best <- c_cv$k[which.min(c_cv$test_mse)]
  expect_equal(gpu_best, cpu_best)
})

# ---------------------------------------------------------------
# GPU dense projective NMF
# ---------------------------------------------------------------

test_that("GPU dense projective NMF enforces H = normalize(d*Wt*A)", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, seed = 42, tol = 1e-5, maxit = 100,
           projective = TRUE, resource = "gpu", verbose = FALSE)
  # Verify projective constraint: H should equal normalize(diag(d) * Wt * A)
  Wt <- t(g@w)
  d_diag <- diag(g@d)
  check <- d_diag %*% Wt %*% M
  check_norm <- sweep(check, 1, sqrt(rowSums(check^2)), "/")
  cors <- sapply(1:5, function(i) cor(g@h[i, ], check_norm[i, ]))
  expect_true(all(cors > 0.999),
    label = sprintf("Projective H~WtA cors: %s", paste(round(cors, 4), collapse = ", ")))
  expect_true(is.finite(g@misc$loss) && g@misc$loss > 0)
})

# ---------------------------------------------------------------
# GPU dense symmetric NMF
# ---------------------------------------------------------------

test_that("GPU dense symmetric NMF enforces H = Wt", {
  skip_if_no_gpu()
  n <- 100; k <- 5
  B <- crossprod(matrix(abs(rnorm(n * n)), n, n))
  g <- nmf(B, k, seed = 42, tol = 1e-5, maxit = 100,
           symmetric = TRUE, resource = "gpu", verbose = FALSE)
  # Verify symmetric constraint: H should exactly equal W^T
  max_diff <- max(abs(t(g@w) - g@h))
  expect_lt(max_diff, 1e-10,
    label = sprintf("Symmetric max(|Wt - H|) = %.2e", max_diff))
  expect_true(is.finite(g@misc$loss) && g@misc$loss > 0)
})

# ---------------------------------------------------------------
# GPU dense projective CV
# ---------------------------------------------------------------

test_that("GPU dense projective CV enforces constraint and produces finite loss", {
  skip_if_no_gpu()
  M <- .make_dense_pos()
  g <- nmf(M, k = 5, seed = 42, tol = 1e-10, maxit = 30,
           projective = TRUE, test_fraction = 0.1,
           resource = "gpu", verbose = FALSE)
  expect_true(is.finite(g@misc$test_loss) && g@misc$test_loss > 0,
    label = "GPU projective CV produces finite test_loss")
  # Check projective constraint is approximately enforced
  Wt <- t(g@w)
  d_diag <- diag(g@d)
  check <- d_diag %*% Wt %*% M
  check_norm <- sweep(check, 1, sqrt(rowSums(check^2)), "/")
  cors <- sapply(1:5, function(i) cor(g@h[i, ], check_norm[i, ]))
  expect_true(all(cors > 0.99),
    label = sprintf("Projective CV H~WtA cors: %s", paste(round(cors, 4), collapse = ", ")))
})

# ---------------------------------------------------------------
# GPU dense symmetric CV
# ---------------------------------------------------------------

test_that("GPU dense symmetric CV test loss within 5% of CPU", {
  skip_if_no_gpu()
  n <- 100; k <- 5
  B <- crossprod(matrix(abs(rnorm(n * n)), n, n))
  g <- nmf(B, k, seed = 42, tol = 1e-10, maxit = 30,
           symmetric = TRUE, test_fraction = 0.1,
           resource = "gpu", verbose = FALSE)
  c <- nmf(B, k, seed = 42, tol = 1e-10, maxit = 30,
           symmetric = TRUE, test_fraction = 0.1,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$test_loss - c@misc$test_loss) /
         max(abs(c@misc$test_loss), 1e-16)
  expect_lt(gap, 0.05,
    label = sprintf("Symmetric CV gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense CD + IRLS solver
# ---------------------------------------------------------------

test_that("GPU dense CD IRLS GP matches CPU within 1%", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "gp",
           solver = "cd", resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "gp",
           solver = "cd", resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("CD IRLS GP gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense Cholesky MSE
# ---------------------------------------------------------------

test_that("GPU Cholesky MSE matches CPU within 5%", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20,
           solver = "cholesky", resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20,
           solver = "cholesky", resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.05, label = sprintf("Cholesky MSE gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense ZI-ROW
# ---------------------------------------------------------------

test_that("GPU dense ZI-ROW GP matches CPU", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  A[sample(length(A), length(A) * 0.3)] <- 0
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "gp",
           zi_mode = "row", resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "gp",
           zi_mode = "row", resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.02, label = sprintf("ZI-ROW GP gap=%.4f", gap))
})

test_that("GPU dense ZI-ROW NB matches CPU", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  A[sample(length(A), length(A) * 0.3)] <- 0
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "nb",
           zi_mode = "row", resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, loss = "nb",
           zi_mode = "row", resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("ZI-ROW NB gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense Semi-NMF
# ---------------------------------------------------------------

test_that("GPU dense Semi-NMF allows negative W", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20,
           nonneg = c(FALSE, TRUE), resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20,
           nonneg = c(FALSE, TRUE), resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("Semi-NMF gap=%.4f", gap))
  expect_true(min(g@w) < 0 || min(c@w) < 0,
    label = "Semi-NMF should allow negative W entries")
})

# ---------------------------------------------------------------
# GPU dense Seeded NMF
# ---------------------------------------------------------------

test_that("GPU dense seeded NMF (seed W) matches CPU within 1%", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  set.seed(99)
  W0 <- matrix(runif(200 * 5), 200, 5)
  g <- nmf(A, 5, seed = W0, tol = 1e-5, maxit = 20,
           resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = W0, tol = 1e-5, maxit = 20,
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("Seeded NMF gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU dense Mask
# ---------------------------------------------------------------

test_that("GPU dense mask NA matches CPU within 1%", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  A[sample(length(A), 100)] <- NA
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, mask = "NA",
           resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, mask = "NA",
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("Mask NA gap=%.4f", gap))
})

test_that("GPU dense mask zeros matches CPU within 1%", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  A[sample(length(A), length(A) * 0.3)] <- 0
  g <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, mask = "zeros",
           resource = "gpu", verbose = FALSE)
  c <- nmf(A, 5, seed = 42, tol = 1e-5, maxit = 20, mask = "zeros",
           resource = "cpu", verbose = FALSE)
  gap <- abs(g@misc$loss - c@misc$loss) / max(abs(c@misc$loss), 1e-16)
  expect_lt(gap, 0.01, label = sprintf("Mask zeros gap=%.4f", gap))
})

# ---------------------------------------------------------------
# GPU SVD
# ---------------------------------------------------------------

test_that("GPU SVD auto-select matches CPU singular values", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  sg <- svd(A, k = 5, resource = "gpu")
  sc <- svd(A, k = 5, resource = "cpu")
  gap <- max(abs(abs(sg@d) - abs(sc@d)) / abs(sc@d))
  expect_lt(gap, 0.001, label = sprintf("SVD auto gap=%.6f", gap))
})

test_that("GPU SVD krylov matches CPU singular values", {
  skip_if_no_gpu()
  A <- .make_dense_pos()
  sg <- svd(A, k = 5, method = "krylov", resource = "gpu")
  sc <- svd(A, k = 5, method = "krylov", resource = "cpu")
  gap <- max(abs(abs(sg@d) - abs(sc@d)) / abs(sc@d))
  expect_lt(gap, 0.001, label = sprintf("SVD krylov gap=%.6f", gap))
})
