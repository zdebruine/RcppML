# GPU SVD parity tests
# Verifies GPU SVD produces results equivalent to CPU SVD for all methods.
# Skipped automatically if GPU is not available.

library(Matrix)

skip_if_not(
  exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(),
  "GPU not available"
)

# --- Shared ground-truth data ---
set.seed(123)
gt_m <- 30; gt_n <- 20; gt_k <- 5
U_raw <- qr.Q(qr(matrix(rnorm(gt_m * gt_k), gt_m, gt_k)))
V_raw <- qr.Q(qr(matrix(rnorm(gt_n * gt_k), gt_n, gt_k)))
D_true <- c(50, 30, 15, 8, 4)
A_gt_dense <- U_raw %*% diag(D_true) %*% t(V_raw)
A_gt_sparse <- as(Matrix(A_gt_dense, sparse = TRUE), "dgCMatrix")

ref_d <- base::svd(A_gt_dense)$d[1:gt_k]

# ===========================================================================
# GPU Sparse SVD tests (P1) — 5 methods
# ===========================================================================

test_that("TEST-GPU-SVD-DEFLATION: GPU deflation matches CPU (sparse)", {
  s_gpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "deflation",
                       resource = "gpu", maxit = 500, tol = 1e-10,
                       seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "deflation",
                       resource = "cpu", maxit = 500, tol = 1e-10,
                       seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((as.matrix(A_gt_sparse) - A_hat)^2)) / sqrt(sum(as.matrix(A_gt_sparse)^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-KRYLOV: GPU krylov matches ground truth (sparse)", {
  s_gpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "krylov",
                       resource = "gpu", maxit = 500, tol = 1e-10,
                       seed = 1, test_fraction = 0)
  # Compare to ground truth (CPU krylov less accurate on small matrices)
  expect_equal(s_gpu@d, ref_d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((as.matrix(A_gt_sparse) - A_hat)^2)) / sqrt(sum(as.matrix(A_gt_sparse)^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-LANCZOS: GPU lanczos matches ground truth (sparse)", {
  s_gpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "lanczos",
                       resource = "gpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  # Compare to ground truth (CPU lanczos less accurate on small matrices)
  expect_equal(s_gpu@d, ref_d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((as.matrix(A_gt_sparse) - A_hat)^2)) / sqrt(sum(as.matrix(A_gt_sparse)^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-IRLBA: GPU irlba matches CPU (sparse)", {
  s_gpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "irlba",
                       resource = "gpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "irlba",
                       resource = "cpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((as.matrix(A_gt_sparse) - A_hat)^2)) / sqrt(sum(as.matrix(A_gt_sparse)^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-RANDOMIZED: GPU randomized matches CPU (sparse)", {
  s_gpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "randomized",
                       resource = "gpu", seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_sparse, k = gt_k, method = "randomized",
                       resource = "cpu", seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((as.matrix(A_gt_sparse) - A_hat)^2)) / sqrt(sum(as.matrix(A_gt_sparse)^2))
  expect_lt(rel_err, 1e-3)
})

# ===========================================================================
# GPU Dense SVD tests (P2) — 4 methods (no krylov for dense GPU)
# ===========================================================================

test_that("TEST-GPU-SVD-DEFLATION-DENSE: GPU deflation matches CPU (dense)", {
  s_gpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "deflation",
                       resource = "gpu", maxit = 500, tol = 1e-10,
                       seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "deflation",
                       resource = "cpu", maxit = 500, tol = 1e-10,
                       seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((A_gt_dense - A_hat)^2)) / sqrt(sum(A_gt_dense^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-LANCZOS-DENSE: GPU lanczos matches CPU (dense)", {
  s_gpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "lanczos",
                       resource = "gpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "lanczos",
                       resource = "cpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((A_gt_dense - A_hat)^2)) / sqrt(sum(A_gt_dense^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-IRLBA-DENSE: GPU irlba matches CPU (dense)", {
  s_gpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "irlba",
                       resource = "gpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "irlba",
                       resource = "cpu", tol = 1e-10,
                       seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((A_gt_dense - A_hat)^2)) / sqrt(sum(A_gt_dense^2))
  expect_lt(rel_err, 1e-3)
})

test_that("TEST-GPU-SVD-RANDOMIZED-DENSE: GPU randomized matches CPU (dense)", {
  s_gpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "randomized",
                       resource = "gpu", seed = 1, test_fraction = 0)
  s_cpu <- RcppML::svd(A_gt_dense, k = gt_k, method = "randomized",
                       resource = "cpu", seed = 1, test_fraction = 0)
  expect_equal(s_gpu@d, s_cpu@d, tolerance = 1e-3)
  A_hat <- s_gpu@u %*% diag(s_gpu@d) %*% t(s_gpu@v)
  rel_err <- sqrt(sum((A_gt_dense - A_hat)^2)) / sqrt(sum(A_gt_dense^2))
  expect_lt(rel_err, 1e-3)
})
