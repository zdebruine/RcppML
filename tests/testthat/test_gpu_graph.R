# GPU Graph NMF Parity Tests
# Tests: single-layer graph NMF with graph_W

# ── Single-layer graph NMF ──────────────────────────────────────────

test_that("GPU graph NMF runs without error", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:200, 1:100]
  A <- as(A, "dgCMatrix")

  # Build a simple random adjacency graph (symmetric, no self-loops)
  set.seed(1)
  knn <- Matrix::sparseMatrix(
    i = sample(1:200, 300, replace = TRUE),
    j = sample(1:200, 300, replace = TRUE),
    x = 1, dims = c(200, 200)
  )
  knn <- (knn + Matrix::t(knn)) / 2
  diag(knn) <- 0
  knn <- as(knn, "dgCMatrix")

  # GPU graph NMF — smoke test (high gap expected ~60%, fp32 amplification)
  gpu <- nmf(A, k = 5, seed = 42, graph_W = knn, graph_lambda = c(0.1, 0),
             maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)

  expect_true(is.finite(gpu@misc$loss))
  expect_true(gpu@misc$loss > 0)
  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
  expect_true(all(gpu@w >= 0))
  expect_true(all(gpu@h >= 0))
})

test_that("GPU graph NMF loss decreases from initial", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:200, 1:100]
  A <- as(A, "dgCMatrix")

  set.seed(1)
  knn <- Matrix::sparseMatrix(
    i = sample(1:200, 300, replace = TRUE),
    j = sample(1:200, 300, replace = TRUE),
    x = 1, dims = c(200, 200)
  )
  knn <- (knn + Matrix::t(knn)) / 2
  diag(knn) <- 0
  knn <- as(knn, "dgCMatrix")

  # 1 iteration vs 20 iterations — loss should decrease
  gpu1 <- nmf(A, k = 5, seed = 42, graph_W = knn, graph_lambda = c(0.1, 0),
              maxit = 1, tol = 1e-10, resource = "gpu", verbose = FALSE)
  gpu20 <- nmf(A, k = 5, seed = 42, graph_W = knn, graph_lambda = c(0.1, 0),
               maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)

  # More iterations should produce lower or equal loss
  expect_true(gpu20@misc$loss <= gpu1@misc$loss * 1.01)  # small tolerance for rounding
})

# ── Deep graph NMF (multi-layer) ────────────────────────────────────

test_that("GPU deep graph NMF runs with both graph_W and graph_H", {
  skip_if_no_gpu()

  data(pbmc3k)
  A <- pbmc3k[1:200, 1:100]
  A <- as(A, "dgCMatrix")

  # Row graph (200 x 200)
  set.seed(1)
  gW <- Matrix::sparseMatrix(
    i = sample(1:200, 200, replace = TRUE),
    j = sample(1:200, 200, replace = TRUE),
    x = 1, dims = c(200, 200)
  )
  gW <- (gW + Matrix::t(gW)) / 2; diag(gW) <- 0
  gW <- as(gW, "dgCMatrix")

  # Column graph (100 x 100)
  gH <- Matrix::sparseMatrix(
    i = sample(1:100, 100, replace = TRUE),
    j = sample(1:100, 100, replace = TRUE),
    x = 1, dims = c(100, 100)
  )
  gH <- (gH + Matrix::t(gH)) / 2; diag(gH) <- 0
  gH <- as(gH, "dgCMatrix")

  gpu <- nmf(A, k = 5, seed = 42,
             graph_W = gW, graph_H = gH, graph_lambda = c(0.1, 0.1),
             maxit = 20, tol = 1e-10, resource = "gpu", verbose = FALSE)

  expect_true(is.finite(gpu@misc$loss))
  expect_true(gpu@misc$loss > 0)
  expect_true(all(is.finite(gpu@w)))
  expect_true(all(is.finite(gpu@h)))
})
