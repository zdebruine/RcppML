# =============================================================================
# Reference tests for svd() / pca()
#
# Every expectation compares against base::svd, stats::prcomp, or math written
# out in the test. Problems are small; fits run on the CPU with two threads.
# Tolerances are for single-precision fits.
# =============================================================================

options(RcppML.gpu = FALSE)

library(Matrix)

fit <- function(A, ...) RcppML::svd(A, ..., resource = "cpu", threads = 2)

# m x n matrix with prescribed singular values
lowrank <- function(m, n, d, seed) {
  set.seed(seed)
  r <- length(d)
  U <- qr.Q(qr(matrix(rnorm(m * r), m, r)))
  V <- qr.Q(qr(matrix(rnorm(n * r), n, r)))
  U %*% diag(d, r) %*% t(V)
}

orth_err <- function(X) max(abs(crossprod(X) - diag(ncol(X))))

d_true <- c(50, 30, 15, 8, 4)
A_r5 <- lowrank(30, 20, d_true, seed = 11)          # exactly rank 5
A_rep <- lowrank(80, 40, c(10, 10, 10, 5, 5, 1), 12) # repeated singular values

# ---------------------------------------------------------------------------
# Exactly low-rank input, k = rank (Lanczos breakdown)
# ---------------------------------------------------------------------------

test_that("REF-SVD-LOWRANK: exactly rank-5 30x20 matrix, k = 5, every method", {
  for (meth in c("auto", "lanczos", "krylov", "irlba", "randomized", "deflation")) {
    s <- suppressWarnings(fit(A_r5, k = 5, method = meth, seed = 1))
    expect_equal(length(s@d), 5L, info = meth)
    expect_lt(max(abs(s@d - d_true)), if (meth == "deflation") 5e-2 else 2e-3)
    expect_lt(orth_err(s@u), 1e-3)
    expect_lt(orth_err(s@v), 1e-3)
    rel <- sqrt(sum((A_r5 - s@u %*% diag(s@d) %*% t(s@v))^2) / sum(A_r5^2))
    expect_lt(rel, if (meth == "deflation") 2e-2 else 1e-4)
  }
})

test_that("REF-SVD-LOWRANK: k larger than the rank returns the rank plus zeros", {
  for (meth in c("lanczos", "irlba")) {
    s <- fit(A_r5, k = 8, method = meth, seed = 1)
    expect_equal(length(s@d), 8L, info = meth)
    expect_lt(max(abs(s@d[1:5] - d_true)), 2e-3)
    expect_lt(max(abs(s@d[6:8])), 1e-3)
  }
})

test_that("REF-SVD-REPEATED: repeated singular values are all returned", {
  ref <- c(10, 10, 10, 5, 5, 1)
  for (meth in c("auto", "lanczos", "krylov", "irlba")) {
    s <- fit(A_rep, k = 5, method = meth, seed = 1)
    expect_equal(length(s@d), 5L, info = meth)
    expect_lt(max(abs(s@d - ref[1:5])), 2e-3)
  }
  s <- fit(A_rep, k = 20, method = "lanczos", seed = 1)
  expect_equal(length(s@d), 20L)
  expect_lt(max(abs(s@d[1:6] - ref)), 2e-3)
  expect_lt(max(abs(s@d[7:20])), 1e-3)
})

test_that("REF-SVD-SCALE: results do not depend on the overall scale (A * 1e-7)", {
  for (meth in c("auto", "lanczos", "krylov", "irlba", "deflation")) {
    s <- suppressWarnings(fit(A_r5 * 1e-7, k = 5, method = meth, seed = 1))
    expect_equal(length(s@d), 5L, info = meth)
    expect_lt(max(abs(s@d * 1e7 - d_true)), if (meth == "deflation") 5e-2 else 2e-3)
  }
})

test_that("REF-SVD-FULLRANK: k = min(m, n), tall and wide", {
  set.seed(21)
  A <- matrix(rnorm(12 * 8), 12, 8)
  for (X in list(A, t(A))) {
    ref <- base::svd(X)$d
    for (meth in c("lanczos", "irlba", "auto")) {
      s <- fit(X, k = 8, method = meth, seed = 1)
      expect_equal(length(s@d), 8L, info = meth)
      expect_lt(max(abs(s@d - ref)), 1e-4 * ref[1])
      expect_lt(orth_err(s@u), 1e-4)
      expect_lt(orth_err(s@v), 1e-4)
    }
  }
})

test_that("REF-SVD-K1: k = 1 matches base::svd up to sign", {
  set.seed(22)
  A <- matrix(rnorm(40 * 25), 40, 25) + 3
  ref <- base::svd(A)
  for (meth in c("lanczos", "irlba", "deflation")) {
    s <- suppressWarnings(fit(A, k = 1, method = meth, seed = 1))
    expect_lt(abs(s@d - ref$d[1]) / ref$d[1], 1e-4)
    expect_gt(abs(sum(s@u * ref$u[, 1])), 1 - 1e-4)
    expect_gt(abs(sum(s@v * ref$v[, 1])), 1 - 1e-4)
  }
})

# ---------------------------------------------------------------------------
# IRLBA restarts on flat spectra
# ---------------------------------------------------------------------------

test_that("REF-SVD-IRLBA: restarted IRLBA returns singular values on flat spectra", {
  set.seed(31)
  A1 <- Matrix::rsparsematrix(500, 300, 0.05)
  A2 <- matrix(as.numeric(rpois(300 * 200, 3)), 300, 200)
  A3 <- matrix(rnorm(60 * 300), 60, 300)
  cases <- list(sparse = A1, poisson = A2, wide = A3)
  for (nm in names(cases)) {
    X <- cases[[nm]]
    ref <- base::svd(as.matrix(X), nu = 0, nv = 0)$d[1:5]
    s <- fit(X, k = 5, method = "irlba", seed = 1)
    expect_lt(max(abs(s@d - ref)), 5e-4 * ref[1])
    # triplets satisfy A v = d u
    Av <- as.matrix(X %*% s@v)
    expect_lt(max(abs(Av - s@u %*% diag(s@d))), 2e-3 * ref[1])
  }
})

test_that("REF-SVD-LANCZOS: default step budget reaches 1e-3 on a flat spectrum, k = 20", {
  set.seed(32)
  A <- Matrix::rsparsematrix(500, 300, 0.05)
  ref <- base::svd(as.matrix(A), nu = 0, nv = 0)$d[1:20]
  s <- suppressWarnings(fit(A, k = 20, method = "lanczos", seed = 1))
  expect_lt(max(abs(s@d - ref)), 1e-3 * ref[1])
})

# ---------------------------------------------------------------------------
# Convergence reporting
# ---------------------------------------------------------------------------

test_that("REF-SVD-WARN: stopping before tol is met raises a warning", {
  set.seed(41)
  A <- matrix(rnorm(200 * 100), 200, 100)
  expect_warning(fit(A, k = 5, method = "lanczos", maxit = 8, seed = 1),
                 "did not reach tol")
  expect_warning(fit(A, k = 5, method = "irlba", maxit = 1, seed = 1),
                 "did not reach tol")
  expect_warning(fit(A, k = 2, method = "deflation", maxit = 2, seed = 1),
                 "did not reach tol")
  # converged fits are silent
  expect_warning(fit(A_r5, k = 5, method = "lanczos", seed = 1), NA)
  expect_warning(fit(A_r5, k = 5, method = "irlba", seed = 1), NA)
})

# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

A_mask <- lowrank(60, 40, d_true, seed = 51)
set.seed(52)
M20 <- Matrix::rsparsematrix(60, 40, 0.2)
M20@x[] <- 1

test_that("REF-SVD-MASK: masked entries are missing, not zero", {
  # The masked entries carry no information: replacing them by garbage must
  # not change the fit.
  A_bad <- A_mask
  A_bad[as.matrix(M20) != 0] <- 1e3

  s <- suppressWarnings(fit(A_mask, k = 5, mask = M20, method = "krylov", seed = 1))
  expect_lt(max(abs(s@d / d_true - 1)), 0.03)
  s_bad <- suppressWarnings(fit(A_bad, k = 5, mask = M20, method = "krylov", seed = 1))
  expect_lt(max(abs(s_bad@d / d_true - 1)), 0.03)

  s <- suppressWarnings(fit(A_mask, k = 5, mask = M20, method = "deflation", seed = 1))
  expect_lt(max(abs(s@d - d_true)) / d_true[1], 0.03)
  expect_lt(max(abs(s@d[1:3] / d_true[1:3] - 1)), 0.05)

  # method = "auto" picks a method that honors the mask
  s <- suppressWarnings(fit(A_mask, k = 5, mask = M20, seed = 1))
  expect_true(s@misc$method %in% c("krylov", "deflation"))
  expect_lt(max(abs(s@d / d_true - 1)), 0.03)
})

test_that("REF-SVD-MASK: methods that cannot honor a mask or a holdout say so", {
  for (meth in c("lanczos", "irlba", "randomized")) {
    expect_error(fit(A_mask, k = 3, mask = M20, method = meth), "missing")
    expect_error(fit(A_mask, k = 3, test_fraction = 0.1, method = meth), "missing")
  }
  expect_error(fit(A_mask, k = 3, mask = "zeros"), "zeros")
})

test_that("REF-SVD-CV: holding out entries does not shrink d", {
  set.seed(53)
  A <- A_mask + matrix(rnorm(60 * 40, sd = 0.01), 60, 40)
  ref <- base::svd(A)$d[1:5]
  for (meth in c("deflation", "krylov")) {
    s <- suppressWarnings(fit(A, k = 5, test_fraction = 0.1, method = meth, seed = 1))
    expect_equal(length(s@misc$test_loss), 5L, info = meth)
    expect_true(all(is.finite(s@misc$test_loss)))
    expect_lt(max(abs(s@d[1:3] / ref[1:3] - 1)), 0.03)
    expect_lt(max(abs(s@d - ref)) / ref[1], 0.03)
    # the held-out error of the rank-5 model is at the noise level, far below
    # the error of predicting zero
    expect_lt(tail(s@misc$test_loss, 1), 0.05 * mean(A^2))
  }
})

test_that("REF-SVD-CV: centering statistics exclude held-out entries (large mean)", {
  # A constant offset of 100 must not leak into the components through the
  # held-out entries.
  set.seed(54)
  A <- A_mask + 100 + matrix(rnorm(60 * 40, sd = 0.01), 60, 40)
  ref <- base::svd(A - rowMeans(A))$d[1:4]
  s <- suppressWarnings(fit(A, k = 4, center = TRUE, test_fraction = 0.1,
                            method = "deflation", seed = 1))
  expect_lt(max(abs(s@d - ref)) / ref[1], 0.05)
})

test_that("REF-SVD-CV: sparse and dense input define the same test set", {
  set.seed(55)
  S <- Matrix::rsparsematrix(50, 40, 0.5)
  D <- as.matrix(S)
  for (mk in list(NULL, "zeros")) {
    a <- suppressWarnings(fit(S, k = 3, test_fraction = 0.1, method = "deflation",
                              seed = 1, mask = mk))
    b <- suppressWarnings(fit(D, k = 3, test_fraction = 0.1, method = "deflation",
                              seed = 1, mask = mk))
    expect_equal(a@misc$test_loss, b@misc$test_loss, tolerance = 1e-3)
    expect_equal(a@d, b@d, tolerance = 1e-3)
  }
})

test_that("REF-SVD-AUTORANK: k = 'auto' recovers the rank of a low-rank matrix", {
  set.seed(56)
  A <- A_mask + matrix(rnorm(60 * 40, sd = 0.02), 60, 40)
  s <- suppressWarnings(fit(A, k = "auto", seed = 1))
  expect_equal(length(s@d), 5L)
})

# ---------------------------------------------------------------------------
# PCA against prcomp (rows = variables, columns = samples)
# ---------------------------------------------------------------------------

set.seed(61)
X_pca <- matrix(rnorm(40 * 6), 40, 6) %*% diag(c(5, 3, 2, 1, 1, 0.5))
X_pca <- sweep(X_pca, 2, c(10, -5, 0, 3, 100, 1), "+")   # samples in rows

test_that("REF-PCA: pca(t(X)) matches prcomp(X), centered and scaled", {
  for (sc in c(FALSE, TRUE)) {
    ref <- stats::prcomp(X_pca, center = TRUE, scale. = sc)
    for (meth in c("auto", "lanczos", "irlba", "deflation")) {
      p <- suppressWarnings(RcppML::pca(t(X_pca), k = 3, scale = sc, method = meth,
                                        seed = 1, resource = "cpu", threads = 2))
      tol <- if (meth == "deflation") 2e-3 else 2e-4
      expect_equal(p@d / sqrt(nrow(X_pca) - 1), ref$sdev[1:3], tolerance = tol)
      # loadings: u spans the same directions as prcomp's rotation
      ov <- abs(diag(crossprod(p@u, ref$rotation[, 1:3])))
      expect_gt(min(ov), if (meth == "deflation") 1 - 1e-2 else 1 - 1e-3)
      # scores
      sc_ref <- abs(ref$x[, 1:3])
      sc_fit <- abs(p@v %*% diag(p@d))
      expect_lt(max(abs(sc_fit - sc_ref)), (if (meth == "deflation") 5e-2 else 5e-3) * max(sc_ref))
      expect_equal(as.numeric(p@misc$row_means), colMeans(X_pca), tolerance = 1e-5)
      if (sc) expect_equal(as.numeric(p@misc$row_sds), apply(X_pca, 2, sd), tolerance = 1e-4)
    }
  }
})

test_that("REF-PCA: variance_explained matches prcomp", {
  ref <- stats::prcomp(X_pca)
  p <- RcppML::pca(t(X_pca), k = 3, method = "irlba", seed = 1,
                   resource = "cpu", threads = 2)
  expect_equal(RcppML::variance_explained(p), (ref$sdev^2 / sum(ref$sdev^2))[1:3],
               tolerance = 1e-3)
})

test_that("REF-PCA: reconstruct() and predict() honor centering and scaling", {
  A <- t(X_pca)                       # 6 variables x 40 samples
  for (sc in c(FALSE, TRUE)) {
    p <- RcppML::pca(A, k = 6, scale = sc, method = "irlba", seed = 1,
                     resource = "cpu", threads = 2)
    # full rank: the model reproduces the data on its original scale
    expect_lt(max(abs(RcppML::reconstruct(p) - A)), 1e-3 * max(abs(A)))
    # projecting the training rows returns u
    expect_lt(max(abs(predict(p, A) - p@u)), 5e-3)
    expect_lt(max(abs(predict(p, as(A, "dgCMatrix")) - p@u)), 5e-3)
  }
})

# ---------------------------------------------------------------------------
# Argument validation and options that used to do nothing
# ---------------------------------------------------------------------------

test_that("REF-SVD-ARGS: k is validated against the matrix size", {
  expect_error(fit(A_r5, k = 21), "cannot exceed")
  expect_error(fit(A_r5, k = 21, method = "lanczos"), "cannot exceed")
  # the default rank is reduced for small matrices
  s <- fit(matrix(rnorm(24), 6, 4), seed = 1)
  expect_equal(length(s@d), 4L)
})

test_that("REF-SVD-ARGS: unsupported options are rejected", {
  expect_error(fit(A_r5, k = 2, upper_bound = 1), "upper_bound")
  expect_error(fit(A_r5, k = 2, L21 = 0.1, method = "deflation"), "L21")
  expect_error(fit(A_r5, k = 2, L2 = c(1, 0), method = "deflation"), "L2")
  expect_error(fit(A_r5, k = 2, L1 = 0.1, method = "lanczos"), "constraints")
})

test_that("REF-SVD-L2: ridge on both factors soft-thresholds d in deflation", {
  A <- lowrank(40, 30, c(20, 10, 5), seed = 71)
  s0 <- suppressWarnings(fit(A, k = 3, method = "deflation", seed = 1))
  s2 <- suppressWarnings(fit(A, k = 3, method = "deflation", L2 = 2, seed = 1))
  expect_lt(max(abs(s0@d - c(20, 10, 5))), 2e-2)
  expect_lt(max(abs(s2@d - c(18, 8, 3))), 2e-2)
  s9 <- suppressWarnings(fit(A, k = 3, method = "deflation", L2 = c(4, 9), seed = 1))
  expect_lt(max(abs(s9@d - pmax(c(20, 10, 5) - 6, 0))), 2e-2)
})

# ---------------------------------------------------------------------------
# Constrained deflation: d, reconstruct() and variance_explained() agree
# ---------------------------------------------------------------------------

test_that("REF-SVD-CONSTRAINED: least-squares d, reconstruction and variance agree", {
  set.seed(81)
  W <- matrix(0, 40, 3); H <- matrix(0, 30, 3)
  W[1:20, 1] <- runif(20); W[11:30, 2] <- runif(20); W[21:40, 3] <- runif(20)
  H[1:15, 1] <- runif(15); H[8:22, 2] <- runif(15); H[16:30, 3] <- runif(15)
  A <- W %*% diag(c(9, 6, 4)) %*% t(H) + matrix(runif(1200, 0, 0.01), 40, 30)

  s <- suppressWarnings(fit(A, k = 3, nonneg = TRUE, seed = 1))
  expect_equal(s@misc$method, "deflation")
  U <- s@u; V <- s@v
  expect_gte(min(U), 0)
  expect_gte(min(V), 0)
  expect_equal(as.numeric(colSums(U^2)), rep(1, 3), tolerance = 1e-4)
  expect_equal(as.numeric(colSums(V^2)), rep(1, 3), tolerance = 1e-4)

  # d minimizes ||A - U diag(d) V'||_F:  (U'U o V'V) d = diag(U' A V)
  d_ls <- solve(crossprod(U) * crossprod(V), diag(t(U) %*% A %*% V))
  expect_equal(s@d, as.numeric(d_ls), tolerance = 1e-3)

  # variance explained sums to the R^2 of reconstruct()
  r2 <- 1 - sum((A - RcppML::reconstruct(s))^2) / sum(A^2)
  ve <- RcppML::variance_explained(s)
  expect_equal(sum(ve), r2, tolerance = 1e-3)
  expect_true(all(ve >= -1e-6))
  expect_gt(r2, 0.5)

  # each component's share is the drop in residual variance of the nested refit
  for (i in 1:3) {
    Ui <- U[, 1:i, drop = FALSE]; Vi <- V[, 1:i, drop = FALSE]
    di <- solve(crossprod(Ui) * crossprod(Vi), diag(t(Ui) %*% A %*% Vi))
    r2_i <- 1 - sum((A - Ui %*% diag(as.numeric(di), i) %*% t(Vi))^2) / sum(A^2)
    expect_equal(sum(ve[1:i]), r2_i, tolerance = 2e-3)
  }
})

test_that("REF-SVD-L1: sparse factors keep exact zeros and unit norm", {
  A <- lowrank(40, 30, c(20, 10, 5), seed = 91)
  s <- suppressWarnings(fit(A, k = 2, L1 = c(0, 2), seed = 1))
  expect_equal(s@misc$method, "deflation")
  expect_gt(sum(s@v == 0), 0)
  expect_equal(as.numeric(colSums(s@v^2)), rep(1, 2), tolerance = 1e-4)
  d_ls <- solve(crossprod(s@u) * crossprod(s@v), diag(t(s@u) %*% A %*% s@v))
  expect_equal(s@d, as.numeric(d_ls), tolerance = 1e-3)
})

test_that("REF-SVD-L1: a penalty that removes a whole factor stops the decomposition", {
  A <- lowrank(40, 30, c(20, 10, 5), seed = 91)
  # the threshold L1 / 2 exceeds every entry of sigma * v (at most 20)
  expect_warning(s <- fit(A, k = 2, L1 = c(0, 1e4), seed = 1), "removed all entries")
  expect_equal(length(s@d), 0L)
  expect_equal(ncol(s@u), 0L)
  expect_equal(ncol(s@v), 0L)
})

test_that("REF-SVD-PRINT: head() and tail() work on svd objects", {
  s <- fit(A_r5, k = 3, method = "lanczos", seed = 1)
  expect_output(head(s), "svd")
  expect_output(tail(s, 2), "last 2 rows")
  expect_equal(dim(s[1:2]), c(30L, 20L, 2L))
})
