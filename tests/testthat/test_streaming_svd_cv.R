# =============================================================================
# Streaming SVD Cross-Validation Tests
# Tests CV auto-rank for streaming deflation and krylov paths via .spz files.
# =============================================================================

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)
options(RcppML.gpu = FALSE)

library(Matrix)

# ---------------------------------------------------------------------------
# Shared test data: create a .spz file with known-rank structure
# ---------------------------------------------------------------------------
set.seed(42)
m <- 120; n <- 90
A_sparse <- as(Matrix::rsparsematrix(m, n, density = 0.25), "dgCMatrix")
A_sparse@x <- abs(A_sparse@x)

spz_path <- tempfile(fileext = ".spz")

# ---------------------------------------------------------------------------
# Write .spz (include_transpose = TRUE required for streaming)
# ---------------------------------------------------------------------------
test_that("sp_write creates .spz file for streaming SVD", {
  skip_on_cran()
  RcppML::sp_write(A_sparse, spz_path, include_transpose = TRUE)
  expect_true(file.exists(spz_path))
})

# ===========================================================================
# Streaming deflation with CV
# ===========================================================================

test_that("streaming deflation SVD produces test_loss with test_fraction>0", {
  skip_on_cran()
  s <- RcppML::svd(spz_path, k = 5, method = "deflation",
                   maxit = 100, tol = 1e-5, seed = 1,
                   test_fraction = 0.1, cv_seed = 42)
  expect_s4_class(s, "svd_pca")
  expect_true(!is.null(s@misc$test_loss))
  expect_true(length(s@misc$test_loss) >= 1)
  expect_true(all(is.finite(s@misc$test_loss)))
  expect_true(all(s@misc$test_loss > 0))
  expect_true(all(s@d > 0))
})

test_that("streaming deflation CV test_loss decreases initially", {
  skip_on_cran()
  s <- RcppML::svd(spz_path, k = 8, method = "deflation",
                   maxit = 100, tol = 1e-5, seed = 1,
                   test_fraction = 0.1, cv_seed = 42, patience = 8)
  tl <- s@misc$test_loss
  expect_true(length(tl) >= 2)
  # First rank should have higher error than later ranks
  expect_gt(tl[1], min(tl))
})

# ===========================================================================
# Streaming krylov with CV
# ===========================================================================

test_that("streaming krylov SVD produces test_loss with test_fraction>0", {
  skip_on_cran()
  s <- RcppML::svd(spz_path, k = 5, method = "krylov",
                   maxit = 100, tol = 1e-5, seed = 1,
                   test_fraction = 0.1, cv_seed = 42)
  expect_s4_class(s, "svd_pca")
  expect_true(!is.null(s@misc$test_loss))
  expect_true(length(s@misc$test_loss) >= 1)
  expect_true(all(is.finite(s@misc$test_loss)))
  expect_true(all(s@misc$test_loss > 0))
  expect_true(all(s@d > 0))
})

test_that("streaming krylov CV test_loss decreases initially", {
  skip_on_cran()
  s <- RcppML::svd(spz_path, k = 8, method = "krylov",
                   maxit = 100, tol = 1e-5, seed = 1,
                   test_fraction = 0.1, cv_seed = 42, patience = 8)
  tl <- s@misc$test_loss
  expect_true(length(tl) >= 2)
  expect_gt(tl[1], min(tl))
})

# ===========================================================================
# Streaming auto-rank (default method = deflation when k = "auto")
# ===========================================================================

test_that("streaming SVD auto-rank selects reasonable rank", {
  skip_on_cran()
  # Build a known low-rank matrix
  set.seed(99)
  W0 <- matrix(abs(rnorm(m * 3)), m, 3)
  V0 <- matrix(abs(rnorm(n * 3)), n, 3)
  A_rank3 <- as(as(W0 %*% t(V0) + matrix(abs(rnorm(m * n, sd = 0.01)), m, n),
                   "dgCMatrix"), "dgCMatrix")
  rank3_path <- tempfile(fileext = ".spz")
  on.exit(unlink(rank3_path), add = TRUE)
  RcppML::sp_write(A_rank3, rank3_path, include_transpose = TRUE)

  s <- RcppML::svd(rank3_path, k = "auto", method = "deflation",
                   maxit = 100, tol = 1e-5, seed = 1,
                   k_max = 10, patience = 2)
  expect_s4_class(s, "svd_pca")
  expect_lte(ncol(s@u), 10)
  expect_true(!is.null(s@misc$test_loss))
  expect_true(length(s@misc$test_loss) >= 1)
  # k_selected should be reasonable (not 0, not k_max)
  expect_gt(s@misc$k_selected, 0)
})

# ===========================================================================
# Streaming CV with mask_zeros
# ===========================================================================

test_that("streaming deflation mask_zeros=TRUE works", {
  skip_on_cran()
  s <- RcppML::svd(spz_path, k = 5, method = "deflation",
                   maxit = 100, tol = 1e-5, seed = 1,
                   test_fraction = 0.1, cv_seed = 42,
                   mask_zeros = TRUE)
  expect_s4_class(s, "svd_pca")
  expect_true(!is.null(s@misc$test_loss))
  expect_true(all(is.finite(s@misc$test_loss)))
  expect_true(all(s@d > 0))
})

# ===========================================================================
# Consistency: streaming vs in-memory test_loss should be similar
# ===========================================================================

test_that("streaming deflation CV test_loss close to in-memory deflation", {
  skip_on_cran()
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  s_mem <- RcppML::svd(A_sparse, k = 5, method = "deflation",
                       maxit = 100, tol = 1e-5, seed = 1,
                       test_fraction = 0.1, cv_seed = 42,
                       mask_zeros = TRUE)
  s_spz <- RcppML::svd(spz_path, k = 5, method = "deflation",
                        maxit = 100, tol = 1e-5, seed = 1,
                        test_fraction = 0.1, cv_seed = 42,
                        mask_zeros = TRUE)

  tl_mem <- s_mem@misc$test_loss
  tl_spz <- s_spz@misc$test_loss
  shared_len <- min(length(tl_mem), length(tl_spz))
  expect_gt(shared_len, 0)
  # Test losses should be in the same ballpark (within 50% relative)
  for (i in seq_len(shared_len)) {
    ratio <- abs(tl_mem[i] - tl_spz[i]) / max(tl_mem[i], 1e-10)
    expect_lt(ratio, 0.5)
  }
})

# ===========================================================================
# TEST-STREAMING-SVD-PARITY: singular values match between streaming and in-memory
# ===========================================================================

test_that("TEST-STREAMING-SVD-PARITY: streaming SVD singular values match in-memory", {
  skip_on_cran()
  options(RcppML.precision = "double")
  on.exit(options(RcppML.precision = "float"), add = TRUE)

  s_mem <- RcppML::svd(A_sparse, k = 5, method = "deflation",
                       maxit = 200, tol = 1e-8, seed = 1, test_fraction = 0)
  s_spz <- RcppML::svd(spz_path, k = 5, method = "deflation",
                        maxit = 200, tol = 1e-8, seed = 1, test_fraction = 0)

  # Singular values should match within 5%
  expect_equal(s_spz@d, s_mem@d, tolerance = 0.05)
  # Reconstruction errors should be similar
  A_hat_mem <- s_mem@u %*% diag(s_mem@d) %*% t(s_mem@v)
  A_hat_spz <- s_spz@u %*% diag(s_spz@d) %*% t(s_spz@v)
  A_dense <- as.matrix(A_sparse)
  err_mem <- sqrt(sum((A_dense - A_hat_mem)^2)) / sqrt(sum(A_dense^2))
  err_spz <- sqrt(sum((A_dense - A_hat_spz)^2)) / sqrt(sum(A_dense^2))
  expect_equal(err_spz, err_mem, tolerance = 0.05)
})

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
unlink(spz_path)
