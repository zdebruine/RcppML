# tests/testthat/test_norm.R
# Tests for the norm parameter (L1, L2, none) across NMF

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

library(Matrix)

# Shared test data
set.seed(42)
A_sparse <- abs(rsparsematrix(50, 30, density = 0.3))
A_dense <- as.matrix(A_sparse)
k <- 3

# ---- norm parameter acceptance ----
test_that("nmf accepts all three norm types", {
  expect_s4_class(nmf(A_sparse, k, maxit = 5, norm = "L1", seed = 1, v = FALSE), "nmf")
  expect_s4_class(nmf(A_sparse, k, maxit = 5, norm = "L2", seed = 1, v = FALSE), "nmf")
  expect_s4_class(nmf(A_sparse, k, maxit = 5, norm = "none", seed = 1, v = FALSE), "nmf")
})

test_that("nmf rejects invalid norm", {
  expect_error(nmf(A_sparse, k, maxit = 5, norm = "L3", seed = 1, v = FALSE))
})

# ---- default is L1 ----
test_that("default norm is L1", {
  m_default <- nmf(A_sparse, k, maxit = 10, seed = 1, threads = 1, v = FALSE)
  m_l1 <- nmf(A_sparse, k, maxit = 10, norm = "L1", seed = 1, threads = 1, v = FALSE)
  expect_equal(m_default@w, m_l1@w)
  expect_equal(m_default@d, m_l1@d)
  expect_equal(m_default@h, m_l1@h)
})

# ---- L1 normalization correctness: colSums of W should equal d ----
test_that("L1 norm: column sums of W equal d (up to tolerance)", {
  m <- nmf(A_sparse, k, maxit = 50, norm = "L1", seed = 1, threads = 1, v = FALSE)
  w_col_l1 <- unname(colSums(abs(m@w)))
  # After L1 normalization, W columns should be L1-normalized (sum to 1) and d holds the scale
  # So colSums(abs(W)) should be ~1 for each factor
  expect_equal(w_col_l1, rep(1, k), tolerance = 1e-6)
})

test_that("L2 norm: column-wise L2 norms of W equal 1", {
  m <- nmf(A_sparse, k, maxit = 50, norm = "L2", seed = 1, threads = 1, v = FALSE)
  w_col_l2 <- unname(sqrt(colSums(m@w^2)))
  expect_equal(w_col_l2, rep(1, k), tolerance = 1e-6)
})

test_that("none norm: d is all ones", {
  m <- nmf(A_sparse, k, maxit = 50, norm = "none", seed = 1, threads = 1, v = FALSE)
  expect_equal(m@d, rep(1, k), tolerance = 1e-6)
})

# ---- Reconstruction equivalence: all norms should give same A_hat ----
test_that("reconstruction is similar across norm types", {
  m_l1 <- nmf(A_sparse, k, maxit = 30, norm = "L1", seed = 1, threads = 1, v = FALSE, tol = 1e-10)
  m_l2 <- nmf(A_sparse, k, maxit = 30, norm = "L2", seed = 1, threads = 1, v = FALSE, tol = 1e-10)
  m_none <- nmf(A_sparse, k, maxit = 30, norm = "none", seed = 1, threads = 1, v = FALSE, tol = 1e-10)

  recon_l1 <- m_l1@w %*% diag(m_l1@d) %*% m_l1@h
  recon_l2 <- m_l2@w %*% diag(m_l2@d) %*% m_l2@h
  recon_none <- m_none@w %*% diag(m_none@d) %*% m_none@h

  # Different norm types follow slightly different convergence paths,
  # so reconstructions won't be identical but should be close
  expect_lt(norm(recon_l1 - recon_l2, "F") / norm(recon_l1, "F"), 0.5)
  expect_lt(norm(recon_l1 - recon_none, "F") / norm(recon_l1, "F"), 0.5)
})

# ---- Dense input works with all norms ----
test_that("all norm types work with dense input", {
  m_l1 <- nmf(A_dense, k, maxit = 10, norm = "L1", seed = 1, threads = 1, v = FALSE)
  m_l2 <- nmf(A_dense, k, maxit = 10, norm = "L2", seed = 1, threads = 1, v = FALSE)
  m_none <- nmf(A_dense, k, maxit = 10, norm = "none", seed = 1, threads = 1, v = FALSE)
  expect_s4_class(m_l1, "nmf")
  expect_s4_class(m_l2, "nmf")
  expect_s4_class(m_none, "nmf")
})

# ---- Convergence with each norm type ----
test_that("NMF converges with L1 norm", {
  m1 <- nmf(A_sparse, k, maxit = 1, norm = "L1", seed = 1, tol = 1e-10, v = FALSE)
  m50 <- nmf(A_sparse, k, maxit = 50, norm = "L1", seed = 1, tol = 1e-10, v = FALSE)
  expect_gt(evaluate(m1, A_sparse), evaluate(m50, A_sparse))
})

test_that("NMF converges with L2 norm", {
  m1 <- nmf(A_sparse, k, maxit = 1, norm = "L2", seed = 1, tol = 1e-10, v = FALSE)
  m50 <- nmf(A_sparse, k, maxit = 50, norm = "L2", seed = 1, tol = 1e-10, v = FALSE)
  expect_gt(evaluate(m1, A_sparse), evaluate(m50, A_sparse))
})

test_that("NMF converges with no normalization", {
  m1 <- nmf(A_sparse, k, maxit = 1, norm = "none", seed = 1, tol = 1e-10, v = FALSE)
  m50 <- nmf(A_sparse, k, maxit = 50, norm = "none", seed = 1, tol = 1e-10, v = FALSE)
  expect_gt(evaluate(m1, A_sparse), evaluate(m50, A_sparse))
})

# ---- Seed reproducibility with each norm type ----
test_that("seed reproducibility with L1 norm", {
  m1 <- nmf(A_sparse, k, maxit = 5, norm = "L1", seed = 1, threads = 1, v = FALSE)
  m2 <- nmf(A_sparse, k, maxit = 5, norm = "L1", seed = 1, threads = 1, v = FALSE)
  expect_equal(m1@w, m2@w)
  expect_equal(m1@d, m2@d)
  expect_equal(m1@h, m2@h)
})

test_that("seed reproducibility with L2 norm", {
  m1 <- nmf(A_sparse, k, maxit = 5, norm = "L2", seed = 1, threads = 1, v = FALSE)
  m2 <- nmf(A_sparse, k, maxit = 5, norm = "L2", seed = 1, threads = 1, v = FALSE)
  expect_equal(m1@w, m2@w)
  expect_equal(m1@d, m2@d)
  expect_equal(m1@h, m2@h)
})

test_that("seed reproducibility with no normalization", {
  m1 <- nmf(A_sparse, k, maxit = 5, norm = "none", seed = 1, threads = 1, v = FALSE)
  m2 <- nmf(A_sparse, k, maxit = 5, norm = "none", seed = 1, threads = 1, v = FALSE)
  expect_equal(m1@w, m2@w)
  expect_equal(m1@d, m2@d)
  expect_equal(m1@h, m2@h)
})

# ---- Different norms produce different d values ----
test_that("different norm types produce different d values", {
  m_l1 <- nmf(A_sparse, k, maxit = 20, norm = "L1", seed = 1, threads = 1, v = FALSE)
  m_l2 <- nmf(A_sparse, k, maxit = 20, norm = "L2", seed = 1, threads = 1, v = FALSE)
  m_none <- nmf(A_sparse, k, maxit = 20, norm = "none", seed = 1, threads = 1, v = FALSE)
  # d should differ between norm types (unless degenerate)
  expect_false(all(abs(m_l1@d - m_l2@d) < 1e-8))
  expect_false(all(abs(m_l1@d - m_none@d) < 1e-8))
})

# ---- Cross-validation with norm ----
test_that("cross-validation works with all norm types", {
  # Multi-rank CV uses test_fraction > 0
  cv_l1 <- nmf(A_sparse, k = 2:3, test_fraction = 0.1, reps = 1,
                norm = "L1", seed = 1, maxit = 10, v = FALSE)
  cv_l2 <- nmf(A_sparse, k = 2:3, test_fraction = 0.1, reps = 1,
                norm = "L2", seed = 1, maxit = 10, v = FALSE)
  cv_none <- nmf(A_sparse, k = 2:3, test_fraction = 0.1, reps = 1,
                 norm = "none", seed = 1, maxit = 10, v = FALSE)
  expect_s3_class(cv_l1, "data.frame")
  expect_s3_class(cv_l2, "data.frame")
  expect_s3_class(cv_none, "data.frame")
})
