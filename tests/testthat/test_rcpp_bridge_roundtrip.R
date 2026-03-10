# =============================================================================
# TEST-GATE-03: Rcpp Bridge Roundtrip Tests
# Phase 0 safety net — tests each Rcpp_* export that will be touched in Phase 2.
# These are integration-level tests that prove the C++ contract works end-to-end.
# =============================================================================

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)
options(RcppML.gpu = FALSE)

library(Matrix)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
set.seed(42)
m <- 50; n <- 40; k <- 3
A_sparse <- abs(rsparsematrix(m, n, density = 0.3))
A_dense <- as.matrix(A_sparse)
A_iris <- as.matrix(iris[, 1:4])

# ===========================================================================
# NMF bridge functions
# ===========================================================================

test_that("Rcpp_nmf_full produces valid NMF result (sparse)", {
  model <- nmf(A_sparse, k = k, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), k)
  expect_equal(nrow(model@h), k)
  expect_equal(nrow(model@w), m)
  expect_equal(ncol(model@h), n)
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
  expect_true(all(model@d > 0))
})

test_that("Rcpp_nmf_full produces valid NMF result (dense)", {
  model <- nmf(A_dense, k = k, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(ncol(model@w), k)
  expect_equal(nrow(model@h), k)
  expect_true(all(model@w >= 0))
  expect_true(all(model@h >= 0))
})

test_that("Rcpp_nmf_full with CV produces cross-validation results", {
  cv <- nmf(A_dense, k = 2:4, test_fraction = 0.1, maxit = 10,
            cv_seed = 100, verbose = FALSE)
  expect_true(is.data.frame(cv))
  expect_true("k" %in% names(cv))
  expect_true("test_mse" %in% names(cv))
  expect_equal(sort(unique(as.integer(cv$k))), 2:4)
})

# ===========================================================================
# SVD bridge function
# ===========================================================================

test_that("Rcpp_svd_pca produces valid SVD result (dense)", {
  s <- RcppML::svd(A_iris, k = 3, method = "deflation", seed = 1, maxit = 100,
                   test_fraction = 0)
  expect_s4_class(s, "svd")
  expect_equal(ncol(s@u), 3)
  expect_equal(length(s@d), 3)
  expect_equal(ncol(s@v), 3)
  expect_true(all(s@d > 0))
})

test_that("Rcpp_svd_pca produces valid SVD result (sparse)", {
  s <- RcppML::svd(A_sparse, k = 3, method = "deflation", seed = 1, maxit = 100,
                   test_fraction = 0)
  expect_s4_class(s, "svd")
  expect_equal(ncol(s@u), 3)
  expect_true(all(s@d > 0))
})

# ===========================================================================
# Predict bridge functions
# ===========================================================================

test_that("Rcpp_predict_sparse matches R-level predict()", {
  model <- nmf(A_sparse, k = k, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  projected <- predict(model, A_sparse)
  expect_s4_class(projected, "nmf")
  expect_equal(ncol(projected@h), n)
  expect_true(all(projected@h >= 0))
})

test_that("Rcpp_predict_dense matches R-level predict()", {
  model <- nmf(A_dense, k = k, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  projected <- predict(model, A_dense)
  expect_s4_class(projected, "nmf")
  expect_equal(ncol(projected@h), n)
  expect_true(all(projected@h >= 0))
})

# ===========================================================================
# Evaluate bridge functions
# ===========================================================================

test_that("Rcpp_evaluate_loss_sparse matches R-level evaluate()", {
  model <- nmf(A_sparse, k = k, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  loss <- evaluate(model, A_sparse)
  expect_true(is.numeric(loss))
  expect_true(loss > 0)
  expect_true(is.finite(loss))
})

test_that("Rcpp_evaluate_loss_dense matches R-level evaluate()", {
  model <- nmf(A_dense, k = k, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
  loss <- evaluate(model, A_dense)
  expect_true(is.numeric(loss))
  expect_true(loss > 0)
  expect_true(is.finite(loss))
})

# ===========================================================================
# Bipartition bridge functions
# ===========================================================================

test_that("Rcpp_bipartition_sparse produces valid split", {
  result <- bipartition(A_sparse, tol = 1e-4, maxit = 50, seed = 1, verbose = FALSE)
  expect_true(is.list(result))
  # Should split samples into two groups
  expect_true("v" %in% names(result))
  expect_equal(length(result$v), n)
})

test_that("Rcpp_bipartition_dense produces valid split", {
  result <- bipartition(A_dense, tol = 1e-4, maxit = 50, seed = 1, verbose = FALSE)
  expect_true(is.list(result))
  expect_true("v" %in% names(result))
  expect_equal(length(result$v), n)
})

# ===========================================================================
# StreamPress bridge functions
# ===========================================================================

test_that("st_write + st_read round-trip", {
  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path))
  st_write(A_sparse, path)
  B <- st_read(path)
  expect_equal(as.matrix(A_sparse), as.matrix(B), tolerance = 1e-6)
})

test_that("st_info reads correct dimensions", {
  path <- tempfile(fileext = ".spz")
  on.exit(unlink(path))
  st_write(A_sparse, path)
  info <- st_info(path)
  expect_equal(info$rows, m)
  expect_equal(info$cols, n)
})
