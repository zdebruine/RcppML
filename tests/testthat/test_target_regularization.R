# Tests for target regularization in nmf() and factor_net

test_that("nmf with target_H and target_lambda runs without error", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  model <- nmf(A, k, target_H = target_H, target_lambda = c(0, 0.5),
               seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(nrow(model@h), k)
  expect_equal(ncol(model@h), n)
})

test_that("nmf with target_lambda=0 gives same result as no target", {
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  model_no_target <- nmf(A, k, seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_zero_lambda <- nmf(A, k, target_H = target_H, target_lambda = c(0, 0),
                           seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)

  # With lambda=0, target has no effect → should get identical result
  expect_equal(model_zero_lambda@misc$loss, model_no_target@misc$loss, tolerance = 1e-4)
})

test_that("nmf with positive target_lambda gives different result than without", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  model_none <- nmf(A, k, seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_target <- nmf(A, k, target_H = target_H, target_lambda = c(0, 1.0),
                      seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)

  # Results should differ
  expect_false(identical(model_target@h, model_none@h))
})

test_that("nmf with sparse matrix and target works", {
  skip_on_cran()
  set.seed(42)
  k <- 3; n <- 30
  A <- abs(Matrix::rsparsematrix(50, n, 0.3))
  target_H <- matrix(runif(k * n), k, n)

  model <- nmf(A, k, target_H = target_H, target_lambda = c(0, 0.5),
               seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("W() and H() accept target parameters in factor_net", {
  set.seed(42)
  k <- 3; n <- 20
  tgt <- matrix(runif(k * n), k, n)

  h <- H(target = tgt, target_lambda = 0.5)
  expect_s3_class(h, "fn_factor_config")
  expect_equal(h$target_lambda, 0.5)
  expect_equal(dim(h$target), c(k, n))

  # W also accepts (even though W-side target is less common)
  w <- W(target = matrix(runif(k * 50), k, 50), target_lambda = 0.3)
  expect_s3_class(w, "fn_factor_config")
  expect_equal(w$target_lambda, 0.3)
})

# --- PROJ_ADV (negative target_lambda) tests ---

test_that("nmf with negative target_lambda (PROJ_ADV) runs without error", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  model <- nmf(A, k, target_H = target_H, target_lambda = c(0, -0.5),
               seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_equal(nrow(model@h), k)
  expect_equal(ncol(model@h), n)
  expect_true(all(model@h >= 0))
})

test_that("PROJ_ADV gives different result from enrichment", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  model_pos <- nmf(A, k, target_H = target_H, target_lambda = c(0, 0.5),
                   seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)
  model_neg <- nmf(A, k, target_H = target_H, target_lambda = c(0, -0.5),
                   seed = 1, maxit = 30, tol = 1e-8, verbose = FALSE)

  # Positive and negative lambda should produce different H
  expect_false(identical(model_pos@h, model_neg@h))
})

test_that("PROJ_ADV with sparse matrix works", {
  skip_on_cran()
  set.seed(42)
  k <- 3; n <- 30
  A <- abs(Matrix::rsparsematrix(50, n, 0.3))
  target_H <- matrix(runif(k * n), k, n)

  model <- nmf(A, k, target_H = target_H, target_lambda = c(0, -0.5),
               seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
  expect_true(all(model@h >= 0))
})

test_that("PROJ_ADV scalar lambda expands correctly", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  # Scalar -0.5 should expand to c(0, -0.5)
  model <- nmf(A, k, target_H = target_H, target_lambda = -0.5,
               seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

# --- NNLS target tests ---

test_that("nnls with positive target_lambda (enrichment) works", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  w <- abs(matrix(rnorm(m * k), m, k))
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  h <- nnls(w = w, A = A, target_H = target_H, target_lambda = 0.5)
  expect_equal(dim(h), c(k, n))
  expect_true(all(is.finite(h)))
})

test_that("nnls with negative target_lambda (PROJ_ADV) works", {
  skip_on_cran()
  set.seed(42)
  k <- 3; m <- 50; n <- 30
  w <- abs(matrix(rnorm(m * k), m, k))
  A <- abs(matrix(rnorm(m * n), m, n))
  target_H <- matrix(runif(k * n), k, n)

  h <- nnls(w = w, A = A, target_H = target_H, target_lambda = -0.5)
  expect_equal(dim(h), c(k, n))
  expect_true(all(is.finite(h)))
})
