# Tests for refine()

test_that("refine returns nmf object when given nmf input", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  refined <- refine(model, labels = labels)
  expect_s4_class(refined, "nmf")
  expect_equal(dim(refined@h), dim(model@h))
  expect_equal(dim(refined@w), dim(model@w))
  expect_equal(length(refined@d), length(model@d))
})

test_that("refine returns matrix when given matrix input", {
  set.seed(42)
  H <- matrix(runif(5 * 30), 5, 30)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  refined <- refine(H, labels = labels)
  expect_true(is.matrix(refined))
  expect_equal(dim(refined), c(5, 30))
})

test_that("refine with lambda=0 returns nearly unchanged embedding", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  refined <- refine(model, labels = labels, lambda = 0)
  expect_equal(refined@h, model@h, tolerance = 1e-10)
})

test_that("refine enforces non-negativity by default", {
  set.seed(42)
  H <- matrix(runif(4 * 20, -1, 1), 4, 20)  # Has negatives
  labels <- factor(rep(c("X", "Y"), each = 10))

  refined <- refine(H, labels = labels, lambda = 0.8)
  expect_true(all(refined >= 0))
})

test_that("refine with nonneg=FALSE allows negatives", {
  set.seed(42)
  H <- matrix(runif(4 * 20, 0, 1), 4, 20)
  labels <- factor(rep(c("X", "Y"), each = 10))

  refined <- refine(H, labels = labels, lambda = 0.8, nonneg = FALSE)
  # Could have negatives (from centroid shifts), but test it runs without error
  expect_true(is.matrix(refined))
})

test_that("refine errors on mismatched label length", {
  H <- matrix(runif(20), 4, 5)
  labels <- factor(c("A", "B", "C"))
  expect_error(refine(H, labels = labels), "length.*labels")
})

test_that("refine errors on invalid lambda", {
  H <- matrix(runif(20), 4, 5)
  labels <- factor(rep("A", 5))
  expect_error(refine(H, labels = labels, lambda = -0.5), "lambda")
  expect_error(refine(H, labels = labels, lambda = 2.0), "lambda")
})

test_that("refine with cycles>0 requires data", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 10, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  expect_error(refine(model, labels = labels, cycles = 1), "data.*required")
})

test_that("refine with cycles>0 runs and returns updated model", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  refined <- refine(model, data = A, labels = labels, cycles = 2)
  expect_s4_class(refined, "nmf")
  # W should be different after W-refit cycles
  expect_false(identical(refined@w, model@w))
  # All factors should be non-negative
  expect_true(all(refined@w >= 0))
  expect_true(all(refined@h >= 0))
  expect_true(all(refined@d > 0))
})

test_that("refine with batch labels uses PROJ_ADV (cycles>0)", {
  skip_on_cran()
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))
  batch <- factor(rep(c("b1", "b2"), 15))

  # With cycles > 0 and batch, should use PROJ_ADV internally
  refined <- refine(model, data = A, labels = labels,
                    batch = batch, cycles = 1)
  expect_s4_class(refined, "nmf")
  expect_equal(dim(refined@h), dim(model@h))
  expect_true(all(refined@h >= 0))
})

test_that("refine with batch labels (cycles=0) still does post-hoc correction", {
  set.seed(42)
  A <- abs(matrix(rnorm(50 * 30), 50, 30))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))
  batch <- factor(rep(c("b1", "b2"), 15))

  refined <- refine(model, labels = labels, batch = batch)
  expect_s4_class(refined, "nmf")
  expect_equal(dim(refined@h), dim(model@h))
})

test_that("refine errors on non-nmf non-matrix input", {
  expect_error(refine("not_a_matrix", labels = factor("A")), "must be an nmf")
})

test_that("refine with sparse data and cycles>0 works", {
  skip_on_cran()
  set.seed(42)
  A <- abs(Matrix::rsparsematrix(50, 30, 0.3))
  model <- nmf(A, 3, seed = 1, maxit = 20, tol = 1e-4, verbose = FALSE)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  refined <- refine(model, data = A, labels = labels, cycles = 1)
  expect_s4_class(refined, "nmf")
  expect_true(all(refined@h >= 0))
})
