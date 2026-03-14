# Tests for compute_target()

test_that("compute_target returns correct dimensions", {
  set.seed(42)
  k <- 5; n <- 30
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("A", "B", "C"), each = 10))

  tgt <- compute_target(H, labels)
  expect_true(is.matrix(tgt))
  expect_equal(nrow(tgt), k)
  expect_equal(ncol(tgt), n)
})

test_that("compute_target without whitening produces sensible values", {
  set.seed(42)
  k <- 4; n <- 20
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("X", "Y"), each = 10))

  tgt <- compute_target(H, labels, whiten = FALSE)
  expect_true(all(is.finite(tgt)))
  # Target shifts should be zero-mean across all samples (grand mean removed)
  expect_equal(rowMeans(tgt), rep(0, k), tolerance = 1e-10)
})

test_that("compute_target with whitening produces finite values", {
  set.seed(42)
  k <- 5; n <- 40
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("A", "B", "C", "D"), each = 10))

  tgt <- compute_target(H, labels, whiten = TRUE)
  expect_true(all(is.finite(tgt)))
  expect_equal(dim(tgt), c(k, n))
})

test_that("compute_target for batch labels produces different target than class labels", {
  set.seed(42)
  k <- 4; n <- 40
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("A", "B"), each = 20))
  batch <- factor(rep(c("batch1", "batch2"), 20))

  tgt_class <- compute_target(H, labels, whiten = FALSE)
  tgt_batch <- compute_target(H, batch, whiten = FALSE)

  expect_equal(dim(tgt_batch), c(k, n))
  expect_true(all(is.finite(tgt_batch)))
  # They should differ because class and batch structure differ
  expect_false(all(abs(tgt_batch - tgt_class) < 1e-12))
})

test_that("compute_target with single class produces zero target", {
  set.seed(42)
  k <- 3; n <- 15
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep("only_class", n))

  tgt <- compute_target(H, labels, whiten = FALSE)
  # All centroids equal grand mean → shifts should be zero
  expect_equal(max(abs(tgt)), 0, tolerance = 1e-12)
})

test_that("compute_target errors on mismatched dimensions", {
  H <- matrix(runif(20), 4, 5)
  labels <- factor(c("A", "B", "C"))  # length 3, not 5
  expect_error(compute_target(H, labels), "length.*labels")
})

test_that("compute_target with k=2 works", {
  set.seed(42)
  k <- 2; n <- 20
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("A", "B"), each = 10))

  tgt <- compute_target(H, labels, whiten = TRUE)
  expect_equal(dim(tgt), c(k, n))
  expect_true(all(is.finite(tgt)))
})

test_that("compute_target accepts integer and character labels", {
  set.seed(42)
  k <- 3; n <- 12
  H <- matrix(runif(k * n), k, n)

  tgt_int <- compute_target(H, rep(1:3, each = 4))
  tgt_chr <- compute_target(H, rep(c("a", "b", "c"), each = 4))

  expect_equal(dim(tgt_int), c(k, n))
  expect_equal(dim(tgt_chr), c(k, n))
})

test_that("compute_target handles NA labels (unguided samples)", {
  set.seed(42)
  k <- 4; n <- 20
  H <- matrix(runif(k * n), k, n)
  labels <- factor(rep(c("A", "B"), each = 10))
  labels[c(1, 5, 15)] <- NA

  tgt <- compute_target(H, labels, whiten = FALSE)
  expect_equal(dim(tgt), c(k, n))

  # NA-labeled samples should have zero columns
  expect_true(all(tgt[, 1] == 0))
  expect_true(all(tgt[, 5] == 0))
  expect_true(all(tgt[, 15] == 0))

  # Non-NA samples should have non-zero columns
  expect_true(any(tgt[, 2] != 0))
})
