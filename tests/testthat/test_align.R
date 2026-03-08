# Tests for align() S4 method
# align() resolves permutation and scaling ambiguity between two NMF models

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("align returns same class as input", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 50, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 3, maxit = 50, seed = 2, verbose = FALSE)

  aligned <- align(m2, m1)
  expect_s4_class(aligned, "nmf")
})

test_that("align preserves factor dimensions", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 4, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 4, maxit = 50, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 4, maxit = 50, seed = 2, verbose = FALSE)

  aligned <- align(m2, m1)
  expect_equal(dim(aligned@w), dim(m2@w))
  expect_equal(dim(aligned@h), dim(m2@h))
  expect_equal(length(aligned@d), length(m2@d))
})

test_that("align improves factor correlation with reference", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 100, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 3, maxit = 100, seed = 2, verbose = FALSE)

  aligned <- align(m2, m1)

  # Aligned model should have higher column-wise correlation with ref
  cor_before <- sapply(1:3, function(i) {
    max(abs(sapply(1:3, function(j) cor(m2@w[, i], m1@w[, j]))))
  })
  cor_after <- sapply(1:3, function(i) {
    abs(cor(aligned@w[, i], m1@w[, i]))
  })

  # After alignment, the diagonal correlations should match the best
  # possible pairing (which they already were matched to)
  expect_true(mean(cor_after) >= mean(cor_before) - 0.01)
})

test_that("align works with k=2", {
  set.seed(42)
  data <- simulateNMF(30, 20, k = 2, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 2, maxit = 50, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 2, maxit = 50, seed = 2, verbose = FALSE)

  aligned <- align(m2, m1)
  expect_equal(ncol(aligned@w), 2)
  expect_equal(nrow(aligned@h), 2)
})

test_that("align with cor method", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 50, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 3, maxit = 50, seed = 2, verbose = FALSE)

  aligned_cos <- align(m2, m1, method = "cosine")
  aligned_cor <- align(m2, m1, method = "cor")

  # Both should produce valid results
  expect_s4_class(aligned_cos, "nmf")
  expect_s4_class(aligned_cor, "nmf")
})

test_that("align errors on dimension mismatch", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 10, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 4, maxit = 10, seed = 2, verbose = FALSE)

  expect_error(align(m2, m1))
})

test_that("align with identical models is identity-like", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 50, seed = 1, verbose = FALSE)

  aligned <- align(m1, m1)

  # Aligned to itself should preserve high correlation
  for (i in 1:3) {
    expect_gt(abs(cor(aligned@w[, i], m1@w[, i])), 0.99)
  }
})

test_that("align reconstruction error is preserved", {
  set.seed(42)
  data <- simulateNMF(50, 40, k = 3, noise = 0.1, seed = 42)
  A <- data$A

  m1 <- nmf(A, 3, maxit = 100, seed = 1, verbose = FALSE)
  m2 <- nmf(A, 3, maxit = 100, seed = 2, verbose = FALSE)

  # Reconstruction before alignment
  recon_before <- m2@w %*% diag(m2@d) %*% m2@h
  mse_before <- mean((as.matrix(A) - recon_before)^2)

  aligned <- align(m2, m1)

  # Reconstruction after alignment — align permutes factors but
  # prod() should give same reconstruction
  recon_after <- prod(aligned)
  mse_after <- mean((as.matrix(A) - as.matrix(recon_after))^2)

  # MSE should be identical (alignment is a permutation, not a change in fit)
  expect_equal(mse_before, mse_after, tolerance = 1e-6)
})
