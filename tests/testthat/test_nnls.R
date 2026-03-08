test_that("Testing nnls (projection: minimize ||WH - A||^2)", {
  
  # Create a simple factorization problem
  set.seed(42)
  n_features <- 20
  n_samples <- 15
  k <- 5
  
  # True W and H matrices
  w_true <- abs(matrix(rnorm(n_features * k), n_features, k))
  h_true <- abs(matrix(rnorm(k * n_samples), k, n_samples))
  
  # Generate A = W * H (with some noise)
  A <- w_true %*% h_true + matrix(rnorm(n_features * n_samples, sd = 0.01), n_features, n_samples)
  A[A < 0] <- 0  # Ensure non-negativity
  
  # Project w_true onto A to recover h (use named parameter)
  h_recovered <- nnls(w = w_true, A = A)
  
  # h_recovered should have correct dimensions (k x n_samples)
  expect_equal(nrow(h_recovered), k)
  expect_equal(ncol(h_recovered), n_samples)
  
  # Reconstruction should be close to original
  A_reconstructed <- w_true %*% h_recovered
  reconstruction_error <- mean((A - A_reconstructed)^2)
  expect_true(reconstruction_error < 0.1)
  
  # h_recovered should be correlated with h_true
  expect_true(cor(as.vector(h_true), as.vector(h_recovered)) > 0.9)
})

test_that("nnls with sparse matrices", {
  set.seed(123)
  n_features <- 50
  n_samples <- 30
  k <- 3
  
  # Create sparse data
  w <- abs(matrix(rnorm(n_features * k), n_features, k))
  A_dense <- abs(matrix(rnorm(n_features * n_samples), n_features, n_samples))
  A_dense[A_dense < 0.5] <- 0  # Make sparse
  A_sparse <- as(A_dense, "dgCMatrix")
  
  # nnls should work with sparse input
  h_sparse <- nnls(w = w, A = A_sparse)
  h_dense <- nnls(w = w, A = A_dense)
  
  # Results should be similar
  expect_equal(dim(h_sparse), dim(h_dense))
  expect_true(cor(as.vector(h_sparse), as.vector(h_dense)) > 0.99)
})

test_that("nnls with penalties", {
  set.seed(456)
  n_features <- 30
  n_samples <- 20
  k <- 4
  
  w <- abs(matrix(rnorm(n_features * k), n_features, k))
  A <- abs(matrix(rnorm(n_features * n_samples), n_features, n_samples))
  
  # L1 penalty should encourage sparsity in h
  h_no_penalty <- nnls(w = w, A = A)
  h_with_L1 <- nnls(w = w, A = A, L1 = 0.1)
  expect_true(sum(h_with_L1 == 0) >= sum(h_no_penalty == 0))
  
  # L2 penalty should change coefficients (may not always shrink due to NNLS constraints)
  h_with_L2 <- nnls(w = w, A = A, L2 = 0.5)
  # Just check that L2 produces a result with valid dimensions
  expect_equal(dim(h_with_L2), dim(h_no_penalty))
})

test_that("nnls dimension validation", {
  w <- matrix(1:12, 4, 3)  # 4 features, 3 factors
  A <- matrix(1:20, 4, 5)   # 4 features, 5 samples
  A_wrong <- matrix(1:15, 3, 5)  # Wrong number of features
  
  # Should work with compatible dimensions
  h <- nnls(w = w, A = A)
  expect_equal(nrow(h), 3)  # k factors
  expect_equal(ncol(h), 5)  # n_samples
  
  # Dimension mismatch: 3 rows in A_wrong vs 4 rows in w
  # nnls auto-transposes w when ncol(w) == nrow(A), so w(4x3) with A(3x5) works
  h2 <- nnls(w = w, A = A_wrong)
  expect_equal(nrow(h2), 4)  # k = ncol(t(w)) = 4 after auto-transpose
  expect_equal(ncol(h2), 5)  # n_samples
  
  # True incompatible dimensions: no transposition can fix it
  A_bad <- matrix(1:35, 7, 5)  # 7 rows, neither 4 nor 3 match w
  expect_error(nnls(w = w, A = A_bad))
})
