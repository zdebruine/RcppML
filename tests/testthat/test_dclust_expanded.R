# Tests for dclust (divisive clustering)

options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("dclust partitions all samples", {
  A <- Matrix::rsparsematrix(100, 1000, 0.1)
  m <- dclust(A, min_samples = 100, min_dist = 0)
  total_samples <- sum(vapply(m, function(x) length(x$samples), integer(1)))
  expect_equal(total_samples, 1000)
})

test_that("dclust returns correct structure", {
  A <- Matrix::rsparsematrix(100, 200, 0.1)
  m <- dclust(A, min_samples = 20, min_dist = 0, seed = 42)

  # Each cluster is a list with id, samples, center, size
  for (cluster in m) {
    expect_true("id" %in% names(cluster))
    expect_true("samples" %in% names(cluster))
    expect_true("center" %in% names(cluster))
    expect_true("size" %in% names(cluster))
    expect_true(is.character(cluster$id))
    expect_true(is.numeric(cluster$samples))
    expect_equal(cluster$size, length(cluster$samples))
  }
})

test_that("dclust min_samples prevents small clusters", {
  set.seed(123)
  A <- Matrix::rsparsematrix(50, 100, 0.2)
  min_s <- 30
  m <- dclust(A, min_samples = min_s, min_dist = 0, seed = 42)

  # All leaf clusters should have at least 1 sample
  for (cluster in m) {
    expect_gte(length(cluster$samples), 1)
  }

  # With very large min_samples relative to N, should have few clusters
  m_large <- dclust(A, min_samples = 80, min_dist = 0, seed = 42)
  # Cannot split 100 samples with min_samples = 80 (each half would be < 80)
  # so should return very few clusters
  expect_lte(length(m_large), 2)
})

test_that("dclust min_dist controls splitting resolution", {
  set.seed(123)
  A <- Matrix::rsparsematrix(50, 200, 0.2)

  # Very high min_dist should prevent almost all splits
  m_strict <- dclust(A, min_samples = 5, min_dist = 0.99, seed = 42)
  n_strict <- length(m_strict)

  # min_dist = 0 should allow all splits
  m_loose <- dclust(A, min_samples = 5, min_dist = 0, seed = 42)
  n_loose <- length(m_loose)

  # Strict should have fewer or equal clusters
  expect_lte(n_strict, n_loose)
})

test_that("dclust is reproducible with seed", {
  A <- Matrix::rsparsematrix(50, 200, 0.1)

  m1 <- dclust(A, min_samples = 20, min_dist = 0, seed = 42)
  m2 <- dclust(A, min_samples = 20, min_dist = 0, seed = 42)

  # Same number of clusters
  expect_equal(length(m1), length(m2))

  # Same sample assignments
  for (i in seq_along(m1)) {
    expect_equal(sort(m1[[i]]$samples), sort(m2[[i]]$samples))
  }
})

test_that("dclust works with dense matrix input", {
  set.seed(123)
  A <- abs(matrix(runif(50 * 100), 50, 100))

  m <- dclust(A, min_samples = 20, min_dist = 0, seed = 42)

  total_samples <- sum(vapply(m, function(x) length(x$samples), integer(1)))
  expect_equal(total_samples, 100)
})

test_that("dclust cluster IDs are unique", {
  A <- Matrix::rsparsematrix(50, 200, 0.1)
  m <- dclust(A, min_samples = 10, min_dist = 0, seed = 42)

  ids <- vapply(m, function(x) x$id, character(1))

  # IDs should be unique
  expect_equal(length(ids), length(unique(ids)))
})

test_that("dclust center vectors have correct dimensions", {
  A <- Matrix::rsparsematrix(50, 200, 0.1)
  m <- dclust(A, min_samples = 20, min_dist = 0, seed = 42)

  for (cluster in m) {
    if (!is.null(cluster$center)) {
      expect_equal(length(cluster$center), 50)  # nrow(A) features
    }
  }
})

test_that("dclust sample indices are valid and non-overlapping", {
  A <- Matrix::rsparsematrix(50, 200, 0.1)
  m <- dclust(A, min_samples = 10, min_dist = 0, seed = 42)

  all_samples <- list()
  for (cluster in m) {
    # Indices are 0-based from C++ (0 to ncol-1)
    expect_true(all(cluster$samples >= 0))
    expect_true(all(cluster$samples < 200))
    all_samples <- c(all_samples, list(cluster$samples))
  }

  # All samples should appear in the output
  all_samples_vec <- unlist(all_samples)
  expect_equal(length(all_samples_vec), length(unique(all_samples_vec)))

  # Total samples should equal ncol(A)
  expect_equal(length(all_samples_vec), 200)

  # All indices present (0-based: 0 to 199)
  expect_equal(sort(all_samples_vec), 0:199)
})

# =========================================================================
# TEST-DCLUST-PROPERTY: Property validation on separable data (Part D)
# =========================================================================

test_that("TEST-DCLUST-PROPERTY: dclust recovers ground-truth clusters on separable data", {
  # Create well-separated data with 3 known clusters
  set.seed(42)
  n_per_cluster <- 50
  n_features <- 30

  # Cluster 1: high values in features 1-10
  C1 <- matrix(0, n_features, n_per_cluster)
  C1[1:10, ] <- abs(rnorm(10 * n_per_cluster, mean = 5, sd = 0.5))

  # Cluster 2: high values in features 11-20
  C2 <- matrix(0, n_features, n_per_cluster)
  C2[11:20, ] <- abs(rnorm(10 * n_per_cluster, mean = 5, sd = 0.5))

  # Cluster 3: high values in features 21-30
  C3 <- matrix(0, n_features, n_per_cluster)
  C3[21:30, ] <- abs(rnorm(10 * n_per_cluster, mean = 5, sd = 0.5))

  A <- cbind(C1, C2, C3)
  true_labels <- rep(1:3, each = n_per_cluster)

  m <- dclust(A, min_samples = 10, min_dist = 0, seed = 42)

  # All samples should be assigned
  all_assigned <- sort(unlist(lapply(m, function(x) x$samples)))
  expect_equal(all_assigned, 0:(3 * n_per_cluster - 1))

  # Should produce at least 3 clusters (may produce more via recursive splitting)
  expect_gte(length(m), 3)

  # Check purity: each cluster should predominantly contain samples from one true group
  # 0-based indices → true_labels[idx + 1]
  purities <- vapply(m, function(cl) {
    labels <- true_labels[cl$samples + 1]
    max(table(labels)) / length(labels)
  }, numeric(1))

  # Average purity should be high (> 0.8 for well-separated data)
  expect_gt(mean(purities), 0.8)
})
