# Comprehensive Parameter Testing for NMF Function
# Tests all parameters for expected behavior

options(RcppML.verbose = FALSE)

# Setup test matrix for multiple initialization tests
set.seed(123)
test_matrix <- matrix(abs(rnorm(50 * 30)), 50, 30)

# ============================================================================
# Basic Parameters: k, tol, maxit
# ============================================================================

test_that("k parameter accepts single integer", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, k = 1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, k = 5, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, k = 10, maxit = 5, verbose = FALSE))
  
  # Check output dimensions
  m <- nmf(A, k = 7, maxit = 5, verbose = FALSE)
  expect_equal(ncol(m@w), 7)
  expect_equal(nrow(m@h), 7)
})

test_that("k parameter accepts vector for CV", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  cv_result <- nmf(A, k = 2:5, test_fraction = 0.1, maxit = 10, verbose = FALSE)
  expect_true(is.data.frame(cv_result))
  expect_true("k" %in% names(cv_result))
  expect_true("test_mse" %in% names(cv_result))
  expect_equal(sort(unique(cv_result$k)), 2:5)
})

test_that("k = 'auto' performs automatic rank search", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, k = "auto", cv_k_range = c(2, 6), test_fraction = 0.1,
           maxit = 10, verbose = FALSE)
  expect_s4_class(m, "nmf")
  expect_true(!is.null(m@misc$rank_search))
  expect_true("k_optimal" %in% names(m@misc$rank_search))
})

test_that("tol parameter controls convergence", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Strict tolerance should iterate more
  m_strict <- nmf(A, k = 3, tol = 1e-8, maxit = 100, verbose = FALSE)
  m_loose <- nmf(A, k = 3, tol = 1e-4, maxit = 100, seed = m_strict@misc$w_init, verbose = FALSE)
  
  # Strict tolerance typically runs more iterations
  expect_gte(m_strict@misc$iter, m_loose@misc$iter)
})

test_that("maxit parameter limits iterations", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, k = 5, tol = 1e-10, maxit = 10, verbose = FALSE)
  expect_lte(m@misc$iter, 10)
  
  m <- nmf(A, k = 5, tol = 1e-10, maxit = 50, verbose = FALSE)
  expect_lte(m@misc$iter, 50)
})

# ============================================================================
# Penalty Parameters: L1, L2, L21, angular
# ============================================================================

test_that("L1 accepts single value or vector", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, L1 = 0.1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L1 = c(0.1, 0.2), maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L1 = c(0, 0.1), maxit = 5, verbose = FALSE))
})

test_that("L2 accepts single value or vector", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, L2 = 0.1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L2 = c(0.1, 0.2), maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L2 = c(0, 0.1), maxit = 5, verbose = FALSE))
})

test_that("L21 accepts single value or vector", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, L21 = 0.1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L21 = c(0.1, 0.2), maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L21 = c(0, 0.1), maxit = 5, verbose = FALSE))
})

test_that("angular accepts single value or vector", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, angular = 0.1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, angular = c(0.1, 0.2), maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, angular = c(0, 0.1), maxit = 5, verbose = FALSE))
})

test_that("penalties can be combined", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, L1 = 0.01, L2 = 0.01, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L1 = 0.01, L21 = 0.01, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L2 = 0.01, angular = 0.1, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, L1 = 0.01, L2 = 0.01, L21 = 0.01, angular = 0.1, maxit = 5, verbose = FALSE))
})

# ============================================================================
# Seed Parameter
# ============================================================================

test_that("seed = NULL gives random initialization", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m1 <- nmf(A, 3, seed = NULL, maxit = 5, verbose = FALSE)
  m2 <- nmf(A, 3, seed = NULL, maxit = 5, verbose = FALSE)
  
  # Different random initializations
  expect_false(all(m1@w == m2@w))
})

test_that("seed = integer gives reproducible results", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m1 <- nmf(A, 3, seed = 456, maxit = 5, verbose = FALSE)
  m2 <- nmf(A, 3, seed = 456, maxit = 5, verbose = FALSE)
  
  expect_equal(m1@w, m2@w)
  expect_equal(m1@h, m2@h)
})

test_that("seed = matrix gives custom initialization", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  W_custom <- matrix(abs(rnorm(50 * 3)), 50, 3)
  
  m <- nmf(A, 3, seed = W_custom, maxit = 5, verbose = FALSE)
  expect_equal(ncol(m@w), 3)
})

test_that("seed = 'random' gives random initialization", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, seed = "random", maxit = 5, verbose = FALSE))
})

# ============================================================================
# Loss Functions
# ============================================================================

test_that("loss parameter accepts all valid options", {
  set.seed(123)
  A <- matrix(abs(rnorm(40 * 25)), 40, 25)
  
  expect_silent(nmf(A, 3, loss = "mse", maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, loss = "mae", maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, loss = "huber", maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, loss = "gp", maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, loss = "nb", maxit = 5, verbose = FALSE))
  # "kl" still works but emits a deprecation warning
  expect_warning(nmf(A, 3, loss = "kl", maxit = 5, verbose = FALSE), "deprecated")
})

test_that("huber_delta parameter works", {
  set.seed(123)
  A <- matrix(abs(rnorm(40 * 25)), 40, 25)
  
  expect_silent(nmf(A, 3, loss = "huber", huber_delta = 0.5, maxit = 5, verbose = FALSE))
  expect_silent(nmf(A, 3, loss = "huber", huber_delta = 2.0, maxit = 5, verbose = FALSE))
})

# ============================================================================
# Cross-Validation Parameters
# ============================================================================

test_that("test_fraction enables cross-validation", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, 3, test_fraction = 0.1, maxit = 10, verbose = FALSE)
  expect_true(!is.null(m@misc$train_loss))
  expect_true(!is.null(m@misc$test_loss))
  expect_true(!is.null(m@misc$best_iter))
})

test_that("test_fraction = 0 disables CV", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, 3, test_fraction = 0, maxit = 10, verbose = FALSE)
  expect_null(m@misc$train_loss)
  expect_null(m@misc$test_loss)
})

test_that("cv_seed controls CV holdout pattern", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Same cv_seed should give same CV results
  m1 <- nmf(A, 3, test_fraction = 0.1, cv_seed = 789, maxit = 10, verbose = FALSE, seed = 456)
  m2 <- nmf(A, 3, test_fraction = 0.1, cv_seed = 789, maxit = 10, verbose = FALSE, seed = 456)
  
  expect_equal(m1@misc$test_loss, m2@misc$test_loss, tolerance = 1e-10)
})

test_that("cv_seed vector enables multiple folds", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Multiple cv_seeds = multiple replicates
  cv_result <- nmf(A, k = 3:5, test_fraction = 0.1, cv_seed = c(100, 101, 102),
                   maxit = 10, verbose = FALSE)
  
  expect_true(is.data.frame(cv_result))
  # Convert to integer for comparison if rep is a factor
  expect_equal(max(as.integer(cv_result$rep)), 3)  # 3 replicates
})

test_that("patience parameter enables early stopping", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # With patience, should stop before maxit if no improvement
  m <- nmf(A, 3, test_fraction = 0.1, patience = 3, maxit = 100, verbose = FALSE)
  expect_lte(m@misc$best_iter, m@misc$iter)
})

test_that("cv_k_range controls automatic rank search range", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, k = "auto", cv_k_range = c(3, 7), test_fraction = 0.1,
           maxit = 10, verbose = FALSE)
  
  expect_gte(ncol(m@w), 3)
  expect_lte(ncol(m@w), 7)
})

# ============================================================================
# Matrix Format Parameters
# ============================================================================

test_that("sparse parameter treats zeros as missing", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  A[sample(length(A), 500)] <- 0  # Add explicit zeros
  
  m_sparse <- nmf(A, 3, sparse = TRUE, maxit = 10, verbose = FALSE)
  m_dense <- nmf(A, 3, sparse = FALSE, maxit = 10, verbose = FALSE)
  
  # Both should complete without error
  expect_s4_class(m_sparse, "nmf")
  expect_s4_class(m_dense, "nmf")
})

# ============================================================================
# Precision (always fp32 by default, options override)
# ============================================================================

test_that("default precision (fp32) works", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  expect_silent(nmf(A, 3, maxit = 5, verbose = FALSE))
})

test_that("options(RcppML.precision = 'double') escape hatch works", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  old_opt <- getOption("RcppML.precision")
  on.exit(options(RcppML.precision = old_opt))
  
  options(RcppML.precision = "double")
  expect_silent(m_double <- nmf(A, 3, maxit = 5, verbose = FALSE))
  
  options(RcppML.precision = "float")
  expect_silent(m_float <- nmf(A, 3, maxit = 5, verbose = FALSE))
  
  # Both should produce valid models
  expect_s4_class(m_double, "nmf")
  expect_s4_class(m_float, "nmf")
})

test_that("precision gives consistent model structure", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  old_opt <- getOption("RcppML.precision")
  on.exit(options(RcppML.precision = old_opt))
  
  options(RcppML.precision = "float")
  m_float <- nmf(A, 5, seed = 123, maxit = 10, verbose = FALSE)
  
  options(RcppML.precision = "double")
  m_double <- nmf(A, 5, seed = 123, maxit = 10, verbose = FALSE)
  
  # Should have same dimensions
  expect_equal(dim(m_float@w), dim(m_double@w))
  expect_equal(dim(m_float@h), dim(m_double@h))
  
  # Results should be similar (not identical due to precision)
  expect_lt(abs(evaluate(m_float, A) - evaluate(m_double, A)) / evaluate(m_double, A), 0.15)
})

# ============================================================================
# Constraint Parameters
# ============================================================================

test_that("nonneg parameter controls non-negativity", {
  set.seed(123)
  A <- matrix(rnorm(50 * 30), 50, 30)  # Can have negative values
  
  # Both non-negative (default)
  m_nn <- nmf(A, 3, nonneg = c(TRUE, TRUE), maxit = 10, verbose = FALSE)
  expect_gte(min(m_nn$w), 0)
  expect_gte(min(m_nn$h), 0)
  
  # W unconstrained (semi-NMF)
  m_semi <- nmf(A, 3, nonneg = c(FALSE, TRUE), maxit = 10, verbose = FALSE)
  expect_gte(min(m_semi$h), 0)
  # W can be negative in semi-NMF
  
  # Both unconstrained
  m_unconst <- nmf(A, 3, nonneg = c(FALSE, FALSE), maxit = 10, verbose = FALSE)
  expect_s4_class(m_unconst, "nmf")
})

test_that("upper_bound parameter constrains factor values", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30) * 10  # Scale up
  
  # With upper bound
  m_bounded <- nmf(A, 3, upper_bound = c(5, 5), maxit = 20, verbose = FALSE)
  expect_lte(max(m_bounded$w), 5.1)  # Allow small numerical tolerance
  expect_lte(max(m_bounded$h), 5.1)
  
  # Separate bounds for W and H
  m_sep <- nmf(A, 3, upper_bound = c(3, 10), maxit = 20, verbose = FALSE)
  expect_lte(max(m_sep$w), 3.1)
  expect_lte(max(m_sep$h), 10.1)
})

# ============================================================================
# Graph Regularization Parameters
# ============================================================================

test_that("graph_W parameter works", {
  skip_if_not_installed("Matrix")
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Create simple feature graph (adjacent features connected)
  graph_W <- Matrix::bandSparse(50, 50, k = c(-1, 0, 1), 
                                diagonals = list(rep(1, 49), rep(2, 50), rep(1, 49)))
  graph_W <- Matrix::drop0(graph_W)
  
  expect_silent(nmf(A, 3, graph_W = graph_W, graph_lambda = c(0.1, 0),
                    maxit = 10, verbose = FALSE))
})

test_that("graph_H parameter works", {
  skip_if_not_installed("Matrix")
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Create simple sample graph
  graph_H <- Matrix::bandSparse(30, 30, k = c(-1, 0, 1),
                                diagonals = list(rep(1, 29), rep(2, 30), rep(1, 29)))
  graph_H <- Matrix::drop0(graph_H)
  
  expect_silent(nmf(A, 3, graph_H = graph_H, graph_lambda = c(0, 0.1),
                    maxit = 10, verbose = FALSE))
})

test_that("graph_lambda controls graph regularization strength", {
  skip_if_not_installed("Matrix")
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  graph_W <- Matrix::bandSparse(50, 50, k = c(-1, 0, 1),
                                diagonals = list(rep(1, 49), rep(2, 50), rep(1, 49)))
  
  # Single value applies to both
  expect_silent(nmf(A, 3, graph_W = graph_W, graph_lambda = 0.1,
                    maxit = 10, verbose = FALSE))
  
  # Vector specifies W and H separately
  expect_silent(nmf(A, 3, graph_W = graph_W, graph_lambda = c(0.1, 0.2),
                    maxit = 10, verbose = FALSE))
})

# ============================================================================
# Mask Parameter
# ============================================================================

test_that("mask parameter accepts matrix", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  mask <- matrix(FALSE, 50, 30)
  mask[sample(length(mask), 100)] <- TRUE
  
  expect_silent(nmf(A, 3, mask = mask, maxit = 10, verbose = FALSE))
})

test_that("mask = 'zeros' masks zero entries", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  A[sample(length(A), 200)] <- 0
  
  expect_silent(nmf(A, 3, mask = "zeros", maxit = 10, verbose = FALSE))
})

test_that("mask = 'NA' masks NA entries", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  A[sample(length(A), 100)] <- NA
  
  expect_warning(
    nmf(A, 3, mask = "NA", maxit = 10, verbose = FALSE, seed = 456),
    "Detected.*NA values"
  )
})

# ============================================================================
# Output Parameters
# ============================================================================

test_that("sort_model parameter sorts factors", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  # Force CPU for deterministic sorting behavior
  m_sorted <- nmf(A, 5, sort_model = TRUE, maxit = 10, verbose = FALSE, resource = "cpu")
  m_unsorted <- nmf(A, 5, sort_model = FALSE, seed = m_sorted@misc$w_init,
                    maxit = 10, verbose = FALSE, resource = "cpu")
  
  # Sorted model should have decreasing diagonal values
  d_sorted <- m_sorted@d
  expect_true(all(diff(d_sorted) <= 0))
  
  # Both should be valid
  expect_s4_class(m_sorted, "nmf")
  expect_s4_class(m_unsorted, "nmf")
})

# ============================================================================
# Comprehensive Integration Tests
# ============================================================================

test_that("all parameters can be combined", {
  skip_if_not_installed("Matrix")
  set.seed(123)
  A <- Matrix::rsparsematrix(60, 40, 0.3)
  
  graph_W <- Matrix::bandSparse(60, 60, k = c(-1, 0, 1),
                                diagonals = list(rep(1, 59), rep(2, 60), rep(1, 59)))
  
  # Kitchen sink test
  expect_silent(
    nmf(A, k = 4, 
        tol = 1e-5, 
        maxit = 20,
        L1 = c(0.01, 0.01), 
        L2 = c(0.01, 0.01),
        L21 = c(0.01, 0), 
        angular = c(0.1, 0.1),
        seed = 123,
        loss = "huber", 
        huber_delta = 1.0,
        graph_W = graph_W, 
        graph_lambda = c(0.05, 0),
        sort_model = TRUE,
        sparse = FALSE,
        nonneg = c(TRUE, TRUE),
        upper_bound = c(10, 10),
        test_fraction = 0.1,
        patience = 5,
        cv_seed = 456,
        verbose = FALSE)
  )
})

test_that("model output has expected structure", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  m <- nmf(A, 5, maxit = 20, seed = 123, verbose = FALSE)
  
  # Check S4 class
  expect_s4_class(m, "nmf")
  
  # Check slots
  expect_true(!is.null(m@w))
  expect_true(!is.null(m@d))
  expect_true(!is.null(m@h))
  expect_true(!is.null(m@misc))
  
  # Check dimensions
  expect_equal(dim(m@w), c(50, 5))
  expect_equal(length(m@d), 5)
  expect_equal(dim(m@h), c(5, 30))
  
  # Check misc list
  expect_true("iter" %in% names(m@misc))
  expect_true("tol" %in% names(m@misc))
  expect_true("loss" %in% names(m@misc))
  expect_true("runtime" %in% names(m@misc))
})

test_that("CV output has expected structure", {
  set.seed(123)
  A <- matrix(abs(rnorm(50 * 30)), 50, 30)
  
  cv <- nmf(A, k = 2:5, test_fraction = 0.1, maxit = 10, cv_seed = c(100, 101),
            verbose = FALSE)
  
  # Check data frame structure
  expect_true(is.data.frame(cv))
  expect_true("rep" %in% names(cv))
  expect_true("k" %in% names(cv))
  expect_true("train_mse" %in% names(cv))
  expect_true("test_mse" %in% names(cv))
  expect_true("best_iter" %in% names(cv))
  expect_true("total_iter" %in% names(cv))
  
  # Check values
  expect_equal(nrow(cv), 4 * 2)  # 4 ranks × 2 replicates
  expect_equal(sort(unique(as.integer(cv$k))), 2:5)
  expect_equal(sort(unique(as.integer(cv$rep))), 1:2)
})

# ============================================================================
# Multiple Initialization Tests
# ============================================================================

test_that("Seed: Multiple random seeds for multiple runs", {
  set.seed(123)
  seeds <- sample.int(1e6, 3)
  
  model <- nmf(test_matrix, k = 2, seed = seeds, maxit = 10, verbose = FALSE)
  
  expect_s4_class(model, "nmf")
  expect_true("all_inits" %in% names(model@misc))
  expect_equal(nrow(model@misc$all_inits), 3)
  expect_true(sum(model@misc$all_inits$selected) == 1)
})

test_that("Seed: Multiple custom initialization matrices", {
  k <- 2
  n <- nrow(test_matrix)
  init1 <- matrix(runif(n * k), n, k)
  init2 <- matrix(runif(n * k), n, k)
  init3 <- matrix(runif(n * k), n, k)
  
  model <- nmf(test_matrix, k = k, seed = list(init1, init2, init3), maxit = 10, verbose = FALSE)
  
  expect_s4_class(model, "nmf")
  expect_true("all_inits" %in% names(model@misc))
  expect_equal(nrow(model@misc$all_inits), 3)
  expect_true(sum(model@misc$all_inits$selected) == 1)
})

test_that("Seed: Rank validation for custom initialization", {
  k <- 3
  n <- nrow(test_matrix)
  wrong_init <- matrix(runif(n * 2), n, 2)  # rank 2 instead of 3
  
  expect_error(
    nmf(test_matrix, k = k, seed = wrong_init, maxit = 10),
    "Rank mismatch"
  )
})

test_that("Seed: Rank validation for list of custom initializations", {
  k <- 2
  n <- nrow(test_matrix)
  init1 <- matrix(runif(n * k), n, k)  # correct rank
  init2 <- matrix(runif(n * 3), n, 3)  # wrong rank
  
  expect_error(
    nmf(test_matrix, k = k, seed = list(init1, init2), maxit = 10),
    "Rank mismatch"
  )
})

test_that("Seed: Multiple inits incompatible with CV", {
  k <- 2
  n <- nrow(test_matrix)
  inits <- list(
    matrix(runif(n * k), n, k),
    matrix(runif(n * k), n, k)
  )
  
  expect_error(
    nmf(test_matrix, k = k, seed = inits, test_fraction = 0.1, maxit = 10),
    "not compatible with cross-validation"
  )
})

# ============================================================================
# NA Detection Tests
# ============================================================================

test_that("NA detection: Warns and auto-masks NA values in dense matrix", {
  test_na <- test_matrix
  test_na[1:3, 1:2] <- NA
  
  expect_warning(
    model <- nmf(test_na, k = 2, maxit = 10, verbose = FALSE, seed = 123),
    "Detected.*NA values"
  )
  
  expect_s4_class(model, "nmf")
})

test_that("NA detection: Works with sparse matrix containing NAs", {
  test_sparse <- as(test_matrix, "dgCMatrix")
  test_sparse[1:3, 1:2] <- NA
  
  expect_warning(
    model <- nmf(test_sparse, k = 2, maxit = 10, verbose = FALSE, seed = 123),
    "Detected.*NA values"
  )
  
  expect_s4_class(model, "nmf")
})
