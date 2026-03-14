# Tests for parameter validation error messages
# Ensures all stop() calls in nmf_validation.R produce clear errors

# ---- Setup ----

set.seed(42)
A_dense <- abs(matrix(rnorm(50 * 20), nrow = 50, ncol = 20))
A_sparse <- as(A_dense, "dgCMatrix")

# ============================================================================
# 1. Data validation (validate_data)
# ============================================================================

test_that("nmf() errors on non-coercible data", {
  # An environment cannot be coerced to a matrix
  expect_error(nmf(new.env(), 3), "coercible")
})

test_that("nmf() errors on non-existent file path", {
  expect_error(nmf("/no/such/file.rds", 3), "not found|does not exist")
})

test_that("nmf() errors on unsupported file extension", {
  # Create a temp file with unsupported extension
  tmp <- tempfile(fileext = ".xyz")
  writeLines("dummy", tmp)
  on.exit(unlink(tmp))
  expect_error(nmf(tmp, 3), "Unsupported file format")
})

# ============================================================================
# 2. Penalty validation (validate_all_penalties)
# ============================================================================

test_that("L1 penalty must be in [0, 1)", {
  expect_error(nmf(A_dense, 3, L1 = 1.0), "range \\[0,1\\)")
  expect_error(nmf(A_dense, 3, L1 = 1.5), "range \\[0,1\\)")
  expect_error(nmf(A_dense, 3, L1 = -0.1), "range \\[0,1\\)|non-negative")
})

test_that("L1 penalty must be length 1 or 2", {
  expect_error(nmf(A_dense, 3, L1 = c(0.1, 0.1, 0.1)), "length 1 or 2")
})

test_that("L2 penalty must be non-negative", {
  expect_error(nmf(A_dense, 3, L2 = -0.1), "non-negative|>= 0")
})

test_that("L2 penalty must be length 1 or 2", {
  expect_error(nmf(A_dense, 3, L2 = c(0.1, 0.1, 0.1)), "length 1 or 2")
})

test_that("L21 penalty must be non-negative", {
  expect_error(nmf(A_dense, 3, L21 = -0.1), "non-negative|>= 0")
})

test_that("angular penalty must be non-negative", {
  expect_error(nmf(A_dense, 3, angular = -0.1), "non-negative|>= 0")
})

test_that("graph_lambda must be non-negative", {
  expect_error(nmf(A_dense, 3, graph_lambda = -0.5), "non-negative")
})

test_that("upper_bound must be non-negative", {
  expect_error(nmf(A_dense, 3, upper_bound = -1), "non-negative")
})

# ============================================================================
# 3. CV parameter validation (validate_cv_params)
# ============================================================================

test_that("test_fraction must be a single numeric in [0, 1)", {
  expect_error(nmf(A_dense, 3, test_fraction = 1.5), "range \\[0, 1\\)")
  expect_error(nmf(A_dense, 3, test_fraction = -0.1), "range \\[0, 1\\)")
  expect_error(nmf(A_dense, 3, test_fraction = "half"), "single numeric")
  expect_error(nmf(A_dense, 3, test_fraction = c(0.1, 0.2)), "single numeric")
})

test_that("patience must be a single numeric", {
  expect_error(nmf(A_dense, 3, patience = "five"), "single numeric")
  expect_error(nmf(A_dense, 3, patience = c(1, 2)), "single numeric")
})

# ============================================================================
# 4. Simple parameter validation (validate_simple_params)
# ============================================================================

test_that("sort_model must be a single logical", {
  expect_error(nmf(A_dense, 3, sort_model = 2), "single logical")
  expect_error(nmf(A_dense, 3, sort_model = "yes"), "single logical")
})

test_that("nonneg must be logical", {
  expect_error(nmf(A_dense, 3, nonneg = "yes"), "must be logical")
  expect_error(nmf(A_dense, 3, nonneg = c(TRUE, TRUE, FALSE)), "length 1 or 2")
})

# ============================================================================
# 5. Mask validation (validate_mask)
# ============================================================================

test_that("mask must be a matrix, 'zeros', 'NA', or NULL", {
  expect_error(nmf(A_dense, 3, mask = "foo"), "must be NULL")
  expect_error(nmf(A_dense, 3, mask = "random"), "must be NULL")
})

# ============================================================================
# 6. Graph validation (validate_graphs)
# ============================================================================

test_that("graph_W must have correct dimensions", {
  wrong_graph <- Matrix::sparseMatrix(i = 1:5, j = 1:5, x = 1, dims = c(5, 5))
  # A_dense is 50x20, graph_W must be 50x50
  expect_error(
    nmf(A_dense, 3, graph_W = wrong_graph, graph_lambda = c(1, 0)),
    "must be a .* matrix"
  )
})

test_that("graph_H must have correct dimensions", {
  wrong_graph <- Matrix::sparseMatrix(i = 1:5, j = 1:5, x = 1, dims = c(5, 5))
  # A_dense is 50x20, graph_H must be 20x20
  expect_error(
    nmf(A_dense, 3, graph_H = wrong_graph, graph_lambda = c(0, 1)),
    "must be a .* matrix"
  )
})

# ============================================================================
# 7. Loss/convergence string validation (T10)
# ============================================================================

test_that("invalid loss string is rejected", {
  expect_error(nmf(A_sparse, 3, loss = "invalid"), "should be one of")
  expect_error(nmf(A_sparse, 3, loss = "MSE"), "should be one of")     # case-sensitive
  expect_error(nmf(A_sparse, 3, loss = "l2"), "should be one of")
})
