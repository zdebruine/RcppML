# Test bipartiteMatch
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("bipartiteMatch returns correct structure", {
  cost <- matrix(c(1, 0.5, 0.2,
                    0.5, 1, 0.3,
                    0.2, 0.3, 1), nrow = 3, byrow = TRUE)
  result <- bipartiteMatch(cost)
  
  expect_type(result, "list")
  expect_true("cost" %in% names(result))
  expect_true("assignment" %in% names(result))
  expect_type(result$cost, "double")
  expect_true(is.numeric(result$assignment))
})

test_that("bipartiteMatch minimizes cost", {
  # Cost matrix where optimal is off-diagonal matching
  # Row 0 -> Col 2 (cost 0.2), Row 1 -> Col 0 (cost 0.5), Row 2 -> Col 1 (cost 0.3)
  cost <- matrix(c(1, 0.5, 0.2,
                    0.5, 1, 0.3,
                    0.2, 0.3, 1), nrow = 3, byrow = TRUE)
  result <- bipartiteMatch(cost)
  
  # Minimum cost matching should be less than diagonal (sum = 3)
  expect_true(result$cost < 3)
  
  # Assignment should be a permutation (0-indexed)
  expect_length(result$assignment, 3)
  expect_true(setequal(result$assignment, c(0, 1, 2)))
})

test_that("bipartiteMatch with identity cost matches off-diagonal", {
  # With identity matrix, minimum cost is 0 (all off-diagonal)
  cost <- diag(3)
  result <- bipartiteMatch(cost)
  
  expect_equal(result$cost, 0)
  # No row should be matched to itself
  for (i in seq_along(result$assignment)) {
    expect_false(result$assignment[i] == (i - 1))
  }
})

test_that("bipartiteMatch with zero matrix has zero cost", {
  cost <- matrix(0, 3, 3)
  result <- bipartiteMatch(cost)
  expect_equal(result$cost, 0)
})

test_that("bipartiteMatch handles single element", {
  cost <- matrix(0.5, nrow = 1, ncol = 1)
  result <- bipartiteMatch(cost)
  expect_equal(result$cost, 0.5)
  expect_equal(result$assignment, 0)
})

# =========================================================================
# TEST-BIPARTITE-MATCH-PROPERTY: Optimality verification (Part D)
# =========================================================================

test_that("TEST-BIPARTITE-MATCH-PROPERTY: optimal assignment on known cost matrix", {
  # 4x4 cost matrix with a known unique optimal assignment
  # Optimal: row0→col2 (1), row1→col3 (2), row2→col0 (3), row3→col1 (4) = 10
  cost <- matrix(c(
    10,  8, 1, 9,    # row 0: cheapest is col 2 (cost 1)
     7, 10, 6, 2,    # row 1: cheapest is col 3 (cost 2)
     3,  5, 8, 7,    # row 2: cheapest is col 0 (cost 3)
     6,  4, 9, 8     # row 3: cheapest is col 1 (cost 4)
  ), nrow = 4, byrow = TRUE)

  result <- bipartiteMatch(cost)

  # Optimal total cost = 1 + 2 + 3 + 4 = 10
  expect_equal(result$cost, 10)

  # Verify assignment is valid permutation (0-indexed)
  expect_length(result$assignment, 4)
  expect_true(setequal(result$assignment, 0:3))

  # Verify the returned assignment produces the claimed cost
  actual_cost <- sum(vapply(seq_along(result$assignment), function(i) {
    cost[i, result$assignment[i] + 1]
  }, numeric(1)))
  expect_equal(actual_cost, result$cost)

  # Brute-force verify optimality over all 24 permutations (4! = 24)
  all_perms <- function(v) {
    if (length(v) == 1) return(list(v))
    res <- list()
    for (i in seq_along(v)) {
      rest <- all_perms(v[-i])
      for (p in rest) res[[length(res) + 1]] <- c(v[i], p)
    }
    res
  }
  perms <- all_perms(0:3)
  costs <- vapply(perms, function(p) {
    sum(vapply(seq_along(p), function(i) cost[i, p[i] + 1], numeric(1)))
  }, numeric(1))
  expect_equal(result$cost, min(costs))
})
