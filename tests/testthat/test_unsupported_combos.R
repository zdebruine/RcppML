# Test unsupported parameter combinations — guards
# Owned by: Agent 1 (CPU Core)

test_that("GUARD-MSE-ZI: MSE + zero-inflation is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "mse", zi = "row"),
    "requires loss='gp' or loss='nb'"
  )
  expect_error(
    nmf(aml, k = 2, loss = "mse", zi = "col"),
    "requires loss='gp' or loss='nb'"
  )
})

test_that("GUARD-GAMMA-ZI: Gamma + zero-inflation is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "gamma", zi = "row"),
    "requires loss='gp' or loss='nb'"
  )
  expect_error(
    nmf(aml, k = 2, loss = "gamma", zi = "col"),
    "requires loss='gp' or loss='nb'"
  )
})

test_that("GUARD-INVGAUSS-ZI: InvGauss + zero-inflation is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "inverse_gaussian", zi = "row"),
    "requires loss='gp' or loss='nb'"
  )
})

test_that("GUARD-TWEEDIE-ZI: Tweedie + zero-inflation is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "tweedie", zi = "row"),
    "requires loss='gp' or loss='nb'"
  )
})

test_that("GUARD-CHOLESKY-IRLS: Cholesky + non-MSE distribution is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "gp", solver = "cholesky"),
    "solver='cholesky' is not supported with non-MSE"
  )
  expect_error(
    nmf(aml, k = 2, loss = "nb", solver = "cholesky"),
    "solver='cholesky' is not supported with non-MSE"
  )
  expect_error(
    nmf(aml, k = 2, loss = "gamma", solver = "cholesky"),
    "solver='cholesky' is not supported with non-MSE"
  )
})

test_that("GUARD-CHOLESKY-ROBUST: Cholesky + robust is rejected", {
  data(aml, package = "RcppML")
  expect_error(
    nmf(aml, k = 2, loss = "mse", solver = "cholesky", robust = TRUE),
    "solver='cholesky' is not supported with robust"
  )
})

test_that("AUTO-SOLVER-IRLS: auto solver selects CD for IRLS distributions", {
  data(aml, package = "RcppML")
  # GP with auto solver should work (auto selects CD)
  result <- nmf(aml, k = 2, loss = "gp", maxit = 3, tol = 1e-10, verbose = FALSE)
  expect_s4_class(result, "nmf")
})

test_that("GUARD-ZI-NOT-GP-NB: ZI with generic non-GP/NB also rejected", {
  data(aml, package = "RcppML")
  # Verify the catch-all guard still works
  expect_error(
    nmf(aml, k = 2, loss = "mse", zi = "row"),
    "requires loss='gp' or loss='nb'"
  )
})
