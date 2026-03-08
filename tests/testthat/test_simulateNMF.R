# Test simulateNMF
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("simulateNMF returns correct structure", {
  result <- simulateNMF(50, 30, k = 3, seed = 42)
  
  expect_type(result, "list")
  expect_named(result, c("A", "w", "h"))
})

test_that("simulateNMF returns matrices with correct dimensions", {
  nrow <- 50
  ncol <- 30
  k <- 5
  result <- simulateNMF(nrow, ncol, k, seed = 42)
  
  expect_equal(dim(result$A), c(nrow, ncol))
  expect_equal(dim(result$w), c(nrow, k))
  expect_equal(dim(result$h), c(k, ncol))
})

test_that("simulateNMF with no noise produces non-negative matrix", {
  result <- simulateNMF(50, 30, k = 3, noise = 0, dropout = 0, seed = 42)
  
  expect_true(all(result$A >= 0))
  expect_true(all(result$w >= 0))
  expect_true(all(result$h >= 0))
})

test_that("simulateNMF with noise still produces non-negative matrix", {
  result <- simulateNMF(50, 30, k = 3, noise = 0.5, seed = 42)
  
  # Negative values from noise should be clipped to 0
  expect_true(all(result$A >= 0))
})

test_that("simulateNMF with dropout introduces zeros", {
  result_no_dropout <- simulateNMF(50, 30, k = 3, noise = 0, dropout = 0, seed = 42)
  result_dropout <- simulateNMF(50, 30, k = 3, noise = 0, dropout = 0.5, seed = 42)
  
  # Dropout version should have more zeros
  zeros_no <- sum(result_no_dropout$A == 0)
  zeros_yes <- sum(result_dropout$A == 0)
  expect_true(zeros_yes > zeros_no)
})

test_that("simulateNMF is reproducible with seed", {
  result1 <- simulateNMF(50, 30, k = 3, seed = 123)
  result2 <- simulateNMF(50, 30, k = 3, seed = 123)
  
  expect_equal(result1$A, result2$A)
  expect_equal(result1$w, result2$w)
  expect_equal(result1$h, result2$h)
})

test_that("simulateNMF different seeds give different results", {
  result1 <- simulateNMF(50, 30, k = 3, seed = 1)
  result2 <- simulateNMF(50, 30, k = 3, seed = 2)
  
  expect_false(identical(result1$A, result2$A))
})

test_that("simulateNMF w columns sum to 1", {
  result <- simulateNMF(50, 30, k = 3, noise = 0, dropout = 0, seed = 42)
  
  # w is normalized so columns sum to 1
  col_sums <- colSums(result$w)
  expect_equal(col_sums, rep(1, 3), tolerance = 1e-10)
})
