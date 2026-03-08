# Test simulateSwimmer
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("simulateSwimmer returns matrix with correct dimensions", {
  library(Matrix)
  result <- simulateSwimmer(seed = 42)
  
  expect_true(is(result, "sparseMatrix") || is.matrix(result))
  expect_equal(dim(result), c(256, 1024))  # 256 images, 32x32 pixels
})

test_that("simulateSwimmer custom n_images", {
  result <- simulateSwimmer(n_images = 50, seed = 42)
  expect_equal(nrow(result), 50)
  expect_equal(ncol(result), 1024)
})

test_that("simulateSwimmer gaussian style works and is non-negative", {
  result <- simulateSwimmer(n_images = 10, style = "gaussian", seed = 42)
  expect_equal(nrow(result), 10)
  result_dense <- as.matrix(result)
  expect_true(all(result_dense >= 0))
})

test_that("simulateSwimmer stick style produces output", {
  result <- simulateSwimmer(n_images = 10, style = "stick", seed = 42)
  expect_equal(nrow(result), 10)
  expect_equal(ncol(result), 1024)
})

test_that("simulateSwimmer with return_factors returns list", {
  result <- simulateSwimmer(n_images = 50, style = "gaussian", 
                           return_factors = TRUE, seed = 42)
  
  expect_type(result, "list")
  expect_true("A" %in% names(result))
  expect_true("w" %in% names(result))
  expect_true("h" %in% names(result))
  expect_true("limb_positions" %in% names(result))
  
  expect_equal(nrow(result$A), 50)
  expect_equal(ncol(result$A), 1024)
  expect_equal(nrow(result$w), 50)
  expect_equal(ncol(result$h), 1024)
  # 4 limbs x 4 positions = 16 factors
  expect_equal(ncol(result$w), 16)
  expect_equal(nrow(result$h), 16)
  expect_equal(dim(result$limb_positions), c(50, 4))
})

test_that("simulateSwimmer is reproducible with seed", {
  result1 <- simulateSwimmer(n_images = 20, style = "gaussian", seed = 123)
  result2 <- simulateSwimmer(n_images = 20, style = "gaussian", seed = 123)
  
  expect_equal(as.matrix(result1), as.matrix(result2))
})

test_that("simulateSwimmer gaussian with noise produces non-negative output", {
  result <- simulateSwimmer(n_images = 10, style = "gaussian", noise = 0.1, seed = 42)
  result_dense <- as.matrix(result)
  expect_true(all(result_dense >= 0))
})

test_that("simulateSwimmer different seeds produce different results", {
  result1 <- simulateSwimmer(n_images = 10, style = "gaussian", seed = 1)
  result2 <- simulateSwimmer(n_images = 10, style = "gaussian", seed = 2)
  expect_false(identical(as.matrix(result1), as.matrix(result2)))
})
