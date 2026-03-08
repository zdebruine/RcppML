# Test consensus_nmf
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

test_that("consensus_nmf returns correct structure with hard method", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(30, 20, 0.5))
  
  result <- consensus_nmf(A, k = 3, reps = 5, method = "hard", seed = 123, verbose = FALSE)
  
  expect_s3_class(result, "consensus_nmf")
  expect_true(is.matrix(result$consensus))
  expect_equal(dim(result$consensus), c(30, 30))
  expect_true(is.list(result$models))
  expect_length(result$models, 5)
  expect_true(is.integer(result$clusters) || is.numeric(result$clusters))
  expect_length(result$clusters, 30)
  expect_true(is.numeric(result$cophenetic))
  expect_equal(result$method, "hard")
  expect_equal(result$k, 3)
  expect_equal(result$reps, 5)
})

test_that("consensus_nmf hard method produces valid consensus matrix", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))
  
  result <- consensus_nmf(A, k = 2, reps = 5, method = "hard", seed = 123, verbose = FALSE)
  
  # Consensus values should be between 0 and 1
  expect_true(all(result$consensus >= 0 & result$consensus <= 1))
  # Diagonal should be 1 (each sample always co-clusters with itself)
  expect_equal(diag(result$consensus), rep(1, 20), tolerance = 1e-10)
  # Should be symmetric
  expect_equal(result$consensus, t(result$consensus))
})

test_that("consensus_nmf knn_jaccard method works", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))
  
  result <- consensus_nmf(A, k = 2, reps = 5, method = "knn_jaccard", knn = 5, 
                          seed = 123, verbose = FALSE)
  
  expect_s3_class(result, "consensus_nmf")
  expect_equal(result$method, "knn_jaccard")
  expect_equal(result$knn, 5)
  expect_true(all(result$consensus >= 0 & result$consensus <= 1))
})

test_that("consensus_nmf cluster assignments have correct range", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(30, 20, 0.5))
  k <- 4
  
  result <- consensus_nmf(A, k = k, reps = 5, method = "hard", seed = 123, verbose = FALSE)
  
  # All cluster assignments should be between 1 and k
  expect_true(all(result$clusters >= 1 & result$clusters <= k))
})

test_that("consensus_nmf cophenetic is between -1 and 1", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))
  
  result <- consensus_nmf(A, k = 2, reps = 5, seed = 123, verbose = FALSE)
  
  expect_true(result$cophenetic >= -1 && result$cophenetic <= 1)
})

test_that("summary.consensus_nmf returns object and prints stats", {
  skip_on_cran()
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(20, 15, 0.5))
  
  result <- consensus_nmf(A, k = 2, reps = 5, seed = 123, verbose = FALSE)
  
  # summary() should return the input object invisibly
  s <- summary(result)
  expect_s3_class(s, "consensus_nmf")
  expect_identical(s, result)
  
  # Verify that summary() produces console output
  output <- capture.output(summary(result))
  expect_true(any(grepl("Rank \\(k\\):", output)))
  expect_true(any(grepl("Replicates:", output)))
  expect_true(any(grepl("Cophenetic correlation:", output)))
  expect_true(any(grepl("Cluster sizes:", output)))
  expect_true(any(grepl("Consensus summary:", output)))
})
