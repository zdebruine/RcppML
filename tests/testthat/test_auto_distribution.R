test_that("auto_nmf_distribution runs with default distributions", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, maxit = 20)

  expect_type(res, "list")
  expect_true("best" %in% names(res))
  expect_true("comparison" %in% names(res))
  expect_true("models" %in% names(res))

  expect_true(res$best %in% c("mse", "gp", "nb"))
  expect_s3_class(res$comparison, "data.frame")
  expect_equal(nrow(res$comparison), 3)
  expect_true(all(c("distribution", "nll", "df", "aic", "bic", "selected") %in%
                    names(res$comparison)))
  expect_equal(sum(res$comparison$selected), 1)
  expect_equal(length(res$models), 3)

  # All models should be nmf objects
  for (m in res$models) {
    expect_s4_class(m, "nmf")
  }
})

test_that("auto_nmf_distribution works with subset of distributions", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, distributions = c("mse", "gp"), maxit = 20)

  expect_equal(nrow(res$comparison), 2)
  expect_equal(length(res$models), 2)
  expect_true(res$best %in% c("mse", "gp"))
})

test_that("auto_nmf_distribution AIC criterion works", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, criterion = "aic", maxit = 20)

  expect_true(res$best %in% c("mse", "gp", "nb"))
  # AIC-selected should have lowest AIC
  best_row <- res$comparison[res$comparison$selected, ]
  expect_equal(best_row$aic, min(res$comparison$aic))
})

test_that("auto_nmf_distribution BIC penalizes complexity", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, maxit = 20)

  # BIC-selected should have lowest BIC
  best_row <- res$comparison[res$comparison$selected, ]
  expect_equal(best_row$bic, min(res$comparison$bic))

  # GP and NB have more parameters than MSE
  mse_df <- res$comparison$df[res$comparison$distribution == "mse"]
  gp_df  <- res$comparison$df[res$comparison$distribution == "gp"]
  nb_df  <- res$comparison$df[res$comparison$distribution == "nb"]
  expect_gt(gp_df, mse_df)
  expect_gt(nb_df, mse_df)
})

test_that("auto_nmf_distribution works with dense matrix", {
  set.seed(42)
  A <- matrix(abs(rnorm(50 * 30, mean = 5)), nrow = 50, ncol = 30)

  res <- auto_nmf_distribution(A, k = 3, maxit = 20)

  expect_type(res, "list")
  expect_true(res$best %in% c("mse", "gp", "nb"))
  expect_equal(nrow(res$comparison), 3)
})

test_that("auto_nmf_distribution verbose mode prints output", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(30, 20, density = 0.3)) * 10

  expect_output(
    auto_nmf_distribution(A, k = 2, maxit = 10, verbose = TRUE),
    "Fitting NMF"
  )
})

test_that("auto_nmf_distribution NLL values are finite", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, maxit = 20)

  expect_true(all(is.finite(res$comparison$nll)))
  expect_true(all(is.finite(res$comparison$aic)))
  expect_true(all(is.finite(res$comparison$bic)))
})

test_that("auto_nmf_distribution single distribution works", {
  library(Matrix)
  set.seed(42)
  A <- abs(rsparsematrix(50, 30, density = 0.3)) * 10

  res <- auto_nmf_distribution(A, k = 3, distributions = "gp", maxit = 20)

  expect_equal(nrow(res$comparison), 1)
  expect_equal(res$best, "gp")
  expect_true(res$comparison$selected[1])
})
