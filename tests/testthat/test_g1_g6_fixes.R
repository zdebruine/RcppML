# Tests for G1-G6 audit fixes
# G1: CV path Gamma/IG/Tweedie dispersion estimation
# G3: distribution="auto" with custom power -> Tweedie mapping
# G4: Graph NMF Tweedie support
# G5: Multi-rank CV distribution parameter columns
# G6: distribution_config$power_param alias
# (G2 is GPU-only, tested in test_gpu_*.R)

options(RcppML.verbose = FALSE)

# ============================================================
# G1: CV path returns dispersion for Gamma/IG/Tweedie
# ============================================================

test_that("G1: Gamma CV returns dispersion vector", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(60 * 40, shape = 2, rate = 1), 60, 40)
  A[A < 1e-8] <- 1e-8
  model <- nmf(A, 2, distribution = "gamma", dispersion = "per_row",
               test_fraction = 0.1, maxit = 20, tol = 1e-4, seed = 1,
               verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp), info = "Gamma CV should return dispersion")
  expect_equal(length(disp), nrow(A))
  expect_true(all(disp > 0))
})

test_that("G1: Inverse Gaussian CV returns dispersion vector", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(60 * 40, shape = 2, rate = 1), 60, 40)
  A[A < 1e-8] <- 1e-8
  model <- nmf(A, 2, distribution = "inverse_gaussian", dispersion = "per_row",
               test_fraction = 0.1, maxit = 20, tol = 1e-4, seed = 1,
               verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp), info = "IG CV should return dispersion")
  expect_equal(length(disp), nrow(A))
  expect_true(all(disp > 0))
})

test_that("G1: Tweedie CV returns dispersion vector", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(60 * 40, shape = 2, rate = 1), 60, 40)
  A[A < 1e-8] <- 1e-8
  model <- nmf(A, 2, distribution = "tweedie", tweedie_power = 1.5,
               dispersion = "per_row",
               test_fraction = 0.1, maxit = 20, tol = 1e-4, seed = 1,
               verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp), info = "Tweedie CV should return dispersion")
  expect_equal(length(disp), nrow(A))
  expect_true(all(disp > 0))
})

test_that("G1: Gamma CV per_col dispersion returns n-length vector", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(50 * 30, shape = 2, rate = 1), 50, 30)
  A[A < 1e-8] <- 1e-8
  model <- nmf(A, 2, distribution = "gamma", dispersion = "per_col",
               test_fraction = 0.1, maxit = 20, tol = 1e-4, seed = 1,
               verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp))
  expect_equal(length(disp), ncol(A))
})

test_that("G1: Gamma CV with sparse data returns dispersion", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(50 * 30, shape = 2, rate = 1), 50, 30)
  A[A < 1e-8] <- 1e-8
  A_sp <- Matrix::Matrix(A, sparse = TRUE)
  model <- nmf(A_sp, 2, distribution = "gamma", dispersion = "per_row",
               test_fraction = 0.1, maxit = 20, tol = 1e-4, seed = 1,
               verbose = FALSE)
  disp <- model@misc$dispersion
  expect_true(!is.null(disp), info = "Gamma CV sparse should return dispersion")
  expect_equal(length(disp), nrow(A))
})

# ============================================================
# G3: distribution="auto" maps custom power to Tweedie
# ============================================================

test_that("G3: auto-distribution with custom power does not crash", {
  skip_on_cran()
  # Simulate data where Tweedie might be best
  set.seed(99)
  A <- matrix(rgamma(60 * 40, shape = 0.5, rate = 0.5), 60, 40)
  A[A < 1e-8] <- 1e-8
  # distribution="auto" runs score_test_distribution which could return
  # a custom power (e.g. "power_1.5"); the fix maps it to "tweedie"
  model <- nmf(A, 3, distribution = "auto",
               maxit = 20, tol = 1e-3, seed = 42, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("G3: score_test with non-standard powers returns valid result", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(40 * 25, shape = 2, rate = 1), 40, 25)
  model <- nmf(A, 2, maxit = 10, tol = 1e-3, seed = 1, verbose = FALSE)
  # Use custom powers that include non-standard values
  diag <- score_test_distribution(A, model, powers = c(0.5, 1.5, 2.5))
  expect_true(is.data.frame(diag$scores))
  expect_equal(nrow(diag$scores), 3)
  # Non-standard powers should be labeled "power_X"
  expect_true(any(grepl("^power_", diag$scores$distribution)))
})

# ============================================================
# G4: Graph NMF accepts loss="tweedie"
# ============================================================

test_that("G4: factor_net with loss='tweedie' runs", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(40 * 25, shape = 2, rate = 1), 40, 25)
  A[A < 1e-8] <- 1e-8
  inp <- factor_input(A, "X")
  L1 <- inp |> nmf_layer(k = 2)
  gc <- factor_config(maxit = 10, tol = 1e-3, loss = "tweedie", seed = 1L)
  net <- factor_net(inp, L1, config = gc)
  result <- fit(net)
  expect_true(is.finite(result$layers$L1$loss))
})

# ============================================================
# G5: Multi-rank CV includes distribution parameter columns
# ============================================================

test_that("G5: multi-rank CV with GP includes mean_theta column", {
  skip_on_cran()
  set.seed(42)
  A <- Matrix::rsparsematrix(50, 30, density = 0.4)
  A@x <- abs(A@x) * 10
  A <- ceiling(A)
  cv_result <- nmf(A, k = 2:3, test_fraction = 0.1, loss = "gp",
                   dispersion = "per_row",
                   maxit = 15, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_true(is.data.frame(cv_result))
  expect_true("mean_theta" %in% names(cv_result),
              info = "Multi-rank CV should have mean_theta column")
  expect_true("mean_dispersion" %in% names(cv_result),
              info = "Multi-rank CV should have mean_dispersion column")
})

test_that("G5: multi-rank CV with Gamma includes mean_dispersion column", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(50 * 30, shape = 2, rate = 1), 50, 30)
  A[A < 1e-8] <- 1e-8
  cv_result <- nmf(A, k = 2:3, test_fraction = 0.1, loss = "gamma",
                   dispersion = "per_row",
                   maxit = 15, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_true(is.data.frame(cv_result))
  expect_true("mean_dispersion" %in% names(cv_result))
  # Gamma should have non-NA mean_dispersion values
  gamma_disp <- cv_result$mean_dispersion
  expect_true(any(!is.na(gamma_disp)),
              info = "Gamma multi-rank CV should have non-NA dispersion")
})

test_that("G5: multi-rank CV with MSE has NA distribution columns", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(abs(rnorm(40 * 25, 2)), 40, 25)
  cv_result <- nmf(A, k = 2:3, test_fraction = 0.1, loss = "mse",
                   maxit = 10, tol = 1e-3, seed = 1, verbose = FALSE)
  expect_true(is.data.frame(cv_result))
  expect_true("mean_theta" %in% names(cv_result))
  expect_true("mean_dispersion" %in% names(cv_result))
  # MSE has no distribution params, so both should be NA
  expect_true(all(is.na(cv_result$mean_theta)))
  expect_true(all(is.na(cv_result$mean_dispersion)))
})

# ============================================================
# G6: distribution_config$power_param alias
# ============================================================

test_that("G6: power_param alias sets tweedie_power", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(40 * 25, shape = 2, rate = 1), 40, 25)
  A[A < 1e-8] <- 1e-8
  # Use power_param instead of tweedie_power
  model <- nmf(A, 2, distribution = "tweedie",
               distribution_config = list(power_param = 2.5),
               maxit = 20, tol = 1e-4, seed = 1, verbose = FALSE)
  expect_s4_class(model, "nmf")
})

test_that("G6: power_param overrides tweedie_power in distribution_config", {
  skip_on_cran()
  set.seed(42)
  A <- matrix(rgamma(40 * 25, shape = 2, rate = 1), 40, 25)
  A[A < 1e-8] <- 1e-8
  # Both specified; power_param comes after tweedie_power so wins
  model_alias <- nmf(A, 2, distribution = "tweedie",
                     distribution_config = list(tweedie_power = 2.0,
                                                power_param = 2.5),
                     maxit = 30, tol = 1e-6, seed = 1, verbose = FALSE)
  model_direct <- nmf(A, 2, distribution = "tweedie",
                      distribution_config = list(tweedie_power = 2.5),
                      maxit = 30, tol = 1e-6, seed = 1, verbose = FALSE)
  # Both should use power 2.5, so losses should match
  expect_equal(model_alias@misc$loss, model_direct@misc$loss, tolerance = 1e-6)
})
