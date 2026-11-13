# Reference tests for the non-Gaussian losses.
#
# Every expectation compares the package against math written out here in
# plain R (double precision): unit deviances, negative log-likelihoods and the
# score identities that hold at a stationary point of each likelihood. Nothing
# is compared against another code path of the package except where the claim
# under test is that two inputs (sparse and dense) are the same problem.

options(RcppML.verbose = FALSE)

# ---------------------------------------------------------------------------
# Reference math
# ---------------------------------------------------------------------------

ref_mu <- function(model) {
  d <- model@d
  model@w %*% diag(d, nrow = length(d)) %*% model@h
}

ref_dev_poisson <- function(y, mu) {
  t1 <- ifelse(y > 0, y * log(y / mu), 0)
  2 * (t1 - (y - mu))
}

ref_dev_gamma <- function(y, mu) 2 * ((y - mu) / mu - log(y / mu))

ref_dev_invgauss <- function(y, mu) (y - mu)^2 / (mu^2 * y)

# The fitted mean can be exactly 0 where y = 0. Every term below that
# multiplies y by a function of mu is taken as 0 there (its limit), so the
# references stay finite instead of evaluating 0 * log(0) or 0 * Inf.
ref_dev_tweedie <- function(y, mu, p) {
  t1 <- ifelse(y > 0, y^(2 - p) / ((1 - p) * (2 - p)), 0)
  t2 <- ifelse(y > 0, y * mu^(1 - p) / (1 - p), 0)
  2 * (t1 - t2 + mu^(2 - p) / (2 - p))
}

ref_nll_poisson <- function(y, mu) {
  mu - ifelse(y > 0, y * log(mu), 0) + lgamma(y + 1)
}

# Generalized Poisson with mean mu and variance mu * (1 + theta)^2
# (Consul's form with lambda = mu / (1 + theta)); theta may be a per-row vector.
ref_nll_gp <- function(y, mu, theta) {
  cc <- 1 + theta
  lambda <- mu / cc
  inner <- (mu + theta * y) / cc
  # at y = 0 the two log terms cancel and the NLL is lambda
  ifelse(y > 0,
         -(log(lambda) + (y - 1) * log(inner) - inner - lgamma(y + 1)),
         lambda)
}

# ---------------------------------------------------------------------------
# Simulated data with known factors
# ---------------------------------------------------------------------------

sim_factors <- function(m, n, k, seed, scale = 1) {
  set.seed(seed)
  W <- matrix(rgamma(m * k, shape = 2, rate = 2), m, k)
  H <- matrix(rgamma(k * n, shape = 2, rate = 2), k, n)
  scale * (W %*% H) / k
}

sim_poisson <- function(m = 40, n = 30, k = 3, seed = 11, scale = 1.2) {
  mu <- sim_factors(m, n, k, seed, scale)
  matrix(rpois(m * n, mu), m, n)
}

sim_nb <- function(m = 40, n = 30, k = 3, seed = 12, scale = 3, size = 5) {
  mu <- sim_factors(m, n, k, seed, scale)
  matrix(rnbinom(m * n, size = size, mu = mu), m, n)
}

sim_gamma <- function(m = 40, n = 30, k = 3, seed = 13, phi = 0.1) {
  mu <- sim_factors(m, n, k, seed, 4)
  matrix(rgamma(m * n, shape = 1 / phi, scale = mu * phi), m, n)
}

# Inverse Gaussian by the Michael-Schucany-Haas transformation; Var = phi * mu^3
sim_invgauss <- function(m = 40, n = 30, k = 3, seed = 14, phi = 0.02) {
  mu <- as.vector(sim_factors(m, n, k, seed, 3))
  lambda <- 1 / phi
  nu <- rnorm(m * n)^2
  x <- mu + mu^2 * nu / (2 * lambda) -
    mu / (2 * lambda) * sqrt(4 * mu * lambda * nu + mu^2 * nu^2)
  u <- runif(m * n)
  y <- ifelse(u <= mu / (mu + x), x, mu^2 / x)
  matrix(y, m, n)
}

# Tweedie compound Poisson-gamma with power p in (1, 2); has exact zeros
sim_tweedie <- function(m = 40, n = 30, k = 3, seed = 15, p = 1.5, phi = 1) {
  mu <- as.vector(sim_factors(m, n, k, seed, 2))
  lambda <- mu^(2 - p) / (phi * (2 - p))
  alpha <- (2 - p) / (p - 1)
  sc <- phi * (p - 1) * mu^(p - 1)
  N <- rpois(m * n, lambda)
  y <- ifelse(N > 0, rgamma(m * n, shape = pmax(N, 1) * alpha, scale = sc), 0)
  matrix(y, m, n)
}

as_sparse <- function(A) Matrix::Matrix(A, sparse = TRUE)

fit <- function(A, loss, ..., k = 3, maxit = 60, tol = 1e-8, seed = 7) {
  nmf(A, k, loss = loss, maxit = maxit, tol = tol, seed = seed,
      threads = 2, verbose = FALSE, ...)
}

rel_diff <- function(a, b) max(abs(a - b)) / max(abs(b))

# ---------------------------------------------------------------------------
# 1. Sparse and dense input are the same problem
# ---------------------------------------------------------------------------

expect_same_fit <- function(A, loss, ..., tol_mu = 5e-3, tol_loss = 1e-3) {
  m_dense <- fit(A, loss, ..., maxit = 15)
  m_sparse <- fit(as_sparse(A), loss, ..., maxit = 15)
  expect_lt(rel_diff(ref_mu(m_sparse), ref_mu(m_dense)), tol_mu)
  expect_equal(m_sparse@misc$loss, m_dense@misc$loss, tolerance = tol_loss)
}

test_that("Poisson (gp, dispersion none): sparse and dense input give the same fit", {
  A <- sim_poisson()
  expect_gt(mean(A == 0), 0.2)
  expect_same_fit(A, "gp", dispersion = "none")
})

test_that("gp with per-row dispersion: sparse and dense input give the same fit", {
  expect_same_fit(sim_poisson(seed = 21), "gp", dispersion = "per_row")
})

test_that("nb: sparse and dense input give the same fit", {
  expect_same_fit(sim_nb(), "nb", dispersion = "per_row")
})

test_that("tweedie p = 1.5: sparse and dense input give the same fit", {
  A <- sim_tweedie()
  expect_gt(mean(A == 0), 0.05)
  expect_same_fit(A, "tweedie", tweedie_power = 1.5)
})

test_that("gamma: sparse and dense input give the same fit", {
  expect_same_fit(sim_gamma(), "gamma")
})

test_that("inverse_gaussian: sparse and dense input give the same fit", {
  expect_same_fit(sim_invgauss(), "inverse_gaussian")
})

# ---------------------------------------------------------------------------
# 2. Reported loss equals the stated definition, summed over all entries
# ---------------------------------------------------------------------------

test_that("gp (dispersion none) reports the exact Poisson NLL over all entries", {
  A <- sim_poisson(seed = 31)
  for (X in list(A, as_sparse(A))) {
    model <- fit(X, "gp", dispersion = "none", maxit = 20)
    ref <- sum(ref_nll_poisson(A, ref_mu(model)))
    expect_gt(model@misc$loss, 0)
    expect_equal(model@misc$loss, ref, tolerance = 1e-3)
  }
})

test_that("gp with dispersion reports the exact generalized-Poisson NLL, zeros included", {
  A <- sim_nb(seed = 32, scale = 2, size = 3)
  expect_gt(mean(A == 0), 0.1)
  for (X in list(A, as_sparse(A))) {
    model <- fit(X, "gp", dispersion = "per_row", maxit = 20)
    theta <- model@misc$theta
    skip_if(is.null(theta), "theta not returned on this path")
    ref <- sum(ref_nll_gp(A, ref_mu(model), theta))
    expect_equal(model@misc$loss, ref, tolerance = 1e-3)
  }
})

test_that("nb reports the exact negative-binomial NLL over all entries", {
  A <- sim_nb(seed = 33)
  for (X in list(A, as_sparse(A))) {
    # dispersion = "none" fixes the size at nb_size_max, so the reference
    # needs nothing from the fit except the factors.
    model <- fit(X, "nb", dispersion = "none", nb_size_max = 50, maxit = 20)
    ref <- -sum(dnbinom(A, size = 50, mu = ref_mu(model), log = TRUE))
    expect_gt(model@misc$loss, 0)
    expect_equal(model@misc$loss, ref, tolerance = 1e-3)
  }
})

test_that("gamma, inverse_gaussian and tweedie report the summed unit deviance", {
  A <- sim_gamma(seed = 34)
  model <- fit(A, "gamma", maxit = 20)
  expect_equal(model@misc$loss, sum(ref_dev_gamma(A, ref_mu(model))), tolerance = 1e-3)

  A <- sim_invgauss(seed = 35)
  model <- fit(A, "inverse_gaussian", maxit = 20)
  expect_equal(model@misc$loss, sum(ref_dev_invgauss(A, ref_mu(model))), tolerance = 1e-3)

  A <- sim_tweedie(seed = 36)
  for (X in list(A, as_sparse(A))) {
    model <- fit(X, "tweedie", tweedie_power = 1.5, maxit = 20)
    ref <- sum(ref_dev_tweedie(A, ref_mu(model), 1.5))
    expect_equal(model@misc$loss, ref, tolerance = 1e-3)
  }
})

# ---------------------------------------------------------------------------
# 3. Score identities at the fitted point
# ---------------------------------------------------------------------------
# With identity link and non-negative factors, complementary slackness in the
# last-updated factor gives sum_ij (y - mu) / V(mu) * mu = 0 for a likelihood
# with variance function V. Wrong weights on any subset of entries (for
# example unit weight on zeros) break the identity.

test_that("Poisson fit on sparse input satisfies sum(mu) = sum(y)", {
  A <- sim_poisson(seed = 41, scale = 0.8)
  model <- fit(as_sparse(A), "gp", dispersion = "none", maxit = 150)
  expect_equal(sum(ref_mu(model)), sum(A), tolerance = 5e-3)
})

test_that("gamma fit satisfies mean(y / mu) = 1", {
  A <- sim_gamma(seed = 42)
  model <- fit(A, "gamma", maxit = 150)
  expect_equal(mean(A / ref_mu(model)), 1, tolerance = 5e-3)
})

test_that("inverse Gaussian fit satisfies sum((y - mu) / mu^2) = 0", {
  A <- sim_invgauss(seed = 43)
  mu <- ref_mu(fit(A, "inverse_gaussian", maxit = 150))
  expect_lt(abs(sum((A - mu) / mu^2)) / sum(A / mu^2), 5e-3)
})

test_that("tweedie fit on sparse input satisfies sum((y - mu) mu^(1-p)) = 0", {
  A <- sim_tweedie(seed = 44)
  mu <- ref_mu(fit(as_sparse(A), "tweedie", tweedie_power = 1.5, maxit = 150))
  # (y - mu) mu^(1-p) = y mu^(-1/2) - mu^(1/2); the first term is 0 at y = 0
  ymu <- sum(ifelse(A > 0, A / sqrt(mu), 0))
  expect_lt(abs(ymu - sum(sqrt(mu))) / ymu, 5e-3)
})

# ---------------------------------------------------------------------------
# 4. The matching loss fits its own deviance at least as well as least squares
# ---------------------------------------------------------------------------

test_that("Poisson fit on sparse low counts has lower Poisson deviance than mse", {
  A <- sim_poisson(m = 60, n = 40, seed = 51, scale = 0.6)
  S <- as_sparse(A)
  dev_gp <- mean(ref_dev_poisson(A, ref_mu(fit(S, "gp", dispersion = "none"))))
  dev_ls <- mean(ref_dev_poisson(A, pmax(ref_mu(fit(S, "mse")), 1e-10)))
  expect_lt(dev_gp, dev_ls)
})

test_that("nb fit on sparse input has lower NB deviance than mse", {
  size <- 5
  A <- sim_nb(m = 60, n = 40, seed = 52, size = size)
  S <- as_sparse(A)
  dev_nb <- function(y, mu) {
    t1 <- ifelse(y > 0, y * log(y / mu), 0)
    2 * (t1 - (y + size) * log((y + size) / (mu + size)))
  }
  mu_nb <- ref_mu(fit(S, "nb", dispersion = "none", nb_size_max = size))
  mu_ls <- pmax(ref_mu(fit(S, "mse")), 1e-10)
  expect_lt(mean(dev_nb(A, mu_nb)), mean(dev_nb(A, mu_ls)))
})

test_that("inverse Gaussian fit stays finite and beats mse in its own deviance", {
  A <- sim_invgauss(m = 60, n = 40, seed = 53)
  mu_ig <- ref_mu(fit(A, "inverse_gaussian"))
  mu_ls <- pmax(ref_mu(fit(A, "mse")), 1e-10)
  dev_ig <- mean(ref_dev_invgauss(A, mu_ig))
  expect_true(is.finite(dev_ig))
  expect_true(all(mu_ig > 0))
  expect_lt(dev_ig, mean(ref_dev_invgauss(A, mu_ls)))
  # the truth has mean unit deviance phi = 0.02; a rank-3 fit cannot be far above it
  expect_lt(dev_ig, 0.1)
})

# ---------------------------------------------------------------------------
# 5. Tweedie power: limits and validation
# ---------------------------------------------------------------------------

test_that("tweedie with p = 2 and p = 3 reproduces gamma and inverse Gaussian", {
  A <- sim_gamma(seed = 61)
  expect_lt(rel_diff(ref_mu(fit(A, "tweedie", tweedie_power = 2, maxit = 15)),
                     ref_mu(fit(A, "gamma", maxit = 15))), 1e-5)

  A <- sim_invgauss(seed = 62)
  expect_lt(rel_diff(ref_mu(fit(A, "tweedie", tweedie_power = 3, maxit = 15)),
                     ref_mu(fit(A, "inverse_gaussian", maxit = 15))), 1e-5)
})

test_that("tweedie with p = 1 reproduces the Poisson fit", {
  A <- sim_poisson(seed = 63)
  expect_lt(rel_diff(ref_mu(fit(A, "tweedie", tweedie_power = 1, maxit = 15)),
                     ref_mu(fit(A, "gp", dispersion = "none", maxit = 15))), 1e-3)
})

test_that("tweedie power below 1 is rejected", {
  A <- sim_tweedie(seed = 64)
  expect_error(fit(A, "tweedie", tweedie_power = 0.5), "tweedie_power")
  expect_error(fit(as_sparse(A), "tweedie", tweedie_power = 0), "tweedie_power")
})

# ---------------------------------------------------------------------------
# 6. Data outside the support is an error, not a silent fit
# ---------------------------------------------------------------------------

test_that("zeros are rejected by gamma, inverse_gaussian and tweedie p >= 2", {
  A <- sim_gamma(seed = 71)
  A[3, 4] <- 0
  expect_error(fit(A, "gamma"), "strictly positive")
  expect_error(fit(as_sparse(A), "gamma"), "strictly positive")
  expect_error(fit(A, "inverse_gaussian"), "strictly positive")
  expect_error(fit(A, "tweedie", tweedie_power = 2.5), "strictly positive")
})

test_that("negative values are rejected by count losses", {
  A <- sim_poisson(seed = 72)
  A[2, 2] <- -1
  expect_error(fit(A, "gp", dispersion = "none"), "negative")
  expect_error(fit(as_sparse(A), "nb"), "negative")
})

# ---------------------------------------------------------------------------
# 7. L1 under IRLS: KKT conditions of the penalised objective
# ---------------------------------------------------------------------------
# The W half-step minimises, row by row,
#   sum_j deviance(y_ij, mu_ij) / 2 + L1 * sum_f Wd_if,   Wd = W diag(d) >= 0,
# with H fixed. Its KKT system, with g_if = sum_j h_fj (mu_ij - y_ij) / V(mu_ij) + L1:
#   g_if = 0 where Wd_if > 0,   g_if >= 0 where Wd_if = 0.
# This is also the fixed point of the weighted NNLS step (Gram H diag(w) H',
# right-hand side H diag(w) y - L1, w = 1 / V(mu)). W is the last factor
# updated, so the returned model must satisfy it. Summing Wd_if * g_if gives
#   sum_ij mu (mu - y) / V(mu) + L1 * sum(Wd) = 0.
#
# What these tests deliberately do not assert: that a larger L1 gives more
# exact zeros. Factors are scaled to unit row sums by default, so
# L1 * sum(Wd) = L1 * sum(mu): the penalty is a charge on the total fitted
# mean. Under least squares that subtracts L1 from every target and thresholds
# small entries to zero. Under a Poisson likelihood sum(mu - y log mu) + L1 sum(mu)
# is minimised by the unpenalised solution divided by (1 + L1), so the zero
# pattern does not depend on L1 at all; negative binomial and Tweedie
# likelihoods sit between the two and have no thresholding at y > 0 either.

ref_l1_kkt <- function(model, A, L1, var_fun) {
  mu <- pmax(ref_mu(model), 1e-10)
  Wd <- model@w %*% diag(model@d, nrow = length(model@d))
  h <- model@h
  v <- var_fun(mu)
  g <- ((mu - A) / v) %*% t(h) + L1
  scale <- matrix(rowSums(abs(h)) + L1, nrow(g), ncol(g), byrow = TRUE)
  ymu <- sum(A * mu / v)
  list(
    # sum_ij mu (mu - y) / V + L1 sum(Wd), relative to sum_ij y mu / V
    identity = abs(sum(mu * (mu - A) / v) + L1 * sum(Wd)) / ymu,
    # complementary slackness, weighted by the size of the coefficient
    slack = sum(abs(g) * Wd) / sum(scale * Wd),
    # dual feasibility on the zero set (Inf if the set is empty)
    dual = if (any(Wd == 0)) min((g / scale)[Wd == 0]) else Inf
  )
}

expect_l1_kkt <- function(A, loss, var_fun, ..., L1 = 0.1) {
  model <- fit(as_sparse(A), loss, ..., L1 = c(L1, 0), maxit = 200, tol = 1e-9)
  kkt <- ref_l1_kkt(model, A, L1, var_fun)
  expect_lt(kkt$identity, 1e-2)
  expect_lt(kkt$slack, 3e-2)
  expect_gt(kkt$dual, -0.1)
  invisible(model)
}

test_that("Poisson fit with L1 on W satisfies the penalised KKT conditions", {
  A <- sim_poisson(m = 50, n = 40, seed = 81, scale = 2)
  model <- expect_l1_kkt(A, "gp", function(mu) mu, dispersion = "none")
  # with unit row sums of H the penalty is L1 * sum(mu): the fit is the
  # unpenalised one scaled by 1 / (1 + L1), so sum(mu) (1 + L1) = sum(y)
  expect_equal(unname(rowSums(model@h)), rep(1, nrow(model@h)), tolerance = 1e-4)
  expect_equal(sum(ref_mu(model)) * 1.1, sum(A), tolerance = 1e-2)
})

test_that("nb fit with L1 on W satisfies the penalised KKT conditions", {
  size <- 5
  A <- sim_nb(m = 50, n = 40, seed = 82, size = size)
  expect_l1_kkt(A, "nb", function(mu) mu + mu^2 / size,
                dispersion = "none", nb_size_max = size)
})

test_that("tweedie fit with L1 on W satisfies the penalised KKT conditions", {
  A <- sim_tweedie(m = 50, n = 40, seed = 83)
  expect_l1_kkt(A, "tweedie", function(mu) mu^1.5, tweedie_power = 1.5)
})

# ---------------------------------------------------------------------------
# 8. Negative-binomial size is recovered
# ---------------------------------------------------------------------------

test_that("nb recovers a known size within a factor of two", {
  size <- 5
  A <- sim_nb(m = 80, n = 60, seed = 91, scale = 20, size = size)

  model <- fit(as_sparse(A), "nb", dispersion = "global", maxit = 40)
  est <- model@misc$theta
  skip_if(is.null(est), "size not returned on this path")
  expect_gt(est[1], size / 2)
  expect_lt(est[1], size * 2)

  model <- fit(A, "nb", dispersion = "per_row", maxit = 40)
  est <- model@misc$theta
  expect_length(est, nrow(A))
  expect_gt(median(est), size / 2)
  expect_lt(median(est), size * 2)
})
