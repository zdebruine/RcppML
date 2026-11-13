# R-layer behavior of nmf() and nnls(), checked against references written
# in plain R (no second package path is used as the reference unless stated).
#
# Covers: thread resolution, missing-value masks, input validation, removed
# arguments, initialization (h_init, SVD seeds, RNG state), cross-validation
# outputs, projective/symmetric restrictions and 0.3.7 compatibility.

library(Matrix)

# ---- helpers ---------------------------------------------------------------

# Exact non-negative rank-k matrix with well separated, partly sparse factors
# (so the matrix is not dominated by a single rank-1 component)
ref_lowrank <- function(m = 40, n = 30, k = 3, seed = 1) {
  set.seed(seed)
  W <- matrix(runif(m * k) * rbinom(m * k, 1, 0.5) + 0.05, m, k) %*%
    diag(seq(3, 1, length.out = k), k)
  H <- matrix(runif(k * n) * rbinom(k * n, 1, 0.5) + 0.05, k, n)
  list(A = W %*% H, W = W, H = H)
}

ref_recon <- function(model) model@w %*% diag(model@d, length(model@d)) %*% model@h

# Exact solution of min_x x'ax/2 - b'x subject to x >= 0 by enumerating the
# active sets (only for small problems). `a` symmetric positive definite.
ref_qp_nonneg <- function(a, b) {
  k <- ncol(a)
  best <- rep(0, k); best_obj <- 0
  for (code in seq_len(2^k - 1)) {
    free <- which(bitwAnd(code, 2^(seq_len(k) - 1)) != 0)
    x <- rep(0, k)
    x[free] <- solve(a[free, free, drop = FALSE], b[free])
    if (any(x < 0)) next
    obj <- 0.5 * sum(x * (a %*% x)) - sum(b * x)
    if (obj < best_obj) { best <- x; best_obj <- obj }
  }
  best
}

with_envvars <- function(vars, code) {
  old <- Sys.getenv(names(vars), unset = NA)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm) else do.call(Sys.setenv, as.list(old[nm]))
    }
  })
  for (nm in names(vars)) {
    if (is.na(vars[[nm]])) Sys.unsetenv(nm) else do.call(Sys.setenv, as.list(vars[nm]))
  }
  force(code)
}

# ---- thread resolution -----------------------------------------------------

test_that("thread count: argument, then option, then all cores", {
  resolve <- RcppML:::.resolve_threads
  old <- options(RcppML.threads = 3L)
  on.exit(options(old))
  with_envvars(c(`_R_CHECK_LIMIT_CORES_` = NA, OMP_THREAD_LIMIT = NA), {
    expect_identical(resolve(5), 5L)      # explicit argument wins
    expect_identical(resolve(0), 3L)      # falls back to the option
    expect_identical(resolve(NULL), 3L)
    options(RcppML.threads = 0L)
    expect_identical(resolve(0), 0L)      # 0 = all cores
    options(RcppML.threads = NULL)
    expect_identical(resolve(0), 0L)
  })
})

test_that("thread count is capped under R CMD check and OMP_THREAD_LIMIT", {
  resolve <- RcppML:::.resolve_threads
  old <- options(RcppML.threads = 0L)
  on.exit(options(old))
  with_envvars(c(`_R_CHECK_LIMIT_CORES_` = "TRUE", OMP_THREAD_LIMIT = NA), {
    expect_identical(resolve(0), 2L)
    expect_identical(resolve(16), 2L)
    expect_identical(resolve(1), 1L)
  })
  with_envvars(c(`_R_CHECK_LIMIT_CORES_` = "false", OMP_THREAD_LIMIT = NA), {
    expect_identical(resolve(16), 16L)
  })
  with_envvars(c(`_R_CHECK_LIMIT_CORES_` = NA, OMP_THREAD_LIMIT = "4"), {
    expect_identical(resolve(0), 4L)
    expect_identical(resolve(16), 4L)
    expect_identical(resolve(3), 3L)
  })
})

# ---- masks -----------------------------------------------------------------

test_that("NA mask matrix marks exactly the NA/NaN positions (dense and sparse)", {
  set.seed(1)
  A <- matrix(runif(30), 6, 5)
  A[c(2, 9, 17)] <- NA
  A[23] <- NaN
  ref <- which(is.na(A), arr.ind = TRUE)

  M <- RcppML:::.na_mask_matrix(A)
  expect_s4_class(M, "dgCMatrix")
  expect_equal(dim(M), dim(A))
  got <- which(as.matrix(M) != 0, arr.ind = TRUE)
  expect_equal(unname(got), unname(ref))

  As <- as(Matrix(replace(A, is.na(A), 0), sparse = TRUE), "dgCMatrix")
  As[ref] <- NA
  As <- as(As, "dgCMatrix")
  Ms <- RcppML:::.na_mask_matrix(As)
  expect_equal(unname(which(as.matrix(Ms) != 0, arr.ind = TRUE)), unname(ref))

  # zero filling leaves every observed value untouched
  Z <- RcppML:::.zero_na(A)
  expect_false(anyNA(Z))
  expect_equal(Z[!is.na(A)], A[!is.na(A)])
})

test_that("mask = 'NA' excludes missing entries from the fit (matrix completion)", {
  skip_on_cran()
  sim <- ref_lowrank(m = 60, n = 40, k = 3, seed = 2)
  set.seed(3)
  miss <- sample(length(sim$A), round(0.2 * length(sim$A)))
  A_na <- sim$A
  A_na[miss] <- NA
  A_zero <- replace(sim$A, miss, 0)

  fit_na <- suppressWarnings(
    nmf(A_na, 3, seed = 1, tol = 1e-8, maxit = 200, solver = "cd"))
  fit_zero <- nmf(A_zero, 3, seed = 1, tol = 1e-8, maxit = 200, solver = "cd")

  truth <- sim$A[miss]
  rel_err <- function(fit) {
    sqrt(mean((ref_recon(fit)[miss] - truth)^2)) / sqrt(mean(truth^2))
  }
  # Treating missing values as zeros pulls the held-out reconstruction toward
  # zero; masking them recovers the exact low-rank values.
  e_na <- rel_err(fit_na)
  e_zero <- rel_err(fit_zero)
  expect_lt(e_na, 0.08)
  expect_gt(e_zero, 0.1)
  expect_gt(e_zero, 2 * e_na)
  expect_false(isTRUE(all.equal(fit_na@w, fit_zero@w)))
})

test_that("NA positions are added to a custom mask", {
  sim <- ref_lowrank(m = 30, n = 20, k = 2, seed = 4)
  A <- sim$A
  na_pos <- cbind(c(1, 5, 9), c(2, 3, 4))
  A[na_pos] <- NA
  user_pos <- cbind(c(2, 6), c(7, 8))
  user_mask <- sparseMatrix(i = user_pos[, 1], j = user_pos[, 2], x = 1, dims = dim(A))
  both <- rbind(na_pos, user_pos)
  full_mask <- sparseMatrix(i = both[, 1], j = both[, 2], x = 1, dims = dim(A))

  # same fit, expressed once with NAs and once with an explicit mask
  fit1 <- suppressWarnings(nmf(A, 2, seed = 1, mask = user_mask, maxit = 20, tol = 1e-10))
  fit2 <- nmf(replace(A, is.na(A), 0), 2, seed = 1, mask = full_mask, maxit = 20, tol = 1e-10)
  expect_equal(fit1@w, fit2@w)
  expect_equal(fit1@h, fit2@h)
})

test_that("mask forms are parsed into a zeros flag and a mask matrix", {
  vm <- RcppML:::validate_mask
  M <- sparseMatrix(i = c(1, 2), j = c(1, 3), x = c(1, 5), dims = c(4, 3))

  z <- vm("zeros")
  expect_true(z$mask_zeros); expect_equal(length(z$mask_matrix@x), 0)

  m <- vm(M, dims = c(4, 3))
  expect_false(m$mask_zeros); expect_equal(dim(m$mask_matrix), c(4, 3))

  lz <- vm(list("zeros", M), dims = c(4, 3))
  expect_true(lz$mask_zeros)
  expect_equal(unname(which(as.matrix(lz$mask_matrix) != 0, arr.ind = TRUE)),
               unname(which(as.matrix(M) != 0, arr.ind = TRUE)))

  expect_error(vm(M, dims = c(3, 4)), "must match")
  expect_error(vm("bogus"), "mask")
})

# ---- input validation ------------------------------------------------------

test_that("nmf() rejects data it cannot fit, before any C++ call", {
  set.seed(1)
  A <- matrix(runif(60, 0.5, 2), 10, 6)

  expect_error(nmf(replace(A, 3, Inf), 2), "infinite")
  expect_error(nmf(replace(A, 3, -Inf), 2), "infinite")
  expect_error(nmf(matrix(0, 5, 4), 2), "no non-zero")
  expect_error(nmf(A, 7), "cannot exceed")
  expect_error(nmf(A, 0), "positive")
  expect_error(nmf(A, "many"), "auto")

  # negative values: fine for least squares, an error for every other loss
  A_neg <- replace(A, 5, -1)
  expect_s4_class(nmf(A_neg, 2, maxit = 3, nonneg = FALSE, seed = 1), "nmf")
  for (loss in c("gp", "nb", "gamma", "inverse_gaussian", "tweedie")) {
    expect_error(nmf(A_neg, 2, loss = loss), "negative", info = loss)
  }

  # strictly positive models reject zeros unless zeros are masked
  A_zero <- replace(A, 5, 0)
  expect_error(nmf(A_zero, 2, loss = "gamma"), "zeros")
  expect_error(nmf(A_zero, 2, loss = "inverse_gaussian"), "zeros")
  expect_error(nmf(as(Matrix(A_zero, sparse = TRUE), "dgCMatrix"), 2, loss = "gamma"), "zeros")
  expect_error(nmf(A_zero, 2, loss = "tweedie", tweedie_power = 2.5), "zeros")

  # Tweedie power range
  expect_error(nmf(A, 2, loss = "tweedie", tweedie_power = 0.5), "tweedie_power")
  expect_error(nmf(A, 2, loss = "tweedie", tweedie_power = 3.5), "tweedie_power")

  # count models warn on non-integer data, and do not on counts
  expect_warning(nmf(A, 2, loss = "gp", maxit = 2, seed = 1), "non-integer")
  expect_warning(nmf(A, 2, loss = "nb", maxit = 2, seed = 1), "non-integer")
  counts <- matrix(rpois(60, 4), 10, 6)
  expect_warning(nmf(counts, 2, loss = "gp", maxit = 2, seed = 1), NA)
})

test_that("nmf() checks the shape of auxiliary matrices and penalty ranges", {
  set.seed(1)
  A <- matrix(runif(60, 0.5, 2), 10, 6)
  k <- 2

  expect_error(nmf(A, k, target_H = matrix(1, 3, 6), target_lambda = 0.1), "target_H")
  expect_error(nmf(A, k, target_H = matrix(1, k, 5), target_lambda = 0.1), "target_H")
  expect_error(nmf(A, k, w_init = matrix(1, 9, k)), "w_init")
  expect_error(nmf(A, k, h_init = matrix(1, k, 5)), "h_init")
  expect_error(nmf(A, k, h_init = matrix(c(NA, rep(1, k * 6 - 1)), k, 6)), "h_init")
  expect_error(nmf(A, k, seed = matrix(1, 10, 3)), "[Rr]ank")
  expect_error(nmf(A, k, graph_W = Diagonal(9), graph_lambda = c(1, 0)), "graph_W")
  expect_error(nmf(A, k, graph_H = Diagonal(7), graph_lambda = c(0, 1)), "graph_H")
  expect_error(nmf(A, k, mask = Matrix(0, 6, 10, sparse = TRUE)), "mask")
  expect_error(nmf(A, k, L1 = 1), "L1")
  expect_error(nmf(A, k, L1 = -0.1), "L1")
  expect_error(nmf(A, k, L2 = -1), "L2")
  expect_error(nmf(A, k, not_an_argument = 1), "Unknown parameter")
  expect_error(nmf(A, k, 1e-4, 10, 0, 0, NULL, NULL, "mse", TRUE, 0, FALSE,
                   FALSE, FALSE, "none", FALSE, 99), "named")
})

test_that("projective and symmetric NMF refuse settings they would ignore", {
  set.seed(1)
  A <- matrix(runif(100, 0.5, 2), 10, 10)
  S <- crossprod(A)

  expect_error(nmf(A, 2, projective = TRUE, L1 = 0.1), "H-side")
  expect_error(nmf(A, 2, projective = TRUE, L2 = c(0, 0.1)), "L2\\[2\\]")
  expect_error(nmf(A, 2, projective = TRUE, nonneg = c(TRUE, FALSE)), "nonneg")
  expect_error(nmf(A, 2, projective = TRUE, mask = "zeros"), "mask")
  expect_s4_class(nmf(A, 2, projective = TRUE, L1 = c(0.1, 0), maxit = 3, seed = 1), "nmf")

  expect_error(nmf(S, 2, symmetric = TRUE, loss = "gamma"), "mse")
  # H = t(W): an H-side setting of its own has no effect and is refused,
  # a single value recycled to both sides is accepted
  expect_error(nmf(S, 2, symmetric = TRUE, angular = c(0, 0.1)), "angular\\[2\\]")
  expect_error(nmf(S, 2, symmetric = TRUE, upper_bound = c(0, 1)), "upper_bound\\[2\\]")
  expect_error(nmf(S, 2, symmetric = TRUE, graph_H = Diagonal(10), graph_lambda = c(0, 1)),
               "graph_H")
  expect_error(nmf(S, 2, symmetric = TRUE, target_H = matrix(1, 2, 10), target_lambda = 0.1),
               "target_H")
  expect_s4_class(nmf(S, 2, symmetric = TRUE, L1 = 0.01, maxit = 3, seed = 1), "nmf")
  expect_error(nmf(S, 2, symmetric = TRUE, mask = "zeros"), "mask")
  expect_error(suppressWarnings(nmf(replace(S, 2, NA), 2, symmetric = TRUE)), "missing")
})

# ---- initialization --------------------------------------------------------

test_that("h_init is used: the true H reproduces an exact factorization at once", {
  sim <- ref_lowrank(m = 40, n = 30, k = 3, seed = 5)
  # Reference: with H fixed at the truth, W = A H' (H H')^-1 is non-negative,
  # so the non-negative solve for W is that least-squares solution and W H = A.
  W_ref <- sim$A %*% t(sim$H) %*% solve(tcrossprod(sim$H))
  expect_true(all(W_ref > -1e-8))

  rel <- function(fit) sqrt(sum((ref_recon(fit) - sim$A)^2) / sum(sim$A^2))
  fit_h <- nmf(sim$A, 3, h_init = sim$H, maxit = 1, tol = 1e-10, solver = "cd")
  fit_r <- nmf(sim$A, 3, seed = 7, maxit = 1, tol = 1e-10, solver = "cd")
  expect_lt(rel(fit_h), 0.02)
  expect_gt(rel(fit_r), 3 * rel(fit_h))

  # transposed orientation is accepted; sparse input too
  fit_t <- nmf(as(Matrix(sim$A, sparse = TRUE), "dgCMatrix"), 3, h_init = t(sim$H),
               maxit = 1, tol = 1e-10, solver = "cd")
  expect_lt(rel(fit_t), 0.02)

  expect_error(nmf(sim$A, 3, h_init = sim$H, seed = sim$W), "h_init")
  expect_error(nmf(sim$A, 2:3, h_init = sim$H), "single fixed rank")
  expect_error(nmf(sim$A, 3, h_init = sim$H, seed = "lanczos"), "h_init")
})

test_that("SVD-based seed strings start from the SVD, not from the R RNG", {
  skip_on_cran()
  sim <- ref_lowrank(m = 60, n = 40, k = 3, seed = 6)
  one_step_loss <- function(...) {
    fit <- nmf(sim$A, 3, maxit = 1, tol = 1e-10, solver = "cd", ...)
    sum((ref_recon(fit) - sim$A)^2) / sum(sim$A^2)
  }
  # The leading singular vectors are unique, so the start (and the loss after
  # one update) must not depend on the state of R's random number generator.
  for (s in c("lanczos", "irlba", "svd")) {
    set.seed(1);   l1 <- one_step_loss(seed = s)
    set.seed(999); l2 <- one_step_loss(seed = s)
    expect_equal(l1, l2, tolerance = 1e-3, info = s)
  }
  # ...whereas an unseeded random start does depend on it
  set.seed(1);   r1 <- nmf(sim$A, 3, maxit = 1, tol = 1e-10)
  set.seed(999); r2 <- nmf(sim$A, 3, maxit = 1, tol = 1e-10)
  expect_false(isTRUE(all.equal(r1@w, r2@w)))
  expect_error(nmf(sim$A, 3, seed = "bogus"), "Unknown seed")
})

test_that("a numeric seed does not change the caller's RNG stream", {
  sim <- ref_lowrank(m = 20, n = 15, k = 2, seed = 7)

  set.seed(123); expected <- runif(3)
  set.seed(123)
  m1 <- nmf(sim$A, 2, seed = 42, maxit = 5)
  cv <- nmf(sim$A, 2:3, seed = 42, test_fraction = 0.1, maxit = 5)
  mm <- nmf(sim$A, 2, seed = c(1, 2, 3), maxit = 5)
  expect_equal(runif(3), expected)

  # and the fit itself is reproducible regardless of the stream
  set.seed(999)
  m2 <- nmf(sim$A, 2, seed = 42, maxit = 5)
  expect_equal(m1@w, m2@w)
})

# ---- cross-validation outputs ----------------------------------------------

test_that("CV data frame has the documented columns and consistent iterations", {
  sim <- ref_lowrank(m = 40, n = 30, k = 3, seed = 8)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.05), nrow(sim$A))

  cv <- nmf(A, k = 2:4, test_fraction = 0.1, cv_seed = 1:2, maxit = 3, tol = 1e-12)
  expect_s3_class(cv, "nmfCrossValidate")
  expect_true(all(c("rep", "k", "train_mse", "test_mse", "best_iter", "total_iter") %in% names(cv)))
  expect_equal(nrow(cv), 3 * 2)
  expect_true(all(cv$best_iter >= 1))
  expect_true(all(cv$best_iter <= cv$total_iter))   # was total_iter + 1
  expect_true(all(is.finite(cv$test_mse) & cv$test_mse >= 0))
})

test_that("train_mse is NA when it is not tracked, and finite when it is", {
  sim <- ref_lowrank(m = 40, n = 30, k = 3, seed = 9)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.05), nrow(sim$A))

  off <- nmf(A, k = 2:3, test_fraction = 0.1, cv_seed = 1, maxit = 30,
             track_train_loss = FALSE)
  expect_true(all(is.na(off$train_mse)))
  expect_true(all(is.finite(off$test_mse)))

  on <- nmf(A, k = 2:3, test_fraction = 0.1, cv_seed = 1, maxit = 30,
            track_train_loss = TRUE)
  expect_true(all(is.finite(on$train_mse)))

  single <- nmf(A, k = 3, test_fraction = 0.1, cv_seed = 1, seed = 1, maxit = 30,
                track_train_loss = FALSE)
  expect_true(is.na(single@misc$train_loss))
})

test_that("best_iter is the 1-based position of the lowest test loss", {
  sim <- ref_lowrank(m = 40, n = 30, k = 3, seed = 10)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.2), nrow(sim$A))
  fit <- nmf(A, k = 3, test_fraction = 0.1, cv_seed = 1, seed = 1, maxit = 25,
             tol = 1e-12, patience = 100)
  hist <- fit@misc$test_loss_history
  skip_if(is.null(hist), "test loss history not returned")
  expect_equal(fit@misc$best_iter, which.min(hist))
  expect_equal(fit@misc$test_loss, min(hist), tolerance = 1e-6)
})

test_that("k = 'auto' honors the data size and the rest of the configuration", {
  skip_on_cran()
  sim <- ref_lowrank(m = 30, n = 12, k = 2, seed = 11)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.05), nrow(sim$A))

  fit <- nmf(A, k = "auto", seed = 1, cv_seed = 1, maxit = 20)
  expect_s4_class(fit, "nmf")
  expect_lte(ncol(fit@w), min(dim(A)))        # default range c(2, 50) is clamped
  expect_true(!is.null(fit@misc$rank_search$k_optimal))

  # arguments that the search used to drop now reach it
  expect_error(nmf(replace(A, 3, -1), k = "auto", loss = "gp"), "negative")
  A_na <- replace(A, c(4, 40, 100), NA)
  fit_na <- suppressWarnings(nmf(A_na, k = "auto", seed = 1, cv_seed = 1, maxit = 20,
                                 cv_k_range = c(1, 4)))
  expect_lte(ncol(fit_na@w), 4)
  expect_true(all(is.finite(fit_na@w)))
})

# ---- removed and deprecated API --------------------------------------------

test_that("mask_zeros and A are accepted with a deprecation warning", {
  sim <- ref_lowrank(m = 30, n = 20, k = 2, seed = 12)
  A <- sim$A
  A[sample(length(A), 100)] <- 0

  new <- nmf(A, 2, seed = 1, mask = "zeros", maxit = 10)
  expect_warning(old <- nmf(A, 2, seed = 1, mask_zeros = TRUE, maxit = 10), "deprecated")
  expect_equal(old@w, new@w)

  expect_warning(old_a <- nmf(A = A, k = 2, seed = 1, mask = "zeros", maxit = 10), "deprecated")
  expect_equal(old_a@w, new@w)
})

test_that("nnls(a, b) keeps the 0.3.7 meaning: a x = b with x >= 0", {
  set.seed(13)
  W <- matrix(runif(40 * 3), 40, 3)
  a <- crossprod(W)                       # symmetric positive definite

  # (1) non-negative exact solution: x solves a x = b
  x_true <- matrix(c(1, 2, 0.5, 0.2, 0.1, 3), 3, 2)
  b <- a %*% x_true
  expect_warning(x1 <- nnls(a, b), "deprecated")
  expect_equal(unname(x1), x_true, tolerance = 1e-4)

  # (2) active constraints: compare with the exact quadratic program
  b2 <- cbind(c(1, -2, 0.5), c(-1, 3, -0.2), c(-1, -1, -1))
  x_ref <- apply(b2, 2, function(col) ref_qp_nonneg(a, col))
  x2 <- suppressWarnings(nnls(a, b2, 1000L, 1e-12))
  expect_equal(unname(x2), x_ref, tolerance = 1e-4)
  expect_true(all(x2 >= 0))

  # (3) the old positional and named forms give the same answer
  x3 <- suppressWarnings(nnls(a, b2))
  x4 <- suppressWarnings(nnls(a = a, b = b2, cd_maxit = 1000L, cd_tol = 1e-12))
  expect_equal(unname(x3), x_ref, tolerance = 1e-3)
  expect_equal(unname(x4), x_ref, tolerance = 1e-4)

  # (4) the current API is unaffected and silent
  A <- W %*% x_true
  expect_warning(h <- nnls(w = W, A = A), NA)
  expect_equal(unname(h), x_true, tolerance = 1e-4)
})

test_that("project() and crossValidate() still work, with a deprecation warning", {
  sim <- ref_lowrank(m = 30, n = 20, k = 2, seed = 14)
  project <- getFromNamespace("project", "RcppML")
  crossValidate <- getFromNamespace("crossValidate", "RcppML")

  expect_warning(h <- project(sim$A, w = sim$W), "deprecated")
  expect_equal(unname(h), sim$H, tolerance = 1e-4)
  expect_warning(w <- project(sim$A, h = sim$H), "deprecated")
  expect_equal(unname(w), sim$W, tolerance = 1e-4)

  expect_warning(cv <- crossValidate(sim$A, k = 2:3, reps = 2, n = 0.1, maxit = 5),
                 "deprecated")
  expect_s3_class(cv, "nmfCrossValidate")
  expect_equal(nrow(cv), 4)
})

test_that("streaming = TRUE on a .spz path says once that it loads in memory", {
  skip_on_cran()
  A <- abs(rsparsematrix(40, 30, 0.3))
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  state <- RcppML:::.rcppml_state
  state$streaming_stub <- NULL
  expect_message(m1 <- nmf(f, 2, seed = 1, maxit = 5, streaming = TRUE), "loaded into memory")
  expect_message(m2 <- nmf(f, 2, seed = 1, maxit = 5, streaming = TRUE), NA)
  expect_s4_class(m1, "nmf")
})

# ---- multiple initializations ----------------------------------------------

test_that("a multi-seed fit returns the best start and the factors that belong to it", {
  sim <- ref_lowrank(m = 40, n = 30, k = 3, seed = 22)
  set.seed(23)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.3), nrow(sim$A))
  seeds <- c(11, 12, 13, 14)

  fit <- nmf(A, 3, seed = seeds, maxit = 30, tol = 1e-10)
  ai <- fit@misc$all_inits
  expect_equal(nrow(ai), length(seeds))
  expect_equal(sum(ai$selected), 1)
  sel <- which(ai$selected)
  expect_equal(sel, which.min(ai$loss))
  expect_equal(fit@misc$loss, min(ai$loss))

  # Reference: the same starts fitted one at a time, scored in plain R
  singles <- lapply(seeds, function(s) nmf(A, 3, seed = s, maxit = 30, tol = 1e-10))
  sse <- vapply(singles, function(m) sum((ref_recon(m) - A)^2), numeric(1))
  expect_equal(ai$loss / ai$loss[1], sse / sse[1], tolerance = 1e-3)
  expect_lte(sse[sel], min(sse) * (1 + 1e-3))

  # The returned factors reproduce the selected loss and are that start's fit
  expect_equal(sum((ref_recon(fit) - A)^2), sse[sel], tolerance = 1e-3)
  expect_equal(ref_recon(fit), ref_recon(singles[[sel]]), tolerance = 1e-3)
})

# ---- SVD initialization, checked against base::svd --------------------------

test_that("SVD seed strings start from |U| sqrt(d): one sweep matches plain R", {
  sim <- ref_lowrank(m = 20, n = 12, k = 2, seed = 24)
  set.seed(25)
  A <- sim$A + matrix(runif(length(sim$A), 0, 0.05), nrow(sim$A))
  k <- 2

  # Reference start and one alternating sweep (H given W0, then W given H),
  # with each non-negative least-squares problem solved exactly.
  sv <- base::svd(A, nu = k, nv = k)
  W0 <- abs(sv$u) %*% diag(sqrt(sv$d[seq_len(k)]), k)
  nn_solve <- function(X, B) {
    G <- crossprod(X)
    apply(B, 2, function(b) ref_qp_nonneg(G, as.vector(crossprod(X, b))))
  }
  H1 <- nn_solve(W0, A)
  W1 <- t(nn_solve(t(H1), t(A)))
  ref <- W1 %*% H1

  for (s in c("lanczos", "irlba", "randomized", "svd")) {
    fit <- nmf(A, k, seed = s, maxit = 1, tol = 1e-10, solver = "cd")
    expect_equal(unname(ref_recon(fit)), unname(ref), tolerance = 1e-3, info = s)
  }

  # A random start gives a different model after the same single sweep
  fit_r <- nmf(A, k, seed = 1, maxit = 1, tol = 1e-10, solver = "cd")
  expect_gt(max(abs(ref_recon(fit_r) - ref)) / max(abs(ref)), 1e-2)
})

# ---- solver = "auto" rule ---------------------------------------------------

test_that("the automatic solver rule is the documented one", {
  auto <- RcppML:::.auto_solver
  expect_identical(auto(needs_irls = TRUE,  use_gpu = FALSE, k = 5,  L1 = c(0, 0)), "cd")
  expect_identical(auto(needs_irls = FALSE, use_gpu = TRUE,  k = 5,  L1 = c(0, 0)), "cholesky")
  expect_identical(auto(needs_irls = FALSE, use_gpu = FALSE, k = 5,  L1 = c(0, 0)), "cholesky")
  expect_identical(auto(needs_irls = FALSE, use_gpu = FALSE, k = 5,  L1 = c(0.1, 0)), "cholesky")
  expect_identical(auto(needs_irls = FALSE, use_gpu = FALSE, k = 40, L1 = c(0, 0)), "cholesky")
  expect_identical(auto(needs_irls = FALSE, use_gpu = FALSE, k = "auto", L1 = c(0, 0)), "cholesky")
  # earlier rule, still selectable: Cholesky only for k < 32 and L1 == 0
  expect_identical(auto(FALSE, FALSE, 40, c(0, 0),   cholesky_all_k = FALSE), "cd")
  expect_identical(auto(FALSE, FALSE, 5,  c(0.1, 0), cholesky_all_k = FALSE), "cd")
  # active rule: Cholesky at every k for Gaussian CPU fits
  expect_identical(auto(FALSE, FALSE, 40, c(0.1, 0), cholesky_all_k = TRUE), "cholesky")
  expect_identical(auto(TRUE,  FALSE, 40, c(0, 0),   cholesky_all_k = TRUE), "cd")
  expect_identical(auto(FALSE, TRUE,  40, c(0, 0),   cholesky_all_k = TRUE), "cholesky")
})
