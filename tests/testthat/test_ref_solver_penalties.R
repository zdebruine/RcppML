# Reference tests for the NNLS solvers and penalty terms of nmf().
#
# Every expectation compares the package against the same mathematics written
# in plain R in double precision.  The fits run in single precision, so the
# tolerances are loose (1e-3 to 1e-2 relative).
#
# One ALS iteration from a known W0 is fully determined:
#   H-step:  h~_j = argmin 1/2 x'Gx - b_j'x + penalties,  G = W0'W0, b_j = W0'a_j
#            d = rowSums(|H~|),  H = H~ / d
#   W-step:  w~_i solved the same way against H, with the previous iterate
#            S = d * t(W0) as the reference point of the graph and L21 terms
# so nmf(seed = W0, maxit = 1, sort_model = FALSE) can be checked entry by entry.

# ---------------------------------------------------------------------------
# Reference solvers (double precision)
# ---------------------------------------------------------------------------

# Coordinate descent for all columns at once:
#   min 1/2 x'Gx - b'x + L1*|x|_1 + 1/2*ridge_j*|x|^2,  lower <= x <= ub
ref_cd <- function(G, B, L1 = 0, ridge = 0, nonneg = TRUE, ub = 0, sweeps = 5000) {
  k <- nrow(G)
  n <- ncol(B)
  ridge <- rep_len(ridge, n)
  X <- matrix(0, k, n)
  for (s in seq_len(sweeps)) {
    X_old <- X
    for (i in seq_len(k)) {
      cc <- B[i, ] - as.vector(crossprod(G[-i, i], X[-i, , drop = FALSE]))
      v <- if (nonneg) pmax(cc - L1, 0) else sign(cc) * pmax(abs(cc) - L1, 0)
      # a factor with zero curvature (an all-zero row of the fixed factor)
      # cannot move: keep it at zero instead of dividing by zero
      den <- G[i, i] + ridge
      v <- ifelse(den > 0, v / den, 0)
      if (ub > 0) v <- pmin(v, ub)
      X[i, ] <- v
    }
    if (max(abs(X - X_old)) <= 1e-13 * (1 + max(abs(X)))) break
  }
  X
}

# Exact NNLS for one column by enumerating supports (small k only)
ref_nnls_enum <- function(G, b) {
  k <- length(b)
  best <- rep(0, k)
  best_obj <- 0
  bits <- 2^(0:(k - 1))
  for (code in seq_len(2^k - 1)) {
    P <- which(bitwAnd(code, bits) > 0)
    x <- tryCatch(solve(G[P, P, drop = FALSE], b[P]), error = function(e) NULL)
    if (is.null(x) || any(x < 0)) next
    obj <- -0.5 * sum(b[P] * x)
    if (obj < best_obj) {
      best_obj <- obj
      best <- replace(rep(0, k), P, x)
    }
  }
  best
}

# One ALS iteration from W0 with the penalties of nmf()
ref_iter <- function(A, W0, L1 = c(0, 0), L2 = c(0, 0), ub = c(0, 0),
                     nonneg = c(TRUE, TRUE), L21_w = 0,
                     graph_W = NULL, graph_lambda_w = 0,
                     target_H = NULL, target_lambda = 0) {
  A <- as.matrix(A)
  k <- ncol(W0)
  m <- nrow(A)

  # H-step
  G <- crossprod(W0) + diag(L2[2], k)
  B <- crossprod(W0, A)
  if (!is.null(target_H) && target_lambda > 0) {
    G <- G + diag(target_lambda, k)
    B <- B + target_lambda * target_H
  }
  Ht <- ref_cd(G, B, L1[2], 0, nonneg[2], ub[2])
  d1 <- rowSums(abs(Ht)) + 1e-15
  H1 <- Ht / d1

  # W-step (columns of the k x m system are features)
  G <- tcrossprod(H1) + diag(L2[1], k)
  B <- H1 %*% t(A)
  S <- d1 * t(W0)                      # previous iterate on the scale of W~
  ridge <- rep(0, m)
  if (!is.null(graph_W) && graph_lambda_w > 0) {
    L <- as.matrix(graph_W)
    ridge <- graph_lambda_w * diag(L)
    diag(L) <- 0
    B <- B - graph_lambda_w * S %*% L  # + lambda * sum_{i != j} (-L_ij) s_i
  }
  tb <- if (nonneg[1]) pmax(B - L1[1], 0) else sign(B) * pmax(abs(B) - L1[1], 0)
  bn <- sqrt(colSums(tb^2))
  zero <- rep(FALSE, m)
  if (L21_w > 0) {
    zero <- bn <= L21_w                # exact condition for an all-zero group
    ridge <- ridge + L21_w / sqrt(colSums(S^2))
  }
  Wt <- ref_cd(G, B, L1[1], ridge, nonneg[1], ub[1])
  Wt[, zero] <- 0
  list(Ht = Ht, h = H1, Wt = Wt, bn = bn, zero = zero)
}

# One ALS iteration with missing entries (obs: TRUE = observed), plain NNLS
ref_iter_masked <- function(A, W0, obs) {
  A <- as.matrix(A)
  k <- ncol(W0)
  m <- nrow(A)
  n <- ncol(A)
  Ht <- matrix(0, k, n)
  for (j in seq_len(n)) {
    O <- obs[, j]
    Wo <- W0[O, , drop = FALSE]
    Ht[, j] <- ref_nnls_enum(crossprod(Wo), as.vector(crossprod(Wo, A[O, j])))
  }
  d1 <- rowSums(abs(Ht)) + 1e-15
  H1 <- Ht / d1
  Wt <- matrix(0, k, m)
  for (i in seq_len(m)) {
    O <- obs[i, ]
    Ho <- H1[, O, drop = FALSE]
    Wt[, i] <- ref_nnls_enum(tcrossprod(Ho), as.vector(Ho %*% A[i, O]))
  }
  list(h = H1, Wt = Wt)
}

# W~ = t(w %*% diag(d)) of a fitted model, as a k x m matrix
fit_Wt <- function(model) unname(t(model@w) * model@d)

# ---------------------------------------------------------------------------
# Test problem: well-conditioned, with many active constraints
# ---------------------------------------------------------------------------

make_problem <- function(m = 40, n = 30, k = 6, seed = 11) {
  set.seed(seed)
  Wtrue <- matrix(runif(m * k), m, k) * (runif(m * k) < 0.4)
  Htrue <- matrix(runif(k * n), k, n) * (runif(k * n) < 0.5)
  A <- Wtrue %*% Htrue + matrix(runif(m * n, 0, 0.3), m, n)
  A <- A * (runif(m * n) < 0.6)        # exact zeros, so a sparse copy differs
  W0 <- matrix(runif(m * k), m, k) * (runif(m * k) < 0.4) + 0.01
  list(A = A, As = Matrix::Matrix(A, sparse = TRUE), W0 = W0, m = m, n = n, k = k)
}

one_iter <- function(A, W0, solver, ...) {
  suppressWarnings(nmf(A, ncol(W0), seed = W0, maxit = 1, tol = 1e-10,
                       solver = solver, norm = "L1", sort_model = FALSE,
                       threads = 2, verbose = FALSE, ...))
}

tol_fit <- 5e-3

expect_matches_ref <- function(model, ref, label) {
  expect_equal(unname(as.matrix(model@h)), ref$h, tolerance = tol_fit,
               info = paste(label, "H"))
  expect_equal(fit_Wt(model), ref$Wt, tolerance = tol_fit,
               info = paste(label, "W"))
}

inputs <- function(p) list(dense = p$A, sparse = p$As)
solvers <- c("cd", "cholesky")

# ---------------------------------------------------------------------------
# Solver exactness
# ---------------------------------------------------------------------------

test_that("cd and cholesky solve each half-step exactly (plain NNLS)", {
  p <- make_problem()
  ref <- ref_iter(p$A, p$W0)
  # The reference must have active constraints, or the test says nothing
  # about the clipped solve.
  expect_gt(mean(ref$Ht == 0), 0.05)
  expect_gt(mean(ref$Wt == 0), 0.05)
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv)
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("the default solver satisfies the NNLS KKT conditions at k = 12", {
  p <- make_problem(m = 60, n = 40, k = 12, seed = 3)
  for (nm in names(inputs(p))) {
    fit <- suppressWarnings(nmf(inputs(p)[[nm]], 12, seed = p$W0, maxit = 5,
                                tol = 1e-10, threads = 2))
    H <- unname(as.matrix(fit@h))
    G <- tcrossprod(H)
    B <- H %*% t(p$A)
    X <- fit_Wt(fit)
    grad <- G %*% X - B
    scale <- max(abs(B))
    expect_true(all(X >= 0), info = nm)
    expect_lt(max(abs(grad[X > 0])), 2e-3 * scale)       # stationarity
    expect_gt(min(grad[X == 0]), -2e-3 * scale)          # dual feasibility
    # the fitted H can be poorly conditioned at this rank: looser tolerance
    expect_equal(X, ref_cd(G, B), tolerance = 2e-2, info = nm)
  }
})

test_that("a singular Gram does not break the cholesky solver", {
  p <- make_problem(k = 5)
  W0 <- p$W0
  W0[, 5] <- W0[, 4]                   # duplicate column: W0'W0 is singular
  for (nm in names(inputs(p))) {
    fit <- one_iter(inputs(p)[[nm]], W0, "cholesky")
    expect_true(all(is.finite(fit@w)) && all(is.finite(fit@h)) &&
                  all(is.finite(fit@d)), info = nm)
    # The minimizer need not be unique, the minimum is: compare objectives
    # of the W-step given the returned H.
    H <- unname(as.matrix(fit@h))
    Xref <- ref_cd(tcrossprod(H), H %*% t(p$A))
    err_fit <- sum((p$A - t(fit_Wt(fit)) %*% H)^2)
    err_ref <- sum((p$A - t(Xref) %*% H)^2)
    expect_lt(err_fit, err_ref * (1 + 1e-3))
  }
})

# ---------------------------------------------------------------------------
# L1, L2, bounds
# ---------------------------------------------------------------------------

test_that("L1 is an absolute shift and L2 a ridge, on every Gaussian path", {
  p <- make_problem()
  L1 <- c(0.05, 0.3)
  L2 <- c(0.02, 0.5)
  ref <- ref_iter(p$A, p$W0, L1 = L1, L2 = L2)
  # The penalties must change the reference materially, or the comparison
  # below could not tell a fit that ignores them from one that applies them.
  # (The number of zeros is not a valid check: the ridge removes zeros while
  # L1 adds them.)
  plain <- ref_iter(p$A, p$W0)
  expect_gt(max(abs(ref$h - plain$h)), 1e-2 * max(abs(plain$h)))
  expect_gt(max(abs(ref$Wt - plain$Wt)), 1e-2 * max(abs(plain$Wt)))
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv, L1 = L1, L2 = L2)
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("L1 on a signed factor is soft-thresholding", {
  p <- make_problem()
  L1 <- c(0.05, 0)
  nonneg <- c(FALSE, TRUE)
  ref <- ref_iter(p$A, p$W0, L1 = L1, nonneg = nonneg)
  expect_true(any(ref$Wt < 0))
  expect_gt(mean(ref$Wt == 0), 0.02)
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv, L1 = L1, nonneg = nonneg)
    expect_matches_ref(fit, ref, paste(nm, sv))
    # exact zeros where the reference has them
    expect_gt(mean(fit_Wt(fit)[ref$Wt == 0] == 0), 0.9)
  }
})

test_that("upper_bound is a box constraint inside the solve", {
  p <- make_problem()
  plain <- ref_iter(p$A, p$W0)
  ub <- c(unname(quantile(plain$Wt[plain$Wt > 0], 0.6)),
          unname(quantile(plain$Ht[plain$Ht > 0], 0.6)))
  ref <- ref_iter(p$A, p$W0, ub = ub)
  expect_true(any(ref$Wt >= ub[1] * (1 - 1e-9)))
  expect_true(any(ref$Ht >= ub[2] * (1 - 1e-9)))
  # clipping the unconstrained solution is a different answer
  expect_gt(max(abs(pmin(plain$Ht, ub[2]) - ref$Ht)), 1e-3 * ub[2])
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv, upper_bound = ub)
    expect_matches_ref(fit, ref, paste(nm, sv))
    expect_lte(max(fit_Wt(fit)), ub[1] * (1 + 1e-5))
  }
})

# ---------------------------------------------------------------------------
# L21, graph, target
# ---------------------------------------------------------------------------

test_that("L21 on W removes whole features and matches the reference step", {
  p <- make_problem()
  bn <- sort(ref_iter(p$A, p$W0)$bn)
  lambda <- mean(bn[12:13])            # 12 features below the threshold
  ref <- ref_iter(p$A, p$W0, L21_w = lambda)
  expect_equal(sum(ref$zero), 12)
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv, L21 = c(lambda, 0))
    zero_fit <- rowSums(abs(fit@w)) == 0
    expect_identical(unname(zero_fit), ref$zero, info = paste(nm, sv))
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("L21 keeps features at exactly zero over many iterations", {
  p <- make_problem()
  bn <- sort(ref_iter(p$A, p$W0)$bn)
  lambda <- mean(bn[12:13])
  for (nm in names(inputs(p))) {
    fit <- suppressWarnings(nmf(inputs(p)[[nm]], p$k, seed = p$W0, maxit = 20,
                                tol = 1e-10, L21 = c(lambda, 0), threads = 2))
    # Exact optimality condition of the last W-step, given the returned H
    H <- unname(as.matrix(fit@h))
    bn_last <- sqrt(colSums(pmax(H %*% t(p$A), 0)^2))
    clear <- abs(bn_last - lambda) > 1e-3 * lambda
    zero_fit <- unname(rowSums(abs(fit@w)) == 0)
    expect_identical(zero_fit[clear], (bn_last <= lambda)[clear], info = nm)
    expect_true(any(zero_fit) && !all(zero_fit), info = nm)
  }
})

test_that("graph_W applies the per-feature Laplacian coupling", {
  p <- make_problem()
  m <- p$m
  # ring graph over features: L = D - A
  adj <- matrix(0, m, m)
  idx <- cbind(seq_len(m), c(2:m, 1))
  adj[idx] <- 1
  adj <- adj + t(adj)
  L <- diag(rowSums(adj)) - adj
  Ls <- Matrix::Matrix(L, sparse = TRUE)
  Ls <- methods::as(methods::as(Ls, "generalMatrix"), "CsparseMatrix")
  lambda <- 0.02
  ref <- ref_iter(p$A, p$W0, graph_W = L, graph_lambda_w = lambda)
  plain <- ref_iter(p$A, p$W0)
  expect_gt(max(abs(ref$Wt - plain$Wt)), 1e-2 * max(abs(plain$Wt)))
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv,
                    graph_W = Ls, graph_lambda = c(lambda, 0))
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("a symmetric adjacency matrix is used as its Laplacian D - A", {
  p <- make_problem()
  m <- p$m
  set.seed(31)
  adj <- matrix(runif(m * m) * (runif(m * m) < 0.1), m, m)
  adj <- adj + t(adj)
  diag(adj) <- 0
  L <- diag(rowSums(adj)) - adj
  adj_s <- methods::as(methods::as(Matrix::Matrix(adj, sparse = TRUE),
                                   "generalMatrix"), "CsparseMatrix")
  lambda <- 0.05
  ref <- ref_iter(p$A, p$W0, graph_W = L, graph_lambda_w = lambda)
  for (nm in names(inputs(p))) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, "cd",
                    graph_W = adj_s, graph_lambda = c(lambda, 0))
    expect_matches_ref(fit, ref, paste(nm, "adjacency"))
  }
})

test_that("projective NMF with graph_W stays finite and smooths W", {
  p <- make_problem()
  m <- p$m
  adj <- matrix(0, m, m)
  adj[cbind(seq_len(m), c(2:m, 1))] <- 1
  adj <- adj + t(adj)
  L <- diag(rowSums(adj)) - adj
  Ls <- methods::as(methods::as(Matrix::Matrix(L, sparse = TRUE),
                                "generalMatrix"), "CsparseMatrix")
  energy <- function(fit) {
    Wt <- fit_Wt(fit)                    # k x m
    sum((Wt %*% L) * Wt) / sum(Wt^2)     # Laplacian energy per unit norm
  }
  fits <- lapply(c(0, 1), function(lam) {
    suppressWarnings(nmf(p$As, p$k, seed = p$W0, maxit = 15, tol = 1e-10,
                         projective = TRUE, threads = 2,
                         graph_W = Ls, graph_lambda = c(lam, 0)))
  })
  for (fit in fits) {
    expect_true(all(is.finite(fit@w)) && all(is.finite(fit@h)) &&
                  all(is.finite(fit@d)))
  }
  expect_lt(energy(fits[[2]]), energy(fits[[1]]))
})

test_that("a large angular step does not re-correlate a signed factor", {
  p <- make_problem()
  mean_abs_cor <- function(fit) {
    w <- sweep(fit@w, 2, sqrt(colSums(fit@w^2)), "/")
    cc <- abs(crossprod(w))              # |cosine| between W columns
    mean(cc[upper.tri(cc)])
  }
  fit_at <- function(lam) {
    suppressWarnings(nmf(p$A, p$k, seed = p$W0, maxit = 10, tol = 1e-10,
                         nonneg = c(FALSE, FALSE), angular = c(lam, 0),
                         solver = "cd", threads = 2))
  }
  base <- mean_abs_cor(fit_at(0))
  # (I - lambda * C0) reverses a shared component once lambda * mu > 1; the
  # step is capped so that a large lambda can only decorrelate.
  expect_lt(mean_abs_cor(fit_at(2)), base)
  expect_lt(mean_abs_cor(fit_at(20)), base)
})

test_that("a non-symmetric graph is rejected", {
  p <- make_problem()
  L <- diag(2, p$m)
  L[1, 2] <- -1                        # not symmetric
  Ls <- methods::as(methods::as(Matrix::Matrix(L, sparse = TRUE),
                                "generalMatrix"), "CsparseMatrix")
  expect_error(one_iter(p$A, p$W0, "cd", graph_W = Ls,
                        graph_lambda = c(0.1, 0)), "symmetric")
})

test_that("target_H adds lambda/2 * |h - t|^2 on dense and sparse input", {
  p <- make_problem()
  set.seed(5)
  target <- matrix(runif(p$k * p$n), p$k, p$n)
  lambda <- 2
  ref <- ref_iter(p$A, p$W0, target_H = target, target_lambda = lambda)
  plain <- ref_iter(p$A, p$W0)
  expect_gt(max(abs(ref$h - plain$h)), 1e-2 * max(abs(plain$h)))
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv,
                    target_H = target, target_lambda = c(0, lambda))
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("target_H with the wrong dimensions is an error", {
  p <- make_problem()
  bad <- matrix(0.1, p$k, p$n + 1)
  expect_error(one_iter(p$A, p$W0, "cd", target_H = bad,
                        target_lambda = c(0, 1)))
})

test_that("a strong target dominates under a robust (IRLS) loss too", {
  p <- make_problem()
  set.seed(6)
  target <- matrix(runif(p$k * p$n) + 0.1, p$k, p$n)
  fit <- suppressWarnings(nmf(p$A, p$k, seed = p$W0, maxit = 1, tol = 1e-10,
                              robust = TRUE, norm = "L1", sort_model = FALSE,
                              threads = 2, target_H = target,
                              target_lambda = c(0, 1e4)))
  # (G_w + lambda I) h = b_w + lambda t  ->  h -> t as lambda grows
  expect_equal(unname(as.matrix(fit@h)), target / rowSums(target),
               tolerance = 1e-2)
})

# ---------------------------------------------------------------------------
# Angular
# ---------------------------------------------------------------------------

test_that("angular does not clip a signed factor to non-negative", {
  p <- make_problem()
  nonneg <- c(FALSE, TRUE)
  expect_true(any(ref_iter(p$A, p$W0, nonneg = nonneg)$Wt < 0))
  for (nm in names(inputs(p))) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, "cd", nonneg = nonneg,
                    angular = c(0.05, 0))
    expect_true(any(fit@w < 0), info = nm)
  }
})

# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

test_that("a mask matrix excludes entries from both half-steps", {
  p <- make_problem(k = 5)
  set.seed(8)
  M <- matrix(runif(p$m * p$n) < 0.1, p$m, p$n)
  Ms <- methods::as(methods::as(Matrix::Matrix(M * 1, sparse = TRUE),
                                "generalMatrix"), "CsparseMatrix")
  ref <- ref_iter_masked(p$A, p$W0, !M)
  unmasked <- ref_iter(p$A, p$W0)
  expect_gt(max(abs(ref$Wt - unmasked$Wt)), 1e-2 * max(abs(unmasked$Wt)))
  for (nm in names(inputs(p))) for (sv in solvers) {
    fit <- one_iter(inputs(p)[[nm]], p$W0, sv, mask = Ms)
    expect_matches_ref(fit, ref, paste(nm, sv))
  }
})

test_that("mask = 'zeros' fits the non-zero entries only (sparse input)", {
  p <- make_problem(k = 5)
  ref <- ref_iter_masked(p$A, p$W0, p$A != 0)
  unmasked <- ref_iter(p$A, p$W0)
  expect_gt(max(abs(ref$Wt - unmasked$Wt)), 1e-2 * max(abs(unmasked$Wt)))
  for (sv in solvers) {
    fit <- one_iter(p$As, p$W0, sv, mask = "zeros")
    expect_matches_ref(fit, ref, paste("sparse", sv))
  }
})

test_that("mask = 'zeros' fits the non-zero entries only (dense input)", {
  p <- make_problem(k = 5)
  ref <- ref_iter_masked(p$A, p$W0, p$A != 0)
  for (sv in solvers) {
    fit <- one_iter(p$A, p$W0, sv, mask = "zeros")
    expect_matches_ref(fit, ref, paste("dense", sv))
  }
})

test_that("a mask with a non-Gaussian loss is refused, not ignored", {
  p <- make_problem(k = 5)
  counts <- round(p$A * 10)
  M <- matrix(0, p$m, p$n)
  M[1:5, 1:5] <- 1
  Ms <- methods::as(methods::as(Matrix::Matrix(M, sparse = TRUE),
                                "generalMatrix"), "CsparseMatrix")
  expect_error(
    suppressWarnings(nmf(counts, p$k, seed = p$W0, maxit = 2, loss = "gp",
                         mask = Ms, threads = 2)),
    "mask")
})

test_that("L21 with a non-Gaussian loss is refused, not ignored", {
  p <- make_problem(k = 5)
  counts <- round(p$A * 10)
  expect_error(
    suppressWarnings(nmf(counts, p$k, seed = p$W0, maxit = 2, loss = "gp",
                         L21 = c(0.1, 0), threads = 2)),
    "L21")
})

# ---------------------------------------------------------------------------
# Symmetric NMF
# ---------------------------------------------------------------------------

test_that("symmetric NMF reports the loss of the returned model", {
  set.seed(21)
  X <- matrix(runif(30 * 4), 30, 4) * (runif(30 * 4) < 0.5)
  S <- tcrossprod(X) + 0.05
  W0 <- matrix(runif(30 * 4), 30, 4)
  fit <- suppressWarnings(nmf(S, 4, seed = W0, symmetric = TRUE, maxit = 6,
                              tol = 1e-10, solver = "cd", threads = 2))
  recon <- fit@w %*% (fit@d * t(fit@w))
  err <- sum((S - recon)^2)
  loss <- fit@misc$loss
  # The loss is reported either as a total or as a mean over entries.
  rel <- min(abs(loss / err - 1), abs(loss * length(S) / err - 1))
  expect_lt(rel, 1e-2)
})

# ---------------------------------------------------------------------------
# Reported loss
# ---------------------------------------------------------------------------

# Non-negative data with mean near 1: |A|^2 is far larger than the loss, so
# the Gram-trick loss (|A|^2 - 2 cross + recon) is sensitive to how its terms
# are accumulated.
make_loss_problem <- function(m, n, k, seed) {
  set.seed(seed)
  W <- matrix(runif(m * k), m, k)
  H <- matrix(runif(k * n), k, n)
  A <- (W %*% H) * (4 / k) + matrix(runif(m * n), m, n)
  A <- A * (runif(m * n) < 0.4)
  list(A = A, As = Matrix::Matrix(A, sparse = TRUE),
       W0 = matrix(runif(m * k), m, k), k = k)
}

explicit_loss <- function(fit, A) {
  sum((as.matrix(A) - fit@w %*% (fit@d * fit@h))^2)
}

test_that("misc$loss is the explicit squared error on sparse input", {
  for (case in list(list(m = 300, n = 200, k = 5), list(m = 600, n = 400, k = 12))) {
    p <- make_loss_problem(case$m, case$n, case$k, seed = 40 + case$k)
    run <- function(...) {
      suppressWarnings(nmf(p$As, p$k, seed = p$W0, maxit = 20, tol = 1e-10,
                           threads = 2, ...))
    }
    plain <- run(solver = "cd")
    ub_w <- 0.5 * max(t(plain@w) * plain@d)
    configs <- list(
      cd          = list(solver = "cd"),
      cholesky    = list(solver = "cholesky"),
      L1          = list(solver = "cholesky", L1 = c(0.05, 0.05)),
      L1_cd       = list(solver = "cd", L1 = c(0.05, 0.05)),
      L2          = list(solver = "cholesky", L2 = c(0.01, 0.01)),
      upper_bound = list(solver = "cholesky", upper_bound = c(ub_w, 0)),
      signed_W    = list(solver = "cholesky", nonneg = c(FALSE, TRUE)),
      signed_L1   = list(solver = "cd", nonneg = c(FALSE, TRUE), L1 = c(0.05, 0)),
      projective  = list(solver = "cholesky", projective = TRUE)
    )
    for (nm in names(configs)) {
      fit <- do.call(run, configs[[nm]])
      expect_equal(fit@misc$loss, explicit_loss(fit, p$A), tolerance = 1e-4,
                   info = paste("k =", p$k, nm))
    }
    # angular changes W after the solve, so the loss uses the separate
    # single-precision pass: same quantity, looser tolerance
    fit <- run(solver = "cd", angular = c(0.05, 0))
    expect_equal(fit@misc$loss, explicit_loss(fit, p$A), tolerance = 1e-3,
                 info = paste("k =", p$k, "angular"))
  }
})

test_that("misc$loss is the explicit squared error on dense input", {
  p <- make_loss_problem(300, 200, 5, seed = 45)
  for (sv in solvers) {
    fit <- suppressWarnings(nmf(p$A, p$k, seed = p$W0, maxit = 20, tol = 1e-10,
                                solver = sv, threads = 2))
    expect_equal(fit@misc$loss, explicit_loss(fit, p$A), tolerance = 1e-4,
                 info = sv)
  }
})

# ---------------------------------------------------------------------------
# Parallel sparse setup
# ---------------------------------------------------------------------------

test_that("parallel cast and transpose equal Eigen's serial result exactly", {
  hook <- get0("Rcpp_test_sparse_setup", envir = asNamespace("RcppML"),
               inherits = FALSE)
  skip_if(is.null(hook), "test hook Rcpp_test_sparse_setup not built")
  set.seed(51)
  # Skewed rows: a few dense rows, many nearly empty ones, some empty rows
  # and columns
  m <- 200
  n <- 150
  dens <- c(rep(0.9, 5), rep(0.02, m - 25), rep(0, 20))
  A <- matrix(0, m, n)
  for (r in seq_len(m)) {
    nz <- runif(n) < dens[r]
    A[r, nz] <- rnorm(sum(nz))
  }
  A[, c(3, 77, n)] <- 0
  As <- Matrix::Matrix(A, sparse = TRUE)
  for (threads in 2L) {
    res <- hook(As, threads)
    expect_true(res$cast, info = paste("threads", threads))
    expect_true(res$transpose, info = paste("threads", threads))
    expect_true(res$transpose_from_double, info = paste("threads", threads))
  }
  # degenerate shapes
  tiny <- Matrix::Matrix(matrix(c(0, 1.5, 0, 0, 0, -2), 2, 3), sparse = TRUE)
  res <- hook(tiny, 2L)
  expect_true(res$cast && res$transpose && res$transpose_from_double)
})

# ---------------------------------------------------------------------------
# Nearly exact low-rank data: ill-conditioned Gram, loss far below |A|^2
# ---------------------------------------------------------------------------

# Exact NNLS for all columns at once by enumerating supports (small k only)
ref_nnls_enum_batch <- function(G, B) {
  k <- nrow(G)
  n <- ncol(B)
  X <- matrix(0, k, n)
  best <- rep(0, n)
  bits <- 2^(0:(k - 1))
  for (code in seq_len(2^k - 1)) {
    P <- which(bitwAnd(code, bits) > 0)
    BP <- B[P, , drop = FALSE]
    Z <- tryCatch(solve(G[P, P, drop = FALSE], BP, tol = 1e-30),
                  error = function(e) NULL)
    if (is.null(Z)) next
    feasible <- colSums(Z < 0) == 0
    obj <- -0.5 * colSums(BP * Z)
    upd <- feasible & (obj < best)
    if (any(upd)) {
      X[, upd] <- 0
      X[P, upd] <- Z[, upd]
      best[upd] <- obj[upd]
    }
  }
  X
}

test_that("one iteration on nearly exact low-rank data matches exact ALS", {
  set.seed(77)
  m <- 300
  n <- 1000
  k <- 4
  A <- abs(matrix(rnorm(m * k), m, k)) %*% abs(matrix(rnorm(k * n), k, n))
  A <- A + matrix(abs(rnorm(m * n)), m, n) * (runif(m * n) < 0.01)
  As <- Matrix::Matrix(A, sparse = TRUE)
  W0 <- matrix(runif(m * k), m, k)

  # Exact ALS step in double precision
  Ht <- ref_nnls_enum_batch(crossprod(W0), crossprod(W0, A))
  d1 <- rowSums(Ht) + 1e-15
  H1 <- Ht / d1
  Wt <- ref_nnls_enum_batch(tcrossprod(H1), H1 %*% t(A))
  loss_ref <- sum((A - t(Wt) %*% H1)^2)
  # after one step from a random start the loss is still a few percent of
  # |A|^2; what makes this matrix hard is the ill-conditioned Gram of H
  # (condition number of order 1e3 to 1e4), not a tiny loss
  expect_gt(kappa(tcrossprod(H1), exact = TRUE), 1e3)

  for (nm in c("dense", "sparse")) {
    X <- if (nm == "dense") A else As
    # Default solver: Gram and solve in double, so the single-precision
    # factors are the only limit
    fit <- one_iter(X, W0, "cholesky")
    expect_lt(abs(explicit_loss(fit, A) / loss_ref - 1), 5e-3)
    expect_lt(abs(fit@misc$loss / loss_ref - 1), 5e-3)
    # Coordinate descent: a column that hits the sweep cap is finished by
    # the exact active-set solve
    fit_cd <- one_iter(X, W0, "cd")
    expect_lt(abs(explicit_loss(fit_cd, A) / loss_ref - 1), 5e-3)

    # Last half-step on its own: excess loss of the returned W over the
    # exact W given the returned H
    for (f in list(fit, fit_cd)) {
      H <- unname(as.matrix(f@h))
      W_opt <- ref_nnls_enum_batch(tcrossprod(H), H %*% t(A))
      gap <- explicit_loss(f, A) / sum((A - t(W_opt) %*% H)^2) - 1
      expect_lt(gap, 2e-3)
      expect_gt(gap, -1e-3)
    }
  }
})

test_that("coordinate descent passes a KKT test on ordinary noisy data", {
  set.seed(78)
  m <- 300
  n <- 1000
  k <- 4
  A <- abs(matrix(rnorm(m * k), m, k)) %*% abs(matrix(rnorm(k * n), k, n))
  A <- A + matrix(abs(rnorm(m * n)), m, n) * (runif(m * n) < 0.01)
  A <- pmax(A + matrix(rnorm(m * n, 0, 0.5), m, n), 0)
  W0 <- matrix(runif(m * k), m, k)
  for (sv in solvers) {
    fit <- one_iter(A, W0, sv)
    H <- unname(as.matrix(fit@h))
    W_opt <- ref_nnls_enum_batch(tcrossprod(H), H %*% t(A))
    gap <- explicit_loss(fit, A) / sum((A - t(W_opt) %*% H)^2) - 1
    expect_lt(abs(gap), 1e-6)
  }
})
