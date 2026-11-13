# Reference-based tests for clustering, factor networks and utilities.
#
# Every expectation is checked against math written out in this file
# (brute force or a closed form), not against another code path of the
# package. Problems are small and run on two threads.

options(RcppML.threads = 2)
options(RcppML.verbose = FALSE)

# ---------------------------------------------------------------------------
# helpers (reference math)
# ---------------------------------------------------------------------------

# Exact 2-variable NNLS by enumerating the faces of the non-negative quadrant
ref_nnls2 <- function(G, b, nonneg = TRUE) {
  if (!nonneg) return(as.vector(solve(G, b)))
  cands <- list(c(0, 0))
  if (G[1, 1] > 0) cands <- c(cands, list(c(max(0, b[1] / G[1, 1]), 0)))
  if (G[2, 2] > 0) cands <- c(cands, list(c(0, max(0, b[2] / G[2, 2]))))
  x <- tryCatch(as.vector(solve(G, b)), error = function(e) NULL)
  if (!is.null(x) && all(x >= 0)) cands <- c(cands, list(x))
  obj <- vapply(cands, function(z) 0.5 * sum(z * (G %*% z)) - sum(b * z), numeric(1))
  cands[[which.min(obj)]]
}

# Rank-2 alternating least squares exactly as bipartition() defines it:
# update h, scale rows to unit L1 norm, update w, scale rows to unit L1 norm
ref_bipartition_v <- function(A, w, maxit, nonneg = TRUE) {
  A <- as.matrix(A)
  h <- NULL
  d <- c(1, 1)
  for (it in seq_len(maxit)) {
    G <- tcrossprod(w)
    B <- w %*% A
    h <- apply(B, 2, function(b) ref_nnls2(G, b, nonneg))
    h <- h / (rowSums(abs(h)) + 1e-15)
    G <- tcrossprod(h)
    B <- h %*% t(A)
    w <- apply(B, 2, function(b) ref_nnls2(G, b, nonneg))
    d <- rowSums(abs(w)) + 1e-15
    w <- w / d
  }
  if (d[1] > d[2]) h[1, ] - h[2, ] else h[2, ] - h[1, ]
}

# mean over samples of cos(x, own center) - cos(x, other center)
ref_rel_cosine <- function(A, s1, s2) {
  A <- as.matrix(A)
  c1 <- rowMeans(A[, s1, drop = FALSE])
  c2 <- rowMeans(A[, s2, drop = FALSE])
  cosv <- function(x, y) {
    den <- sqrt(sum(x^2) * sum(y^2))
    if (den > 0) sum(x * y) / den else 0
  }
  t1 <- vapply(s1, function(j) cosv(A[, j], c1) - cosv(A[, j], c2), numeric(1))
  t2 <- vapply(s2, function(j) cosv(A[, j], c2) - cosv(A[, j], c1), numeric(1))
  mean(c(t1, t2))
}

# block-structured non-negative data: `sizes[g]` samples of cluster g
make_clusters <- function(sizes, feat_per = 10, hi = 2, lo = 0.2, jitter = 0.2, seed = 1) {
  set.seed(seed)
  g <- length(sizes)
  m <- feat_per * g
  lab <- rep(seq_len(g), sizes)
  A <- matrix(0, m, length(lab))
  for (j in seq_along(lab)) {
    mu <- rep(lo, m)
    mu[((lab[j] - 1) * feat_per + 1):(lab[j] * feat_per)] <- hi
    A[, j] <- mu * runif(m, 1 - jitter, 1 + jitter)
  }
  perm <- sample(length(lab))
  list(A = A[, perm], labels = lab[perm])
}

make_model <- function(m, n, k, seed = 1, d = NULL, signed_h = FALSE) {
  set.seed(seed)
  w <- matrix(runif(m * k), m, k)
  w <- sweep(w, 2, colSums(w), "/")
  h <- matrix(runif(k * n), k, n)
  if (signed_h) h <- h - 0.5
  if (is.null(d)) d <- seq(k + 1, 2)[seq_len(k)] * 10
  new("nmf", w = w, d = as.numeric(d), h = h, misc = list())
}

# ---------------------------------------------------------------------------
# rank-2 solver and bipartition
# ---------------------------------------------------------------------------

test_that("bipartition matches exact rank-2 NNLS alternating least squares", {
  sim <- make_clusters(c(22, 18), feat_per = 15, jitter = 0.5, seed = 3)
  A <- sim$A
  for (seed in c(1L, 7L)) {
    w0 <- t(RcppML:::c_rmatrix(nrow(A), 2L, seed))   # bipartition()'s initial w
    for (maxit in 1:3) {
      v_ref <- ref_bipartition_v(A, w0, maxit)
      bp <- bipartition(A, tol = 1e-12, maxit = maxit, seed = seed, threads = 2)
      expect_equal(bp$v, v_ref, tolerance = 1e-6)
      bps <- bipartition(Matrix::Matrix(A, sparse = TRUE), tol = 1e-12, maxit = maxit,
                         seed = seed, threads = 2)
      expect_equal(bps$v, v_ref, tolerance = 1e-6)
    }
  }
})

test_that("bipartition with nonneg = FALSE matches unconstrained rank-2 ALS", {
  set.seed(11)
  A <- matrix(rnorm(20 * 25), 20, 25)
  A[1:10, 1:12] <- A[1:10, 1:12] + 3
  w0 <- t(RcppML:::c_rmatrix(nrow(A), 2L, 5L))
  v_ref <- ref_bipartition_v(A, w0, 2, nonneg = FALSE)
  bp <- bipartition(A, tol = 1e-12, maxit = 2, seed = 5L, nonneg = FALSE, threads = 2)
  expect_true(all(is.finite(bp$v)))
  expect_equal(bp$v, v_ref, tolerance = 1e-6)
})

test_that("bipartition returns 0-based indices covering every sample once", {
  sim <- make_clusters(c(20, 15), seed = 2)
  bp <- bipartition(sim$A, seed = 1, threads = 2)
  expect_equal(sort(c(bp$samples1, bp$samples2)), 0:(ncol(sim$A) - 1))
  expect_equal(bp$size1, length(bp$samples1))
  expect_equal(bp$size2, length(bp$samples2))
  # the two clean clusters are separated exactly
  tab <- table(sim$labels[bp$samples1 + 1])
  expect_equal(length(tab), 1L)
  expect_equal(length(table(sim$labels[bp$samples2 + 1])), 1L)
})

test_that("bipartition dist is the mean relative cosine distance", {
  sim <- make_clusters(c(20, 15), jitter = 0.4, seed = 4)
  A <- sim$A
  bp <- bipartition(A, seed = 1, threads = 2)
  ref <- ref_rel_cosine(A, bp$samples1 + 1, bp$samples2 + 1)
  expect_equal(bp$dist, ref, tolerance = 1e-8)
  expect_gt(bp$dist, 0.2)

  # independent of the number of features and of the number of samples
  bp_rows <- bipartition(rbind(A, A), seed = 1, threads = 2)
  expect_equal(bp_rows$dist,
               ref_rel_cosine(rbind(A, A), bp_rows$samples1 + 1, bp_rows$samples2 + 1),
               tolerance = 1e-8)
  expect_equal(bp_rows$dist, bp$dist, tolerance = 1e-6)
  bp_cols <- bipartition(cbind(A, A), seed = 1, threads = 2)
  expect_equal(bp_cols$dist, bp$dist, tolerance = 1e-6)
  # independent of the scale of the data
  expect_equal(bipartition(1000 * A, seed = 1, threads = 2)$dist, bp$dist, tolerance = 1e-6)
})

test_that("bipartition dist is ~1 for disjoint clusters and ~0 for a homogeneous set", {
  # two clusters with disjoint features and identical samples within a cluster
  A <- cbind(matrix(rep(c(1, 2, 3, 0, 0, 0), 8), 6, 8),
             matrix(rep(c(0, 0, 0, 3, 1, 2), 6), 6, 6))
  bp <- bipartition(A, seed = 1, threads = 2)
  expect_equal(sort(c(bp$size1, bp$size2)), c(6, 8))
  expect_equal(bp$dist, 1, tolerance = 1e-8)

  # one homogeneous cluster: any split is arbitrary
  set.seed(5)
  mu <- runif(30, 1, 3)
  H <- mu * matrix(runif(30 * 40, 0.9, 1.1), 30, 40)
  bph <- bipartition(H, seed = 1, threads = 2)
  expect_true(is.finite(bph$dist))
  expect_lt(abs(bph$dist), 0.02)
  if (bph$size1 > 0 && bph$size2 > 0)
    expect_equal(bph$dist, ref_rel_cosine(H, bph$samples1 + 1, bph$samples2 + 1),
                 tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
# dclust
# ---------------------------------------------------------------------------

test_that("dclust recovers four clean clusters and assigns every sample once", {
  sim <- make_clusters(c(60, 50, 40, 30), feat_per = 10, jitter = 0.2, seed = 6)
  A <- Matrix::Matrix(sim$A, sparse = TRUE)
  cl <- dclust(A, min_samples = 10, min_dist = 0.05, seed = 1, threads = 2)

  all_samples <- unlist(lapply(cl, function(x) x$samples))
  expect_equal(sort(all_samples), 0:(ncol(A) - 1))    # 0-based, no sample lost or repeated
  expect_equal(length(cl), 4L)

  assigned <- integer(ncol(A))
  for (i in seq_along(cl)) assigned[cl[[i]]$samples + 1] <- i
  tab <- table(sim$labels, assigned)
  expect_true(all(rowSums(tab > 0) == 1))             # each true cluster in one leaf
  expect_true(all(colSums(tab > 0) == 1))             # each leaf holds one true cluster

  # centers are the feature means of the member samples
  for (x in cl)
    expect_equal(x$center, rowMeans(sim$A[, x$samples + 1, drop = FALSE]), tolerance = 1e-10)
})

test_that("dclust separates four clusters under heavy multiplicative noise", {
  # every sample is its cluster center times independent exponential noise
  set.seed(51)
  m <- 300
  cent <- matrix(rexp(m * 4) * (runif(m * 4) < 0.3), m, 4)
  lab <- rep(1:4, c(60, 50, 40, 30))
  A <- cent[, lab] * matrix(rexp(m * 180, 1), m)
  A <- A + 0.05 * matrix(rexp(m * 180), m)

  # Reference values of the distance (formula written out in ref_rel_cosine):
  # a split between two true clusters scores far above a random split of one
  between <- vapply(list(c(1, 2), c(1, 3), c(1, 4), c(2, 3), c(2, 4), c(3, 4)), function(p) {
    idx <- which(lab %in% p)
    ref_rel_cosine(A[, idx], which(lab[idx] == p[1]), which(lab[idx] == p[2]))
  }, numeric(1))
  expect_gt(min(between), 0.4)

  # with this noise a split of a homogeneous cluster reaches 0.1-0.2, so the
  # threshold has to sit between that and the between-cluster values
  cl <- dclust(Matrix::Matrix(A, sparse = TRUE), min_samples = 10, min_dist = 0.3,
               seed = 1, threads = 2)
  expect_equal(sort(unlist(lapply(cl, function(x) x$samples))), 0:179)
  expect_equal(length(cl), 4L)
  assigned <- integer(180)
  for (i in seq_along(cl)) assigned[cl[[i]]$samples + 1] <- i
  tab <- table(lab, assigned)
  expect_true(all(rowSums(tab > 0) == 1) && all(colSums(tab > 0) == 1))
})

test_that("mse() is deprecated in favor of evaluate()", {
  m <- make_model(10, 8, 2)
  A <- m@w %*% diag(m@d) %*% m@h
  expect_warning(val <- RcppML:::mse(m@w, m@d, m@h, A), "deprecated")
  expect_equal(val, 0, tolerance = 1e-12)
})

test_that("dclust min_dist rejects splits of a homogeneous cluster, not good ones", {
  sim <- make_clusters(c(30, 30), jitter = 0.2, seed = 8)
  A <- Matrix::Matrix(sim$A, sparse = TRUE)
  # a threshold above any attainable separation keeps one cluster
  expect_equal(length(dclust(A, min_samples = 5, min_dist = 0.99, seed = 1, threads = 2)), 1L)
  # a moderate threshold accepts the real split and rejects the noise splits
  expect_equal(length(dclust(A, min_samples = 5, min_dist = 0.05, seed = 1, threads = 2)), 2L)
})

test_that("dclust guards min_samples, min_dist and all-zero columns", {
  sim <- make_clusters(c(20, 20), seed = 9)
  A <- Matrix::Matrix(cbind(sim$A, matrix(0, nrow(sim$A), 4)), sparse = TRUE)
  expect_error(dclust(A, min_samples = 0), "min_samples")
  expect_error(dclust(A, min_samples = 5, min_dist = -1), "min_dist")
  expect_error(dclust(A, min_samples = 5, min_dist = NA_real_), "min_dist")

  cl <- dclust(A, min_samples = 5, min_dist = 0.05, seed = 1, threads = 2)
  all_samples <- unlist(lapply(cl, function(x) x$samples))
  expect_equal(sort(all_samples), 0:(ncol(A) - 1))
  expect_true(all(vapply(cl, function(x) all(is.finite(x$center)), logical(1))))
  expect_true(all(vapply(cl, function(x) x$size >= 5, logical(1))))
})

# ---------------------------------------------------------------------------
# cosine, bipartiteMatch, AUC
# ---------------------------------------------------------------------------

test_that("cosine agrees with the definition in all five call forms", {
  set.seed(1)
  X <- matrix(runif(7 * 4), 7, 4)
  Y <- matrix(runif(7 * 3), 7, 3)
  x <- runif(7)
  y <- runif(7)
  cosv <- function(a, b) sum(a * b) / sqrt(sum(a^2) * sum(b^2))
  ref_mm <- outer(seq_len(ncol(X)), seq_len(ncol(Y)),
                  Vectorize(function(i, j) cosv(X[, i], Y[, j])))
  expect_equal(unname(cosine(X, Y)), ref_mm, tolerance = 1e-10)
  expect_equal(unname(cosine(X)),
               outer(1:4, 1:4, Vectorize(function(i, j) cosv(X[, i], X[, j]))),
               tolerance = 1e-10)
  expect_equal(cosine(X, y), apply(X, 2, cosv, b = y), tolerance = 1e-10)
  expect_equal(cosine(x, Y), apply(Y, 2, cosv, b = x), tolerance = 1e-10)   # was wrong
  expect_equal(cosine(x, y), cosv(x, y), tolerance = 1e-10)
})

test_that("bipartiteMatch is optimal against brute force, also with negative costs", {
  perms <- function(n) {
    if (n == 1) return(matrix(1L, 1, 1))
    p <- perms(n - 1)
    do.call(rbind, lapply(seq_len(n), function(i) cbind(i, ifelse(p >= i, p + 1L, p))))
  }
  P <- perms(4)
  set.seed(3)
  for (shift in c(0, -0.7, -5)) {
    x <- matrix(runif(16), 4, 4) + shift
    best <- min(apply(P, 1, function(p) sum(x[cbind(1:4, p)])))
    res <- bipartiteMatch(x)
    expect_equal(sort(res$assignment), 0:3)
    expect_equal(sum(x[cbind(1:4, res$assignment + 1)]), best, tolerance = 1e-10)
    expect_equal(res$cost, best, tolerance = 1e-10)
  }
})

test_that("AUC counts ties as one half and matches the pairwise definition", {
  auc <- RcppML:::.compute_auc
  expect_equal(auc(c(1, 0), c(0.5, 0.5)), 0.5)
  expect_equal(auc(c(0, 1), c(0.5, 0.5)), 0.5)
  expect_equal(auc(c(1, 1, 0, 0), c(0.9, 0.8, 0.2, 0.1)), 1)
  expect_equal(auc(c(0, 0, 1, 1), c(0.9, 0.8, 0.2, 0.1)), 0)

  set.seed(4)
  y <- rbinom(60, 1, 0.4)
  s <- round(runif(60) + 0.3 * y, 1)          # heavily tied scores
  pos <- s[y == 1]
  neg <- s[y == 0]
  ref <- mean(outer(pos, neg, function(a, b) (a > b) + 0.5 * (a == b)))
  expect_equal(auc(y, s), ref, tolerance = 1e-12)
  o <- sample(60)                              # order of the samples is irrelevant
  expect_equal(auc(y[o], s[o]), ref, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# nmf class methods
# ---------------------------------------------------------------------------

test_that("single-factor subsetting keeps a valid rank-1 model", {
  m <- make_model(12, 9, 3)
  m1 <- m[2]
  expect_s4_class(m1, "nmf")
  expect_equal(dim(m1@w), c(12L, 1L))
  expect_equal(dim(m1@h), c(1L, 9L))
  expect_equal(m1@d, m@d[2])
  expect_equal(m1@w[, 1], m@w[, 2])
  m31 <- m[c(3, 1)]
  expect_equal(m31@h, m@h[c(3, 1), ])
  expect_equal(dim(subset(m, 1)@w), c(12L, 1L))
})

test_that("$ finds misc entries by their exact name", {
  m <- make_model(12, 9, 3)
  m@misc <- list(L1 = c(0.1, 0.2), iter = 7L, Custom = "a")
  expect_equal(m$L1, c(0.1, 0.2))
  expect_equal(m$iter, 7L)
  expect_equal(m$Custom, "a")
  expect_equal(m$w, m@w)
  expect_equal(m$W, m@w)            # case-insensitive fallback for slots
  expect_error(m$nothing_here)
})

test_that("head, tail and show print a model", {
  m <- make_model(12, 9, 3)
  expect_output(head(m), "factor model")
  expect_output(tail(m), "factor model")
  expect_output(show(m), "factor model")
  expect_output(head(m[1]), "12 x 9 x 1")
})

test_that("summary works when a group has a single member", {
  m <- make_model(12, 9, 3)
  rownames(m@h) <- colnames(m@w) <- paste0("nmf", 1:3)
  g <- factor(c("a", rep("b", 4), rep("c", 4)))
  s <- summary(m, group_by = g, stat = "sum")
  expect_equal(nrow(s), 9L)
  get <- function(grp, fac) s$stat[s$group == grp & s$factor == fac]
  for (f in 1:3) {
    expect_equal(get("a", paste0("nmf", f)), unname(m@h[f, 1]))
    expect_equal(get("b", paste0("nmf", f)), unname(sum(m@h[f, 2:5])))
  }
  # a single group and feature-side grouping
  s1 <- summary(m, group_by = factor(rep("all", 9)), stat = "mean")
  expect_equal(s1$stat, unname(rowMeans(m@h)))
  gw <- factor(c("x", rep("y", 11)))
  sw <- summary(m, group_by = gw, stat = "sum")
  expect_equal(sw$stat[sw$group == "x" & sw$factor == "nmf2"], unname(m@w[1, 2]))
})

test_that("align inverts a cyclic permutation of three or more factors", {
  ref <- make_model(30, 10, 4, seed = 2)
  for (p in list(c(2, 3, 1, 4), c(4, 1, 2, 3), c(3, 4, 2, 1), c(2, 1, 4, 3))) {
    obj <- ref[p]
    al <- align(obj, ref)
    expect_equal(al@w, ref@w)
    expect_equal(al@h, ref@h)
    expect_equal(al@d, ref@d)
  }
})

# ---------------------------------------------------------------------------
# predict and evaluate
# ---------------------------------------------------------------------------

test_that("predict returns h on the model scale: w diag(d) h reproduces the data", {
  m <- make_model(40, 12, 3, seed = 5)
  A <- m@w %*% diag(m@d) %*% m@h                 # exact model, h >= 0
  p <- predict(m, A, threads = 2)
  expect_s4_class(p, "nmf")
  expect_equal(p@d, m@d)
  expect_equal(unname(p@h), unname(m@h), tolerance = 1e-4)
  expect_equal(unname(p@w %*% diag(p@d) %*% p@h), A, tolerance = 1e-4)
  # sparse input gives the same projection
  ps <- predict(m, Matrix::Matrix(A, sparse = TRUE), threads = 2)
  expect_equal(unname(ps@h), unname(m@h), tolerance = 1e-4)
})

test_that("predict honors a mask matrix and mask = 'zeros'", {
  m <- make_model(40, 12, 3, seed = 6)
  A <- m@w %*% diag(m@d) %*% m@h
  set.seed(7)
  M <- matrix(0, 40, 12)
  M[sample(length(M), 60)] <- 1

  A_bad <- A
  A_bad[M == 1] <- 100 * max(A)                  # corrupt the masked entries
  p_mask <- predict(m, A_bad, mask = Matrix::Matrix(M, sparse = TRUE), threads = 2)
  expect_equal(unname(p_mask@h), unname(m@h), tolerance = 1e-4)
  p_nomask <- predict(m, A_bad, threads = 2)
  expect_gt(max(abs(p_nomask@h - m@h)), 10 * max(abs(p_mask@h - m@h)) + 1e-3)

  A_drop <- A
  A_drop[M == 1] <- 0                            # dropout: zeros are missing values
  p_zero <- predict(m, Matrix::Matrix(A_drop, sparse = TRUE), mask = "zeros", threads = 2)
  expect_equal(unname(p_zero@h), unname(m@h), tolerance = 1e-4)
  p_zero_dense <- predict(m, A_drop, mask = "zeros", threads = 2)
  expect_equal(unname(p_zero_dense@h), unname(m@h), tolerance = 1e-4)
})

test_that("predict honors the nonneg setting of the model", {
  m <- make_model(40, 12, 3, seed = 8, signed_h = TRUE)
  expect_true(any(m@h < 0))
  A <- m@w %*% diag(m@d) %*% m@h
  m@misc$nonneg <- c(TRUE, FALSE)
  p <- predict(m, A, threads = 2)
  expect_equal(unname(p@h), unname(m@h), tolerance = 1e-4)
  p_nn <- predict(m, A, nonneg = TRUE, threads = 2)
  expect_true(all(p_nn@h >= 0))
  m@misc$nonneg <- NULL                          # models without the record default to TRUE
  expect_true(all(predict(m, A, threads = 2)@h >= 0))
})

test_that("evaluate honors mask, mask = 'zeros' and missing_only", {
  m <- make_model(25, 14, 3, seed = 9)
  set.seed(10)
  A <- m@w %*% diag(m@d) %*% m@h + matrix(rnorm(25 * 14, sd = 0.3), 25, 14)
  A[sample(length(A), 80)] <- 0
  R <- (A - m@w %*% diag(m@d) %*% m@h)^2
  M <- matrix(0, 25, 14)
  M[sample(length(M), 50)] <- 1
  Ms <- Matrix::Matrix(M, sparse = TRUE)
  As <- Matrix::Matrix(A, sparse = TRUE)

  for (X in list(A, As)) {
    expect_equal(evaluate(m, X, threads = 2), mean(R), tolerance = 1e-8)
    expect_equal(evaluate(m, X, mask = Ms, threads = 2), mean(R[M == 0]), tolerance = 1e-8)
    expect_equal(evaluate(m, X, mask = Ms, missing_only = TRUE, threads = 2),
                 mean(R[M == 1]), tolerance = 1e-8)
    expect_equal(evaluate(m, X, mask = "zeros", threads = 2), mean(R[A != 0]), tolerance = 1e-8)
  }
})

# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------

test_that("refine with lambda = 0 keeps the reconstruction (d is not counted twice)", {
  m <- make_model(30, 16, 3, seed = 11, d = c(50, 20, 5))
  A <- m@w %*% diag(m@d) %*% m@h
  labels <- rep(c("a", "b"), each = 8)
  r <- refine(m, data = A, labels = labels, lambda = 0, cycles = 1, nonneg = TRUE)
  recon <- r@w %*% diag(r@d) %*% r@h
  expect_lt(sqrt(sum((recon - A)^2)) / sqrt(sum(A^2)), 1e-3)
  r2 <- refine(m, data = A, labels = labels, lambda = 0, cycles = 3, nonneg = TRUE)
  recon2 <- r2@w %*% diag(r2@d) %*% r2@h
  expect_lt(sqrt(sum((recon2 - A)^2)) / sqrt(sum(A^2)), 1e-3)
})

# ---------------------------------------------------------------------------
# consensus_nmf
# ---------------------------------------------------------------------------

test_that("consensus_nmf clusters the rows of data (pass t(data) for samples in columns)", {
  sim <- make_clusters(c(8, 7, 6), feat_per = 9, jitter = 0.2, seed = 12)   # 27 features x 21 samples
  X <- t(sim$A)                                   # samples in rows
  cons <- consensus_nmf(X, k = 3, reps = 3, seed = 1, threads = 2, tol = 1e-5, maxit = 30)
  expect_equal(dim(cons$consensus), c(nrow(X), nrow(X)))
  expect_equal(length(cons$clusters), nrow(X))

  # the consensus matrix is the mean co-membership of the dominant factor of
  # each row of w
  ref <- matrix(0, nrow(X), nrow(X))
  for (mod in cons$models) {
    cl <- apply(mod@w, 1, which.max)
    ref <- ref + outer(cl, cl, "==")
  }
  expect_equal(cons$consensus, ref / 3)
  expect_equal(diag(cons$consensus), rep(1, nrow(X)))
  # three clean sample clusters are recovered
  tab <- table(sim$labels, cons$clusters)
  expect_true(all(rowSums(tab > 0) == 1) && all(colSums(tab > 0) == 1))
})

# ---------------------------------------------------------------------------
# factor networks
# ---------------------------------------------------------------------------

test_that("multi-modal W blocks follow the order of the inputs, dense and sparse", {
  set.seed(13)
  k <- 3
  n <- 25
  H <- matrix(runif(k * n), k, n)
  A_rna <- matrix(runif(30 * k), 30, k) %*% H          # first input, 30 rows
  A_atac <- 5 * matrix(runif(50 * k), 50, k) %*% H     # second input, 50 rows
  rel_err <- function(A, W, d, H) sqrt(sum((A - W %*% diag(d) %*% H)^2)) / sqrt(sum(A^2))

  for (sparse in c(FALSE, TRUE)) {
    X1 <- if (sparse) Matrix::Matrix(A_rna, sparse = TRUE) else A_rna
    X2 <- if (sparse) Matrix::Matrix(A_atac, sparse = TRUE) else A_atac
    # names deliberately not in alphabetical order
    i1 <- factor_input(X1, name = "rna")
    i2 <- factor_input(X2, name = "atac")
    net <- factor_net(list(i1, i2),
                      nmf_layer(factor_shared(i1, i2), k = k, name = "joint"),
                      factor_config(maxit = 100, tol = 1e-8, seed = 1, threads = 2))
    res <- fit(net)
    lay <- res$joint
    expect_true(res$multi_modal)
    expect_equal(names(lay$W), c("rna", "atac"))
    expect_equal(nrow(lay$W$rna), 30L)
    expect_equal(nrow(lay$W$atac), 50L)
    expect_equal(dim(lay$H), c(k, n))
    # each block reconstructs its own modality
    expect_lt(rel_err(A_rna, lay$W$rna, lay$d, lay$H), 0.1)
    expect_lt(rel_err(A_atac, lay$W$atac, lay$d, lay$H), 0.1)
  }
})

test_that("layers are returned in declared order, not alphabetical order", {
  sim <- make_clusters(c(10, 10, 10), feat_per = 6, seed = 14)
  inp <- factor_input(sim$A, name = "X")
  l1 <- nmf_layer(inp, k = 4, name = "zeta")
  l2 <- nmf_layer(l1, k = 2, name = "alpha")
  res <- fit(factor_net(inp, l2, factor_config(maxit = 5, seed = 1, threads = 2)))
  expect_equal(names(res$layers), c("zeta", "alpha"))
  expect_equal(dim(res$zeta$W), c(nrow(sim$A), 4L))
  expect_equal(dim(res$alpha$H), c(2L, 4L))
})

test_that("unsupported network options raise an error that names them", {
  sim <- make_clusters(c(10, 10), feat_per = 6, seed = 15)
  A <- sim$A
  inp <- factor_input(A, name = "X")
  cfg <- factor_config(maxit = 5, seed = 1, threads = 2)
  fit_layer <- function(layer) fit(factor_net(inp, layer, cfg))

  expect_error(fit_layer(nmf_layer(inp, k = 2, mask = "zeros")), "mask")
  expect_error(fit_layer(nmf_layer(inp, k = 2, zi = "row")), "zi")
  expect_error(fit_layer(nmf_layer(inp, k = 2, projective = TRUE)), "projective")
  expect_error(fit_layer(nmf_layer(inp, k = 2, symmetric = TRUE)), "symmetric")
  expect_error(fit_layer(nmf_layer(inp, k = 2, robust = TRUE)), "robust")
  expect_error(fit_layer(nmf_layer(inp, k = 2, cd_maxit = 5)), "cd_maxit")
  expect_error(fit_layer(svd_layer(inp, k = 2, center = TRUE)), "center")
  expect_error(fit_layer(nmf_layer(factor_condition(inp, matrix(1, 2, ncol(A))), k = 2)),
               "factor_condition")
  # targets: wrong shape, and multi-layer networks
  expect_error(fit_layer(nmf_layer(inp, k = 2,
                                   H = H(target = matrix(1, 3, ncol(A)), target_lambda = 1))),
               "target")
  l1 <- nmf_layer(inp, k = 3, H = H(target = matrix(1, 3, ncol(A)), target_lambda = 1))
  expect_error(fit(factor_net(inp, nmf_layer(l1, k = 2), cfg)), "target")
  expect_error(fit(factor_net(inp, nmf_layer(inp, k = 2),
                              factor_config(maxit = 5, dispersion = "global"))),
               "dispersion")
  # the supported configuration still fits
  expect_s3_class(fit_layer(nmf_layer(inp, k = 2, W = W(L1 = 0.01))), "factor_net_result")
})

test_that("a target on H of a single-layer network changes the fit", {
  sim <- make_clusters(c(12, 12), feat_per = 8, seed = 16)
  A <- sim$A
  inp <- factor_input(A, name = "X")
  cfg <- factor_config(maxit = 20, seed = 1, threads = 2)
  set.seed(17)
  target <- matrix(runif(2 * ncol(A)), 2, ncol(A)) * mean(A) * nrow(A)
  plain <- fit(factor_net(inp, nmf_layer(inp, k = 2, name = "L"), cfg))
  guided <- fit(factor_net(inp, nmf_layer(inp, k = 2, name = "L",
                                          H = H(target = target, target_lambda = 5)), cfg))
  expect_equal(dim(guided$L$H), dim(plain$L$H))
  expect_true(all(is.finite(guided$L$H)))
  expect_gt(max(abs(guided$L$H - plain$L$H)), 1e-4)   # it used to be ignored
})

test_that("cross_validate_graph uses the same mask for every combination in a replicate", {
  sim <- make_clusters(c(15, 15), feat_per = 10, jitter = 0.4, seed = 18)
  inp <- factor_input(sim$A, name = "X")
  cv <- cross_validate_graph(
    inputs = inp,
    layer_fn = function(p) nmf_layer(inp, k = 2, L2 = p$L2),
    params = list(L2 = c(0, 0)),                   # two identical combinations
    config = factor_config(maxit = 10, seed = 1, threads = 2, test_fraction = 0.2),
    reps = 2, seed = 5, verbose = FALSE)
  r <- cv$results
  for (ri in 1:2) {
    tl <- r$test_loss[r$rep == ri]
    expect_equal(length(tl), 2L)
    expect_true(all(is.finite(tl)))
    # identical models on identical held-out entries give identical test loss
    expect_equal(tl[1], tl[2], tolerance = 1e-3)
  }
})
