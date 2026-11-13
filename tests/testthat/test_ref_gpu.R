# GPU backend: discovery, diagnostics, and "no silent drops".
#
# The first block runs everywhere (no GPU needed). The second block needs a
# working GPU library and is skipped otherwise. Numerical checks compare
# against plain R references, not against another path of the package.

library(Matrix)

# ---------------------------------------------------------------------------
# No GPU required
# ---------------------------------------------------------------------------

test_that("TEST-REF-GPU-SEARCH: the GPU library is never looked up in the working directory", {
  cands <- RcppML:::.gpu_lib_candidates()
  expect_true(length(cands) >= 1L)
  wd <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  user_dir <- normalizePath(tools::R_user_dir("RcppML", "cache"),
                            winslash = "/", mustWork = FALSE)
  pkg_dir <- normalizePath(system.file(package = "RcppML"),
                           winslash = "/", mustWork = FALSE)
  for (p in cands) {
    d <- normalizePath(dirname(p), winslash = "/", mustWork = FALSE)
    expect_true(startsWith(d, user_dir) || startsWith(d, pkg_dir),
                info = p)
    expect_false(identical(d, wd), info = p)
  }
})

test_that("TEST-REF-GPU-STATUS: gpu_status() reports every step", {
  st <- gpu_status(quiet = TRUE)
  expect_type(st, "list")
  expect_true(all(c("nvcc", "nvcc_version", "searched", "library", "loaded",
                    "abi_expected", "abi_found", "devices", "available",
                    "reason") %in% names(st)))
  expect_identical(st$available, gpu_available())
  expect_true(is.character(st$reason) && nzchar(st$reason))
  if (!st$available) {
    expect_identical(st$devices, 0L)
    expect_null(gpu_info())
  } else {
    expect_identical(st$abi_found, st$abi_expected)
    expect_true(st$devices >= 1L)
  }
  expect_output(gpu_status(), "RcppML GPU status")
})

test_that("TEST-REF-GPU-ARCH: architecture flags are well formed", {
  fl <- RcppML:::.gpu_gencode_flags("nvcc", arch = c("sm_90", "sm_70"))
  expect_identical(fl[c(TRUE, FALSE)], rep("-gencode", 3L))
  expect_identical(fl[c(FALSE, TRUE)],
                   c("arch=compute_70,code=sm_70", "arch=compute_90,code=sm_90",
                     "arch=compute_70,code=compute_70"))
  expect_error(RcppML:::.gpu_gencode_flags("nvcc", arch = "volta"))
})

test_that("TEST-REF-GPU-INSTALL: install_gpu() fails clearly without a toolchain", {
  skip_on_os("windows")
  expect_error(install_gpu(nvcc = file.path(tempdir(), "no-such-nvcc"),
                           quiet = TRUE),
               "nvcc not found")
})

test_that("TEST-REF-GPU-SPZ: GPU .spz decoding reports that it is unavailable", {
  expect_error(st_read_gpu(tempfile(fileext = ".spz")), "not available in this build")
  expect_error(st_free_gpu(list()), "gpu_sparse_matrix")
})

test_that("TEST-REF-GPU-SVD-GUARD: unsupported SVD requests are named", {
  f <- RcppML:::.gpu_svd_unsupported
  # Runs on GPU: unconstrained, unmasked, no CV, no scale
  for (method in c("lanczos", "krylov", "randomized", "deflation"))
    expect_null(f(scale = FALSE, method = method, k = 5L))
  # Everything else is routed, with the reason named
  expect_match(f(scale = TRUE), "scale")
  expect_match(f(method = "deflation", k = 4L, nonneg = c(TRUE, FALSE)), "constrained")
  expect_match(f(method = "deflation", k = 1L, nonneg = c(TRUE, TRUE)), "constrained")
  expect_match(f(method = "krylov", k = 4L, L1 = c(0.1, 0)), "constrained")
  expect_match(f(method = "krylov", k = 4L, upper_bound = c(0, 1)), "constrained")
  expect_match(f(method = "krylov", k = 4L, L21 = c(0.1, 0)), "constrained")
  expect_match(f(method = "krylov", k = 4L, angular = c(0, 0.1)), "constrained")
  expect_match(f(method = "krylov", k = 4L, graph = TRUE), "constrained")
  expect_match(f(method = "irlba", k = 4L), "irlba")
  expect_match(f(method = "lanczos", k = 4L, mask = matrix(0, 2, 2)), "mask")
  expect_match(f(method = "lanczos", k = 4L, test_fraction = 0.1), "cross-validation")
})

test_that("TEST-REF-GPU-NO-DEVICE: resource = 'gpu' without a GPU warns and fits on CPU", {
  skip_if(gpu_available(), "a GPU is available")
  set.seed(1)
  A <- abs(Matrix::rsparsematrix(40, 30, 0.3))
  expect_warning(
    fit <- nmf(A, k = 3, seed = 1, maxit = 5, resource = "gpu", verbose = FALSE),
    "GPU")
  expect_s4_class(fit, "nmf")
})

# ---------------------------------------------------------------------------
# GPU required
# ---------------------------------------------------------------------------

.ref_gpu_skip <- function() {
  skip_if_not(gpu_available(), "GPU not available")
}

.rel_err <- function(A, fit) {
  R <- as.matrix(A) - fit@w %*% (fit@d * fit@h)
  sqrt(sum(R^2)) / sqrt(sum(as.matrix(A)^2))
}

# Fit on GPU and CPU with the same arguments. `routed` is a regular expression
# for the fallback warning (NULL when the fit is expected to run on GPU).
.ref_gpu_pair <- function(A, ..., routed = NULL) {
  cpu <- nmf(A, ..., resource = "cpu", verbose = FALSE)
  if (is.null(routed)) {
    gpu <- expect_no_warning(nmf(A, ..., resource = "gpu", verbose = FALSE))
  } else {
    expect_warning(gpu <- nmf(A, ..., resource = "gpu", verbose = FALSE), routed)
  }
  list(cpu = cpu, gpu = gpu)
}

.expect_same_fit <- function(pr, info) {
  expect_equal(pr$gpu@w, pr$cpu@w, tolerance = 1e-6, info = info)
  expect_equal(pr$gpu@d, pr$cpu@d, tolerance = 1e-6, info = info)
  expect_equal(pr$gpu@h, pr$cpu@h, tolerance = 1e-6, info = info)
}

test_that("TEST-REF-GPU-FALLBACK: routed options warn and return the CPU fit", {
  .ref_gpu_skip()
  set.seed(2)
  A <- abs(Matrix::rsparsematrix(80, 60, 0.3))
  Ac <- A; Ac@x <- as.numeric(rpois(length(Ac@x), 3) + 1)   # counts
  Ad <- as.matrix(A) + 0.05                                 # strictly positive
  k <- 4
  T_H <- matrix(runif(k * ncol(A)), k, ncol(A))
  L <- as(Matrix::Diagonal(ncol(A)), "CsparseMatrix")
  # solver is fixed: "auto" resolves to "cholesky" for CPU and "cd" for GPU
  # requests in R, before the gateway decides where the fit runs
  base <- list(k = k, seed = 3, maxit = 8, tol = 1e-8, solver = "cd")

  cases <- list(
    target       = list(A, "target", target_H = T_H, target_lambda = 0.5),
    L21          = list(A, "L21", L21 = c(0.1, 0)),
    angular      = list(A, "angular", angular = c(0.1, 0)),
    graph        = list(A, "graph", graph_H = L, graph_lambda = c(0, 0.1)),
    upper_bound  = list(A, "upper_bound", upper_bound = c(1, 1)),
    projective   = list(A, "projective", projective = TRUE),
    cv           = list(A, "cross-validation", test_fraction = 0.1),
    robust       = list(A, "robust", robust = TRUE),
    gp_sparse    = list(Ac, "non-Gaussian", loss = "gp"),
    nb_sparse    = list(Ac, "non-Gaussian", loss = "nb"),
    zi           = list(Ac, "non-Gaussian|zero inflation", loss = "gp", zi = "row"),
    gp_dense     = list(as.matrix(Ac), "non-Gaussian", loss = "gp"),
    gamma_dense  = list(Ad, "non-Gaussian", loss = "gamma"),
    graph_dense  = list(as.matrix(A), "graph", graph_H = L, graph_lambda = c(0, 0.1)),
    L1_dense     = list(as.matrix(A), "dense", L1 = c(0.05, 0.05)),
    L2_dense     = list(as.matrix(A), "dense", L2 = c(0.05, 0.05))
  )
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    args <- c(list(cs[[1]]), base, cs[-(1:2)], list(routed = cs[[2]]))
    pr <- do.call(.ref_gpu_pair, args)
    .expect_same_fit(pr, nm)
  }

  # The target was actually applied (issue #75), not dropped
  plain <- nmf(A, k = k, seed = 3, maxit = 8, tol = 1e-8, resource = "cpu",
               solver = "cd", verbose = FALSE)
  tg <- nmf(A, k = k, seed = 3, maxit = 8, tol = 1e-8, resource = "cpu",
            solver = "cd", target_H = T_H, target_lambda = 0.5, verbose = FALSE)
  expect_gt(max(abs(tg@h - plain@h)), 1e-4)

  # Symmetric input
  S <- as(Matrix::crossprod(A), "CsparseMatrix")
  pr <- .ref_gpu_pair(S, k = k, seed = 3, maxit = 8, tol = 1e-8,
                      solver = "cd", symmetric = TRUE, routed = "symmetric")
  .expect_same_fit(pr, "symmetric")
})

test_that("TEST-REF-GPU-MSE: the supported Gaussian path matches CPU and plain R", {
  .ref_gpu_skip()
  set.seed(4)
  W <- matrix(runif(120 * 5), 120, 5); H <- matrix(runif(5 * 90), 5, 90)
  D <- W %*% H * (matrix(runif(120 * 90), 120, 90) < 0.3)
  # Thinned low-rank data: noisy and well conditioned, so CPU and GPU follow
  # the same trajectory and can be compared with each other
  inputs <- list(sparse = as(Matrix(D, sparse = TRUE), "CsparseMatrix"),
                 dense = D + 0.01)
  W0 <- matrix(runif(120 * 5), 120, 5)
  settings <- list(
    plain    = list(),
    L1       = list(L1 = c(0.05, 0.05)),
    L2       = list(L2 = c(0.05, 0.05)),
    norm_L2  = list(norm = "L2"),
    user_W   = list(seed = W0)
  )
  for (inp in names(inputs)) {
    A <- inputs[[inp]]
    for (nm in names(settings)) {
      # L1 and L2 on dense input are routed to CPU (covered in FALLBACK)
      if (inp == "dense" && nm %in% c("L1", "L2")) next
      info <- paste(inp, nm)
      args <- utils::modifyList(
        list(k = 5, seed = 7, maxit = 20, tol = 0, solver = "cd"),
        settings[[nm]])
      # Fixed iteration count from the same start, exact solver on both sides
      pr <- do.call(.ref_gpu_pair, c(list(A), args))
      g <- pr$gpu; c0 <- pr$cpu

      # Model convention: unit-norm factors under the requested norm, d sorted
      nrm <- if (identical(args$norm, "L2")) function(x) sqrt(sum(x^2))
             else function(x) sum(abs(x))
      expect_equal(unname(apply(g@w, 2, nrm)), rep(1, 5), tolerance = 1e-4, info = info)
      expect_equal(unname(apply(g@h, 1, nrm)), rep(1, 5), tolerance = 1e-4, info = info)
      expect_false(is.unsorted(rev(g@d)), info = info)
      expect_true(all(g@w >= 0) && all(g@h >= 0), info = info)

      # Loss measured in plain R, GPU against the CPU fit
      sse <- function(fit) sum((as.matrix(A) - fit@w %*% (fit@d * fit@h))^2)
      expect_lt(abs(sse(g) - sse(c0)) / sse(c0), 1e-3, label = info)
      # The reported loss is on the CPU definition
      expect_equal(g@misc$loss, c0@misc$loss, tolerance = 1e-3, info = info)
    }
  }

  # solver = "cholesky" is served by coordinate descent on GPU. gc and gd are
  # the same sparse fit requested with solver = "cholesky" and solver = "cd";
  # they agree to solver precision, not bit for bit (reruns of the
  # "cholesky" request themselves differ by 3e-7).
  gc <- nmf(inputs$sparse, k = 5, seed = 7, maxit = 20, tol = 0, resource = "gpu",
            solver = "cholesky", verbose = FALSE)
  gd <- nmf(inputs$sparse, k = 5, seed = 7, maxit = 20, tol = 0, resource = "gpu",
            solver = "cd", verbose = FALSE)
  expect_equal(gc@w, gd@w, tolerance = 1e-4)

  # Same seed, same result to 1e-6. Reruns are not bit-identical: cuBLAS and
  # cuSPARSE reductions and the loss kernels' atomic sums do not fix the
  # order of floating-point additions (observed 9e-9).
  g1 <- nmf(inputs$sparse, k = 5, seed = 7, maxit = 20, tol = 0, resource = "gpu",
            verbose = FALSE)
  g2 <- nmf(inputs$sparse, k = 5, seed = 7, maxit = 20, tol = 0, resource = "gpu",
            verbose = FALSE)
  expect_equal(g1@w, g2@w, tolerance = 1e-6)
})

# Exact non-negative least squares for all columns, in double precision:
# minimize ||A - W X||_F over X >= 0 by coordinate descent on G = W'W, B = W'A.
.ref_nnls <- function(W, A, sweeps = 2000L, tol = 1e-12) {
  G <- crossprod(W)
  B <- crossprod(W, A)
  X <- pmax(solve(G, B), 0)
  for (s in seq_len(sweeps)) {
    moved <- 0
    for (i in seq_len(nrow(X))) {
      new <- pmax(0, X[i, ] - (drop(G[i, , drop = FALSE] %*% X) - B[i, ]) / G[i, i])
      moved <- max(moved, max(abs(new - X[i, ])))
      X[i, ] <- new
    }
    if (moved <= tol * max(abs(X))) break
  }
  X
}

test_that("TEST-REF-GPU-EXACT: one iteration matches exact double-precision ALS", {
  .ref_gpu_skip()
  # W H plus 1% sparse noise, every entry non-zero. After the first H solve
  # the rows of H are close to collinear, so single-precision fits drift from
  # each other on this matrix (CPU and GPU end 2-3x apart in loss after 40
  # iterations, each within its own rounding of the exact path). GPU and CPU
  # are therefore each compared with an exact reference computed here in
  # plain R, not with each other.
  set.seed(77)
  m <- 300; n <- 1000; k <- 4
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  noise <- Matrix::rsparsematrix(m, n, nnz = as.integer(m * n * 0.05))
  D <- W_true %*% H_true + abs(as.matrix(noise)) * 0.01
  W0 <- matrix(runif(m * k), m, k)

  # One ALS iteration from W0: H given W0, then W given H
  H1 <- .ref_nnls(W0, D)
  W1 <- t(.ref_nnls(t(H1), t(D)))
  ref <- mean((D - W1 %*% H1)^2)

  mse <- function(fit) mean((D - fit@w %*% (fit@d * fit@h))^2)
  inputs <- list(dense = D,
                 stored = as(Matrix(D, sparse = TRUE), "CsparseMatrix"))
  for (inp in names(inputs)) {
    A <- inputs[[inp]]
    gpu <- expect_no_warning(
      nmf(A, k = k, seed = W0, maxit = 1, tol = 0, resource = "gpu",
          verbose = FALSE))
    cpu <- nmf(A, k = k, seed = W0, maxit = 1, tol = 0, resource = "cpu",
               verbose = FALSE)
    expect_lt(abs(mse(gpu) / ref - 1), 0.01, label = paste("GPU,", inp))   # measured 0.0051 on a V100S
    expect_lt(abs(mse(cpu) / ref - 1), 0.02, label = paste("CPU,", inp))
  }
})

test_that("TEST-REF-GPU-SVD-FLAT: Lanczos and Krylov resolve a flat spectrum", {
  .ref_gpu_skip()
  set.seed(9)
  m <- 500; n <- 300
  A <- matrix(rnorm(m * n), m, n)
  ref <- base::svd(A, nu = 0, nv = 0)$d[1:20]
  As <- as(Matrix(A, sparse = TRUE), "CsparseMatrix")
  for (method in c("lanczos", "krylov")) {
    s <- RcppML::svd(As, k = 20, method = method, resource = "gpu", seed = 1)
    expect_lt(max(abs(s@d - ref)) / ref[1], 1e-3, label = method)
  }
})

test_that("TEST-REF-GPU-SVD-RANK: Lanczos and Krylov recover an exactly rank-k matrix", {
  .ref_gpu_skip()
  set.seed(123)
  m <- 30; n <- 20; k <- 5
  U <- qr.Q(qr(matrix(rnorm(m * k), m, k)))
  V <- qr.Q(qr(matrix(rnorm(n * k), n, k)))
  d_true <- c(50, 30, 15, 8, 4)
  A <- U %*% diag(d_true) %*% t(V)
  ref <- base::svd(A)$d[seq_len(k)]
  As <- as(Matrix(A, sparse = TRUE), "CsparseMatrix")
  for (method in c("lanczos", "krylov")) {
    for (X in list(As, A)) {
      s <- RcppML::svd(X, k = k, method = method, resource = "gpu",
                       tol = 1e-10, seed = 1, test_fraction = 0)
      expect_equal(s@d, ref, tolerance = 1e-3,
                   info = paste(method, class(X)[1]))
      A_hat <- s@u %*% diag(s@d) %*% t(s@v)
      expect_lt(sqrt(sum((A - A_hat)^2)) / sqrt(sum(A^2)), 1e-3)
    }
  }
  # k larger than the rank: the extra singular values are zero, not garbage
  s <- RcppML::svd(As, k = 8, method = "lanczos", resource = "gpu",
                   tol = 1e-10, seed = 1, test_fraction = 0)
  expect_equal(s@d[1:5], ref, tolerance = 1e-3)
  expect_true(all(abs(s@d[6:8]) < 1e-2))
})
