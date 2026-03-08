#' @title GPU NMF Backend
#'
#' @description Internal functions for GPU-accelerated NMF via CUDA.
#' Automatically detects GPU availability and dispatches to GPU when
#' beneficial, falling back to CPU otherwise.
#'
#' @name gpu-backend
#' @keywords internal
NULL

# Package-level GPU state (initialized once per session)
.gpu_env <- new.env(parent = emptyenv())
.gpu_env$available <- NA       # NA = not checked, TRUE/FALSE after check
.gpu_env$lib_loaded <- FALSE
.gpu_env$num_gpus <- 0L
.gpu_env$gpu_names <- character(0)
.gpu_env$total_mem_mb <- numeric(0)
.gpu_env$free_mem_mb <- numeric(0)

# Helper: call GPU bridge .C() with dynamic symbol name.
# Uses do.call() to prevent Rcpp::compileAttributes() from scanning the 
# symbol name and auto-registering it in the main package DLL.
# We resolve the symbol via getNativeSymbolInfo() to pin to the GPU DLL.
.gpu_call <- function(sym, ...) {
  sym_info <- getNativeSymbolInfo(paste0(sym), .gpu_env$dll)
  do.call(.C, c(list(sym_info), list(...)))
}

# Helper: extract CSC arrays from a dgCMatrix for .C() bridge.
# Returns a named list of (p, i, x, dim, nnz, lambda) suitable for
# splicing into .gpu_call() arguments. If mat is NULL, returns dummy
# arrays with nnz = 0 (sentinel) so the C code skips reconstruction.
.extract_sparse_csc <- function(mat, lambda = 0, symmetric = TRUE) {
  if (is.null(mat)) {
    list(p = as.integer(0L), i = as.integer(0L), x = as.double(0),
         dim = as.integer(0L), nnz = as.integer(0L), lambda = as.double(lambda))
  } else {
    mat <- as(mat, "dgCMatrix")
    list(p = as.integer(mat@p), i = as.integer(mat@i), x = as.double(mat@x),
         dim = as.integer(nrow(mat)), nnz = as.integer(length(mat@x)),
         lambda = as.double(lambda))
  }
}

# Like .extract_sparse_csc but for obs_mask where rows/cols may differ
.extract_sparse_csc_rect <- function(mat) {
  if (is.null(mat)) {
    list(p = as.integer(0L), i = as.integer(0L), x = as.double(0),
         rows = as.integer(0L), cols = as.integer(0L), nnz = as.integer(0L))
  } else {
    mat <- as(mat, "dgCMatrix")
    list(p = as.integer(mat@p), i = as.integer(mat@i), x = as.double(mat@x),
         rows = as.integer(nrow(mat)), cols = as.integer(ncol(mat)),
         nnz = as.integer(length(mat@x)))
  }
}

#' Check if GPU acceleration is available
#'
#' Detects CUDA GPUs at runtime. Result is cached for the session.
#'
#' @param force_recheck Re-probe GPUs even if already cached
#' @return logical TRUE if GPU is available
#' @export
#' @examples
#' gpu_available()
gpu_available <- function(force_recheck = FALSE) {
  if (!is.na(.gpu_env$available) && !force_recheck) {
    return(.gpu_env$available)
  }
  
  # If library is already loaded, skip the find/load step
  # (force_recheck only re-probes GPUs, not the library path)
  if (!.gpu_env$lib_loaded) {
    lib_path <- .find_gpu_lib()
    if (is.null(lib_path)) {
      .gpu_env$available <- FALSE
      return(FALSE)
    }
  }
  
  if (!.gpu_env$lib_loaded) {
    tryCatch({
      # local=FALSE makes symbols globally visible so C++ detect_gpus_via_bridge()
      # can find them via dlsym(RTLD_DEFAULT, "rcppml_gpu_detect")
      .gpu_env$dll <- dyn.load(lib_path, local = FALSE)
      .gpu_env$lib_loaded <- TRUE
    }, error = function(e) {
      .gpu_env$available <- FALSE
    })
  }
  
  if (!.gpu_env$lib_loaded) {
    .gpu_env$available <- FALSE
    return(FALSE)
  }
  
  # Probe GPUs via C bridge (all args are pointers per .C() convention)
  tryCatch({
    result <- .gpu_call("rcppml_gpu_detect",
                  num_gpus = integer(1),
                  total_mem_mb = double(8),
                  free_mem_mb = double(8),
                  max_gpus = as.integer(8L),
                  out_status = integer(1))
    
    if (result$out_status == 0L && result$num_gpus > 0L) {
      ng <- result$num_gpus
      .gpu_env$available <- TRUE
      .gpu_env$num_gpus <- ng
      .gpu_env$total_mem_mb <- result$total_mem_mb[seq_len(ng)]
      .gpu_env$free_mem_mb <- result$free_mem_mb[seq_len(ng)]
      # GPU names not returned via .C() (char* is read-only in .C()),
      # so we just report generic names based on count
      .gpu_env$gpu_names <- paste0("GPU ", seq_len(ng) - 1L)
    } else {
      .gpu_env$available <- FALSE
    }
  }, error = function(e) {
    .gpu_env$available <- FALSE
  })
  
  return(.gpu_env$available)
}

#' Get GPU device information
#'
#' @return data.frame with GPU device details, or NULL if no GPU
#' @export
#' @examples
#' gpu_info()
gpu_info <- function() {
  if (!gpu_available()) return(NULL)
  
  data.frame(
    device = seq_len(.gpu_env$num_gpus) - 1L,
    name = .gpu_env$gpu_names,
    total_mb = .gpu_env$total_mem_mb,
    free_mb = .gpu_env$free_mem_mb,
    stringsAsFactors = FALSE
  )
}

#' Find GPU shared library
#' @keywords internal
.find_gpu_lib <- function() {
  # Search order:
  # 1. Working directory inst/lib/ (development build — takes priority)
  # 2. Working directory src/ (development mode, make without install)
  # 3. Package inst/lib/ directory (installed package)
  # 4. Working directory root

  lib_name <- paste0("RcppML_gpu", .Platform$dynlib.ext)
  
  candidates <- c(
    file.path(getwd(), "inst", "lib", lib_name),
    file.path(getwd(), "src", lib_name),
    file.path(getwd(), lib_name),
    system.file("lib", lib_name, package = "RcppML"),
    file.path(find.package("RcppML", quiet = TRUE), "src", lib_name)
  )
  
  for (path in candidates) {
    if (nzchar(path) && file.exists(path)) return(path)
  }
  
  NULL
}

# Legacy .gpu_nmf_sparse() — REMOVED.
# .gpu_nmf_unified() and .gpu_nmf_cv_unified() — REMOVED (Phase 3).
# NMF GPU dispatch now goes through C++ gateway: Rcpp_nmf_full() →
# nmf/fit.hpp → gpu::bridge_nmf_sparse() → dlsym → gpu_bridge.cu

# -----------------------------------------------------------------------
# GPU NMF zero-copy dispatch (sp_read_gpu → NMF without re-upload)
# -----------------------------------------------------------------------

#' @keywords internal
.gpu_nmf_zerocopy <- function(gpu_mat, k, maxit, tol, seed,
                              L1 = c(0, 0), L2 = c(0, 0),
                              L21 = c(0, 0), ortho = c(0, 0),
                              upper_bound = c(0, 0),
                              cd_maxit = 10L, verbose = FALSE,
                              nonneg = c(TRUE, TRUE),
                              w_init = NULL,
                              loss_every = 1L, patience = 5L,
                              loss = "mse", huber_delta = 1.0,
                              irls_max_iter = 20L, irls_tol = 1e-4,
                              norm = "L1") {

  if (!gpu_available()) stop("GPU not available")
  if (!inherits(gpu_mat, "gpu_sparse_matrix"))
    stop("gpu_mat must be a gpu_sparse_matrix from sp_read_gpu()")

  m <- gpu_mat$m
  n <- gpu_mat$n
  nnz <- gpu_mat$nnz

  # Initialize W (k x m) and H (k x n)
  if (is.null(w_init)) {
    set.seed(seed)
    W <- matrix(runif(k * m), k, m)
  } else {
    W <- t(w_init)  # m x k → k x m
  }
  H <- matrix(runif(k * n), k, n)
  d <- rep(1.0, k)

  # Map loss string to integer: MSE=0, MAE=1, HUBER=2, KL=3
  loss_type_int <- match(tolower(loss), c("mse", "mae", "huber", "kl")) - 1L
  if (is.na(loss_type_int)) loss_type_int <- 0L

  # Map norm string to integer: L1=0, L2=1, none=2
  norm_type_int <- match(tolower(norm), c("l1", "l2", "none")) - 1L
  if (is.na(norm_type_int)) norm_type_int <- 0L

  result <- .gpu_call("rcppml_gpu_nmf_zerocopy_double",
    d_col_ptr_addr = as.double(gpu_mat$.col_ptr),
    d_row_idx_addr = as.double(gpu_mat$.row_idx),
    d_values_addr  = as.double(gpu_mat$.values),
    m = as.integer(m),
    n = as.integer(n),
    nnz_d = as.double(nnz),
    k = as.integer(k),
    W = as.double(W),
    H = as.double(H),
    d = as.double(d),
    max_iter = as.integer(maxit),
    tol = as.double(tol),
    L1_H = as.double(L1[1]),
    L1_W = as.double(L1[2]),
    L2_H = as.double(L2[1]),
    L2_W = as.double(L2[2]),
    L21_H = as.double(L21[1]),
    L21_W = as.double(L21[2]),
    ortho_H = as.double(ortho[1]),
    ortho_W = as.double(ortho[2]),
    ub_H = as.double(upper_bound[1]),
    ub_W = as.double(upper_bound[2]),
    cd_maxit = as.integer(cd_maxit),
    verbose = as.integer(verbose),
    seed = as.integer(seed),
    loss_every = as.integer(loss_every),
    patience = as.integer(patience),
    nonneg_W = as.integer(nonneg[1]),
    nonneg_H = as.integer(nonneg[2]),
    loss_type = as.integer(loss_type_int),
    huber_delta = as.double(huber_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    norm_type = as.integer(norm_type_int),
    out_iter = as.integer(0L),
    out_converged = as.integer(0L),
    out_loss = as.double(0.0),
    out_status = as.integer(0L),
    out_tol = as.double(0.0)
  )

  if (result$out_status != 0L) {
    stop("GPU zero-copy NMF failed (status ", result$out_status, ")")
  }

  # Reshape W from flat k*m back to m x k
  W_out <- matrix(result$W, nrow = k, ncol = m)
  W_out <- t(W_out)

  # Reshape H from flat k*n back to k x n
  H_out <- matrix(result$H, nrow = k, ncol = n)

  list(
    w = W_out,
    d = result$d,
    h = H_out,
    iter = result$out_iter,
    converged = (result$out_converged != 0L),
    loss = result$out_loss,
    tol = result$out_tol,
    backend = "gpu"
  )
}

# -----------------------------------------------------------------------
# GPU SVD/PCA dispatch function
# -----------------------------------------------------------------------

#' @keywords internal
.gpu_svd_pca <- function(A, k_max, tol, max_iter, center, verbose, seed,
                         threads, L1_u, L1_v, L2_u, L2_v,
                         nonneg_u, nonneg_v, upper_bound_u, upper_bound_v,
                         L21_u = 0, L21_v = 0, angular_u = 0, angular_v = 0,
                         test_fraction, cv_seed, patience, mask_zeros,
                         method = "deflation", precision = "double",
                         graph_U = NULL, graph_V = NULL,
                         graph_lambda = c(0, 0),
                         obs_mask = NULL,
                         robust_delta = 0, irls_max_iter = 5L, irls_tol = 1e-4) {

  if (!gpu_available()) stop("GPU not available")

  A <- .to_dgCMatrix(A)
  m <- nrow(A); n <- ncol(A); nnz <- length(A@x)

  # Map method name to algorithm integer:
  # 0 = deflation, 1 = irlba, 2 = lanczos, 3 = randomized, 4 = krylov
  algo_map <- c(deflation = 0L, irlba = 1L, lanczos = 2L, randomized = 3L, krylov = 4L)
  algo_int <- algo_map[method]
  if (is.na(algo_int)) {
    warning("GPU does not support method '", method, "', falling back to deflation")
    algo_int <- 0L
  }

  # Pre-allocate output arrays
  U_buf <- numeric(m * k_max)
  d_buf <- numeric(k_max)
  V_buf <- numeric(n * k_max)
  test_loss_buf <- numeric(k_max)
  iters_buf <- integer(k_max)
  row_means_buf <- numeric(m)

  # Choose precision entry point
  sym <- switch(precision,
    "float" = "rcppml_gpu_svd_pca_float",
    "rcppml_gpu_svd_pca_double"  # default
  )

  # Extract graph Laplacian and obs_mask CSC arrays
  gU <- .extract_sparse_csc(graph_U, lambda = graph_lambda[1])
  gV <- .extract_sparse_csc(graph_V, lambda = graph_lambda[2])
  om <- .extract_sparse_csc_rect(obs_mask)

  result <- .gpu_call(sym,
    col_ptr = as.integer(A@p),
    row_idx = as.integer(A@i),
    values  = as.double(A@x),
    m = as.integer(m),
    n = as.integer(n),
    nnz = as.integer(nnz),
    k_max = as.integer(k_max),
    U = as.double(U_buf),
    d = as.double(d_buf),
    V = as.double(V_buf),
    tol = as.double(tol),
    max_iter = as.integer(max_iter),
    center = as.integer(center),
    verbose = as.integer(verbose),
    seed = as.integer(seed),
    threads = as.integer(threads),
    L1_u = as.double(L1_u),
    L1_v = as.double(L1_v),
    L2_u = as.double(L2_u),
    L2_v = as.double(L2_v),
    nonneg_u = as.integer(nonneg_u),
    nonneg_v = as.integer(nonneg_v),
    ub_u = as.double(upper_bound_u),
    ub_v = as.double(upper_bound_v),
    L21_u = as.double(L21_u),
    L21_v = as.double(L21_v),
    angular_u = as.double(angular_u),
    angular_v = as.double(angular_v),
    test_fraction = as.double(test_fraction),
    cv_seed = as.integer(cv_seed),
    patience = as.integer(patience),
    mask_zeros = as.integer(mask_zeros),
    algorithm = as.integer(algo_int),
    graph_u_p = gU$p, graph_u_i = gU$i, graph_u_x = gU$x,
    graph_u_dim = gU$dim, graph_u_nnz = gU$nnz, graph_u_lambda = gU$lambda,
    graph_v_p = gV$p, graph_v_i = gV$i, graph_v_x = gV$x,
    graph_v_dim = gV$dim, graph_v_nnz = gV$nnz, graph_v_lambda = gV$lambda,
    obs_mask_p = om$p, obs_mask_i = om$i, obs_mask_x = om$x,
    obs_mask_rows = om$rows, obs_mask_cols = om$cols, obs_mask_nnz = om$nnz,
    out_k_selected = as.integer(0L),
    out_wall_time_ms = as.double(0),
    out_test_loss = as.double(test_loss_buf),
    out_iters_per_factor = as.integer(iters_buf),
    out_frobenius_norm_sq = as.double(0),
    out_row_means = as.double(row_means_buf),
    robust_delta = as.double(robust_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    out_status = as.integer(0L)
  )

  if (result$out_status != 0L) {
    stop("GPU SVD/PCA failed (status ", result$out_status, ")")
  }

  ksel <- result$out_k_selected

  # Reshape U from flat m*k_max -> m x ksel
  U_mat <- matrix(result$U, nrow = m, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  V_mat <- matrix(result$V, nrow = n, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  d_vec <- result$d[seq_len(ksel)]

  # k_computed: count non-zero iterations
  k_computed <- sum(result$out_iters_per_factor > 0)
  if (k_computed == 0) k_computed <- ksel

  iters_out <- result$out_iters_per_factor[seq_len(k_computed)]

  ret <- list(
    u = U_mat,
    d = d_vec,
    v = V_mat,
    centered = as.logical(result$center),
    wall_time_ms = result$out_wall_time_ms,
    iters_per_factor = iters_out,
    frobenius_norm_sq = result$out_frobenius_norm_sq
  )

  if (test_fraction > 0) {
    test_loss_out <- result$out_test_loss[seq_len(k_computed)]
    ret$test_loss <- test_loss_out
  }

  if (center) {
    ret$row_means <- result$out_row_means
  }

  ret
}

#' @title Dense-matrix GPU SVD/PCA
#' @description GPU SVD/PCA for a base R dense matrix (is.matrix(A) == TRUE).
#'   For deflation: uses cuBLAS GEMV instead of cuSPARSE SpMV (significantly
#'   faster for dense input). For other algorithms: converts to CSC internally.
#' @keywords internal
.gpu_svd_pca_dense <- function(A, k_max, tol, max_iter, center, verbose, seed,
                                threads, L1_u, L1_v, L2_u, L2_v,
                                nonneg_u, nonneg_v, upper_bound_u, upper_bound_v,
                                L21_u = 0, L21_v = 0, angular_u = 0, angular_v = 0,
                                test_fraction, cv_seed, patience, mask_zeros,
                                method = "deflation",
                                graph_U = NULL, graph_V = NULL,
                                graph_lambda = c(0, 0),
                                obs_mask = NULL,
                                robust_delta = 0, irls_max_iter = 5L, irls_tol = 1e-4) {

  if (!gpu_available()) stop("GPU not available")

  if (!is.matrix(A)) A <- as.matrix(A)
  m <- nrow(A); n <- ncol(A)

  algo_map <- c(deflation = 0L, irlba = 1L, lanczos = 2L, randomized = 3L, krylov = 4L)
  algo_int <- algo_map[method]
  if (is.na(algo_int)) { warning("GPU does not support method '", method, "', using deflation"); algo_int <- 0L }

  U_buf   <- numeric(m * k_max)
  d_buf   <- numeric(k_max)
  V_buf   <- numeric(n * k_max)
  test_loss_buf <- numeric(k_max)
  iters_buf <- integer(k_max)
  row_means_buf <- numeric(m)

  # Extract graph Laplacian and obs_mask CSC arrays
  gU <- .extract_sparse_csc(graph_U, lambda = graph_lambda[1])
  gV <- .extract_sparse_csc(graph_V, lambda = graph_lambda[2])
  om <- .extract_sparse_csc_rect(obs_mask)

  result <- .gpu_call("rcppml_gpu_svd_pca_dense_double",
    A_data      = as.double(A),   # column-major flat vector
    m           = as.integer(m),
    n           = as.integer(n),
    k_max       = as.integer(k_max),
    U           = as.double(U_buf),
    d           = as.double(d_buf),
    V           = as.double(V_buf),
    tol         = as.double(tol),
    max_iter    = as.integer(max_iter),
    center      = as.integer(center),
    verbose     = as.integer(verbose),
    seed        = as.integer(seed),
    threads     = as.integer(threads),
    L1_u        = as.double(L1_u),
    L1_v        = as.double(L1_v),
    L2_u        = as.double(L2_u),
    L2_v        = as.double(L2_v),
    nonneg_u    = as.integer(nonneg_u),
    nonneg_v    = as.integer(nonneg_v),
    ub_u        = as.double(upper_bound_u),
    ub_v        = as.double(upper_bound_v),
    L21_u       = as.double(L21_u),
    L21_v       = as.double(L21_v),
    angular_u   = as.double(angular_u),
    angular_v   = as.double(angular_v),
    test_fraction = as.double(test_fraction),
    cv_seed     = as.integer(cv_seed),
    patience    = as.integer(patience),
    mask_zeros  = as.integer(mask_zeros),
    algorithm   = as.integer(algo_int),
    graph_u_p = gU$p, graph_u_i = gU$i, graph_u_x = gU$x,
    graph_u_dim = gU$dim, graph_u_nnz = gU$nnz, graph_u_lambda = gU$lambda,
    graph_v_p = gV$p, graph_v_i = gV$i, graph_v_x = gV$x,
    graph_v_dim = gV$dim, graph_v_nnz = gV$nnz, graph_v_lambda = gV$lambda,
    obs_mask_p = om$p, obs_mask_i = om$i, obs_mask_x = om$x,
    obs_mask_rows = om$rows, obs_mask_cols = om$cols, obs_mask_nnz = om$nnz,
    out_k_selected      = as.integer(0L),
    out_wall_time_ms    = as.double(0),
    out_test_loss       = as.double(test_loss_buf),
    out_iters_per_factor = as.integer(iters_buf),
    out_frobenius_norm_sq = as.double(0),
    out_row_means       = as.double(row_means_buf),
    robust_delta = as.double(robust_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    out_status          = as.integer(0L)
  )

  if (result$out_status != 0L) stop("GPU SVD/PCA (dense) failed (status ", result$out_status, ")")

  ksel    <- result$out_k_selected
  U_mat   <- matrix(result$U, nrow = m, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  V_mat   <- matrix(result$V, nrow = n, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  d_vec   <- result$d[seq_len(ksel)]
  k_computed <- sum(result$out_iters_per_factor > 0)
  if (k_computed == 0) k_computed <- ksel
  iters_out <- result$out_iters_per_factor[seq_len(k_computed)]

  ret <- list(
    u = U_mat, d = d_vec, v = V_mat,
    centered = as.logical(center),
    wall_time_ms = result$out_wall_time_ms,
    iters_per_factor = iters_out,
    frobenius_norm_sq = result$out_frobenius_norm_sq
  )
  if (test_fraction > 0) ret$test_loss <- result$out_test_loss[seq_len(k_computed)]
  if (center) ret$row_means <- result$out_row_means
  ret
}