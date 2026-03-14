#' @title Truncated SVD / PCA with constraints and regularization
#'
#' @description
#' Compute a truncated SVD or PCA of a matrix using multiple algorithm backends.
#' Supports sparse and dense matrices, implicit PCA centering, per-element
#' regularization (L1/L2/non-negativity/bounds), Gram-level regularization
#' (L2,1/angular/graph Laplacian), and automatic rank selection via speckled
#' holdout cross-validation.
#'
#' @details
#' Multiple algorithms are available via the \code{method} parameter:
#' \describe{
#'   \item{\code{"deflation"}}{Sequential rank-1 ALS with deflation correction.
#'     Supports all constraints/CV/auto-rank. Best for small k with mixed constraints.}
#'   \item{\code{"krylov"}}{Krylov-Seeded Projected Refinement (KSPR). Block
#'     method: computes all k factors simultaneously via Lanczos seed + Gram-solve-then-project.
#'     Faster than deflation for larger k. Supports all regularization except robust. CV is
#'     evaluated post-convergence.}
#'   \item{\code{"lanczos"}}{Unconstrained Lanczos bidiagonalization. Fast for
#'     unconstrained SVD. No regularization support.}
#'   \item{\code{"irlba"}}{Implicitly Restarted Lanczos. Good general-purpose
#'     unconstrained SVD. No regularization support.}
#'   \item{\code{"randomized"}}{Randomized SVD with power iterations. Fast
#'     approximate SVD for large matrices. No regularization support.}
#' }
#'
#' @section Auto-rank:
#' When \code{k = "auto"}, a speckled holdout set is created and test MSE is
#' evaluated after each factor. Rank selection stops when test MSE fails to
#' improve for \code{patience} consecutive factors. The returned model is
#' truncated to the best rank.
#'
#' @section Advanced Parameters (via \code{...}):
#' The following parameters can be passed via \code{...}:
#' \describe{
#'   \item{\code{L21}}{L2,1 (group sparsity) penalty. Single value or length-2 vector (default 0).}
#'   \item{\code{angular}}{Angular (orthogonality) penalty. Single value or length-2 vector (default 0).}
#'   \item{\code{graph_U}}{Sparse graph Laplacian for features (m x m). Default NULL.}
#'   \item{\code{graph_V}}{Sparse graph Laplacian for samples (n x n). Default NULL.}
#'   \item{\code{graph_lambda}}{Graph regularization strength. Single value or length-2 (default 0).}
#'   \item{\code{convergence}}{Convergence criterion: \code{"factor"} (default) or \code{"global"}.}
#'   \item{\code{cv_seed}}{Separate seed for holdout mask. Default NULL.}
#'   \item{\code{patience}}{Auto-rank non-improving factor patience (default 3).}
#'   \item{\code{k_max}}{Maximum rank for auto-rank mode (default 50).}
#'   \item{\code{threads}}{Number of OpenMP threads. 0 = all (default 0).}

#' }
#'
#' @param A Input matrix. May be dense (\code{matrix}), sparse (\code{dgCMatrix}),
#'   or a path to a \code{.spz} file for out-of-core streaming SVD.
#' @param k Number of factors (rank). Use \code{"auto"} for automatic rank selection
#'   via cross-validation. Default: 10.
#' @param tol Convergence tolerance per rank-1 subproblem. Default: 1e-5.
#' @param maxit Maximum ALS iterations per factor. Default: 200.
#' @param center If \code{TRUE}, subtract row means (PCA mode). Default: \code{FALSE}.
#' @param scale If \code{TRUE}, divide each row by its standard deviation after centering. Default: \code{FALSE}.
#' @param verbose Print per-factor diagnostics. Default: \code{FALSE}.
#' @param seed Random seed for initialization and cross-validation. Default: NULL.
#' @param L1 L1 (lasso) penalty. Single value or length-2 vector. Default: 0.
#' @param L2 L2 (ridge) penalty. Single value or length-2 vector. Default: 0.
#' @param nonneg Non-negativity constraints. Single logical or length-2 vector. Default: \code{FALSE}.
#' @param upper_bound Upper bound constraints. Single value or length-2 vector. Default: 0.
#' @param test_fraction Fraction of entries to hold out for cross-validation. Default: 0.
#' @param mask Masking control. \code{NULL} (default, no masking), \code{"zeros"}
#'   (mask zero entries — only non-zero entries can be holdout in CV),
#'   a \code{dgCMatrix} (custom mask matrix where non-zero entries are excluded),
#'   or \code{list("zeros", <dgCMatrix>)} to combine both.
#' @param robust Robustness control: \code{FALSE} (default), \code{TRUE} (Huber delta=1.345),
#'   \code{"mae"} (near-MAE), or a positive numeric Huber delta.
#' @param method Algorithm: \code{"auto"} (default), \code{"deflation"}, \code{"krylov"},
#'   \code{"lanczos"}, \code{"irlba"}, \code{"randomized"}.
#' @param resource Compute backend: \code{"auto"} (default), \code{"cpu"}, or \code{"gpu"}.
#' @param ... Advanced parameters. See \strong{Advanced Parameters} section.
#'
#' @return An S4 object of class \code{svd} with slots:
#' \describe{
#'   \item{\code{u}}{Left singular vectors (scores), m x k matrix}
#'   \item{\code{d}}{Singular values, length-k numeric vector}
#'   \item{\code{v}}{Right singular vectors (loadings), n x k matrix}
#'   \item{\code{misc}}{List with metadata: \code{centered}, \code{row_means},
#'     \code{test_loss}, \code{iters_per_factor}, \code{wall_time_ms}}
#' }
#'
#' @examples
#' \donttest{
#' library(RcppML)
#' data(iris)
#' A <- as.matrix(iris[, 1:4])
#'
#' # Standard SVD (3 factors)
#' s <- svd(A, k = 3)
#'
#' # PCA with centering (convenience wrapper)
#' p <- pca(A, k = 3)
#'
#' # Auto-rank PCA
#' ar <- svd(A, k = "auto", center = TRUE, verbose = TRUE)
#'
#' # Robust PCA (outlier-resistant)
#' rp <- svd(A, k = 3, center = TRUE, robust = TRUE, verbose = TRUE)
#' }
#'
#' @note This function shadows \code{base::svd()}. To use the base R version,
#' call \code{base::svd()} explicitly.
#'
#' @seealso \code{\link{nmf}}, \code{\link{pca}}
#' @export
svd <- function(A, k = 10, tol = 1e-5, maxit = 200,
                center = FALSE, scale = FALSE, verbose = FALSE,
                seed = NULL,
                L1 = 0, L2 = 0, nonneg = FALSE, upper_bound = 0,
                test_fraction = 0,
                mask = NULL, robust = FALSE,
                resource = "auto", method = "auto",
                ...) {

  # Track whether user explicitly set maxit (for method-specific defaults)
  maxit_missing <- missing(maxit)

  # --- Parse advanced parameters from ... ---
  dots <- .parse_svd_dots(list(...))
  L21           <- dots$L21
  angular       <- dots$angular
  graph_U       <- dots$graph_U
  graph_V       <- dots$graph_V
  graph_lambda  <- dots$graph_lambda
  convergence   <- dots$convergence
  cv_seed       <- dots$cv_seed
  patience      <- dots$patience
  k_max         <- dots$k_max
  threads       <- dots$threads


  # Convert NULL seed/cv_seed to 0L (C++ convention: 0 = random)
  if (is.null(seed)) seed <- 0L
  if (is.null(cv_seed)) cv_seed <- 0L

  # =========================================================================
  # Method-specific defaults
  # =========================================================================

  # Lanczos/irlba/krylov pass maxit=0 as sentinel → let C++ pick defaults
  # For "auto", defer defaults until method is resolved (after use_gpu)
  if (method != "auto") {
    if (method == "lanczos" && maxit_missing) maxit <- 0L
    if (method == "randomized" && maxit_missing) maxit <- 3L
    if (method == "irlba" && maxit_missing) maxit <- 0L
    if (method == "irlba" && missing(tol)) tol <- 1e-5
    if (method == "krylov" && maxit_missing) maxit <- 0L
  }

  # =========================================================================
  # Validate method
  # =========================================================================

  valid_methods <- c("auto", "deflation", "krylov", "lanczos", "irlba", "randomized")
  if (!method %in% valid_methods) {
    stop(sprintf("method must be one of: %s", paste(valid_methods, collapse = ", ")))
  }

  # =========================================================================
  # Input validation
  # =========================================================================

  if (is.character(A) && length(A) == 1 && grepl("\\.spz$", A)) {
    # SPZ file path — streaming mode, validate later
  } else if (inherits(A, "dgCMatrix")) {
    # Good: sparse
  } else if (is.matrix(A)) {
    if (!is.numeric(A)) stop("'A' must be a numeric matrix")
  } else if (inherits(A, "sparseMatrix")) {
    A <- as(A, "dgCMatrix")
  } else {
    stop("'A' must be a matrix, dgCMatrix, or path to a .spz file")
  }

  # Handle k = "auto"
  auto_rank <- FALSE
  if (is.character(k) && k == "auto") {
    auto_rank <- TRUE
    k <- k_max
    if (test_fraction <= 0) test_fraction <- 0.05
  } else {
    k <- as.integer(k)
    if (k < 1) stop("'k' must be >= 1")
  }

  # Scale requires center (correlation PCA = center + scale)
  if (scale && !center) {
    center <- TRUE
    if (verbose) message("Setting center=TRUE (required for scale=TRUE)")
  }

  # Expand regularization parameters to length 2
  L1 <- rep_len(as.numeric(L1), 2)
  L2 <- rep_len(as.numeric(L2), 2)
  nonneg <- rep_len(as.logical(nonneg), 2)
  upper_bound <- rep_len(as.numeric(upper_bound), 2)
  L21 <- rep_len(as.numeric(L21), 2)
  angular <- rep_len(as.numeric(angular), 2)
  graph_lambda <- rep_len(as.numeric(graph_lambda), 2)

  # Validate ranges
  if (any(L1 < 0)) stop("L1 penalties must be non-negative")
  if (any(L2 < 0)) stop("L2 penalties must be non-negative")
  if (any(upper_bound < 0)) stop("upper_bound must be non-negative")
  if (any(L21 < 0)) stop("L21 penalties must be non-negative")
  if (any(angular < 0)) stop("angular penalties must be non-negative")
  if (any(graph_lambda < 0)) stop("graph_lambda must be non-negative")
  if (tol < 0) stop("'tol' must be non-negative")
  if ((method %in% c("lanczos", "irlba", "krylov")) && maxit == 0L) {
    # sentinel value: let C++ pick defaults
  } else if (maxit < 1) {
    stop("'maxit' must be >= 1")
  }
  if (test_fraction < 0 || test_fraction >= 1) stop("'test_fraction' must be in [0, 1)")
  if (patience < 1) stop("'patience' must be >= 1")
  if (!convergence %in% c("factor", "loss", "both")) {
    stop("'convergence' must be 'factor', 'loss', or 'both'")
  }

  # Parse robust parameter: FALSE=0, TRUE=1.345, "mae"=1e-4, numeric=value
  if (is.logical(robust)) {
    robust_delta <- if (isTRUE(robust)) 1.345 else 0.0
  } else if (is.character(robust) && tolower(robust) == "mae") {
    robust_delta <- 1e-4
  } else if (is.numeric(robust)) {
    robust_delta <- as.double(robust)
  } else {
    stop("'robust' must be FALSE, TRUE, 'mae', or a positive numeric Huber delta.", call. = FALSE)
  }

  # Validate mask — derive mask_zeros and mask_matrix from mask parameter
  # mask accepts: NULL, "zeros", dgCMatrix, or list("zeros", <dgCMatrix>)
  mask_zeros <- FALSE
  mask_matrix <- NULL
  if (!is.null(mask)) {
    if (is.list(mask) && !inherits(mask, "Matrix")) {
      # list("zeros", <matrix>) — both mask_zeros and custom mask
      if (length(mask) < 2 || !is.character(mask[[1]]) || mask[[1]] != "zeros")
        stop("'mask' list must be list(\"zeros\", <matrix>)")
      mask_zeros <- TRUE
      mask_matrix <- as(mask[[2]], "dgCMatrix")
    } else if (is.character(mask)) {
      if (mask == "zeros") {
        mask_zeros <- TRUE
      } else {
        stop("'mask' string must be \"zeros\". Got: '", mask, "'")
      }
    } else {
      # Assume it's a matrix
      if (!inherits(mask, "dgCMatrix")) {
        if (inherits(mask, "sparseMatrix")) {
          mask_matrix <- as(mask, "dgCMatrix")
        } else {
          stop("'mask' must be NULL, 'zeros', a dgCMatrix, or list(\"zeros\", <dgCMatrix>)")
        }
      } else {
        mask_matrix <- mask
      }
    }
    # Dimension check for in-memory matrices when a custom mask is provided
    if (!is.null(mask_matrix) && !is.character(A)) {
      if (nrow(mask_matrix) != nrow(A) || ncol(mask_matrix) != ncol(A)) {
        stop(sprintf("'mask' dimensions (%d x %d) must match 'A' (%d x %d)",
                     nrow(mask_matrix), ncol(mask_matrix), nrow(A), ncol(A)))
      }
    }
  }
  # Set mask to the matrix (or NULL) for passing to C++
  mask <- mask_matrix

  # =========================================================================
  # Method-feature validation matrix
  # =========================================================================

  # Methods that support element-wise constraints (L1, L2, nonneg, bounds, L21)
  constrained_methods <- c("deflation", "krylov")

  # Methods that support Gram-level regularization (angular, graph)
  gram_methods <- c("krylov", "deflation")

  # Methods that support CV / auto-rank
  cv_methods <- c("deflation", "krylov")

  has_element_constraints <- any(L1 > 0) || any(L2 > 0) || any(nonneg) ||
                             any(upper_bound > 0) || any(L21 > 0)
  has_gram_constraints <- any(angular > 0) ||
                          (!is.null(graph_U) && graph_lambda[1] > 0) ||
                          (!is.null(graph_V) && graph_lambda[2] > 0)
  has_cv <- auto_rank || (test_fraction > 0 && test_fraction < 1)
  has_robust <- robust_delta > 0

  # Skip method-feature validation for "auto" — resolved later
  if (method != "auto") {
    if (has_robust && method != "deflation") {
      stop(sprintf(
        "method '%s' does not support robust SVD. Use 'deflation' (or method='auto').",
        method))
    }
    if (has_element_constraints && !method %in% constrained_methods) {
      stop(sprintf(
        "method '%s' does not support constraints (L1/L2/nonneg/bounds/L21). Use 'deflation' or 'krylov'.",
        method))
    }

    if (has_gram_constraints && !method %in% gram_methods) {
      stop(sprintf(
        "method '%s' does not support Gram-level regularization (angular/graph). Use 'krylov'.",
        method))
    }

    if (has_cv && !method %in% cv_methods) {
      if (auto_rank) {
        stop(sprintf(
          "method '%s' does not support auto-rank. Use 'deflation' or 'krylov'.",
          method))
      }
      # For non-auto-rank, just disable CV silently
      test_fraction <- 0
    }
  }

  # Graph Laplacian validation (skip for streaming — dimensions unknown until C++)
  is_streaming <- is.character(A) && length(A) == 1 && grepl("\\.spz$", A)
  if (!is.null(graph_U) && !is_streaming) {
    if (!inherits(graph_U, "sparseMatrix")) stop("graph_U must be a sparse matrix")
    if (nrow(graph_U) != nrow(A) || ncol(graph_U) != nrow(A)) {
      stop(sprintf("graph_U must be %d x %d (matching rows of A)", nrow(A), nrow(A)))
    }
    graph_U <- as(graph_U, "dgCMatrix")
  }
  if (!is.null(graph_V) && !is_streaming) {
    if (!inherits(graph_V, "sparseMatrix")) stop("graph_V must be a sparse matrix")
    if (nrow(graph_V) != ncol(A) || ncol(graph_V) != ncol(A)) {
      stop(sprintf("graph_V must be %d x %d (matching columns of A)", ncol(A), ncol(A)))
    }
    graph_V <- as(graph_V, "dgCMatrix")
  }

  # =========================================================================
  # Resource resolution
  # =========================================================================

  if (resource == "auto") {
    env_res <- Sys.getenv("RCPPML_RESOURCE", "auto")
    if (env_res != "auto") {
      resource <- tolower(env_res)
    } else {
      gpu_opt <- getOption("RcppML.gpu", "auto")
      if (isTRUE(gpu_opt)) {
        resource <- "gpu"
      } else if (identical(gpu_opt, FALSE)) {
        resource <- "cpu"
      }
    }
  }
  if (!resource %in% c("auto", "cpu", "gpu")) {
    stop("'resource' must be \"auto\", \"cpu\", or \"gpu\"")
  }

  use_gpu <- FALSE
  if (resource == "gpu") {
    if (gpu_available()) {
      use_gpu <- TRUE
    } else {
      warning("GPU requested but not available, falling back to CPU")
    }
  } else if (resource == "auto" && gpu_available() && inherits(A, "dgCMatrix")) {
    use_gpu <- TRUE
  }

  # =========================================================================
  # Auto method selection (mirrors auto_select.hpp logic)
  # =========================================================================

  if (method == "auto") {
    has_any_constraints <- has_element_constraints || has_gram_constraints
    if (has_robust) {
      method <- "deflation"
    } else if (has_any_constraints) {
      method <- if (k >= 8) "krylov" else "deflation"
    } else if (has_cv || auto_rank) {
      # CV / auto-rank only supported by deflation and krylov
      method <- "deflation"
    } else if (use_gpu) {
      # GPU three-tier: Lanczos k<32, Randomized 32<=k<64, IRLBA k>=64
      if (k < 32) method <- "lanczos"
      else if (k < 64) method <- "randomized"
      else method <- "irlba"
    } else {
      # CPU method selection: IRLBA for large k or small matrices (where
      # Lanczos Krylov subspace is too constrained), Lanczos otherwise.
      mat_dim <- if (!is.character(A)) min(nrow(A), ncol(A)) else Inf
      if (k >= 64 || mat_dim < 4 * k) {
        method <- "irlba"
      } else {
        method <- "lanczos"
      }
    }

    # Apply method-specific defaults for the resolved method
    if (maxit_missing) {
      if (method %in% c("lanczos", "irlba", "krylov")) maxit <- 0L
      else if (method == "randomized") maxit <- 3L
    }

    if (verbose) {
      message(sprintf("  auto method: %s (k=%d, gpu=%s)", method, k, use_gpu))
    }
  }

  # =========================================================================
  # Dispatch
  # =========================================================================

  # --- Streaming SPZ shortcut ---
  # If A is a .spz file path, use streaming SVD (out-of-core processing).
  # Requires a v2 .spz file with include_transpose=TRUE.
  if (is.character(A) && length(A) == 1 && grepl("\\.spz$", A)) {
    if (!file.exists(A)) stop("SPZ file not found: ", A)

    result <- Rcpp_svd_streaming_spz(
      path = A,
      k_max = as.integer(k),
      tol = as.double(tol),
      max_iter = as.integer(maxit),
      center = as.logical(center),
      scale = as.logical(scale),
      verbose = as.logical(verbose),
      seed = as.integer(seed),
      threads = as.integer(threads),
      L1_u = L1[1], L1_v = L1[2],
      L2_u = L2[1], L2_v = L2[2],
      nonneg_u = nonneg[1], nonneg_v = nonneg[2],
      upper_bound_u = upper_bound[1], upper_bound_v = upper_bound[2],
      L21_u = L21[1], L21_v = L21[2],
      angular_u = angular[1], angular_v = angular[2],
      graph_u = graph_U,
      graph_v = graph_V,
      graph_u_lambda = graph_lambda[1],
      graph_v_lambda = graph_lambda[2],
      test_fraction = as.double(test_fraction),
      cv_seed = as.integer(cv_seed),
      patience = as.integer(patience),
      mask_zeros = as.logical(mask_zeros),
      obs_mask = mask,
      convergence = convergence,
      method = method,
      robust_delta = as.double(robust_delta),
      irls_max_iter = 5L,
      irls_tol = 1e-4
    )
  } else if (use_gpu && inherits(A, "dgCMatrix")) {
    # GPU sparse path (CSC input, cuSPARSE SpMV/SpMM)
    result <- .gpu_svd_pca(
      A = A,
      k_max = as.integer(k),
      tol = as.double(tol),
      max_iter = as.integer(maxit),
      center = as.logical(center),
      verbose = as.logical(verbose),
      seed = as.integer(seed),
      threads = as.integer(threads),
      L1_u = L1[1], L1_v = L1[2],
      L2_u = L2[1], L2_v = L2[2],
      nonneg_u = nonneg[1], nonneg_v = nonneg[2],
      upper_bound_u = upper_bound[1], upper_bound_v = upper_bound[2],
      L21_u = L21[1], L21_v = L21[2],
      angular_u = angular[1], angular_v = angular[2],
      test_fraction = as.double(test_fraction),
      cv_seed = as.integer(cv_seed),
      patience = as.integer(patience),
      mask_zeros = as.logical(mask_zeros),
      method = method,
      graph_U = graph_U, graph_V = graph_V,
      graph_lambda = graph_lambda,
      obs_mask = mask,
      robust_delta = as.double(robust_delta),
      irls_max_iter = 5L,
      irls_tol = 1e-4
    )
  } else if (use_gpu && is.matrix(A)) {
    # GPU dense path (column-major double matrix, cuBLAS GEMV/GEMM for deflation)
    result <- .gpu_svd_pca_dense(
      A = A,
      k_max = as.integer(k),
      tol = as.double(tol),
      max_iter = as.integer(maxit),
      center = as.logical(center),
      verbose = as.logical(verbose),
      seed = as.integer(seed),
      threads = as.integer(threads),
      L1_u = L1[1], L1_v = L1[2],
      L2_u = L2[1], L2_v = L2[2],
      nonneg_u = nonneg[1], nonneg_v = nonneg[2],
      upper_bound_u = upper_bound[1], upper_bound_v = upper_bound[2],
      L21_u = L21[1], L21_v = L21[2],
      angular_u = angular[1], angular_v = angular[2],
      test_fraction = as.double(test_fraction),
      cv_seed = as.integer(cv_seed),
      patience = as.integer(patience),
      mask_zeros = as.logical(mask_zeros),
      method = method,
      graph_U = graph_U, graph_V = graph_V,
      graph_lambda = graph_lambda,
      obs_mask = mask,
      robust_delta = as.double(robust_delta),
      irls_max_iter = 5L,
      irls_tol = 1e-4
    )
  } else {
    # CPU path (Rcpp)
    result <- Rcpp_svd_pca(
      A_sexp = A,
      k_max = as.integer(k),
      tol = as.double(tol),
      max_iter = as.integer(maxit),
      center = as.logical(center),
      scale = as.logical(scale),
      verbose = as.logical(verbose),
      seed = as.integer(seed),
      threads = as.integer(threads),
      L1_u = L1[1], L1_v = L1[2],
      L2_u = L2[1], L2_v = L2[2],
      nonneg_u = nonneg[1], nonneg_v = nonneg[2],
      upper_bound_u = upper_bound[1], upper_bound_v = upper_bound[2],
      L21_u = L21[1], L21_v = L21[2],
      angular_u = angular[1], angular_v = angular[2],
      graph_u = graph_U,
      graph_v = graph_V,
      graph_u_lambda = graph_lambda[1],
      graph_v_lambda = graph_lambda[2],
      test_fraction = as.double(test_fraction),
      cv_seed = as.integer(cv_seed),
      patience = as.integer(patience),
      mask_zeros = as.logical(mask_zeros),
      obs_mask = mask,
      convergence = convergence,
      method = method,
      robust_delta = as.double(robust_delta),
      irls_max_iter = 5L,
      irls_tol = 1e-4
    )
  }

  # =========================================================================
  # Construct S4 object
  # =========================================================================

  misc <- list(
    centered = result$centered,
    scaled = isTRUE(result$scaled),
    iters_per_factor = result$iters_per_factor,
    wall_time_ms = result$wall_time_ms,
    auto_rank = auto_rank,
    method = method
  )
  if (!is.null(result$k)) misc$k_selected <- result$k
  if (!is.null(result$row_means)) misc$row_means <- result$row_means
  if (!is.null(result$row_sds)) misc$row_sds <- result$row_sds
  if (!is.null(result$test_loss)) misc$test_loss <- result$test_loss
  if (!is.null(result$train_loss)) misc$train_loss <- result$train_loss
  if (!is.null(result$frobenius_norm_sq)) misc$frobenius_norm_sq <- result$frobenius_norm_sq

  # Transfer dimnames
  u_mat <- result$u
  v_mat <- result$v
  if (!is.null(rownames(A))) rownames(u_mat) <- rownames(A)
  if (!is.null(colnames(A))) rownames(v_mat) <- colnames(A)

  new("svd",
      u = u_mat,
      d = as.numeric(result$d),
      v = v_mat,
      misc = misc)
}

#' @title PCA (centered SVD)
#'
#' @description
#' Convenience wrapper around \code{\link{svd}} with \code{center = TRUE}.
#'
#' @inheritParams svd
#' @param ... Additional arguments passed to \code{\link{svd}}.
#' @return An S4 object of class \code{svd} (see \code{\link{svd}}).
#' @seealso \code{\link{svd}}
#' @examples
#' \donttest{
#' library(Matrix)
#' data(aml)
#' result <- pca(aml, k = 5)
#' result
#' }
#' @export
pca <- function(A, k = 10, ...) {
  svd(A, k = k, center = TRUE, ...)
}
