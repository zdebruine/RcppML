#' @title Truncated SVD / PCA with constraints and regularization
#'
#' @description
#' Compute a truncated SVD or PCA of a matrix using multiple algorithm backends.
#' Supports sparse and dense matrices, implicit PCA centering, per-element
#' regularization (L1/L2/non-negativity), Gram-level regularization
#' (L2,1/angular/graph Laplacian), missing data, and automatic rank selection
#' via speckled holdout cross-validation.
#'
#' @details
#' \strong{Orientation.} As in \code{\link{nmf}}, \strong{rows are variables
#' (features) and columns are samples}. \code{center} and \code{scale} act on
#' rows. To reproduce \code{stats::prcomp(X)} for a samples-in-rows matrix
#' \code{X}, call \code{pca(t(X))}: then \code{d / sqrt(ncol(A) - 1)} equals
#' \code{prcomp(X)$sdev}, \code{u} holds the loadings (\code{rotation}) and
#' \code{v \%*\% diag(d)} the scores (\code{x}), up to sign.
#'
#' \strong{Precision.} All fits run in single precision. Expect singular values
#' accurate to roughly \code{1e-5} relative to the largest one and
#' \eqn{\|U^T U - I\|} of about \code{1e-5} for the Lanczos-type methods; later
#' deflation components are accurate to about \code{1e-3}. Signs of singular
#' vectors are not normalized.
#'
#' Multiple algorithms are available via the \code{method} parameter:
#' \describe{
#'   \item{\code{"deflation"}}{Sequential rank-1 alternating least squares.
#'     Supports constraints, masks, cross-validation, auto-rank and robust
#'     fitting. This is the method for constrained problems
#'     (\code{nonneg}, \code{L1}): each factor is fit to the data with earlier
#'     constrained directions projected out. \code{maxit} is the number of
#'     iterations per factor and \code{tol} the change in \eqn{1 - |\cos|}
#'     between successive iterates.}
#'   \item{\code{"krylov"}}{Lanczos seed followed by block
#'     solve-then-project refinement of all k factors. Supports
#'     \code{L2}, \code{L21}, \code{angular} and graph penalties, masks and
#'     cross-validation. With \code{nonneg}/\code{L1} and \code{k > 1} it can
#'     return redundant or all-zero components, so \code{"deflation"} is
#'     preferred there. Without constraints or missing data it is identical to
#'     \code{"lanczos"}. \code{maxit} is the number of refinement passes.}
#'   \item{\code{"lanczos"}}{Golub-Kahan-Lanczos bidiagonalization with full
#'     reorthogonalization. Unconstrained. \code{maxit} is the maximum number
#'     of Lanczos steps (default \code{max(8k, k + 120)}, never more than
#'     \code{min(dim(A))}); \code{tol} bounds the Ritz residual relative to the
#'     largest singular value.}
#'   \item{\code{"irlba"}}{Augmented implicitly restarted Lanczos
#'     bidiagonalization (Baglama and Reichel). Unconstrained. \code{maxit} is
#'     the number of restarts (default 1000).}
#'   \item{\code{"randomized"}}{Randomized SVD (Halko, Martinsson and Tropp)
#'     with oversampling \code{max(10, k/5)} and \code{maxit} power iterations
#'     (default 3). Unconstrained; \code{tol} is not used. Approximate by
#'     design: accurate when the spectrum decays, but on flat spectra (for
#'     example a random matrix) singular values come out a few percent low.
#'     Use \code{"lanczos"} or \code{"irlba"} when accuracy matters.}
#' }
#' A warning is issued when a method stops before the requested tolerance is
#' met, naming the number of unconverged values.
#'
#' @section Missing data and cross-validation:
#' Entries selected by \code{mask} (a sparse matrix whose non-zero entries mark
#' missing values) and entries held out by \code{test_fraction} are treated as
#' \emph{missing}: they are imputed from the current model at every iteration,
#' so the fit is the least-squares fit to the observed entries and \code{d} is
#' not biased towards zero. Row means and standard deviations are computed from
#' observed entries only. This is implemented for \code{"deflation"} and
#' \code{"krylov"} on the CPU; other methods raise an error.
#' The holdout set is the same for sparse and dense input; with
#' \code{mask = "zeros"} only non-zero entries can be held out.
#'
#' @section Auto-rank:
#' When \code{k = "auto"}, a speckled holdout set is created and test MSE is
#' evaluated after each factor. Rank selection stops when test MSE fails to
#' improve for \code{patience} consecutive factors. The returned model is
#' truncated to the best rank.
#'
#' @section Constraints and penalties:
#' \describe{
#'   \item{\code{nonneg}, \code{L1}}{Applied to each factor before it is
#'     normalized. \code{L1} is an absolute soft threshold \code{L1 / 2} on the
#'     unnormalized factor, whose entries are on the scale of the data
#'     (\eqn{\sigma u_i} or \eqn{\sigma v_j}); choose it relative to the
#'     singular values. Constrained factors are not orthogonal. For them
#'     \code{d} is the least-squares scale given \code{u} and \code{v}, so
#'     that \code{\link{reconstruct}} returns the fitted model, and
#'     \code{\link{variance_explained}} reports the drop in residual variance
#'     as each component is added.}
#'   \item{\code{L2}}{With \code{"deflation"}, a ridge penalty on both factors
#'     of a rank-one term is equivalent to soft-thresholding its singular
#'     value by \code{sqrt(L2[1] * L2[2])}; a penalty on one factor only has
#'     no effect on a bilinear model and is rejected. With \code{"krylov"} the
#'     penalty is added to the Gram matrix of each block solve.}
#'   \item{\code{L21}}{Group sparsity across factors; \code{"krylov"} only.}
#'   \item{\code{upper_bound}}{Not supported: factors are returned with unit
#'     norm, so an element-wise bound has no scale-free meaning. A positive
#'     value is an error.}
#' }
#'
#' @section Advanced Parameters (via \code{...}):
#' The following parameters can be passed via \code{...}:
#' \describe{
#'   \item{\code{L21}}{L2,1 (group sparsity) penalty. Single value or length-2 vector (default 0).}
#'   \item{\code{angular}}{Angular (orthogonality) penalty. Single value or length-2 vector (default 0).}
#'   \item{\code{graph_U}}{Sparse graph Laplacian for features (m x m). Default NULL.}
#'   \item{\code{graph_V}}{Sparse graph Laplacian for samples (n x n). Default NULL.}
#'   \item{\code{graph_lambda}}{Graph regularization strength. Single value or length-2 (default 0).}
#'   \item{\code{convergence}}{Convergence criterion for \code{"deflation"} and
#'     \code{"krylov"}: \code{"factor"} (default, change in the factors),
#'     \code{"loss"} (relative change in the singular values) or \code{"both"}
#'     (either). Other methods only accept \code{"factor"}.}
#'   \item{\code{cv_seed}}{Separate seed for holdout mask. Default NULL.}
#'   \item{\code{patience}}{Auto-rank non-improving factor patience (default 3).}
#'   \item{\code{k_max}}{Maximum rank for auto-rank mode (default 50).}
#'   \item{\code{threads}}{Number of OpenMP threads. 0 (default) uses
#'     \code{getOption("RcppML.threads")}, or all cores if that is unset.}
#' }
#'
#' @param A Input matrix with variables (features) in rows and samples in
#'   columns. May be dense (\code{matrix}), sparse (\code{dgCMatrix}),
#'   or a path to a \code{.spz} file for out-of-core streaming SVD.
#' @param k Number of factors (rank), at most \code{min(dim(A))}. Use
#'   \code{"auto"} for automatic rank selection via cross-validation.
#'   Default: 10, or \code{min(dim(A))} if that is smaller.
#' @param tol Convergence tolerance; its meaning depends on \code{method}
#'   (see Details). Default: 1e-5. Values below single-precision resolution are
#'   raised to it.
#' @param maxit Iteration limit; its meaning depends on \code{method} (see
#'   Details). Default: 200 iterations per factor for \code{"deflation"};
#'   method-specific defaults otherwise.
#' @param center If \code{TRUE}, subtract row means (PCA mode). Default: \code{FALSE}.
#' @param scale If \code{TRUE}, divide each row by its sample standard deviation
#'   (denominator \code{n - 1}, as in \code{prcomp}) after centering.
#'   Rows without variance are left unscaled. Default: \code{FALSE}.
#' @param verbose Print per-factor diagnostics. Default: \code{FALSE}.
#' @param seed Random seed for initialization and cross-validation. Default: NULL.
#' @param L1 L1 (lasso) penalty, an absolute threshold on the data scale (see
#'   \strong{Constraints and penalties}). Single value or length-2 vector
#'   \code{c(u, v)}. Default: 0.
#' @param L2 L2 (ridge) penalty. Single value or length-2 vector. Default: 0.
#'   With \code{method = "deflation"} it leaves \code{u} and \code{v} unchanged
#'   and subtracts \code{sqrt(L2[1] * L2[2])} from each singular value
#'   (floored at zero), which is what a ridge penalty on both factors of a
#'   rank-one term amounts to; see \strong{Constraints and penalties}.
#' @param nonneg Non-negativity constraints. Single logical or length-2 vector. Default: \code{FALSE}.
#' @param upper_bound Not supported; must be 0.
#' @param test_fraction Fraction of entries to hold out for cross-validation. Default: 0.
#' @param mask Missing-data control. \code{NULL} (default, no masking), a
#'   \code{dgCMatrix} whose non-zero entries mark missing values,
#'   \code{"zeros"} (only non-zero entries can be held out in
#'   cross-validation; requires \code{test_fraction > 0} or \code{k = "auto"}),
#'   or \code{list("zeros", <dgCMatrix>)} to combine both.
#' @param robust Robustness control: \code{FALSE} (default), \code{TRUE} (Huber delta=1.345),
#'   \code{"mae"} (near-MAE), or a positive numeric Huber delta.
#' @param method Algorithm: \code{"auto"} (default), \code{"deflation"}, \code{"krylov"},
#'   \code{"lanczos"}, \code{"irlba"}, \code{"randomized"}. \code{"auto"} picks
#'   \code{"deflation"} for \code{nonneg}/\code{L1}, robust fits and
#'   cross-validation, \code{"krylov"} for \code{L2}/\code{L21} and for a
#'   \code{mask} without cross-validation, and otherwise
#'   \code{"lanczos"} or \code{"irlba"} depending on \code{k} and the matrix size.
#' @param resource Compute backend: \code{"auto"} (default), \code{"cpu"}, or \code{"gpu"}.
#' @param ... Advanced parameters. See \strong{Advanced Parameters} section.
#'
#' @return An S4 object of class \code{svd} with slots:
#' \describe{
#'   \item{\code{u}}{Left singular vectors (one row per variable), m x k matrix}
#'   \item{\code{d}}{Singular values, length-k numeric vector}
#'   \item{\code{v}}{Right singular vectors (one row per sample), n x k matrix}
#'   \item{\code{misc}}{List with metadata: \code{centered}, \code{row_means},
#'     \code{scaled}, \code{row_sds}, \code{test_loss},
#'     \code{iters_per_factor}, \code{wall_time_ms}}
#' }
#'
#' @examples
#' \donttest{
#' library(RcppML)
#' data(iris)
#' # iris has samples in rows; svd() and pca() expect samples in columns
#' A <- t(as.matrix(iris[, 1:4]))
#'
#' # Standard SVD (3 factors)
#' s <- svd(A, k = 3)
#'
#' # PCA with centering (convenience wrapper)
#' p <- pca(A, k = 3)
#'
#' # Auto-rank PCA
#' ar <- svd(A, k = "auto", center = TRUE)
#'
#' # Robust PCA (outlier-resistant)
#' rp <- svd(A, k = 3, center = TRUE, robust = TRUE)
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
  k_missing <- missing(k)

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
  threads       <- .resolve_threads(dots$threads)


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

  is_streaming <- is.character(A) && length(A) == 1 && grepl("\\.spz$", A)

  if (is_streaming) {
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

  min_dim <- if (is_streaming) Inf else min(nrow(A), ncol(A))
  if (!is_streaming && min_dim < 1) stop("'A' must have at least one row and one column")

  # Handle k = "auto"
  auto_rank <- FALSE
  if (is.character(k) && k == "auto") {
    auto_rank <- TRUE
    k <- as.integer(min(k_max, min_dim))
    if (test_fraction <= 0) test_fraction <- 0.05
  } else {
    k <- as.integer(k)
    if (is.na(k) || k < 1) stop("'k' must be >= 1")
    # The default rank is reduced for small matrices; an explicit k is checked
    if (k_missing && k > min_dim) k <- as.integer(min_dim)
    if (k > min_dim) {
      stop(sprintf("'k' (%d) cannot exceed min(nrow(A), ncol(A)) = %d", k, as.integer(min_dim)))
    }
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
  if (any(upper_bound > 0)) {
    stop("'upper_bound' is not supported by svd(): factors are returned with unit norm, ",
         "so an element-wise bound has no scale-free meaning. Use nmf() for bounded factors.",
         call. = FALSE)
  }
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

  # Methods that support constraints and penalties
  constrained_methods <- c("deflation", "krylov")

  # Methods that can treat entries as missing (masks, CV, auto-rank)
  cv_methods <- c("deflation", "krylov")

  # Constraints enforced by projection (sign / sparsity); these are
  # incompatible with orthogonalizing factors after the fact.
  has_projection_constraints <- any(nonneg) || any(L1 > 0)
  has_l2 <- any(L2 > 0)
  l2_one_sided <- has_l2 && !all(L2 > 0)
  has_l21 <- any(L21 > 0)
  has_element_constraints <- has_projection_constraints || has_l2 || has_l21
  has_gram_constraints <- any(angular > 0) ||
                          (!is.null(graph_U) && graph_lambda[1] > 0) ||
                          (!is.null(graph_V) && graph_lambda[2] > 0)
  has_cv <- auto_rank || (test_fraction > 0 && test_fraction < 1)
  has_mask <- !is.null(mask)
  has_missing <- has_cv || has_mask
  has_robust <- robust_delta > 0

  if (mask_zeros && !has_cv) {
    stop("mask = \"zeros\" only restricts the cross-validation holdout set to non-zero ",
         "entries; it has no effect without test_fraction > 0 or k = \"auto\". ",
         "Treating all zeros as missing is not implemented for svd(); ",
         "pass an explicit sparse 'mask' matrix instead.", call. = FALSE)
  }

  if (is_streaming && has_missing && verbose) {
    message("svd(): masks and cross-validation on .spz input are fit by the in-memory ",
            "solvers (the decoded matrix is assembled in memory).")
  }

  # Skip method-feature validation for "auto" — resolved later
  if (method != "auto") {
    if (has_robust && method != "deflation") {
      stop(sprintf(
        "method '%s' does not support robust SVD. Use 'deflation' (or method='auto').",
        method))
    }
    if (has_element_constraints && !method %in% constrained_methods) {
      stop(sprintf(
        "method '%s' does not support constraints (L1/L2/nonneg/L21). Use 'deflation' or 'krylov'.",
        method))
    }

    if (method == "krylov" && has_projection_constraints &&
        (isTRUE(auto_rank) || (is.numeric(k) && k > 1))) {
      warning("method = 'krylov' with nonneg/L1 constraints and k > 1 ",
              "can return redundant or all-zero components; ",
              "method = 'deflation' (the default for constrained problems) is recommended.",
              call. = FALSE)
    }

    if (has_gram_constraints && !method %in% constrained_methods) {
      stop(sprintf(
        "method '%s' does not support Gram-level regularization (angular/graph). Use 'krylov'.",
        method))
    }

    if (has_missing && !method %in% cv_methods) {
      stop(sprintf(
        paste0("method '%s' cannot treat masked or held-out entries as missing ",
               "(mask, test_fraction, k = \"auto\"). Use 'deflation' or 'krylov' ",
               "(or method = 'auto')."),
        method))
    }

    if (convergence != "factor" && !method %in% constrained_methods) {
      stop(sprintf(
        "convergence = '%s' is only implemented for methods 'deflation' and 'krylov', not '%s'.",
        convergence, method))
    }
  }

  # Graph Laplacian validation (skip for streaming — dimensions unknown until C++)
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

  # The GPU entry points take neither 'scale' nor 'convergence'; run those on
  # the CPU instead of dropping the option.
  if (use_gpu && !is_streaming && (scale || convergence != "factor")) {
    message("svd(): scale = TRUE and convergence = \"loss\"/\"both\" are not ",
            "implemented on the GPU backend; using the CPU.")
    use_gpu <- FALSE
  }

  # =========================================================================
  # Auto method selection
  # =========================================================================

  if (method == "auto") {
    if (has_robust) {
      method <- "deflation"
    } else if (has_l21) {
      # Group sparsity across factors needs the block solver
      method <- "krylov"
    } else if (has_projection_constraints) {
      # Sign/sparsity constraints: Krylov's block solve-then-project can
      # return redundant or all-zero components for k > 1; deflation keeps
      # each factor's constraints and fits it with earlier ones projected out.
      method <- "deflation"
    } else if (has_l2) {
      method <- "krylov"
    } else if (has_gram_constraints) {
      method <- if (k >= 8) "krylov" else "deflation"
    } else if (has_cv || convergence != "factor") {
      # Cross-validation / auto-rank: deflation evaluates the test error
      # after every factor and can stop early
      method <- "deflation"
    } else if (has_mask) {
      # Missing entries without cross-validation: the block solver fits all
      # k factors to the observed entries jointly
      method <- "krylov"
    } else if (use_gpu) {
      # GPU three-tier: Lanczos k<32, Randomized 32<=k<64, IRLBA k>=64
      if (k < 32) method <- "lanczos"
      else if (k < 64) method <- "randomized"
      else method <- "irlba"
    } else {
      # CPU method selection: IRLBA for large k or small matrices (where
      # Lanczos Krylov subspace is too constrained), Lanczos otherwise.
      if (k >= 64 || min_dim < 4 * k) {
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

  # Options that rank-one deflation cannot express
  if (method == "deflation") {
    if (has_l21) {
      stop("L21 (group sparsity across factors) is not defined for method = 'deflation', ",
           "which fits one factor at a time. Use method = 'krylov'.", call. = FALSE)
    }
    if (l2_one_sided) {
      stop("With method = 'deflation', L2 must be positive for both factors: a ridge ",
           "penalty on one factor of a rank-one term has no effect (the scale moves to ",
           "the other factor). Give a single value or two positive values, ",
           "or use method = 'krylov'.", call. = FALSE)
    }
  }

  # =========================================================================
  # Dispatch
  # =========================================================================

  # Requests the GPU SVD cannot honor run on the CPU instead of being altered.
  if (use_gpu && !is_streaming) {
    why <- .gpu_svd_unsupported(scale, method, if (is.numeric(k)) k else 2L,
                                nonneg, L1, rep_len(as.numeric(upper_bound), 2),
                                mask = mask,
                                test_fraction = if (has_cv) max(test_fraction, .Machine$double.eps) else 0,
                                L2 = L2, L21 = L21, angular = angular,
                                graph = !is.null(graph_U) || !is.null(graph_V),
                                robust = robust_delta > 0)
    if (!is.null(why)) {
      message("svd(): ", why, " is not implemented on the GPU backend; using the CPU.")
      use_gpu <- FALSE
    }
  }

  # --- Streaming SPZ shortcut ---
  # If A is a .spz file path, use streaming SVD (out-of-core processing).
  # Requires a v2 .spz file with include_transpose=TRUE.
  if (is_streaming) {
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

  # Convergence warnings recorded by the solver
  if (length(result$warnings) > 0) {
    for (w in result$warnings) warning(w, call. = FALSE)
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

  # Constrained deflation (CPU, in memory) returns non-orthogonal factors with
  # least-squares d; train_loss[i] is the observed mean squared error of the
  # nested rank-i refit. The drop in that error is the variance explained by
  # component i, consistent with reconstruct().
  if (!use_gpu && !is_streaming && method == "deflation" && has_projection_constraints &&
      !is.null(result$train_loss) && length(result$train_loss) == length(result$d) &&
      !is.null(result$frobenius_norm_sq) && result$frobenius_norm_sq > 0) {
    mse0 <- result$frobenius_norm_sq / (as.numeric(nrow(A)) * as.numeric(ncol(A)))
    misc$variance_explained <- -diff(c(mse0, result$train_loss)) / mse0
  }

  # Transfer dimnames
  u_mat <- result$u
  v_mat <- result$v
  if (!is_streaming) {
    if (!is.null(rownames(A))) rownames(u_mat) <- rownames(A)
    if (!is.null(colnames(A))) rownames(v_mat) <- colnames(A)
  }

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
#' @details
#' \strong{Rows are variables and columns are samples}, as in
#' \code{\link{nmf}}: each row is centered (and, with \code{scale = TRUE},
#' divided by its sample standard deviation). This is the transpose of the
#' \code{stats::prcomp} convention. For a matrix \code{X} with samples in rows,
#' \code{p <- pca(t(X), k)} gives \code{p@d / sqrt(nrow(X) - 1)} equal to
#' \code{prcomp(X)$sdev[1:k]}, \code{p@u} equal to the loadings
#' (\code{rotation}) and \code{p@v \%*\% diag(p@d)} equal to the scores
#' (\code{x}), up to sign.
#'
#' @inheritParams svd
#' @param ... Additional arguments passed to \code{\link{svd}}.
#' @return An S4 object of class \code{svd} (see \code{\link{svd}}).
#' @seealso \code{\link{svd}}
#' @examples
#' \donttest{
#' library(Matrix)
#' data(aml)   # variables (regions) in rows, samples in columns
#' result <- pca(aml, k = 5)
#' result
#'
#' # A samples-in-rows data set must be transposed first
#' X <- as.matrix(iris[, 1:4])
#' p <- pca(t(X), k = 2)
#' p@d / sqrt(nrow(X) - 1)       # matches prcomp(X)$sdev[1:2]
#' }
#' @export
pca <- function(A, k = 10, ...) {
  if (missing(k)) {
    # let svd() reduce the default rank for small matrices
    svd(A, center = TRUE, ...)
  } else {
    svd(A, k = k, center = TRUE, ...)
  }
}
