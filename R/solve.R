#' @title Non-negative Least Squares Projection
#'
#' @description Project a factor matrix onto new data to solve for the complementary matrix.
#' Given \eqn{W} and \eqn{A}, find \eqn{H} such that \eqn{||WH - A||^2} is minimized.
#' Alternatively, given \eqn{H} and \eqn{A}, find \eqn{W} such that \eqn{||WH - A||^2} is minimized.
#'
#' @details
#' This function solves NNLS projection problems with full flexibility.
#'
#' **Projection Modes:**
#' - \code{w} provided, \code{h = NULL}: Solve for H given W and A (standard projection)
#' - \code{h} provided, \code{w = NULL}: Solve for W given H and A (transpose projection)
#' - Exactly one of \code{w} or \code{h} must be NULL
#'
#' @section Advanced Parameters (via \code{...}):
#' \describe{
#'   \item{\code{L21}}{L2,1 group sparsity penalty, length 1 or 2 (default \code{c(0,0)})}
#'   \item{\code{angular}}{Angular decorrelation penalty, length 1 or 2 (default \code{c(0,0)})}
#'   \item{\code{cd_maxit}}{Max coordinate descent iterations (default 100)}
#'   \item{\code{cd_tol}}{CD stopping tolerance (default 1e-8)}
#'   \item{\code{warm_start}}{Optional initial solution matrix}
#'   \item{\code{dispersion}}{Dispersion estimation mode: \code{"per_row"} (default) or \code{"global"}}
#'   \item{\code{theta_init}}{Initial GP theta (default 0.1)}
#'   \item{\code{theta_max}}{Maximum GP theta (default 5.0)}
#'   \item{\code{theta_min}}{Minimum GP theta (default 0.0)}
#'   \item{\code{nb_size_init}}{Initial NB size parameter (default 10.0)}
#'   \item{\code{nb_size_max}}{Maximum NB size (default 1e6)}
#'   \item{\code{nb_size_min}}{Minimum NB size (default 0.01)}
#'   \item{\code{gamma_phi_init}}{Initial Gamma shape parameter (default 1.0)}
#'   \item{\code{gamma_phi_max}}{Maximum Gamma shape (default 1e4)}
#'   \item{\code{gamma_phi_min}}{Minimum Gamma shape (default 1e-6)}
#'   \item{\code{tweedie_power}}{Tweedie variance power (default 1.5)}
#'   \item{\code{irls_max_iter}}{Maximum IRLS iterations per solve (default 5)}
#'   \item{\code{irls_tol}}{IRLS convergence tolerance (default 1e-4)}
#' }
#'
#' @param w factor model matrix (n_features x k) for solving H. Set to NULL to solve for W.
#' @param h factor model matrix (k x n_samples) for solving W. Set to NULL to solve for H.
#' @param A data matrix. Can be dense (matrix) or sparse (dgCMatrix).
#' @param L1 L1/LASSO penalty, length 1 or 2. Range [0, 1).
#' @param L2 Ridge penalty, length 1 or 2. Range [0, Inf).
#' @param loss loss function: \code{"mse"} (default) or others via NMF dispatch.
#' @param upper_bound maximum value in solution, length 1 or 2. 0 = no bound.
#' @param nonneg non-negativity constraints, length 1 or 2.
#' @param threads number of threads for OpenMP parallelization (0 = all available).
#' @param verbose print progress information.
#' @param ... advanced parameters. See \strong{Advanced Parameters} section.
#' @return matrix of dimension (k x n_samples) for H or (n_features x k) for W
#'
#' @section Target Regularization:
#' When \code{target_H} and \code{target_lambda} are provided (via \code{...}),
#' \code{nnls()} routes the solve through a single NMF iteration internally.
#' Positive \code{target_lambda} attracts the solution toward \code{target_H}
#' (label enrichment).  Negative \code{target_lambda} uses eigenvalue-projected
#' adversarial removal (PROJ_ADV) to suppress target-correlated structure
#' (batch removal).  See \code{vignette("guided-nmf")} for details.
#'
#' @export
#' @author Zach DeBruine
#' @seealso \code{\link{nmf}}
#' @md
#'
#' @references
#'
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#'
#' Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem."
#'
#' @examples
#' \donttest{
#' # Generate synthetic NMF problem
#' set.seed(123)
#' w <- matrix(runif(100), 20, 5)  # 20 features, 5 factors
#' h_true <- matrix(runif(50), 5, 10)  # 5 factors, 10 samples
#' A <- w %*% h_true + matrix(rnorm(200, 0, 0.1), 20, 10)  # Add noise
#'
#' # Project W onto new data to find H
#' h_recovered <- nnls(w = w, A = A)
#' cor(as.vector(h_true), as.vector(h_recovered))
#'
#' # With L1 penalty for sparse H
#' h_sparse <- nnls(w = w, A = A, L1 = c(0, 0.1))
#' }
nnls <- function(w = NULL, h = NULL, A, 
                 L1 = c(0, 0), L2 = c(0, 0),
                 loss = "mse",
                 upper_bound = c(0, 0), nonneg = c(TRUE, TRUE),
                 threads = 0, verbose = FALSE,
                 ...) {
  
  # --- Parse advanced parameters from ... ---
  dots <- .parse_nnls_dots(list(...))
  L21        <- dots$L21
  angular    <- dots$angular
  cd_maxit   <- dots$cd_maxit
  cd_tol     <- dots$cd_tol
  warm_start <- dots$warm_start
  
  # === BACKWARD COMPATIBILITY ===
  # Old CRAN 0.3.7 API: nnls(factor_matrix, data_matrix) with 2 positional args
  # In the new API: w=factor, h=data, A=missing → detect and redirect
  if (!is.null(w) && !is.null(h) && missing(A)) {
    .Deprecated(msg = paste0(
      "nnls(w, A) 2-positional-arg form is deprecated.\n",
      "Use nnls(w = ..., A = ...) to solve for H, or ",
      "nnls(h = ..., A = ...) to solve for W."))
    A <- h
    h <- NULL
  }
  
  # === PARAMETER VALIDATION ===
  
  # Exactly one of w or h must be provided
  if (is.null(w) && is.null(h)) {
    stop("Either 'w' or 'h' must be provided (not both NULL)")
  }
  if (!is.null(w) && !is.null(h)) {
    stop("Exactly one of 'w' or 'h' must be NULL (cannot provide both)")
  }
  
  # Determine projection mode
  solve_for_h <- !is.null(w)
  factor_matrix <- if (solve_for_h) w else h
  
  # Recycle penalty parameters to length 2 early (needed for NMF dispatch)
  if (length(L1) == 1) L1 <- c(L1, L1)
  if (length(L2) == 1) L2 <- c(L2, L2)
  if (length(L21) == 1) L21 <- c(L21, L21)
  if (length(angular) == 1) angular <- c(angular, angular)
  if (length(upper_bound) == 1) upper_bound <- c(upper_bound, upper_bound)
  if (length(nonneg) == 1) nonneg <- c(nonneg, nonneg)
  
  # === TARGET / NON-MSE LOSS DISPATCH ===
  # For targets or non-MSE distributions (IRLS), delegate to a single NMF iteration
  target_H <- dots$target_H
  target_lambda <- dots$target_lambda
  has_target <- !is.null(target_H) && any(target_lambda != 0)
  if (loss != "mse" || any(L21 != 0) || any(angular != 0) || has_target) {
    if (!is.matrix(factor_matrix)) factor_matrix <- as.matrix(factor_matrix)
    if (is.character(A)) {
      data_info <- validate_data(A)
      A <- data_info$data
    }
    
    if (solve_for_h) {
      model <- nmf(A, k = ncol(factor_matrix), seed = factor_matrix, maxit = 1,
                   loss = loss, L1 = L1, L2 = L2, L21 = L21, angular = angular,
                   upper_bound = upper_bound, nonneg = nonneg,
                   threads = threads, verbose = verbose,
                   target_H = target_H, target_lambda = target_lambda,
                   dispersion = dots$dispersion,
                   theta_init = dots$theta_init,
                   theta_max = dots$theta_max,
                   theta_min = dots$theta_min,
                   nb_size_init = dots$nb_size_init,
                   nb_size_max = dots$nb_size_max,
                   nb_size_min = dots$nb_size_min,
                   gamma_phi_init = dots$gamma_phi_init,
                   gamma_phi_max = dots$gamma_phi_max,
                   gamma_phi_min = dots$gamma_phi_min,
                   tweedie_power = dots$tweedie_power,
                   irls_max_iter = dots$irls_max_iter,
                   irls_tol = dots$irls_tol)
      return(model@h)
    } else {
      model <- nmf(A, k = nrow(factor_matrix), seed = t(A) %*% t(t(factor_matrix)), maxit = 1,
                   loss = loss, L1 = L1, L2 = L2, L21 = L21, angular = angular,
                   upper_bound = upper_bound, nonneg = nonneg,
                   threads = threads, verbose = verbose,
                   target_H = target_H, target_lambda = target_lambda,
                   dispersion = dots$dispersion,
                   theta_init = dots$theta_init,
                   theta_max = dots$theta_max,
                   theta_min = dots$theta_min,
                   nb_size_init = dots$nb_size_init,
                   nb_size_max = dots$nb_size_max,
                   nb_size_min = dots$nb_size_min,
                   gamma_phi_init = dots$gamma_phi_init,
                   gamma_phi_max = dots$gamma_phi_max,
                   gamma_phi_min = dots$gamma_phi_min,
                   tweedie_power = dots$tweedie_power,
                   irls_max_iter = dots$irls_max_iter,
                   irls_tol = dots$irls_tol)
      return(model@w)
    }
  }
  
  # === STREAMING DISPATCH (SPZ file path) ===
  # If A is a file path to .spz, use streaming NNLS (avoids loading full A)
  if (is.character(A) && length(A) == 1 && grepl("\\.spz$", A) && solve_for_h) {
    if (!is.matrix(factor_matrix)) factor_matrix <- as.matrix(factor_matrix)
    
    h <- c_nnls_streaming(normalizePath(A), factor_matrix,
                          as.integer(cd_maxit), cd_tol,
                          L1[2], L2[2], upper_bound[2],
                          nonneg[2], as.integer(threads))
    rownames(h) <- if (!is.null(colnames(factor_matrix))) {
      colnames(factor_matrix)
    } else {
      paste0("nmf", 1:nrow(h))
    }
    return(h)
  }
  
  # Unified input validation for A (supports file paths, sparse, dense)
  if (!is.matrix(factor_matrix)) factor_matrix <- as.matrix(factor_matrix)
  data_info <- validate_data(A)
  A <- data_info$data

  # Validate penalty ranges
  if (any(L1 < 0) || any(L1 >= 1)) stop("L1 penalty must be in range [0, 1)")
  if (any(L2 < 0)) stop("L2 penalty must be >= 0")
  if (any(upper_bound < 0)) stop("upper_bound must be >= 0")
  
  # === DIMENSION MATCHING AND TRANSPOSITION ===
  
  if (solve_for_h) {
    # Solving for H: need nrow(w) == nrow(A)
    target_dim <- nrow(factor_matrix)
    data_dim <- nrow(A)
    dim_name <- "rows"
    factor_names <- rownames(factor_matrix)
    data_names <- rownames(A)
  } else {
    # Solving for W: need ncol(h) == ncol(A) (both are n = number of samples)
    target_dim <- ncol(factor_matrix)
    data_dim <- ncol(A)
    dim_name <- "columns"
    factor_names <- colnames(factor_matrix)
    data_names <- colnames(A)
  }
  
  # Try dimension matching
  if (target_dim != data_dim) {
    # Try auto-transpose (like predict.nmf does)
    if (solve_for_h && ncol(factor_matrix) == data_dim) {
      if (verbose) cat("Auto-transposing w to match A dimensions\n")
      factor_matrix <- t(factor_matrix)
      target_dim <- nrow(factor_matrix)
      factor_names <- rownames(factor_matrix)
    } else if (!solve_for_h && nrow(factor_matrix) == data_dim) {
      if (verbose) cat("Auto-transposing h to match A dimensions\n")
      factor_matrix <- t(factor_matrix)
      target_dim <- ncol(factor_matrix)
      factor_names <- colnames(factor_matrix)
    }
    
    # If still don't match, try name-based matching
    if (target_dim != data_dim) {
      if (!is.null(factor_names) && !is.null(data_names)) {
        # Find intersection
        common_names <- intersect(factor_names, data_names)
        
        if (length(common_names) == 0) {
          stop("Incompatible dimensions and no overlapping names found.\n",
               "  Factor matrix ", dim_name, ": ", target_dim, "\n",
               "  Data matrix ", dim_name, ": ", data_dim)
        }
        
        if (length(common_names) < min(target_dim, data_dim)) {
          warning("Only ", length(common_names), " of ", 
                  min(target_dim, data_dim), 
                  " names overlap. Subsetting to intersection.")
        }
        
        if (verbose) {
          cat("Matching by names: using", length(common_names), "common", dim_name, "\n")
        }
        
        # Reorder both matrices to match
        if (solve_for_h) {
          factor_matrix <- factor_matrix[common_names, , drop = FALSE]
          A <- A[common_names, , drop = FALSE]
        } else {
          factor_matrix <- factor_matrix[, common_names, drop = FALSE]
          A <- A[common_names, , drop = FALSE]
        }
      } else {
        stop("Incompatible dimensions: ",
             if (solve_for_h) "nrow(w)" else "ncol(h)", " = ", target_dim,
             " but ", if (solve_for_h) "nrow(A)" else "ncol(A)", " = ", data_dim, 
             " and no rownames/colnames available for matching")
      }
    }
  }
  
  # === WARM START VALIDATION ===
  
  if (!is.null(warm_start)) {
    if (!is.matrix(warm_start)) warm_start <- as.matrix(warm_start)
    
    expected_rows <- if (solve_for_h) ncol(factor_matrix) else nrow(A)
    expected_cols <- if (solve_for_h) ncol(A) else ncol(factor_matrix)
    
    if (nrow(warm_start) != expected_rows || ncol(warm_start) != expected_cols) {
      stop("warm_start dimensions (", nrow(warm_start), " x ", ncol(warm_start), 
           ") don't match expected output dimensions (", 
           expected_rows, " x ", expected_cols, ")")
    }
  }
  
  # === CALL C++ BACKEND ===
  
  if (solve_for_h) {
    # Standard mode: solve for H given W
    penalty_L1 <- L1[2]  # H penalty
    penalty_L2 <- L2[2]
    penalty_upper <- upper_bound[2]
    constraint_nonneg <- nonneg[2]
    
    h <- c_nnls(factor_matrix, A, as.integer(cd_maxit), cd_tol, 
                penalty_L1, penalty_L2, penalty_upper, 
                constraint_nonneg, as.integer(threads),
                warm_start)
    
    # Add dimnames
    rownames(h) <- if (!is.null(colnames(factor_matrix))) {
      colnames(factor_matrix)
    } else {
      paste0("nmf", 1:nrow(h))
    }
    if (!is.null(colnames(A))) colnames(h) <- colnames(A)
    
    return(h)
    
  } else {
    # Transpose mode: solve for W given H
    # Transpose problem: HH' W' = HA', solve for W'
    penalty_L1 <- L1[1]  # W penalty
    penalty_L2 <- L2[1]
    penalty_upper <- upper_bound[1]
    constraint_nonneg <- nonneg[1]
    
    # Transpose warm_start if provided (W -> W')
    warm_start_T <- if (!is.null(warm_start)) t(warm_start) else NULL
    
    # Transpose A for the solve
    A_T <- if (is(A, "dgCMatrix")) Matrix::t(A) else t(A)
    w_T <- c_nnls(t(factor_matrix), A_T, as.integer(cd_maxit), cd_tol,
                  penalty_L1, penalty_L2, penalty_upper,
                  constraint_nonneg, as.integer(threads),
                  warm_start_T)
    
    # Transpose back to get W
    w <- t(w_T)
    
    # Add dimnames
    if (!is.null(rownames(A))) rownames(w) <- rownames(A)
    colnames(w) <- if (!is.null(rownames(factor_matrix))) {
      rownames(factor_matrix)
    } else {
      paste0("nmf", 1:ncol(w))
    }
    
    return(w)
  }
}
