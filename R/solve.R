#' @title Solve a linear system with coordinate descent
#'
#' @description Solves the equation \code{a \%*\% x = b} for \code{x}, optionally subject to \eqn{x >= 0}.
#'
#' @details
#' This is a very fast implementation of sequential coordinate descent least squares, suitable for very small or very large systems.
#' The algorithm begins with a zero-filled initialization of \code{x}.
#'
#' Least squares by **sequential coordinate descent** is used to ensure the solution returned is exact. This algorithm was
#' introduced by Franc et al. (2005), and our implementation is a vectorized and optimized rendition of that found in the NNLM R package by Xihui Lin (2020).
#'
#' When \code{nonneg = TRUE}, this solves the Non-Negative Least Squares (NNLS) problem.
#' When \code{nonneg = FALSE}, this solves unconstrained least squares.
#'
#' @param a symmetric positive definite matrix giving coefficients of the linear system
#' @param b matrix giving the right-hand side(s) of the linear system
#' @param L1 L1/LASSO penalty to be subtracted from \code{b}
#' @param L2 Ridge penalty to be added to the diagonal of \code{a}
#' @param cd_maxit maximum number of coordinate descent iterations
#' @param cd_tol stopping criteria, difference in \eqn{x} across consecutive solutions over the sum of \eqn{x}
#' @param upper_bound maximum value permitted in solution, set to \code{0} to impose no upper bound
#' @param nonneg if TRUE (default), impose non-negativity constraint on solution
#' @return vector or matrix giving solution for \code{x}
#' @keywords internal
#' @note This function is deprecated. Use \code{\link{nnls}} instead.
#' \code{solve()} previously masked \code{base::solve()}.
#' @author Zach DeBruine
#' @seealso \code{\link{nmf}}, \code{\link{nnls}}
#' @md
#'
#' @references
#'
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#'
#' Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem. Proc. Int'l Conf. Computer Analysis of Images and Patterns."
#'
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#'
#' @examples
#' \dontrun{
#' # compare solution to base::solve for a random system
#' X <- matrix(runif(100), 10, 10)
#' a <- crossprod(X)
#' b <- crossprod(X, runif(10))
#' 
#' # unconstrained solution (same as base::solve)
#' unconstrained_soln <- solve(a, b, nonneg = FALSE)
#' 
#' # non-negative constrained solution
#' nonneg_soln <- solve(a, b, nonneg = TRUE)
#' }
solve <- function(a, b, cd_maxit = 100L, cd_tol = 1e-8, L1 = 0, L2 = 0, upper_bound = 0, nonneg = TRUE) {
  .Deprecated("nnls", msg = "solve() is deprecated. Use nnls() instead.")
  if (!is.matrix(a)) a <- as.matrix(a)
  if (!is.matrix(b)) b <- as.matrix(b)
  c_solve(a, b, as.integer(cd_maxit), cd_tol, L1, L2, upper_bound, nonneg)
}


#' @title Non-negative Least Squares Projection
#'
#' @description Project a factor matrix onto new data to solve for the complementary matrix.
#' Given \eqn{W} and \eqn{A}, find \eqn{H} such that \eqn{||WH - A||^2} is minimized.
#' Alternatively, given \eqn{H} and \eqn{A}, find \eqn{W} such that \eqn{||WH - A||^2} is minimized.
#'
#' @details
#' This function solves NNLS projection problems with full flexibility:
#'
#' **Projection Modes:**
#' - \code{w} provided, \code{h = NULL}: Solve for H given W and A (standard projection)
#' - \code{h} provided, \code{w = NULL}: Solve for W given H and A (transpose projection)
#' - Exactly one of \code{w} or \code{h} must be NULL
#'
#' **Normal Equations:**
#' - For H: \eqn{W'W H = W'A}
#' - For W: \eqn{HH' W' = HA'} (solved as transpose problem)
#'
#' The problem is solved by forming the Gram matrix and right-hand side in C++ with
#' OpenMP parallelization across samples. Sequential coordinate descent with warm starts
#' enables fast convergence.
#'
#' **Dimension Matching:**
#' When dimensions don't match exactly, the function attempts to:
#' 1. Auto-transpose if one orientation matches
#' 2. Match by rownames/colnames and reorder/subset to the intersection
#' 3. Error if no valid dimension alignment is possible
#'
#' **Warm Start:**
#' Provide an optional \code{warm_start} matrix to initialize the solution. This can
#' dramatically accelerate convergence when solutions are expected to be similar across
#' related problems (e.g., incremental updates, time series).
#'
#' **Penalties:**
#' - L1 (LASSO): Encourages sparsity by subtracting L1 from the right-hand side
#' - L2 (Ridge): Shrinks solutions by adding L2 to the diagonal of the Gram matrix
#' - Graph Laplacian: \emph{Not yet implemented in nnls. Use nmf() for graph regularization.}
#' - Length-2 vectors \code{c(w_penalty, h_penalty)} control penalties for W and H separately
#'
#' @param w factor model matrix (n_features x k) for solving H. Set to NULL to solve for W.
#' @param h factor model matrix (k x n_samples) for solving W. Set to NULL to solve for H.
#' @param A data matrix (n_features x n_samples for w->h, n_samples x n_features for h->w).
#'   Can be dense (matrix) or sparse (dgCMatrix from Matrix package).
#' @param warm_start optional matrix to initialize the solution. Must match output dimensions
#'   (k x n_samples for H, n_features x k for W). Can accelerate convergence.
#' @param L1 L1/LASSO penalty, length 1 or 2 for c(w_penalty, h_penalty). Range [0, 1).
#' @param L2 Ridge penalty, length 1 or 2 for c(w_penalty, h_penalty). Range [0, Inf).
#' @param cd_maxit maximum number of coordinate descent iterations (default 100)
#' @param cd_tol stopping tolerance for coordinate descent (default 1e-8)
#' @param upper_bound maximum value in solution, length 1 or 2 for c(w_bound, h_bound). 0 = no bound.
#' @param nonneg non-negativity constraints, length 1 or 2 for c(w_constraint, h_constraint)
#' @param threads number of threads for OpenMP parallelization (0 = all available)
#' @param verbose print progress information
#' @return matrix of dimension (k x n_samples) for H or (n_features x k) for W
#'
#' @note \code{nnls()} uses \code{A} for the data matrix (linear algebra
#' convention), matching \code{\link{svd}()} and the normal equations
#' \eqn{W'WH = W'A}. In contrast, \code{\link{nmf}()} uses \code{data}
#' (statistical modeling convention).
#' @export
#' @author Zach DeBruine
#' @seealso \code{\link{nmf}}, \code{\link{solve}}
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
#' # Project H onto new features to find W (transpose problem)
#' w_recovered <- nnls(h = h_true, A = A)
#' cor(as.vector(w), as.vector(w_recovered))
#' 
#' # With L1 penalty for sparse H
#' h_sparse <- nnls(w = w, A = A, L1 = c(0, 0.1))
#'
#' # Warm start for faster convergence
#' h_init <- matrix(0.5, 5, 10)
#' h_warm <- nnls(w = w, A = A, warm_start = h_init)
#'
#' # Allow negative values (semi-NMF)
#' h_unconstrained <- nnls(w = w, A = A, nonneg = c(TRUE, FALSE))
#'
#' # Dimension matching by names
#' rownames(w) <- paste0("gene_", 1:20)
#' rownames(A) <- paste0("gene_", 20:1)  # Reversed order
#' h_matched <- nnls(w = w, A = A)  # Automatically reorders A by rownames
#' }
nnls <- function(w = NULL, h = NULL, A, 
                 warm_start = NULL,
                 L1 = c(0, 0), L2 = c(0, 0), 
                 cd_maxit = 100L, cd_tol = 1e-8, 
                 upper_bound = c(0, 0), nonneg = c(TRUE, TRUE),
                 threads = 0, verbose = FALSE) {
  
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
  
  # === STREAMING DISPATCH (SPZ file path) ===
  # If A is a file path to .spz, use streaming NNLS (avoids loading full A)
  if (is.character(A) && length(A) == 1 && grepl("\\.spz$", A) && solve_for_h) {
    if (!is.matrix(factor_matrix)) factor_matrix <- as.matrix(factor_matrix)
    if (length(L1) == 1) L1 <- c(L1, L1)
    if (length(L2) == 1) L2 <- c(L2, L2)
    if (length(upper_bound) == 1) upper_bound <- c(upper_bound, upper_bound)
    if (length(nonneg) == 1) nonneg <- c(nonneg, nonneg)
    
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

  # Recycle penalty parameters to length 2
  if (length(L1) == 1) L1 <- c(L1, L1)
  if (length(L2) == 1) L2 <- c(L2, L2)
  if (length(upper_bound) == 1) upper_bound <- c(upper_bound, upper_bound)
  if (length(nonneg) == 1) nonneg <- c(nonneg, nonneg)
  
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
