#' @title Matrix factorization by alternating least squares
#'
#' @description High-performance Non-Negative Matrix Factorization (NMF), one-sided NMF, or unconstrained matrix factorization with optional diagonalization and L1 regularization.
#'
#' @details
#' TBD
#'
#' @param A matrix of features x samples, of or coercible to sparse matrix \code{dgCMatrix} format
#' @param k rank of factorization
#' @param nonneg apply non-negativity constraints?
#' @param tol stopping criterion for alternating least squares iterations, measuring mean correlation distance between \eqn{w} and \eqn{h} in consecutive iterations.
#' @param L1 L1 ("lasso") regularization to apply to either 'w' or 'h', recommended only to use with diagonalization. Specify array of two for \code{c(w, h)}
#' @param seed random seed for initialization of \eqn{w}, uses C++ \code{srand} RNG.
#' @param maxit maximum number of alternating least squares iterations (updates of \eqn{w} and \eqn{h})
#' @param threads number of CPU threads to use for parallelization. Default is "0" to let OpenMP decide and use all available threads.
#' @param verbose print tolerances for each iteration to the console
#' @param diag introduce a scaling diagonal by normalizing rows in \eqn{h} and columns in \eqn{w} to sum to 1, and sorting factors in the final model by decreasing diagonal values.
#' @param symmetric indicate whether \eqn{A} is a symmetric matrix, so \eqn{A} will not be transposed during alternating least squares and tolerance will measure correlation between \eqn{w} and \eqn{h} with each iteration rather than \eqn{h} between consecutive iterations. Diagonalization will also be applied to ensure convergence of \eqn{w} and \eqn{h}.
#' @param fast_maxit maximum number of iterations of pre-conditioning in NNLS solver prior to coordinate descent, see \code{\link{nnls}}
#' @param cd_maxit maximum number of iterations of coordinate descent NNLS solver, see \code{\link{nnls}}
#' @param cd_tol tolerance of NNLS coordinate descent solver, see \code{\link{nnls}}
#' @param calc_mse calculate mean squared error loss of the factorization at each iteration. This is computationally intensive.
#' @param precision "double" or "float", note that "float" will limit the tolerance of most models at convergence to around 1e-6.
#' @returns list of "w", "d", "h" giving matrices and diagonal vector of the learned factor model, and "tol", and "mse" giving the tolerances and loss (if calculated) of the model at each iteration
#' @export
#'
nmf <- function(A, k, nonneg = TRUE, tol = 1e-3, L1 = c(0, 0), seed = NULL, maxit = 100,
                threads = 0, verbose = TRUE, diag = TRUE, symmetric = FALSE, 
                fast_maxit = 0, cd_maxit = 1000, cd_tol = 1e-8, calc_mse = FALSE, precision = "double") {
  
  if(symmetric && nrow(A) != ncol(A)) stop("'symmetric = TRUE' but 'A' is not square")
  if (length(L1) == 1) L1 <- rep(L1, 2)
  if (!is.logical(verbose)) ifelse(as.numeric(verbose) > 0, verbose <- TRUE, verbose <- FALSE)
  if (is.null(seed) || is.na(seed) || !is.finite(seed)) seed <- 0
  if(class(A) != "dgCMatrix") as(A, "dgCMatrix")
  
  if (precision == "float") {
    return(Rcpp_nmf_float(A, k, seed, tol, nonneg, nonneg, L1[1], L1[2], maxit,
                          threads, verbose, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol))
  } else {
    return(Rcpp_nmf_double(A, k, seed, tol, nonneg, nonneg, L1[1], L1[2], maxit,
                           threads, verbose, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol))
  }
}
