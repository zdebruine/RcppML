#' @title Rank-2 non-negative matrix factorization
#'
#' @description High-performance rank-2 matrix factorization with optional non-negativity constraints.
#'
#' @details
#' here
#' 
#' @section Advanced parameters:
#' Several parameters hidden in the \code{...} argument may be adjusted (although defaults should entirely satisfy) in addition to those documented explicitly:
#' * \code{diag}, default TRUE. Enable model diagonalization to normalize factors to sum to 1, guarantee symmetry for symmetric factorizations, and distribute L1 penalties consistently and equally across all factors. 
#' * \code{samples}, samples (from 1 to \code{ncol(A)}) in \code{A} to include in factorization. Default \code{1:ncol(A)}.
#'
#' @param A matrix of features x samples, may be dense or sparse (a class of the "Matrix" package coercible to \code{Matrix::dgCMatrix})
#' @param tol correlation distance between \eqn{w} across consecutive iterations at which to stop factorization
#' @param maxit maximum number of alternating updates of \eqn{w} and \eqn{h}
#' @param verbose print tolerances after each iteration to the console
#' @param nonneg apply non-negativity constraints
#' @param seed random seed for initializing \eqn{h}.
#' @param ... advanced parameters, see details
#' @return
#' A list giving the factorization model:
#' 	\itemize{
#'    \item w    : feature factor matrix
#'    \item d    : scaling diagonal vector
#'    \item h    : sample factor matrix
#'    \item tol  : tolerance between models after final update
#'    \item iter : number of alternating iterations completed
#'  }
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nmf}}, \code{\link{bipartition}}
#' @md
#'
nmf2 <- function(A, tol = 1e-4, maxit = 100, verbose = TRUE, nonneg = TRUE, seed = NULL, ...) {
  
  if(is.null(seed) || is.na(seed) || seed < 0) seed <- 0

  diag <- TRUE
  samples <- 1:ncol(A)
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$samples)) samples <- params$samples

  samples <- samples - 1
  
  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    Rcpp_nmf2_sparse(A, seed, tol, nonneg, maxit, verbose, diag, samples);
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_nmf2_dense(A, seed, tol, nonneg, maxit, verbose, diag, samples)
  }
}