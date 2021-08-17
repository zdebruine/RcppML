#' @title Bipartition a sample set
#'
#' @description Spectral biparitioning by rank-2 matrix factorization
#'
#' @details
#' The difference between both factors in a rank-2 matrix factorization is approximately linear to the second-right singular vector, which is often used in spectral bipartitioning as a method of sample classification and clustering.
#' 
#' This function runs a rank-2 matrix factorization and returns the difference between sample loadings in the first and second factors (i.e. \eqn{h} matrix). Rank-2 matrix factorization by alternating least squares is faster than any current truncated SVD implementation (i.e. \eqn{irlba}).
#' 
#' There are backend specializations for dense or sparse input matrices. If a sparse matrix is provided, sparse optimizations will be used and generally are worthwhile if a matrix is >70% sparse.
#'
#' @section Advanced parameters:
#' * \code{diag}, default TRUE. Enable model diagonalization to normalize factors to sum to 1, guarantee symmetry for symmetric factorizations, and achieve linearity of the bipartitioning vector with the second-left vector in rank-2 SVD.
#' * \code{maxit}, default 100. Maximum number of alternating updates of \eqn{w} and \eqn{h}.
#' * \code{calc_centers}, default TRUE. Calculate centroids for each cluster, giving mean feature loadings for all samples in each cluster. If \code{calc_centers = FALSE} and \code{calc_dist = TRUE}, then \code{calc_centers} will be set to \code{TRUE}.
#'
#' @param A matrix of features x samples, may be dense or sparse (a class of the "Matrix" package coercible to \code{Matrix::dgCMatrix})
#' @param samples samples to include in bipartition, numbered from 1 to \code{ncol(A)}. Default is \code{NULL} for all samples.
#' @param tol correlation distance between \eqn{w} across consecutive iterations at which to stop factorization
#' @param nonneg apply non-negativity constraints
#' @param seed random seed for initializing \eqn{h} with R RNG. May also \code{set.seed()} prior to calling this function.
#' @param calc_dist calculate the relative cosine distance of samples within a cluster to either cluster centroid. If \code{TRUE}, \code{calc_centers} will be set to \code{TRUE}.
#' @param verbose print NMF tolerances after each iteration to the console
#' @param ... advanced parameters, see details
#' @return
#' A list giving the bipartition and useful statistics:
#' 	\itemize{
#'    \item v       : vector giving difference between sample loadings between factors in a rank-2 factorization
#'    \item dist    : relative cosine distance of samples within a cluster to centroids of assigned vs. not-assigned cluster
#'    \item size1   : number of samples in first cluster (positive loadings in 'v')
#'    \item size2   : number of samples in second cluster (negative loadings in 'v')
#'    \item samples1: indices of samples in first cluster
#'    \item samples2: indices of samples in second cluster
#'    \item center1 : mean feature loadings across samples in first cluster
#'    \item center2 : mean feature loadings across samples in second cluster
#'  }
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nmf}}
#' @md
#'
bipartition <- function(A, tol = 1e-4, nonneg = TRUE, samples = NULL, seed = NULL, verbose = FALSE, calc_dist = FALSE, ...) {
  
  diag <- TRUE
  calc_centers <- TRUE
  maxit <- 100
  params <- list(...)
  if(is.null(samples)) samples <- 1:ncol(A)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$calc_centers)) calc_centers <- params$calc_centers
  if(!is.null(params$maxit)) maxit <- params$maxit
  if(calc_dist) calc_centers <- TRUE
  if(max(samples) > ncol(A) || min(samples) < 1) stop("'samples' were not strictly in the range 1 to ncol(A)")
  samples <- samples - 1
  if(is.null(seed) || is.na(seed) || seed < 0) seed <- 0

  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    Rcpp_bipartition_sparse(A, samples, tol, nonneg, calc_centers, calc_dist, maxit, verbose, diag, seed)
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_bipartition_dense(A, samples, tol, nonneg, calc_centers, calc_dist, maxit, verbose, diag, seed)
  }
}