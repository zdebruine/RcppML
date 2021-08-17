#' @title Divisive clustering
#'
#' @description High-performance divisive clustering by recursive bipartitioning with rank-2 matrix factorization
#'
#' @details
#' Divisive clustering is a sensitive and fast method for sample classification. Samples are recursively bipartitioned until a stopping criteria is satisfied. 
#'
#' This stopping criteria prevents indefinite division of clusters. \code{dclust} supports two stopping criteria:
#' * \code{min_samples}: Minimum number of samples permitted in a cluster
#' * \code{min_dist}: Minimum cosine distance of samples to their cluster center relative to their unassigned cluster center (an approximation of Newman-Girvan modularity)
#'
#' There are specialized implementations for sparse and dense input matrices.
#' 
#' @section Advanced parameters:
#' Several parameters hidden in the \code{...} argument may be adjusted (although defaults should entirely satisfy) in addition to those documented explicitly:
#' * \code{diag}, default TRUE. Enable model diagonalization to normalize factors to sum to 1, guarantee symmetry for symmetric factorizations, and distribute L1 penalties consistently and equally across all factors. 
#'
#' @param A matrix of features x samples, may be dense or sparse (a class of the "Matrix" package coercible to \code{Matrix::dgCMatrix})
#' @param min_dist stopping criteria giving the minimum relative cosine distance of samples within a cluster to their assigned vs. unassigned cluster centroid
#' @param min_samples stopping criteria giving the minimum number of samples permitted in a cluster
#' @param verbose print number of divisions in each generation
#' @param threads number of CPU threads for parallelization, default \code{0} for all available threads
#' @param tol correlation distance between \eqn{w} across consecutive iterations at which to stop factorization
#' @param nonneg apply non-negativity constraints on rank-2 NMF for determining bipartition
#' @param seed random seed for initializing rank-2 NMF models
#' @param ... advanced parameters, see details
#' @return
#' A list of lists containing information about each cluster:
#' 	\itemize{
#'    \item id      : character sequence of "0" and "1" giving position of clusters along splitting hierarchy
#'    \item samples : indices of samples in the cluster
#'    \item center  : mean feature expression of all samples in the cluster
#'    \item dist    : if applicable, relative cosine distance of samples in cluster to assigned/unassigned cluster center.
#'    \item leaf    : is cluster a leaf node
#'  }
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{bipartition}}, \code{\link{nmf2}}
#' @md
dclust <- function(A, min_dist = 0, min_samples = 100, verbose = TRUE, threads = 0, tol = 1e-4, nonneg = TRUE, seed = NULL, ...) {

  if(is.null(seed) || is.na(seed) || seed < 0) seed <- 0

  diag <- TRUE
  maxit <- 100
  calc_centers <- TRUE
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$maxit)) maxit <- params$maxit
  if(!is.null(params$calc_centers)) calc_centers <- params$calc_centers

  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    Rcpp_dclust_sparse(A, min_dist, min_samples, verbose, threads, tol, nonneg, maxit, calc_centers, diag, seed)
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_dclust_dense(A, min_dist, min_samples, verbose, threads, tol, nonneg, maxit, calc_centers, diag, seed)
  }
}