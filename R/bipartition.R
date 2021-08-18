#' @title Bipartition a sample set
#'
#' @description "Spectral" biparitioning by rank-2 matrix factorization
#'
#' @details
#' Spectral bipartitioning is a popular subroutine in divisive clustering. The sign of the difference between sample loadings in factors of a rank-2 matrix factorization
#' gives a bipartition that is nearly identical to an SVD.
#' 
#' Rank-2 matrix factorization by alternating least squares is faster than rank-2-truncated SVD (i.e. _irlba_).
#'
#' This function is a wrapper of rank-2 \code{\link{nmf}} with added support for factorization of only a subset of samples, and with additional calculations on the factorization model relevant to bipartitioning. See \code{\link{nmf}} for details regarding rank-2 factorization.
#'
#' @section Advanced parameters:
#' The \code{...} argument hides several parameters that may be adjusted, although defaults should entirely satisfy:
#' * \code{diag}, set to \code{FALSE} to disable model scaling by the diagonal. When \code{diag = FALSE}, symmetric factorization cannot occur and the difference between factors may not be linear with the second-left vector of an SVD.
#' * \code{maxit}, default 100. Maximum number of alternating updates of \eqn{w} and \eqn{h}.
#' * \code{calc_centers}, default TRUE. Calculate centroids for each cluster, giving mean feature loadings for all samples in each cluster. If \code{calc_centers = FALSE} and \code{calc_dist = TRUE}, then \code{calc_centers} will be set to \code{TRUE}.
#'
#' @inheritParams nmf
#' @param samples samples to include in bipartition, numbered from 1 to \code{ncol(A)}. Default is \code{NULL} for all samples.
#' @param calc_dist calculate the relative cosine distance of samples within a cluster to either cluster centroid. If \code{TRUE}, \code{calc_centers} will be set to \code{TRUE}.
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
#' @references
#' 
#' Kuang, D, Park, H. (2013). "Fast rank-2 nonnegative matrix factorization for hierarchical document clustering." Proc. 19th ACM SIGKDD intl. conf. on Knowledge discovery and data mining.
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nmf}}, \code{\link{dclust}}
#' @md
#' @examples
#' data(iris)
#' bipartition(iris[,1:4])
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