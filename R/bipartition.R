#' @title Bipartition a sample set
#'
#' @description Spectral biparitioning by rank-2 matrix factorization
#'
#' @details
#' Spectral bipartitioning is a popular subroutine in divisive clustering. The sign of the difference between sample loadings in factors of a rank-2 matrix factorization
#' gives a bipartition that is nearly identical to an SVD.
#' 
#' Rank-2 matrix factorization by alternating least squares is faster than rank-2-truncated SVD (i.e. _irlba_).
#'
#' This function is a specialization of rank-2 \code{\link{nmf}} with support for factorization of only a subset of samples, and with additional calculations on the factorization model relevant to bipartitioning. See \code{\link{nmf}} for details regarding rank-2 factorization.
#'
#' @inheritParams nmf
#' @param samples samples to include in bipartition, numbered from 1 to \code{ncol(A)}. Default is \code{NULL} for all samples.
#' @param calc_dist calculate the relative cosine distance of samples within a cluster to either cluster centroid. If \code{TRUE}, centers for clusters will also be calculated.
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
#' \dontrun{
#' library(Matrix)
#' data(iris)
#' A <- as(as.matrix(iris[,1:4]), "dgCMatrix")
#' bipartition(A, calc_dist = TRUE)
#' }
bipartition <- function(A, tol = 1e-5, maxit = 100, nonneg = TRUE, samples = 1:ncol(A), seed = NULL, verbose = FALSE, calc_dist = FALSE, diag = TRUE){

  if (is(A, "sparseMatrix")) {
    A <- as(A, "dgCMatrix")
  } else if (canCoerce(A, "matrix")) {
    A <- as.matrix(A)
  } else stop("'A' was not coercible to a matrix")

    if(!is.numeric(seed)) seed <- 0
    if(min(samples) == 0) stop("sample indices must be strictly positive")
    if(max(samples) > ncol(A)) stop("sample indices must be strictly less than the number of columns in 'A'")
    samples <- samples - 1

    if(class(A) == "dgCMatrix"){
        Rcpp_bipartition_sparse(A, tol, maxit, nonneg, samples, seed, verbose, calc_dist, diag)
    } else {
        Rcpp_bipartition_dense(A, tol, maxit, nonneg, samples, seed, verbose, calc_dist, diag)
    }
}