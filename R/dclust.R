#' @title Divisive clustering
#'
#' @description Recursive bipartitioning by rank-2 matrix factorization with an efficient modularity-approximate stopping criteria
#'
#' @details
#' Divisive clustering is a sensitive and fast method for sample classification. Samples are recursively partitioned into two groups until a stopping criteria is satisfied and prevents successful partitioning. 
#'
#' See \code{\link{nmf}} and \code{\link{bipartition}} for technical considerations and optimizations relevant to bipartitioning.
#'
#' @section Stopping criteria:
#' Two stopping criteria are used to prevent indefinite division of clusters and tune the clustering resolution to a desirable range:
#' * \code{min_samples}: Minimum number of samples permitted in a cluster
#' * \code{min_dist}: Minimum cosine distance of samples to their cluster center relative to their unassigned cluster center (an approximation of Newman-Girvan modularity)
#'
#' Newman-Girvan modularity (\eqn{Q}) is an interpretable and widely used measure of modularity for a bipartition. However, it requires the calculation of distance between all within-cluster and between-cluster sample pairs. This is computationally intensive, especially for large sample sets. 
#'
#' \code{dclust} uses a measure which linearly approximates Newman-Girvan modularity, and simply requires the calculation of distance between all samples in a cluster and both cluster centers (the assigned and unassigned center), which is orders of magnitude faster to compute. Cosine distance is used instead of Euclidean distance since it handles outliers and sparsity well.
#'
#' A bipartition is rejected if either of the two clusters contains fewer than \code{min_samples} or if the mean relative cosine distance of the bipartition is less than \code{min_dist}. 
#'
#' A bipartition will only be attempted if there are more than \code{2 * min_samples} samples in the cluster, meaning that \code{dist} may not be calculated for some clusters.
#'
#' @section Sparse optimization:
#' Sparse optimization is automatically applied if \code{A} is a \code{Matrix::dgCMatrix}, or inherits from the \code{Matrix::sparseMatrix} superclass. 
#' Generally, these optimizations are worthwhile if \code{A} is >70% sparse.
#'
#' When \code{A} is <70% sparse, it should be supplied as a dense matrix to avoid the unnecessary overhead of the sparse matrix format.
#'
#' @section Reproducibility:
#' Because rank-2 NMF is approximate and requires random initialization, results may vary slightly across restarts. Therefore, specify a \code{seed} to guarantee absolute reproducibility.
#' 
#' Other than setting the seed, reproducibility may be improved by setting \code{tol} to a smaller number to increase the exactness of each bipartition.
#'
#' @section Advanced parameters:
#' The \code{...} argument hides several parameters that may be adjusted, although defaults should entirely satisfy:
#' * \code{diag}, set to \code{FALSE} to disable model scaling by the diagonal. When \code{diag = FALSE}, symmetric factorization cannot occur and the difference between factors may not be linear with the second-left vector of an SVD.
#' * \code{maxit}, default 100. In rank-2 NMF, maximum number of alternating updates of \eqn{w} and \eqn{h}.
#' * \code{calc_centers}, default TRUE. Calculate centroids for each cluster, giving mean feature loadings for all samples in each cluster. If \code{min_dist != 0}, then \code{calc_centers} will always be \code{TRUE}.
#'
#' @inheritParams nmf
#' @param min_dist stopping criteria giving the minimum cosine distance of samples within a cluster to the center of their assigned vs. unassigned cluster
#' @param min_samples stopping criteria giving the minimum number of samples permitted in a cluster
#' @param verbose print number of divisions in each generation
#' @param tol in rank-2 NMF, the correlation distance (\eqn{1 - R^2}) between \eqn{w} across consecutive iterations at which to stop factorization
#' @param nonneg in rank-2 NMF, enforce non-negativity
#' @param seed random seed for rank-2 NMF model initialization
#' @param threads number of CPU threads, default \code{0} for all available threads
#' @return
#' A list of lists corresponding to individual clusters:
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
#' @references
#' 
#' Schwartz, G. et al. "TooManyCells identifies and visualizes relationships of single-cell clades". Nature Methods (2020).
#'
#' Newman, MEJ. "Modularity and community structure in networks". PNAS (2006)  
#'
#' Kuang, D, Park, H. (2013). "Fast rank-2 nonnegative matrix factorization for hierarchical document clustering." Proc. 19th ACM SIGKDD intl. conf. on Knowledge discovery and data mining.
#'
#' @export
#' @seealso \code{\link{bipartition}}, \code{\link{nmf}}
#' @md
#' @examples
#' clusters <- dclust(t(USArrests), min_samples = 2)
#'
dclust <- function(A, min_dist = 0, min_samples = 100, verbose = TRUE, threads = 0, tol = 1e-4, nonneg = TRUE, seed = NULL, ...) {

  if(!is.null(seed) && !is.na(seed) && seed > 0) set.seed(seed)
  w <- matrix(runif(2 * nrow(A)), 2, nrow(A))

  diag <- TRUE
  maxit <- 100
  calc_centers <- TRUE
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$maxit)) maxit <- params$maxit
  if(!is.null(params$calc_centers)) calc_centers <- params$calc_centers

  if(min_dist > 0) calc_centers <- TRUE

  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    Rcpp_dclust_sparse(A, w, min_dist, min_samples, verbose, threads, tol, nonneg, maxit, calc_centers, diag)
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_dclust_dense(A, w, min_dist, min_samples, verbose, threads, tol, nonneg, maxit, calc_centers, diag)
  }
}