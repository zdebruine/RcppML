#' @title Divisive clustering
#'
#' @description Recursive bipartitioning by rank-2 matrix factorization with an efficient modularity-approximate stopping criteria
#'
#' @details
#' Divisive clustering is a sensitive and fast method for sample classification. Samples are recursively partitioned into two groups until a stopping criteria is satisfied and prevents successful partitioning.
#'
#' See \code{\link{nmf}} and \code{\link{bipartition}} for technical considerations and optimizations relevant to bipartitioning.
#'
#' **Stopping criteria**. Two stopping criteria are used to prevent indefinite division of clusters and tune the clustering resolution to a desirable range:
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
#' **Reproducibility.** Because rank-2 NMF is approximate and requires random initialization, results may vary slightly across restarts. Therefore, specify a \code{seed} to guarantee absolute reproducibility.
#'
#' Other than setting the seed, reproducibility may be improved by setting \code{tol} to a smaller number to increase the exactness of each bipartition.
#'
#' @inheritParams nmf
#' @param A matrix of features-by-samples in sparse format (preferred class is "Matrix::dgCMatrix")
#' @param min_dist stopping criteria giving the minimum cosine distance of samples within a cluster to the center of their assigned vs. unassigned cluster. If \code{0}, neither this distance nor cluster centroids will be calculated.
#' @param min_samples stopping criteria giving the minimum number of samples permitted in a cluster
#' @param tol in rank-2 NMF, the correlation distance (\eqn{1 - R^2}) between \eqn{w} across consecutive iterations at which to stop factorization
#' @param nonneg in rank-2 NMF, enforce non-negativity
#' @param seed random seed for rank-2 NMF model initialization
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
#' \dontrun{
#' data(USArrests)
#' A <- Matrix::as(as.matrix(t(USArrests)), "dgCMatrix")
#' clusters <- dclust(A, min_samples = 2, min_dist = 0.001)
#' str(clusters)
#' }
dclust <- function(A, min_samples, min_dist = 0, tol = 1e-5, maxit = 100, nonneg = TRUE, seed = NULL) {
    if (!is.numeric(seed)) seed <- 0

    if (canCoerce(A, "dgCMatrix")) {
        A <- as(A, "dgCMatrix")
    } else if (canCoerce(A, "matrix")) {
        A <- as.matrix(A)
        A <- as(A, "dgCMatrix")
    } else {
        stop("'A' could not be coerced to a dgCMatrix")
    }

    Rcpp_dclust_sparse(A, min_samples, min_dist, getOption("RcppML.verbose"), tol, maxit, nonneg, seed, getOption("RcppML.threads"))
}
