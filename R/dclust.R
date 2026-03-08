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
#' @param threads number of threads for OpenMP parallelization (default 0 = all available)
#' @param verbose print progress information (default FALSE)
#' @return
#' A list of lists corresponding to individual clusters:
#' 	\itemize{
#'    \item id      : character sequence of "0" and "1" giving position of clusters along splitting hierarchy
#'    \item samples : 0-indexed integer indices of samples in the cluster (add 1 for R-style indexing)
#'    \item center  : mean feature expression of all samples in the cluster
#'    \item size    : number of samples in the cluster
#'  }
#'
#' @note \code{dclust()} uses \code{A} for the data matrix and scalar
#' \code{nonneg}/\code{tol} parameters (matching \code{\link{bipartition}()}).
#' The default \code{tol = 1e-5} is tighter than \code{\link{nmf}()}'s
#' \code{1e-4} because rank-2 subproblems converge faster and benefit
#' from higher precision.
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
#' \donttest{
#' data(USArrests)
#' A <- as(as.matrix(t(USArrests)), "dgCMatrix")
#' clusters <- dclust(A, min_samples = 2, min_dist = 0.001)
#' str(clusters)
#' }
dclust <- function(A, min_samples, min_dist = 0, tol = 1e-5, maxit = 100, nonneg = TRUE, seed = NULL, threads = 0, verbose = FALSE) {
    if (!is.numeric(seed)) seed <- 0

    # Unified input validation (supports file paths, sparse, dense)
    data_info <- validate_data(A)
    A <- data_info$data
    # dclust requires sparse input — coerce dense to dgCMatrix
    if (is.matrix(A)) A <- .to_dgCMatrix(A)

    # GPU fast path
    gpu_result <- .try_gpu_dclust(A, min_samples, min_dist, tol, maxit, nonneg, seed, verbose)
    if (!is.null(gpu_result)) return(gpu_result)

    Rcpp_dclust_sparse(A, min_samples, min_dist, verbose, tol, maxit, nonneg, seed, as.integer(threads))
}

#' Try GPU dispatch for divisive clustering
#' @keywords internal
.try_gpu_dclust <- function(A, min_samples, min_dist, tol, maxit, nonneg, seed, verbose) {
  use_gpu <- getOption("RcppML.gpu", "auto")
  if (identical(use_gpu, FALSE)) return(NULL)

  gpu_ok <- if (identical(use_gpu, TRUE)) {
    gpu_available()
  } else {
    (length(A@x) >= 100000 || ncol(A) >= 5000) && gpu_available()
  }
  if (!gpu_ok) return(NULL)

  tryCatch({
    m <- nrow(A)
    n <- ncol(A)
    nnz <- length(A@x)
    max_clusters <- as.integer(0)  # no limit

    assignments <- integer(n)
    out_num_clusters <- integer(1)

    ret <- .gpu_call("rcppml_gpu_dclust_double",
              col_ptr = as.integer(A@p),
              row_idx = as.integer(A@i),
              values = as.double(A@x),
              m = as.integer(m), n = as.integer(n), nnz = as.integer(nnz),
              max_clusters = max_clusters,
              min_samples = as.integer(min_samples),
              min_dist = as.double(min_dist),
              max_iter = as.integer(maxit),
              tol = as.double(tol),
              nonneg = as.integer(nonneg),
              seed = as.double(if (is.numeric(seed)) seed else 0),
              assignments = assignments,
              out_num_clusters = out_num_clusters,
              out_status = integer(1))

    if (ret$out_status != 0L) return(NULL)

    # Convert to list-of-clusters format matching CPU output
    cluster_ids <- unique(ret$assignments)
    cluster_ids <- cluster_ids[cluster_ids >= 0]  # drop unassigned
    clusters <- lapply(cluster_ids, function(cid) {
      samples <- which(ret$assignments == cid)
      center <- Matrix::rowMeans(A[, samples, drop = FALSE])
      list(
        samples = as.integer(samples - 1L),
        center = as.numeric(center),
        id = as.integer(cid),
        size = length(samples)
      )
    })
    clusters
  }, error = function(e) {
    if (verbose) message("GPU dclust failed: ", conditionMessage(e), "\nFalling back to CPU.")
    NULL
  })
}
