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

    result <- Rcpp_dclust_sparse(A, min_samples, min_dist, verbose, tol, maxit, nonneg, seed, as.integer(threads))
    structure(result, class = "dclust")
}

#' Try GPU dispatch for divisive clustering
#' @return A \code{dclust} object if GPU succeeds, or \code{NULL}.
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
    clusters <- lapply(seq_along(cluster_ids), function(i) {
      cid <- cluster_ids[i]
      samples <- which(ret$assignments == cid)
      center <- Matrix::rowMeans(A[, samples, drop = FALSE])
      list(
        samples = as.integer(samples - 1L),
        center = as.numeric(center),
        id = as.character(cid),  # GPU path: integer string (no tree info)
        size = length(samples)
      )
    })
    structure(clusters, class = "dclust")
  }, error = function(e) {
    if (verbose) message("GPU dclust failed: ", conditionMessage(e), "\nFalling back to CPU.")
    NULL
  })
}


#' @title Plot divisive clustering hierarchy
#'
#' @description Reconstructs and plots the binary splitting tree from a
#'   \code{dclust} result. Each cluster's binary path ID encodes its
#'   position in the hierarchy (e.g., \code{"01"} = root->left->right).
#'   If \code{labels} are provided, a stacked composition bar is drawn
#'   below each leaf showing label proportions.
#'
#' @param x a \code{dclust} object (list of clusters with binary path IDs)
#' @param labels optional character or factor vector of class labels, one per
#'   sample in the original data matrix passed to \code{\link{dclust}}
#' @param palette optional named character vector mapping label levels to colors.
#'   If \code{NULL}, generated automatically.
#' @param main plot title
#' @param ... additional arguments passed to \code{\link[stats]{plot.dendrogram}}
#' @return \code{x} (invisibly)
#' @export
#' @method plot dclust
#' @seealso \code{\link{dclust}}
#' @importFrom grDevices hcl.colors
#' @importFrom graphics par layout plot.new plot.window segments rect text legend
plot.dclust <- function(x, labels = NULL, palette = NULL,
                        main = "Divisive Clustering Hierarchy", ...) {
  n <- length(x)
  if (n < 2) {
    message("Single cluster -- nothing to plot.")
    return(invisible(x))
  }

  dendro <- .dclust_to_dendro(x)

  if (is.null(labels)) {
    plot(dendro, main = main, ylab = "Split depth", ...)
  } else {
    opar <- par(no.readonly = TRUE)
    on.exit(par(opar))
    layout(matrix(1:2, nrow = 2), heights = c(3, 1.5))

    par(mar = c(0.5, 4, 3, 10))
    plot(dendro, main = main, leaflab = "none", ylab = "Split depth", ...)

    par(mar = c(5, 4, 0.5, 10))
    .plot_dclust_composition(x, dendro, labels, palette)
  }

  invisible(x)
}


#' Convert dclust result to dendrogram
#' @return A \code{\link{dendrogram}} object.
#' @keywords internal
.dclust_to_dendro <- function(clusters) {
  ids <- vapply(clusters, `[[`, character(1), "id")
  sizes <- vapply(clusters, function(cl) as.integer(cl$size), integer(1))

  .build <- function(path) {
    idx <- which(ids == path)
    if (length(idx) == 1) {
      lbl <- if (nchar(path) > 0) path else "root"
      d <- sizes[idx]
      attr(d, "label") <- paste0(lbl, " (n=", sizes[idx], ")")
      attr(d, "members") <- 1L
      attr(d, "height") <- 0
      attr(d, "leaf") <- TRUE
      attr(d, "midpoint") <- 0
      class(d) <- "dendrogram"
      return(d)
    }

    left <- .build(paste0(path, "0"))
    right <- .build(paste0(path, "1"))

    h <- max(attr(left, "height"), attr(right, "height")) + 1
    d <- list(left, right)
    attr(d, "members") <- attr(left, "members") + attr(right, "members")
    attr(d, "height") <- h
    attr(d, "midpoint") <- (attr(left, "midpoint") +
                            attr(left, "members") +
                            attr(right, "midpoint")) / 2
    class(d) <- "dendrogram"
    return(d)
  }

  .build("")
}


#' Plot composition bars aligned with dendrogram leaves
#' @return Called for its side effect (plotting). Returns \code{NULL} invisibly.
#' @keywords internal
.plot_dclust_composition <- function(clusters, dendro, labels, palette) {
  ids <- vapply(clusters, `[[`, character(1), "id")

  # Get leaf order from dendrogram (left-to-right)
  leaf_order <- labels(dendro)
  leaf_paths <- sub(" \\(n=.*", "", leaf_order)
  leaf_paths[leaf_paths == "root"] <- ""
  n <- length(leaf_paths)

  ulabels <- sort(unique(labels))
  if (is.null(palette))
    palette <- setNames(hcl.colors(length(ulabels), "Set2"), ulabels)

  # Build composition matrix in dendrogram order
  comp <- matrix(0, nrow = length(ulabels), ncol = n)
  totals <- integer(n)
  for (i in seq_len(n)) {
    idx <- which(ids == leaf_paths[i])
    if (length(idx) != 1) next
    sidx <- clusters[[idx]]$samples + 1L
    tab <- table(factor(labels[sidx], levels = ulabels))
    comp[, i] <- as.numeric(tab)
    totals[i] <- sum(tab)
  }

  # Normalize to proportions
  props <- sweep(comp, 2, pmax(totals, 1), "/")

  # Match dendrogram x-coordinates (leaves at 1, 2, ..., n)
  plot.new()
  plot.window(xlim = c(0.5, n + 0.5), ylim = c(-0.25, 1.05))

  bar_w <- 0.4
  for (i in seq_len(n)) {
    cumh <- 0
    for (j in seq_len(nrow(props))) {
      if (props[j, i] == 0) next
      rect(i - bar_w, cumh, i + bar_w, cumh + props[j, i],
           col = palette[ulabels[j]], border = "white", lwd = 0.5)
      cumh <- cumh + props[j, i]
    }
  }

  # Leaf labels
  text(seq_len(n), -0.1,
       labels = paste0(leaf_paths, "\n(n=", totals, ")"),
       cex = 0.65, adj = c(0.5, 1))

  # Legend (in right margin, offset to avoid overlapping bars)
  usr <- par("usr")
  legend_x <- usr[2] + (usr[2] - usr[1]) * 0.02
  legend(legend_x, usr[4], legend = rev(ulabels),
         fill = rev(palette[ulabels]),
         bty = "n", cex = 0.65, ncol = 1, xpd = TRUE)
}
