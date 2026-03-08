#' Consensus Clustering for NMF
#' 
#' Run multiple NMF replicates and compute consensus matrix showing 
#' co-clustering frequency of samples.
#' 
#' @param data input matrix (samples x features for clustering samples)
#' @param k rank of factorization
#' @param reps number of replicates (default 50)
#' @param method consensus method: "hard" for hard cluster assignments (default), 
#'   or "knn_jaccard" for KNN-based Jaccard overlap of factor loadings
#' @param knn number of nearest neighbors to use for KNN Jaccard method (default 10)
#' @param seed random seed for reproducibility
#' @param threads number of threads for OpenMP parallelization (default 0 = all available)
#' @param verbose print progress information (default FALSE)
#' @param ... additional arguments passed to \code{\link{nmf}}
#' 
#' @importFrom stats hclust as.dist cutree cophenetic cor
#' @importFrom utils txtProgressBar setTxtProgressBar
#' 
#' @return List with:
#' \itemize{
#'   \item \code{consensus} - consensus matrix (samples x samples)
#'   \item \code{models} - list of fitted nmf objects
#'   \item \code{clusters} - final cluster assignments
#'   \item \code{cophenetic} - cophenetic correlation coefficient
#'   \item \code{method} - consensus method used
#' }
#' 
#' @details
#' Consensus clustering runs NMF multiple times with different random initializations.
#' 
#' **Hard clustering method** (method = "hard"):
#' For each run, samples are clustered based on their dominant factor in W.
#' The consensus matrix C[i,j] gives the proportion of runs where samples i and j 
#' were assigned to the same cluster. This is the traditional consensus clustering approach.
#' 
#' **KNN Jaccard method** (method = "knn_jaccard"):
#' For each run, the k-nearest neighbors of each sample are computed based on 
#' factor loadings (W matrix). The consensus matrix C[i,j] is the average Jaccard 
#' similarity between the KNN sets of samples i and j across all replicates.
#' This approach is more robust to ambiguous cluster assignments and captures 
#' neighborhood structure rather than hard cluster membership.
#' 
#' High consensus values (near 1) indicate stable co-clustering or neighborhood overlap.
#' Intermediate values suggest ambiguous relationships.
#' 
#' The cophenetic correlation coefficient measures cluster stability - higher values
#' (closer to 1) indicate more stable/reproducible clustering.
#' 
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.3)
#' 
#' # Traditional hard clustering consensus
#' cons_hard <- consensus_nmf(A, k = 5, reps = 10, method = "hard", seed = 123)
#' 
#' # KNN Jaccard consensus (more robust)
#' cons_knn <- consensus_nmf(A, k = 5, reps = 10, method = "knn_jaccard", knn = 15, seed = 123)
#' 
#' # Plot consensus heatmap
#' plot(cons_hard)
#' plot(cons_knn)
#' 
#' # Check cophenetic coefficient (higher = more stable)
#' print(cons_hard$cophenetic)
#' print(cons_knn$cophenetic)
#' 
#' # Get cluster assignments
#' print(table(cons_hard$clusters))
#' }
#'
#' @seealso \code{\link{nmf}}, \code{\link{plot.consensus_nmf}}, \code{\link{summary.consensus_nmf}}
#' @export
consensus_nmf <- function(data, k, reps = 50, method = c("hard", "knn_jaccard"), 
                         knn = 10, seed = NULL, threads = 0, verbose = FALSE, ...) {
  if (!is.null(seed)) set.seed(seed)
  
  method <- match.arg(method)
  
  # Get n_samples: handle file paths (streaming) or in-memory data
  if (is.character(data) && length(data) == 1 && file.exists(data)) {
    info <- st_info(data)
    n_samples <- info$rows
  } else {
    n_samples <- nrow(data)
  }
  consensus <- matrix(0, n_samples, n_samples)
  connectivity <- matrix(0, n_samples, n_samples)
  models <- vector("list", reps)
  
  if (verbose) {
    cat("Running", reps, "NMF replicates for consensus clustering (method:", method, ")...\n")
  }
  pb <- if (verbose) txtProgressBar(min = 0, max = reps, style = 3) else NULL
  
  for (i in 1:reps) {
    # Run NMF with different random seed each time
    rep_seed <- if (!is.null(seed)) seed + i else NULL
    models[[i]] <- nmf(data, k, seed = rep_seed, threads = threads, verbose = FALSE, ...)
    
    if (method == "hard") {
      # Hard clustering: Get cluster assignments (max factor in W)
      clusters <- apply(models[[i]]@w, 1, which.max)
      
      # Update connectivity matrix
      for (cluster_id in unique(clusters)) {
        members <- which(clusters == cluster_id)
        connectivity[members, members] <- connectivity[members, members] + 1
      }
    } else if (method == "knn_jaccard") {
      # KNN Jaccard: Compute pairwise distances and find KNN for each sample
      W <- models[[i]]@w
      
      # Compute cosine similarity matrix
      W_norm <- W / sqrt(rowSums(W^2))
      sim_matrix <- tcrossprod(W_norm)
      
      # Use C++ implementation for KNN + Jaccard computation
      jaccard_matrix <- c_knn_jaccard(sim_matrix, knn)
      connectivity <- connectivity + jaccard_matrix
    }
    
    if (!is.null(pb)) setTxtProgressBar(pb, i)
  }
  if (!is.null(pb)) close(pb)
  
  # Compute consensus as average across replicates
  consensus <- connectivity / reps
  
  # Final clustering from consensus matrix
  # Use hierarchical clustering on 1 - consensus as distance
  hc <- hclust(as.dist(1 - consensus), method = "average")
  final_clusters <- cutree(hc, k = k)
  
  # Compute cophenetic correlation
  cophenetic_corr <- cor(as.dist(1 - consensus), cophenetic(hc))
  
  if (verbose) {
    cat("\nCophenetic correlation:", round(cophenetic_corr, 4), 
        "(higher = more stable clustering)\n")
  }
  
  result <- list(
    consensus = consensus,
    models = models,
    clusters = final_clusters,
    cophenetic = cophenetic_corr,
    hclust = hc,
    k = k,
    reps = reps,
    method = method,
    knn = if (method == "knn_jaccard") knn else NULL
  )
  
  class(result) <- c("consensus_nmf", "list")
  return(result)
}

#' Plot Consensus Matrix Heatmap
#' 
#' @param x consensus_nmf object
#' @param cluster_rows whether to reorder rows by hierarchical clustering (default TRUE)
#' @param cluster_cols whether to reorder columns (default TRUE, same as rows)
#' @param show_clusters whether to show cluster assignments as sidebar (default TRUE)
#' @param color_palette color palette name or vector of colors
#' @param interactive whether to make interactive plotly heatmap (default FALSE)
#' @param ... additional arguments (unused)
#' @return A \code{ggplot2} object (or \code{plotly} object if \code{interactive = TRUE}) showing the consensus heatmap.
#' @seealso \code{\link{consensus_nmf}}, \code{\link{summary.consensus_nmf}}
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(50, 30, 0.3)
#' cons <- consensus_nmf(A, k = 3, reps = 5, seed = 42)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(cons)
#' }
#' }
#' 
#' @method plot consensus_nmf
#' @export
plot.consensus_nmf <- function(x, cluster_rows = TRUE, cluster_cols = TRUE,
                                show_clusters = TRUE, 
                                color_palette = c("white", "#fee5d9", "#fcae91", 
                                                 "#fb6a4a", "#de2d26", "#a50f15"),
                                interactive = FALSE, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package required for plotting")
  }
  
  consensus <- x$consensus
  n <- nrow(consensus)
  
  # Reorder by hierarchical clustering
  if (cluster_rows) {
    row_order <- x$hclust$order
    consensus <- consensus[row_order, row_order]
    if (show_clusters) {
      cluster_labels <- x$clusters[row_order]
    }
  } else {
    if (show_clusters) {
      cluster_labels <- x$clusters
    }
  }
  
  # Convert to long format for ggplot2
  df_list <- list()
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      df_list[[length(df_list) + 1]] <- data.frame(
        row = i,
        col = j,
        consensus = consensus[i, j],
        stringsAsFactors = FALSE
      )
    }
  }
  df <- do.call(rbind, df_list)
  
  # Create base heatmap
  p <- ggplot2::ggplot(df, ggplot2::aes(x = col, y = n - row + 1, fill = consensus)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradientn(
      colors = color_palette,
      limits = c(0, 1),
      name = "Consensus",
      breaks = seq(0, 1, 0.2)
    ) +
    ggplot2::labs(
      title = paste0("Consensus Matrix (", x$reps, " replicates, k=", x$k, ")"),
      subtitle = paste0("Cophenetic correlation: ", round(x$cophenetic, 3)),
      x = "Sample",
      y = "Sample"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      legend.position = "right"
    ) +
    ggplot2::coord_fixed()
  
  # Add cluster boundaries if requested
  if (show_clusters && cluster_rows) {
    # Find cluster boundaries
    boundaries <- which(diff(cluster_labels) != 0)
    if (length(boundaries) > 0) {
      for (b in boundaries) {
        p <- p + 
          ggplot2::geom_hline(yintercept = n - b + 0.5, color = "black", linewidth = 1) +
          ggplot2::geom_vline(xintercept = b + 0.5, color = "black", linewidth = 1)
      }
    }
  }
  
  if (interactive && requireNamespace("plotly", quietly = TRUE)) {
    p <- plotly::ggplotly(p)
  }
  
  return(p)
}

#' Summary for Consensus NMF
#' 
#' @method summary consensus_nmf
#' @param object consensus_nmf object
#' @param ... additional arguments (unused)
#' @return Invisibly returns the \code{consensus_nmf} object. Summary statistics are printed to the console.
#' @seealso \code{\link{consensus_nmf}}, \code{\link{plot.consensus_nmf}}
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(50, 30, 0.3)
#' cons <- consensus_nmf(A, k = 3, reps = 5, seed = 42)
#' summary(cons)
#' }
#'
#' @importFrom stats median
#' @export
summary.consensus_nmf <- function(object, ...) {
  cat("Consensus NMF Results\n")
  cat("=====================\n")
  cat("Rank (k):", object$k, "\n")
  cat("Replicates:", object$reps, "\n")
  cat("Samples:", nrow(object$consensus), "\n")
  cat("Cophenetic correlation:", round(object$cophenetic, 4), "\n\n")
  
  cat("Cluster sizes:\n")
  print(table(object$clusters))
  cat("\n")
  
  cat("Consensus summary:\n")
  consensus_values <- object$consensus[upper.tri(object$consensus)]
  cat("  Min:", round(min(consensus_values), 3), "\n")
  cat("  Median:", round(median(consensus_values), 3), "\n")
  cat("  Mean:", round(mean(consensus_values), 3), "\n")
  cat("  Max:", round(max(consensus_values), 3), "\n")
  
  invisible(object)
}
