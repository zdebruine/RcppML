#' @title K-Means Clustering
#'
#' @description Perform k-means clustering on sparse or dense matrices.
#'
#' @details
#' This fast kmeans implementation provides efficient sparse matrix support, and often outperforms \code{stats::kmeans} and the Armadillo k-means solver for comparable dense matrix problems.
#'
#' The Hartigan-Wong algorithm is used. Multiple random restarts are recommended (to do this, specify an array of seeds).
#' 
#' Parallelization is applied based on the number of threads specified in \code{options(RcppML.threads})}. 
#'
#' Verbose outputs (iteration, number of cluster reassignments, and tolerance of wss) are given based on \code{options(RcppML.verbose)}
#'
#' The tolerance-based stopping criteria measures the change in total within-cluster sum of squares (\code{twss}) between consecutive iterations \code{i} and \code{i-1}:
#'
#' \deqn{abs(twss_i - twss_{i-1})/(2 * (twss_i + twss_{i-1}))}
#'
#' @inheritParams nmf
#' @param k number of clusters
#' @param tol relative change in total within-cluster sum of squares between consecutive iterations. See details.
#' @param maxit maximum number of iterations allowed
#' @param seed either no seed (\code{NULL}), a single seed, or multiple seeds corresponding to multiple random starts from which the model with the lowest \code{tot_withinss} will be returned.
#' @return 
#' object of class \code{kmeans}, as in \code{stats::kmeans}. It is a list with the following components:
#' \itemize{
#'   \item{"cluster"}{A vector of integers (from \code{1:k}) indicating the cluster to which each point is allocated.}
#'   \item{"centers"}{A matrix of cluster centers.}
#'   \item{"totss"}{The total sum of squares.}
#'   \item{"withinss"}{Vector of within-cluster sum of squares, one component per cluster.}
#'   \item{"tot_withinss"}{Total within-cluster sum of squares, i.e. \code{sum(withinss)}.}
#'   \item{"betweenss"}{The between-cluster sum of squares, i.e. \code{totss-tot.withinss}.}
#'   \item{"size"}{The number of points in each cluster.}
#'   \item{"iter"}{The number of iterations.}
#' }
#' @author Zach DeBruine
#'
#' @export
#' @seealso \code{\link{dclust}}
#' @md
kmeans <- function(data, k, tol = 1e-6, maxit = 100, seed = NULL) {

  if (is.null(seed)) seed <- 0

  # get 'data' in either sparse or dense matrix format and look for NA's
  if (is(data, "sparseMatrix")) {
    if (class(data)[[1]] != "dgCMatrix")
      data <- as(data, "dgCMatrix")
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
    if (!is.numeric(data)) data <- as.numeric(data)
  } else stop("'data' was not coercible to a matrix")

  # call C++ routines
  if (length(seed) > 0) {
    best_withinss <- 0
    for (s in seed) {
      if (getOption("RcppML.verbose")) cat("\nSEED:", seed, "\n")
      if (class(data)[[1]] == "dgCMatrix") {
        model_ <- Rcpp_kmeans(data, matrix(), k, maxit, tol, seed, getOption("RcppML.verbose"), getOption("RcppML.threads"))
      } else {
        model_ <- Rcpp_kmeans(new("dgCMatrix"), data, k, maxit, tol, seed, getOption("RcppML.verbose"), getOption("RcppML.threads"))
      }
      if (best_withinss == 0 || model_$tot_withinss < best_withinss)
        model <- model_
    }
  } else {
    if (class(data)[[1]] == "dgCMatrix") {
      model <- Rcpp_kmeans(data, matrix(), k, maxit, tol, seed, getOption("RcppML.verbose"), getOption("RcppML.threads"))
    } else {
      model <- Rcpp_kmeans(new("dgCMatrix"), data, k, maxit, tol, seed, getOption("RcppML.verbose"), getOption("RcppML.threads"))
    }
  }

  # add back dimnames
  colnames(centers) <- paste0(1:k)
  rownames(centers) <- rownames(data)
  names(clusters) <- colnames(data)
  class(model) <- "kmeans"
  model
}