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
#' @section Advanced Parameters:
#' Several parameters may be specified in the \code{...} argument:
#'
#' * \code{diag = TRUE}: scale factors in \eqn{w} and \eqn{h} to sum to 1 by introducing a diagonal, \eqn{d}. This should generally never be set to \code{FALSE}. Diagonalization enables symmetry of models in factorization of symmetric matrices, convex L1 regularization, and consistent factor scalings.
#' * \code{samples = 1:ncol(A)}: samples to include in bipartition, numbered from 1 to \code{ncol(A)}. Default is all samples.
#' * \code{calc_dist = TRUE}: calculate the relative cosine distance of samples within a cluster to either cluster centroid. If \code{TRUE}, centers for clusters will also be calculated.
#' * \code{seed = NULL}: random seed for model initialization, generally not needed for rank-2 factorizations because robust solutions are recovered when \code{diag = TRUE}
#' * \code{maxit = 100}: maximum number of alternating updates of \eqn{w} and \eqn{h}. Generally, rank-2 factorizations converge quickly and this should not need to be adjusted.
#'
#' @inheritParams nmf
#' @param nonneg enforce non-negativity of the rank-2 factorization used for bipartitioning
#' @param threads number of threads for OpenMP parallelization (default 0 = all available)
#' @param verbose print progress information (default FALSE)
#' @param ... additional arguments (see Advanced Parameters section)
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
#' @importFrom methods is
#' @references
#' 
#' Kuang, D, Park, H. (2013). "Fast rank-2 nonnegative matrix factorization for hierarchical document clustering." Proc. 19th ACM SIGKDD intl. conf. on Knowledge discovery and data mining.
#'
#' @note \code{bipartition()} uses scalar \code{nonneg} (not length-2)
#' because rank-2 factorizations apply the same constraint to both factors.
#' The default \code{tol = 1e-5} (inherited from \code{\link{nmf}()}'s
#' internal rank-2 path) is tighter than \code{nmf()}'s global \code{1e-4}
#' because single rank-2 subproblems converge faster.
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nmf}}, \code{\link{dclust}}
#' @md
#' @examples
#' \donttest{
#' library(Matrix)
#' data(iris)
#' A <- as(as.matrix(iris[,1:4]), "dgCMatrix")
#' bipartition(A, calc_dist = TRUE)
#' }
bipartition <- function(data, tol = 1e-5, nonneg = TRUE, threads = 0, verbose = FALSE, ...){

  p <- list(...)
  defaults <- list("diag" = TRUE, "samples" = 1:ncol(data), "seed" = NULL, "calc_dist" = TRUE, "maxit" = 100)
  for(i in 1:length(defaults))
    if(is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]

  # Unified input validation (supports file paths, sparse, dense)
  data_info <- validate_data(data)
  data <- data_info$data

  if(!is.numeric(p$seed)) p$seed <- 0
  if(min(p$samples) == 0) stop("sample indices must be strictly positive")
  if(max(p$samples) > ncol(data)) stop("sample indices must be strictly less than the number of columns in 'data'")

  # GPU fast path for sparse bipartition
  if (inherits(data, "dgCMatrix")) {
    gpu_result <- .try_gpu_bipartition(data, tol, p, nonneg, verbose)
    if (!is.null(gpu_result)) return(gpu_result)
  }
  Rcpp_bipartition(data, tol, p$maxit, nonneg, p$samples - 1, p$seed, verbose, p$calc_dist, p$diag, as.integer(threads))
}

#' Try GPU dispatch for bipartition
#' @return A list with elements \code{v} and \code{d}, or \code{NULL}.
#' @keywords internal
.try_gpu_bipartition <- function(data, tol, p, nonneg, verbose) {
  use_gpu <- getOption("RcppML.gpu", "auto")
  if (identical(use_gpu, FALSE)) return(NULL)

  gpu_ok <- if (identical(use_gpu, TRUE)) {
    gpu_available()
  } else {
    # auto: use GPU for large matrices
    (length(data@x) >= 100000 || ncol(data) >= 5000) && gpu_available()
  }
  if (!gpu_ok) return(NULL)

  tryCatch({
    m <- nrow(data)
    n <- ncol(data)
    nnz <- length(data@x)

    partition <- integer(n)
    v_out <- double(m)
    center_out <- double(m * 2)
    dist_out <- double(1)

    ret <- .gpu_call("rcppml_gpu_bipartition_double",
              col_ptr = as.integer(data@p),
              row_idx = as.integer(data@i),
              values = as.double(data@x),
              m = as.integer(m), n = as.integer(n), nnz = as.integer(nnz),
              max_iter = as.integer(p$maxit),
              tol = as.double(tol),
              nonneg = as.integer(nonneg),
              seed = as.double(if (is.numeric(p$seed)) p$seed else 0),
              partition = partition,
              v = v_out,
              center = center_out,
              dist = dist_out,
              out_status = integer(1))

    if (ret$out_status != 0L) return(NULL)

    s1 <- which(ret$partition == 0)
    s2 <- which(ret$partition == 1)
    list(
      v = ret$v,
      dist = ret$dist,
      size1 = length(s1),
      size2 = length(s2),
      samples1 = s1,
      samples2 = s2,
      center1 = ret$center[1:m],
      center2 = ret$center[(m + 1):(2 * m)]
    )
  }, error = function(e) {
    if (verbose) message("GPU bipartition failed: ", conditionMessage(e), "\nFalling back to CPU.")
    NULL
  })
}