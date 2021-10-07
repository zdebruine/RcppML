#' @title Non-negative matrix factorization
#'
#' @description Sparse matrix factorization of the form \eqn{A = wdh} by alternating least squares with optional non-negativity constraints.
#'
#' @details
#' This fast non-negative matrix factorization (NMF) implementation decomposes a matrix \eqn{A} into lower-rank
#'  non-negative matrices \eqn{w} and \eqn{h}, with factors scaled to sum to 1 via multiplication by a diagonal, \eqn{d}: \deqn{A = wdh}
#'
#' The scaling diagonal ensures convex L1 regularization, consistent factor scalings regardless of random initialization, and model symmetry in factorizations of symmetric matrices.
#' 
#' The factorization model is randomly initialized, and \eqn{w} and \eqn{h} are updated alternately using least squares. 
#'
#' **Parallelization:** Least squares projections in factorizations of rank-3 and greater are parallelized using the number of threads set by \code{\link{setRcppMLthreads}}. 
#' By default, all available threads are used, see \code{\link{getRcppMLthreads}}.
#' The overhead of parallelization is too great to benefit rank-1 and rank-2 factorization.
#'
#' **Specializations.** There are specialized backends for dense and sparse matrices (be explicit about what type of matrix you provide), symmetric, rank-1, and rank-2 factorization.
#' 
#' **L1 regularization**. L1 penalization increases the sparsity of factors, but does not change the information content of the model 
#' or the relative contributions of the leading coefficients in each factor to the model. L1 regularization only slightly increases the loss of a model. 
#' L1 penalization should be used to assist interpretability. Penalty values should range from 0 to 1, where 1 gives complete sparsity. In this implementation of NMF, 
#' a scaling diagonal ensures that the L1 penalty is equally applied across all factors regardless of random initialization and the distribution of the model. 
#' Many other implementations of matrix factorization claim to apply L1, but the magnitude of the penalty is at the mercy of the random distribution and 
#' more significantly affects factors with lower overall contribution to the model. L1 regularization of rank-1 and rank-2 factorizations has no effect.
#'
#' **Publication reference.** Please cite our manuscript: "DeBruine ZJ, Melcher K, Triche TJ (2021) 
#' Fast and robust non-negative matrix factorization for single-cell experiments." on BioRXiv.
#'
#' @section Advanced Parameters:
#' Several parameters may be specified in the \code{...} argument:
#'
#' * \code{nonneg = TRUE}: enforce non-negativity
#' * \code{diag = TRUE}: scale factors in \eqn{w} and \eqn{h} to sum to 1 by introducing a diagonal, \eqn{d}. This should generally never be set to \code{FALSE}. Diagonalization enables symmetry of models in factorization of symmetric matrices, convex L1 regularization, and consistent factor scalings.
#' * \code{mask_zeros = FALSE}: handle zeros as missing values. \code{A} must be a sparse matrix. This algorithm is slower and tolerances will fluctuate more than with the standard algorithm.
#' * \code{w_init = NULL}: initial "w" matrix (or list of matrices), supersedes the \code{seed} parameter. If a list of matrices is provided, factorizations will be run for each matrix, and the model with the lowest Mean Squared Error will be returned.
#' * \code{update_in_place = FALSE}: avoid transposition of 'A' (and associated memory usage) by using a slower in-place algorithm for alternating least squares updates of 'w'. \code{update_in_place} is always \code{FALSE} when \code{A} is symmetric.
#' * \code{samples = 1:ncol(A)}: samples to include in factorization, numbered between 1 and \code{ncol(A)}. Default is all samples. This can make the factorization significantly slower. Generally, prefer subsetting prior to factorization.
#' * \code{features = 1:nrow(A)}: features to include in factorization, numbered between 1 and \code{nrow(A)}. Default is all features. This can make the factorization significantly slower. Generally, prefer subsetting prior to factorization.
#'
#' @param A matrix of features-by-samples in dense or sparse format (preferred classes are \code{matrix} or \code{Matrix::dgCMatrix}, respectively). Prefer sparse storage when >50% of values are zero.
#' @param k rank
#' @param tol stopping criteria for alternating least squares updates, the correlation distance between \eqn{w} across consecutive iterations, \eqn{1 - cor(w_i, w_{i-1})}. Prefer \code{tol < 1e-5} for publication runs, \code{1e-5 < tol < 1e-3} for rapid experimentation.
#' @param maxit stopping criteria, maximum number of alternating least squares updates of \eqn{w} and \eqn{h}
#' @param L1 L1/LASSO penalties between 0 and 1, array of length two for \code{c(w, h)}
#' @param seed random seed for model initialization. If an array of seeds is provided, multiple factorizations will be run and the model with the lowest Mean Squared Error will be returned.
#' @param verbose print model tolerances between iterations
#' @param ... see "advanced parameters" section
#' @return
#' A list giving the factorization model:
#' 	\itemize{
#'    \item w    : feature factor matrix
#'    \item d    : scaling diagonal vector
#'    \item h    : sample factor matrix
#'    \item tol  : tolerance between models at final update
#'    \item iter : number of alternating updates
#'  }
#' @references
#' 
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' 
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#' 
#' Lee, D, and Seung, HS (1999). "Learning the parts of objects by non-negative matrix factorization." Nature.
#'
#' Franc, VC, Hlavac, VC, Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem". Proc. Int'l Conf. Computer Analysis of Images and Patterns. Lecture Notes in Computer Science.
#'
#' @author Zach DeBruine
#'
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{project}}, \code{\link{mse}}
#' @md
#' @examples
#' \dontrun{
#' library(Matrix)
#' # basic NMF
#' model <- nmf(rsparsematrix(1000, 100, 0.1), k = 10)
#' 
#' # compare rank-2 NMF to second left vector in an SVD
#' data(iris)
#' A <- as(as.matrix(iris[,1:4]), "dgCMatrix")
#' nmf_model <- nmf(A, 2, tol = 1e-5)
#' bipartitioning_vector <- apply(nmf_model$w, 1, diff)
#' second_left_svd_vector <- base::svd(A, 2)$u[,2]
#' abs(cor(bipartitioning_vector, second_left_svd_vector))
#' 
#' # compare rank-1 NMF with first singular vector in an SVD
#' abs(cor(nmf(A, 1)$w[,1], base::svd(A, 2)$u[,1]))
#'
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf(A, 10, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' # see package vignette for more examples
#' }
nmf <- function(A, k, tol = 1e-4, maxit = 100, verbose = TRUE, L1 = c(0, 0), seed = NULL, ...) {

  # apply defaults to advanced parameters
  p <- list(...)
  defaults <- list("mask_zeros" = FALSE, "diag" = TRUE, "nonneg" = TRUE, "samples" = 1:ncol(A), "features" = 1:nrow(A), "w_init" = NULL, "update_in_place" = FALSE)
  for(i in 1:length(defaults))
    if(is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]

  if (length(L1) != 2) stop("'L1' must be an array of two values, the first for the penalty on 'w', the second for the penalty on 'h'")
  if (max(L1) >= 1 || min(L1) < 0) stop("L1 penalties must be strictly in the range [0,1)")
  threads <- getRcppMLthreads()

  # get 'A' in either sparse or dense matrix format
  if (is(A, "sparseMatrix")) {
    A <- as(A, "dgCMatrix")
  } else if (canCoerce(A, "matrix")) {
    A <- as.matrix(A)
  } else stop("'A' was not coercible to a matrix")

  # randomly initialize "w", or check dimensions of provided initialization
  if (!is.null(p$w_init)) {
    if(!is.list(p$w_init)) {
      w_ <- p$w_init
      p$w_init <- list()
      p$w_init[[1]] <- w_
    }
    for(i in 1:length(p$w_init)){
      if (nrow(p$w_init[[i]]) == length(p$features)) p$w_init <- t(p$w_init[[i]])
      if (ncol(p$w_init[[i]]) != length(p$features)) stop("a matrix was specified for 'w_init', but dimensions are not compatible with number of features")
      if (k != nrow(p$w_init[[i]])) stop("'k' is not equal to rank of the 'w_init' matrix")
    }
  } else {
    p$w_init <- list()
    if(is.null(seed)) {
      p$w_init[[1]] <- matrix(runif(k * length(p$features)), k, length(p$features))
    } else {
      if(!is.list(seed)) seed <- as.list(seed)
      for(i in 1:length(seed)){
        if (is.numeric(seed[[i]])) set.seed(seed[[i]])
        p$w_init[[i]] <- matrix(runif(k * length(p$features)), k, length(p$features))
      }
    }
  }

  # check samples and features arrays
  if(length(p$samples) != ncol(A)){
    if(is.unsorted(p$samples)) p$samples <- sort(p$samples)
    if(min(p$samples) < 1 || max(p$samples) > ncol(A)) stop("'samples' must strictly be between 1 and ncol(A)")
    if(length(p$samples) < 2 || length(p$samples) > ncol(A)) stop("'samples' must be a vector of integers of length >= 2 and <= ncol(A)")
    if(any(duplicated(p$samples))) stop("'samples' array contains duplicate values")
  }
  if(length(p$features) != nrow(A)){
    if(is.unsorted(p$features)) p$features <- sort(p$features)
    if(min(p$features) < 1 || max(p$features) > ncol(A)) stop("'features' must strictly be between 1 and nrow(A)")
    if(length(p$features) < 2 || length(p$features) > nrow(A)) stop("'features' must be a vector of integers of length >= 2 and <= nrow(A)")
    if(any(duplicated(p$features))) stop("'features' array contains duplicate values")
  }

  # call C++ routines
  if (class(A)[[1]] == "dgCMatrix") {
    model <- Rcpp_nmf_sparse(A, tol, maxit, verbose, p$nonneg, L1, p$diag, p$mask_zeros, threads, p$samples - 1, p$features - 1, p$w_init, p$update_in_place)
  } else {
    if (p$mask_zeros) stop("mask_zeros = TRUE not supported when 'A' is in dense format")
    model <- Rcpp_nmf_dense(A, tol, maxit, verbose, p$nonneg, L1, p$diag, p$mask_zeros, threads, p$samples - 1, p$features - 1, p$w_init, p$update_in_place)
  }
  if(!is.null(rownames(A))) rownames(model$w) <- rownames(A)
  if(!is.null(colnames(A))) colnames(model$h) <- colnames(A)
  return(model)
}