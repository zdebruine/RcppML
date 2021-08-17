#' @title Non-negative matrix factorization
#'
#' @description Matrix factorization of the form \eqn{A = wdh} by alternating least squares with optional non-negativity constraints.
#'
#' @details
#' This fast non-negative matrix factorization (NMF) implementation decomposes a matrix \eqn{A} into lower-rank
#'  non-negative matrices \eqn{w} and \eqn{h}, with factors scaled to sum to 1 via multiplication by a diagonal, \eqn{d}:
#' \deqn{A = wdh}.
#' 
#' The scaling diagonal enables symmetric factorization, convex L1 regularization, and consistent factor scalings regardless of random initialization.
#' 
#' The factorization model is randomly initialized, and \eqn{w} and \eqn{h} are updated alternately using least squares. 
#' Given \eqn{A} and \eqn{w}, \eqn{h} is updated according to the equation:
#'
#' \deqn{w^Twh = wA_j}
#'
#' This equation is in the form \eqn{ax = b} where \eqn{a = w^Tw} \eqn{x = h} and \eqn{b = wA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' The corresponding update for \eqn{w} is \deqn{hh^Tw = hA^T_j}.
#'
#' The alternating least squares projections (see \code{\link{project}} subroutine) are repeated until a stopping criteria is satisfied, which in this implementation is either a maximum number of 
#' iterations or based on the correlation of models between consecutive iterations.
#'
#' For theoretical and practical considerations, please see our manuscript: "DeBruine ZJ, Melcher K, Triche TJ (2021) 
#' High-performance non-negative matrix factorization for large single cell data." on BioRXiv.
#'
#' The implementation is carefully optimized for parallelization in high-performance computing environments. There are specializations for dense and sparse, and symmetric
#' input matrices.
#' 
#' @section Stopping criteria:
#' Use the \code{tol} parameter to control the stopping criteria for alternating updates
#' * \code{tol = 1e-2} is appropriate for approximate mean squared error determination and coarse cross-validation, for example in rank determination
#' * \code{tol = 1e-3} to \code{1e-4} are suitable for rapid expermentation, cross-validation, and preliminary analysis
#' * \code{tol = 1e-5} and smaller for publication-quality runs
#' 
#' The \code{maxit} parameter is a secondary stopping criterion that takes effect only if \code{tol} is not satisfied 
#' by the maximum number of specified iterations.
#' 
#' @section Sparse optimization:
#' Sparse optimization is automatically applied if \code{A} is a \code{Matrix::dgCMatrix}, or inherits from the \code{Matrix::sparseMatrix} superclass. 
#' Generally, these optimizations are worthwhile if \code{A} is >70% sparse.
#'
#' When \code{A} is <70% sparse, it should be supplied as a dense matrix to avoid the unnecessary overhead of the sparse matrix format.
#'
#' @section L1 regularization:
#' L1 penalization increases the sparsity of factors, but does not change the information content of the model 
#' or the relative contributions of the leading coefficients in each factor. L1 regularization only minorly increases the error of a model.
#'
#' L1 penalization should be used to aid interpretability as needed. Penalty values should range from 0 to 1, where 1 is 
#' complete sparsity. Note that non-negativity constraints already introduce significant sparsity, so L1 regularization may not always be needed.
#'
#' In this implementation of NMF, a scaling diagonal ensures that the L1 penalty is equally applied across all factors regardless 
#' of random initialization and the distribution of the model. Many other implementations of matrix factorization claim to apply L1, but 
#' the magnitude of the penalty is at the mercy of the random distribution and more significantly affects factors with lower overall contribution to the model.
#'
#' @section Symmetric factorization:
#' Special optimization for symmetric matrices is automatically applied. Specifically, alternating updates of \code{w} and \code{h} 
#' require transposition of \code{A}, but \code{A == t(A)} when \code{A} is symmetric, thus no up-front transposition is performed.
#'
#' @section Reproducibility:
#' Specify a \code{seed} to guarantee absolute reproducibility between restarts. Only random initialization is supported, as other 
#' methods incur unnecessary overhead and sometimes trap updates into local minima.
#' 
#' Because random initializations are used, models may be different across restarts. Factors with lower diagonal weights tend to be most different. 
#' Note that comparable loss values (mean squared error) are achieved even if these models may differ. Finding the exact solution is NP-hard.
#'
#' In ongoing work, we are exploring robust factorization solution paths using a novel algorithm similar to rank-truncated SVD.
#' 
#' @section Rank determination:
#' This function does not attempt to suggest a mechanism for rank determination. Like any clustering algorithm or dimensional reduction, 
#' finding the optimal rank can be subjective. An easy way to 
#' estimate rank uses the "elbow method", where the inflection point on a plot of Mean Squared Error loss (MSE) vs. rank 
#' gives a good idea of the rank at which most of the signal has been captured in the model. Unfortunately, this inflection point
#' is not often as obvious for NMF as it is for SVD or PCA.
#' 
#' k-fold cross-validation is a better method. Missing value of imputation has previously been proposed, but is arguably no less subjective 
#' than test-training splits and requires computationally
#' slower factorization updates using missing values, which are not supported here.
#' 
#' @section Advanced parameters:
#' The \code{...} argument hides several parameters that may be adjusted, although defaults should entirely satisfy:
#' * \code{cd_maxit}, default 1000. See \code{\link{nnls}}.
#' * \code{fast_maxit}, default 10. See \code{\link{nnls}}.
#' * \code{cd_tol}, default 1e-8. See \code{\link{nnls}}.
#' * \code{diag}, set to \code{FALSE} to disable model scaling by the diagonal. When \code{diag = FALSE}, symmetric factorization cannot occur 
#' and L1 regularization is not convex with respect to the factorization model.
#'
#' @inheritParams project
#' @param k rank
#' @param tol correlation distance (\eqn{1 - R^2}) between \eqn{w} across consecutive iterations at which to stop factorization
#' @param L1 L1/LASSO penalties between 0 and 1, array of length two for \code{c(w, h)}
#' @param seed random seed for model initialization
#' @param maxit maximum number of alternating updates of \eqn{w} and \eqn{h}
#' @param verbose print model tolerances between iterations
#' @return
#' A list giving the factorization model:
#' 	\itemize{
#'    \item w    : feature factor matrix
#'    \item d    : scaling diagonal vector
#'    \item h    : sample factor matrix
#'    \item tol  : tolerance between models at final update
#'    \item iter : number of alternating updates run
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
#' @seealso \code{\link{nnls}}, \code{\link{project}}, \code{\link{mse}}, \code{\link{nmf2}}
#' @md
#' @examples
#' library(Matrix)
#' # basic NMF
#' model <- nmf(rsparsematrix(1000, 100, 0.1), k = 10)
#' 
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf(A, 10, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' # see package vignette for more examples
#'
nmf <- function(A, k, tol = 1e-3, maxit = 100, verbose = TRUE, nonneg = TRUE, L1 = c(0, 0), seed = NULL, threads = 0, ...) {
  
  if(is.null(seed) || is.na(seed) || seed < 0) seed <- 0

  diag <- TRUE
  fast_maxit <- 10
  cd_maxit <- 1000
  cd_tol <- 1e-8
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$fast_maxit)) fast_maxit <- params$fast_maxit
  if(!is.null(params$cd_maxit)) fast_maxit <- params$cd_maxit
  if(!is.null(params$cd_tol)) cd_tol <- params$cd_tol

  if (length(L1) == 1) L1 <- rep(L1, 2)
  
  # check for symmetry (approximately)
  is_symmetric <- dim(A)[1] == dim(A)[2]
  if(is_symmetric) is_symmetric <- (all(A[1, ] == A[ ,1])) && all((A[nrow(A), ] == A[, ncol(A)]))

  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    if(is_symmetric){
      # just throw in a small random sparse matrix as a place-holder in the second argument, since Rcpp functions don't like NULL arguments
      Rcpp_nmf_sparse(A, Matrix::rsparsematrix(10, 10, 0.1), is_symmetric, k, seed, tol, nonneg, L1[1], L1[2], maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads)
    } else {
      # we want to use Matrix::t() because it's extremely optimized, rather than transposing on the C++ side of things with Rcpp::dgCMatrix
      Rcpp_nmf_sparse(A, t(A), is_symmetric, k, seed, tol, nonneg, L1[1], L1[2], maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads)
    }
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_nmf_dense(A, is_symmetric, k, seed, tol, nonneg, L1[1], L1[2], maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads)
  }
}