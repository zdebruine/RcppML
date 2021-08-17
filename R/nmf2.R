#' @title Rank-2 non-negative matrix factorization
#'
#' @description Rank-2 matrix factorization of the form \eqn{A = wdh} by alternating least squares with optional non-negativity constraints.
#'
#' @details
#' This specialization of matrix factorization derives much in common from \code{\link{nmf}}, but uses the rank-2 guarantee to 
#' solve least squares problems by extremely efficient substitution, in addition to compile-time optimizations.
#'
#' Rank-2 NMF is useful for bipartitioning, and is a subroutine in \code{\link{bipartition}}, where the sign of the difference between sample loadings
#' in both factors gives the partitioning.
#'
#' In rank-2 NMF, a matrix \eqn{A} is decomposed into non-negative matrices \eqn{w} and \eqn{h}, with both factors scaled to sum to 1 via
#' multiplication by a diagonal, \eqn{d}:
#' \deqn{A = wdh}.
#' 
#' The scaling diagonal enables symmetric factorization, consistent factor scalings regardless of random initialization, 
#' and approximate linearity of the difference between factors with the second-left vector of a Singular Value Decomposition (SVD).
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
#' The alternating least squares projections are repeated until a stopping criteria is satisfied, which in this implementation is either a maximum number of 
#' iterations or based on the correlation of models between consecutive iterations.
#'
#' The implementation is single-threaded because parallelization incurs too much overhead.
#'
#' @section Rank-2 optimization:
#' The rank-2 guarantee enables several optimizations over the generalized matrix factorization algorithm in \code{\link{nmf}}.
#'
#' First, two-variable least squares solutions to the problem \eqn{ax = b} are found by direct substitution:
#' \deqn{x_1 = \frac{a_{22}b_1 - a_{12}b_2}{a_{11}a_{22} - a_{12}^2}}
#' \deqn{x_2 = \frac{a_{11}b_2 - a_{12}b_1}{a_{11}a_{22} - a_{12}^2}}
#'
#' Note that the denominator is constant, and thus needs to be calculated only once.
#' 
#' Additionally, if non-negativity constraints are to be imposed, if \eqn{x_1 < 0} then \eqn{x_1 = 0} and \eqn{x_2 = \frac{b_1}{a_{11}}}. 
#' Similarly, if \eqn{x_2 < 0} then \eqn{x_2 = 0} and \eqn{x_1 = \frac{b_2}{a_{22}}}.
#'
#' Second, no up-front transposition of \code{A} is needed. While transposition of \code{A} is desirable in higher-rank matrix factorization to 
#' facilitate a contiguous column-major access pattern while calculating \eqn{b} for updates of \eqn{w}, the benefit gained when \eqn{b} is considerably smaller,
#' to a point it is much cheaper to simply calculate \eqn{b} in-place on \code{A} at all times. 
#'
#' Thirdly, improved vectorization and compile-time optimization may be achieved for length-2 vectors and matrices of _double_ types.
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
#' L1 regularization of a rank-2 factorization of the form \eqn{A = wdh} has no effect, and thus is not supported.
#'
#' @section Reproducibility:
#' Specify a \code{seed} to guarantee absolute reproducibility between restarts. Only random initialization is supported, as other 
#' methods incur unnecessary overhead and sometimes trap updates into local minima.
#' 
#' @section Advanced parameters:
#' The \code{...} argument hides several parameters that may be adjusted, although defaults should entirely satisfy:
#' * \code{diag}, set to \code{FALSE} to disable model scaling by the diagonal. When \code{diag = FALSE}, symmetric factorization cannot occur and the difference between factors may not be linear with the second-left vector of an SVD.
#' * \code{samples}, samples (from 1 to \code{ncol(A)}) in \code{A} to include in factorization. Default \code{1:ncol(A)}.
#'
#' @inheritParams nmf
#' @return
#' A list giving the factorization model:
#' 	\itemize{
#'    \item w    : feature factor matrix
#'    \item d    : scaling diagonal vector
#'    \item h    : sample factor matrix
#'    \item tol  : tolerance between models at final update
#'    \item iter : number of alternating updates run
#'  }
#'
#' @references
#' 
#' Kuang, D, Park, H. (2013). "Fast rank-2 nonnegative matrix factorization for hierarchical document clustering." Proc. 19th ACM SIGKDD intl. conf. on Knowledge discovery and data mining.
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nmf}}, \code{\link{bipartition}}, \code{\link{dclust}}
#' @md
#' @examples
#' library(Matrix)
#' model <- nmf2(rsparsematrix(1000, 100, 0.1))
#' 
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf2(A, tol = 1e-10, maxit = 100)
#' plot(model$w, t(model$h))
#'
nmf2 <- function(A, tol = 1e-4, maxit = 100, verbose = TRUE, nonneg = TRUE, seed = NULL, ...) {
  
  if(is.null(seed) || is.na(seed) || seed < 0) seed <- 0

  diag <- TRUE
  samples <- 1:ncol(A)
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$samples)) samples <- params$samples

  samples <- samples - 1
  
  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    Rcpp_nmf2_sparse(A, seed, tol, nonneg, maxit, verbose, diag, samples);
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    Rcpp_nmf2_dense(A, seed, tol, nonneg, maxit, verbose, diag, samples)
  }
}