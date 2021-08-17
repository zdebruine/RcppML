#' @title Project a linear factor model
#'
#' @description Solves the equation \eqn{A = wh} for \eqn{h}
#'
#' @details
#' For the classical alternating matrix factorization update problem \eqn{A = wh}, the updates 
#' (or projection) of \eqn{h} is given by the equations: \deqn{w^Twh = wA_j} 
#' in the form \eqn{ax = b} where \eqn{a = w^Tw} \eqn{x = h} and \eqn{b = wA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' To solve \eqn{w} instead of \eqn{h}, input \code{t(A)} in place of \code{A} and \code{h} in place of \code{w}.
#'
#' Sparse optimization is automatically applied if \code{A} is a \code{Matrix::dgCMatrix}, or inherits from the \code{Matrix::sparseMatrix} superclass.
#'
#' Parallelization is applied across columns of \eqn{A} using OpenMP.
#'
#' Any L1 penalty is subtracted from \eqn{b} and should generally be scaled to \code{max(b)}, where \eqn{b = WA_j} for all columns \eqn{j} in \eqn{A}. An easy way to properly scale an L1 penalty is to normalize all columns in \eqn{w} to sum to 1.
#'
#' @section Advanced parameters:
#' The \code{...} argument hides several parameters that may be adjusted, although defaults should entirely satisfy:
#' * \code{cd_maxit}, default 1000. See \code{\link{nnls}}.
#' * \code{fast_maxit}, default 10. See \code{\link{nnls}}.
#' * \code{cd_tol}, default 1e-8. See \code{\link{nnls}}.
#'
#' @param A dense or sparse matrix of features-by-samples
#' @param w dense matrix of features x factors giving the linear model to be projected
#' @param nonneg enforce non-negativity
#' @param L1 L1/LASSO penalty to be applied to \eqn{h}
#' @param threads number of CPU threads, default \code{0} for all available threads
#' @param ... advanced parameters, see details
#' @returns matrix \eqn{h}
#' @references
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' @author Zach DeBruine
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @md
#' @examples
#' library(Matrix)
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' A <- (w %*% h_true) * (rsparsematrix(1000, 100, 0.5) > 0)
#' h <- project(A, w)
#' cor(as.vector(h_true), as.vector(h))
#' 
#' # alternating projections refine solution (like NMF)
#' mse(A, w, h = h) # mse before alternating updates
#' h <- project(A, w)
#' w <- project(t(A), t(h))
#' h <- project(A, w)
#' w <- project(t(A), t(h))
#' h <- project(A, w)
#' w <- t(project(t(A), t(h)))
#' mse(A, w, h = h) # mse after alternating updates
#'
project <- function(A, w, nonneg = TRUE, L1 = 0, threads = 0, ...) {

  fast_maxit <- 10
  cd_maxit <- 1000
  cd_tol <- 1e-8
  params <- list(...)
  if(!is.null(params$fast_maxit)) fast_maxit <- params$fast_maxit
  if(!is.null(params$cd_maxit)) fast_maxit <- params$cd_maxit
  if(!is.null(params$cd_tol)) cd_tol <- params$cd_tol
  w <- as.matrix(w)

  if(is(A, "sparseMatrix")) {
    # input matrix "A" is sparse
    A <- as(A, "dgCMatrix")
    if(A@Dim[1] == nrow(w)) w <- t(w)
    if(A@Dim[1] != ncol(w)) stop("dimensions of 'A' and 'w' are incompatible!")
    Rcpp_project_sparse(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads)
  } else {
    # input matrix "A" is dense
    A <- as.matrix(A)
    if(nrow(A) == nrow(w)) w <- t(w)
    if(nrow(A) != ncol(w)) stop("dimensions of 'A' and 'w' are incompatible!")
    Rcpp_project_dense(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads)
  }
}