#' @title Project a linear factor model
#'
#' @description Solves the equation \eqn{A = wh} for \eqn{h}
#'
#' @details
#' For the classical alternating matrix factorization update problem \eqn{A = wh}, the updates 
#' (or projection) of \eqn{h} is given by the equations: \deqn{w^Twh = wA_j} 
#' in the form \eqn{ax = b} where \eqn{a = w^Tw} \eqn{x = h} and \eqn{b = wA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' Given \eqn{A} and \eqn{w}, \code{RcppML::project} solves for \eqn{h} using the above formulas and 
#' \code{RcppML::nnls}.
#' 
#' The corresponding equation for updating \eqn{w} in block-pivoting is: \deqn{hh^Tw^T = hA^T_j}
#'
#' One may also solve for \eqn{w} by inputting the transpose of \eqn{A} and \eqn{h} in place of \eqn{w}.
#'
#' @param A sparse matrix of features x samples, of or coercible to class \code{Matrix::dgCMatrix}
#' @param w dense matrix of features x factors giving the linear model to be projected
#' @param nonneg apply non-negativity constraints
#' @param L1 L1/LASSO penalty to be applied to \eqn{h}. Generally should be scaled to \code{max(b)} where \eqn{b = WA_j} for all columns \eqn{j} in \eqn{A}
#' @param threads number of CPU threads for parallelization, default \code{0} for all available threads.
#' @param ... expert parameters that shouldn't need to be changed (cd_maxit = 1000, fast_maxit = 10, cd_tol = 1e-8)
#' @returns matrix \eqn{h}
#' @author Zach DeBruine
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @md
#' @examples
#' \dontrun{
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' A <- (w %*% h_true) * (rsparsematrix(1000, 100, 0.5) > 0)
#' h <- project(A, w)
#' cor(as.vector(h_true), as.vector(h))
#' }
project <- function(A, w, nonneg = TRUE, L1 = 0, threads = 0, ...) {
  
  fast_maxit <- 10
  cd_maxit <- 1000
  cd_tol <- 1e-8
  params <- list(...)
  if(!is.null(params$fast_maxit)) fast_maxit <- params$fast_maxit
  if(!is.null(params$cd_maxit)) fast_maxit <- params$cd_maxit
  if(!is.null(params$cd_tol)) cd_tol <- params$cd_tol
  
  A <- as(A, "dgCMatrix")
  if(nrow(A) == nrow(w)) w <- t(w)
  if(nrow(A) != ncol(w)) stop("dimensions of 'A' and 'w' are incompatible!")

  Rcpp_project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads)
}