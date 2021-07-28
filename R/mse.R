#' Mean Squared Error loss of a factor model
#'
#' @description MSE of factor models \code{w} and \code{h} given sparse matrix \eqn{A}
#'
#' @details Calculates the cross-product of \eqn{wh} or \eqn{wdh} (if \eqn{d} is specified), 
#' subtracts that from \eqn{A}, squares the result, and calculates the mean of all values. 
#' 
#' Parallelization is used across all available threads as determined by OpenMP. 
#' 
#' Sparse matrix iterators are used in a C++ backend for high performance.
#' 
#' @param A sparse matrix (of or coercible to \code{dgCMatrix}) of samples (columns) by features (rows)
#' @param w dense matrix of class \code{matrix} with factors (columns) by features (rows)
#' @param d optional diagonal scaling vector (if other than 1's) of rank length
#' @param h dense matrix of class \code{matrix} with samples (columns) by factors (rows)
#' @param threads number of CPU threads for parallelization, default \code{0} for all available threads.
#' @return scalar, MSE of the factorization
#' @export
#' @md
#' @author Zach DeBruine
#' @examples
#' library(Matrix)
#' A <- Matrix::rsparsematrix(1000, 1000, 0.1)
#' model <- nmf(A, k = 10)
#' mse(A, model$w, model$d, model$h)
#'
mse <- function(A, w, d = NULL, h, threads = 0) {
  if (is.null(d)) d <- rep(1, ncol(w))
  if (ncol(w) != nrow(h)) stop("number of columns in w are not equal to number of rows in h!")
  if (nrow(w) != A@Dim[1]) stop("number of rows in w are not equal to the number of rows in A")
  if (ncol(h) != A@Dim[2]) stop("number of columns in h is not equal to the number of columns in A")
  if (length(d) != ncol(w)) stop("length of 'd' is not equal to the rank of 'w' and 'h'")
  A <- as(A, "dgCMatrix")
  return(Rcpp_mse(A, w, d, h, threads))
}
