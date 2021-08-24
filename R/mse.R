#' Mean Squared Error loss of a factor model
#'
#' @description MSE of factor models \code{w} and \code{h} given sparse matrix \eqn{A}
#'
#' @details Mean squared error of a matrix factorization of the form \eqn{A = wdh} is given by \deqn{\frac{(A - wdh)^2}{ij}} where \eqn{i} and \eqn{j} are the number of rows and columns in \eqn{A}. 
#'
#' Thus, this function simply calculates the cross-product of \eqn{wh} or \eqn{wdh} (if \eqn{d} is specified), 
#' subtracts that from \eqn{A}, squares the result, and calculates the mean of all values.
#' 
#' **Parallelization.** Calculation of mean squared error is performed in parallel across columns in \code{A} using the number of threads set by \code{\link{setRcppMLthreads}}. 
#' By default, all available threads are used, see \code{\link{getRcppMLthreads}}.
#' 
#' **Specializations.** There are specializations for dense and sparse input matrices.
#' 
#' @param A sparse matrix (of or coercible to \code{dgCMatrix}) of samples (columns) by features (rows)
#' @param w dense matrix of class \code{matrix} with factors (columns) by features (rows)
#' @param d optional diagonal scaling vector (if other than 1's) of rank length
#' @param h dense matrix of class \code{matrix} with samples (columns) by factors (rows)
#' @return mean squared error of the factorization model
#' @export
#' @md
#' @author Zach DeBruine
#' @examples
#' library(Matrix)
#' A <- Matrix::rsparsematrix(1000, 1000, 0.1)
#' model <- nmf(A, k = 10, tol = 0.01)
#' c_mse <- mse(A, model$w, model$d, model$h)
#' R_mse <- mean((A - model$w %*% Diagonal(x = model$d) %*% model$h)^2)
#' all.equal(c_mse, R_mse)
#'
mse <- function(A, w, d = NULL, h) {
  if (is.null(d)) d <- rep(1, ncol(w));
  threads <- getRcppMLthreads()
  if(is(A, "sparseMatrix") && canCoerce(A, "dgCMatrix")) { # input matrix "A" is sparse
    Rcpp_mse_sparse(as(A, "dgCMatrix"), w, d, h, threads)
  } else if(canCoerce(A, "matrix")) { # input matrix "A" is dense
    Rcpp_mse_dense(A, w, d, h, threads)
  } else stop("could not coerce 'A' to class 'matrix' or 'dgCMatrix'")
}
