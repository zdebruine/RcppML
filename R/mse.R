#' Mean Squared Error loss of a factor model
#'
#' @description MSE of factor models \code{w} and \code{h} given sparse matrix \eqn{A}
#'
#' @details Mean squared error of a matrix factorization of the form \eqn{A = wdh} is given by \deqn{\frac{\sum_{i,j}{(A - wdh)^2}}{ij}} where \eqn{i} and \eqn{j} are the number of rows and columns in \eqn{A}. 
#'
#' Thus, this function simply calculates the cross-product of \eqn{wh} or \eqn{wdh} (if \eqn{d} is specified), 
#' subtracts that from \eqn{A}, squares the result, and calculates the mean of all values.
#' 
#' If no diagonal scaling vector is present in the model, input \code{d = rep(1, k)} where \code{k} is the rank of the model.
#' 
#' **Parallelization.** Calculation of mean squared error is performed in parallel across columns in \code{A} using the number of threads set by \code{\link{setRcppMLthreads}}. 
#' By default, all available threads are used, see \code{\link{getRcppMLthreads}}.
#' 
#' @inheritParams nmf
#' @param w dense matrix of class \code{matrix} with factors (columns) by features (rows)
#' @param d diagonal scaling vector of rank length
#' @param h dense matrix of class \code{matrix} with samples (columns) by factors (rows)
#' @return mean squared error of the factorization model
#' @export
#' @md
#' @author Zach DeBruine
#' @examples
#' \dontrun{
#' library(Matrix)
#' A <- Matrix::rsparsematrix(1000, 1000, 0.1)
#' model <- nmf(A, k = 10, tol = 0.01)
#' c_mse <- mse(A, model$w, model$d, model$h)
#' R_mse <- mean((A - model$w %*% Diagonal(x = model$d) %*% model$h)^2)
#' all.equal(c_mse, R_mse)
#' }
mse <- function(A, w, d = NULL, h, mask_zeros = FALSE) {
  threads <- getRcppMLthreads()

  if (is(A, "sparseMatrix")) {
    A <- as(A, "dgCMatrix")
  } else if (canCoerce(A, "matrix")) {
    A <- as.matrix(A)
  } else stop("'A' was not coercible to a matrix")

  if (nrow(w) == nrow(A)) w <- t(w)
  if (nrow(h) == ncol(A)) h <- t(h)
  if (nrow(w) != nrow(h)) stop("'w' and 'h' are not of equal rank")
  if (ncol(w) != nrow(A)) stop("dimensions of 'w' and 'A' are incompatible")
  if (ncol(h) != ncol(A)) stop("dimensions of 'h' and 'A' are incompatible")
  if (is.null(d)) {
    d <- rep(1, ncol(w))
  } else {
    if (length(d) != nrow(w)) stop("length of 'd' and rank of 'w' and 'h' are not equivalent")
  }

  if (class(A) == "dgCMatrix") {
    Rcpp_mse_sparse(A, w, d, h, mask_zeros, threads)
  } else {
    Rcpp_mse_dense(A, w, d, h, mask_zeros, threads)
  }
}