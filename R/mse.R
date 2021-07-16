#' Mean Squared Error (MSE) loss of a factor model
#'
#' @description Mean squared error of factor models \eqn{W} and \eqn{H} given sparse matrix \eqn{A}
#' 
#' @param A sparse matrix (of or coercible to \code{dgCMatrix}) of samples (columns) by features (rows)
#' @param w dense matrix of class \code{matrix} with factors (columns) by features (rows)
#' @param d vector (if applicable) giving giving coefficients of the diagonal
#' @param h dense matrix of class \code{matrix} with samples (columns) by factors (rows)
#' @param precision "float" or "double" precision
#' @return mean squared error of the factorization
#' @export
#'
mse <- function(A, w, d = NULL, h, precision = "float") {
  if (is.null(d)) d <- rep(1, ncol(w))
  if (ncol(w) != nrow(h)) stop("number of columns in w are not equal to number of rows in h!")
  if (nrow(w) != nrow(A)) stop("number of rows in w are not equal to the number of rows in A")
  if (ncol(h) != ncol(A)) stop("number of columns in h is not equal to the number of columns in A")
  if (length(d) != ncol(w)) stop("length of 'd' is not equal to the rank of 'w' and 'h'")
  if(class(A) != "dgCMatrix") as(A, "dgCMatrix")
  
  ifelse (precision == "float", return(Rcpp_mse_float(A, w, d, h)), return(Rcpp_mse_double(A, w, d, h)))
}