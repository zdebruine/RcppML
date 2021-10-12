#' @title Project a factor model with least squares
#'
#' @description Solves the equation \eqn{A = wh} for \eqn{h} given \eqn{w} and \eqn{A}
#'
#' @details
#' For the alternating least squares matrix factorization update problem \eqn{A = wh}, the updates 
#' (or projection) of \eqn{h} is given by the equation: \deqn{w^Twh = wA_j} 
#' which is in the form \eqn{ax = b} where \eqn{a = w^Tw} \eqn{x = h} and \eqn{b = wA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' Least squares projections in factorizations of rank-3 and greater are parallelized using the number of threads set by \code{\link{setRcppMLthreads}}. 
#'
#' Any L1 penalty is subtracted from \eqn{b} and should generally be scaled to \code{max(b)}, where \eqn{b = WA_j} for all columns \eqn{j} in \eqn{A}. An easy way to properly scale an L1 penalty is to normalize all columns in \eqn{w} to sum to the same value (e.g. 1). No scaling is applied in this function. Such scaling guarantees that \code{L1 = 1} gives a completely sparse solution.
#'
#' There are specializations for dense and sparse input matrices, symmetric input matrices, and for rank-1 and rank-2 projections. See documentation for \code{\link{nmf}} for theoretical details and guidance.
#' 
#' @param model \eqn{w}, matrix of features (rows) by factors (columns) containing the linear model to project
#' @param newdata \eqn{A}, data containing features (rows) by samples (columns) containing the new samples onto which to project the model
#' @param L1 L1/LASSO penalty to be applied. No scaling is performed. See details.
#' @param nonneg enforce non-negativity
#' @param mask_zeros handle zeros as missing values
#' @returns matrix \eqn{h}
#' @references
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' @author Zach DeBruine
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @md
#' @examples
#' \dontrun{
#' library(Matrix)
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' # A is the crossproduct of "w" and "h" with 10% signal dropout
#' A <- (w %*% h_true) * (rsparsematrix(1000, 100, 0.9) > 0)
#' h <- project(A, w)
#' cor(as.vector(h_true), as.vector(h))
#' 
#' # alternating projections refine solution (like NMF)
#' mse_bad <- mse(A, w, rep(1, ncol(w)), h) # mse before alternating updates
#' h <- project(A, w = w)
#' w <- project(A, h = h)
#' h <- project(A, w)
#' w <- project(A, h = h)
#' h <- project(A, w)
#' w <- t(project(A, h = h))
#' mse_better <- mse(A, w, rep(1, ncol(w)), h) # mse after alternating updates
#' mse_better < mse_bad
#'
#' # two ways to solve for "w" that give the same solution
#' w <- project(A, h = h)
#' w2 <- project(t(A), w = t(h))
#' all.equal(w, w2)
#' }
project <- function(model, newdata, L1 = 0, nonneg = TRUE, mask_zeros = FALSE) {
  
  threads <- getRcppMLthreads()

  # get 'A' in either sparse or dense matrix format
  if (is(newdata, "sparseMatrix")) {
    newdata <- as(newdata, "dgCMatrix")
  } else if (canCoerce(newdata, "matrix")) {
    newdata <- as.matrix(newdata)
    if(mask_zeros) newdata <- as(newdata, "dgCMatrix")
  } else stop("'newdata' was not coercible to a matrix")
  
  # check that dimensions of w or h are compatible with A
  if (nrow(newdata) == nrow(model) || nrow(newdata) == ncol(model)){
    if(nrow(model) == nrow(newdata)) model <- t(model)
  } else stop("dimensions of 'newdata' and 'model' are incompatible!")

  # select backend based on whether 'newdata' is dense or sparse
  if (class(newdata)[[1]] == "dgCMatrix") {
    Rcpp_projectW_sparse(newdata, model, nonneg, L1, threads, mask_zeros)
  } else {
    Rcpp_projectW_dense(newdata, model, nonneg, L1, threads)
  }
}