#' @title Project a linear factor model
#'
#' @description Solves the equation \eqn{A = WH} for \eqn{H}
#'
#' @details
#' For the classical alternating matrix factorization update problem \eqn{A = WH}, the updates 
#' (or projection) of \eqn{H} is given by the systems of equations: \deqn{W^TWH = WA_j}
#' \deqn{a = W^TW} \deqn{x = H} \deqn{b = WA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' Given \eqn{A} and \eqn{w}, \code{RcppML::project} solves for \eqn{H} using the above formulas and 
#' \code{RcppML::solve}, applying regularizations up-front and parallelizing solutions across columns in 
#' \eqn{A}.
#'
#' The corresponding equation for updating \eqn{W} in block-pivoting is: \deqn{HH^TW^T = HA^T_j}
#'
#' One may also solve for \eqn{W} by inputting the transpose of \eqn{A} and \eqn{H} in place of \eqn{W}.
#'
#' @param A square numeric matrix containing the coefficients of the linear system
#' @param w numeric vector or matrix giving the right-hand side(s) of the linear system
#' @param nonneg apply non-negativity constraints
#' @param cd_tol tolerance for convergence of the least squares sequential coordinate descent solver
#' @param fast_maxit maximum number of iterations of pre-conditioning in NNLS solver prior to coordinate descent, see \code{\link{nnls}}
#' @param cd_maxit maximum number of iterations of coordinate descent NNLS solver, see \code{\link{nnls}}
#' @param L1 L1/LASSO regularization to be subtracted from \eqn{b}. Generally this should be scaled to \code{max(b)}
#' @param precision "float" or "double"
#' @param threads number of CPU threads to use for parallelization. Default is "0" to let OpenMP decide and use all available threads.
#' @returns matrix giving \eqn{h}
#' @export
#' @md
#'
project <- function(A, w, nonneg = TRUE, L1 = 0, fast_maxit = 10, cd_maxit = 100, cd_tol = 1e-8, threads = 0, precision = "double") {
  if(class(A) != "dgCMatrix") as(A, "dgCMatrix")
  if(nrow(A) == nrow(w)) w <- t(w)
  if(nrow(A) != ncol(w)) stop("dimensions of 'A' and 'w' are incompatible!")

  if (precision == "float") {
    return(Rcpp_project_float(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads))
  } else {
    return(Rcpp_project_double(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads))
  }
}