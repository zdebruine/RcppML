#' Project a linear factor model
#'
#' Solves the equation \eqn{A = WH} for either \eqn{W} or \eqn{H}
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
#' See \code{\link{solve}} for details on NNLS and L0 regularization.
#' 
#' @param A square numeric matrix containing the coefficients of the linear system
#' @param W numeric vector or matrix giving the right-hand side(s) of the linear system
#' @param H optional initial numeric vector for x (if using exclusively coordinate descent)
#' @param nonneg apply non-negativity constraints
#' @param tol tolerance for convergence of the least squares sequential coordinate descent solver
#' @param maxit maximum number of iterations for least squares sequential coordinate descent
#' @param L0 L0 truncation to be incrementally enforced for each column in b. Use 0 or -1 to apply no penalty.
#' @param L1 L1/LASSO regularization to be subtracted from \eqn{b}. Generally this should be scaled to \code{max(b)}
#' @param L2 L2/Ridge regression to be added to the diagonal of \eqn{a}. Generally this should be scaled to \code{mean(diag(a))} or similar.
#' @param RL2 Reciprocal L2/angular/Pattern Extraction penalty to be added to off-diagonal of \eqn{a}. Generally this should be scaled as in \eqn{L2}.
#' @param L0_method algorithm for solving L0-regularized solution, one of "convex", "ggperm", "eperm", or "exact". See details.
#' @param precision "float" or "double"
#' @param threads number of cPU threads to use for parallelization when \eqn{b} is a matrix. Default is "0" to let OpenMP decide and use all available threads.
#' @returns matrix giving either \eqn{W} or \eqn{H}
#' @export
#' @md
#'
project <- function(A, W = NULL, H = NULL, nonneg = FALSE, L0 = 0, L1 = 0, L2 = 0, RL2 = 0, maxit = 100,
                    tol = 1e-9, L0_method = "eperm", precision = "double", threads = 0) {

  if (L0 <= 0) L0 = -1
  if (class(A) != "dgCMatrix") A <- as(A, "dgCMatrix")
  if (!(L0_method %in% c("ggperm", "eperm", "exact", "convex"))) stop("L0_method was not valid!")
  if (is.null(H) && is.null(W)) stop("one of 'H' and 'W' must be specified")
  if (!is.null(W) && !is.null(H)) stop("both 'H' and 'W' were specified. Please specify just one.")
  if (!is.null(W)){
    if(nrow(W) == nrow(A)) W <- t(W)
    if(ncol(W) != nrow(A)) stop("dimensions of 'A' and 'W' are incompatible")
    if(precision == "float"){
      return(Rcpp_projectW_float(A, W, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
    } else {
      return(Rcpp_projectW_double(A, W, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
    }
  } else {
    if (nrow(H) == ncol(A)) H <- t(H)
    if (ncol(H) != ncol(A)) stop("dimensions of 'A' and 'H' are incompatible")
    if(precision == "float"){
      return(Rcpp_projectH_float(A, H, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
    } else {
      return(Rcpp_projectH_double(A, H, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
    }
  }
}