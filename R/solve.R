#' Solve a system of equations
#'
#' Least squares solutions with refinement by coordinate descent with support for non-negativity constraints, L0 regularization, and common convex regularizations.
#'
#' Solve the equation \code{a \%*\% x = b} for \code{x} where \code{b} is a vector.
#'
#' If \eqn{b} is a vector, a single solution will be returned. If \eqn{b} is a matrix, the Cholesky decomposition of \eqn{a} will be computed only once. 
#'
#' The algorithm for sequential coordinate descent least squares was first introduced by Xihui Lin, R package \code{NNLM}. \code{tlearn} generalizes the non-negativity constraint in the original code to single- and multi-range constraints and improves speed.
#'
#' @param a square numeric matrix containing the coefficients of the linear system
#' @param b numeric vector giving the right-hand side of the linear system
#' @param x optional initial numeric vector for x (if using exclusively coordinate descent)
#' @param tol tolerance for convergence of the least squares sequential coordinate descent solver
#' @param maxit maximum number of iterations for least squares sequential coordinate descent
#' @param L0 L0 truncation to be incrementally enforced for each column in b. Use 0 or -1 to apply no penalty.
#' @param L1 L1/LASSO regularization to be subtracted from \eqn{b}. Generally this should be scaled to \code{max(b)}
#' @param L2 L2/Ridge regression to be added to the diagonal of \eqn{a}. Generally this should be scaled to \code{mean(diag(a))} or similar.
#' @param RL2 Reciprocal L2/Trough/angular/Pattern Extraction penalty to be added to off-diagonal of \eqn{a}. Generally this should be scaled as in \eqn{L2}.
#' @param L0_method algorithm for solving L0-regularized solution, one of "convex", "ggperm", "eperm", or "exact". See details.
#' @param precision "float" or "double".
#' @returns vector of least squares solutions, \eqn{x}
#' @export
#'
solve <- function(a, b, x = NULL, nonneg = FALSE, L0 = 0, L1 = 0, L2 = 0, RL2 = 0, maxit = 100, tol = 1e-9, L0_method = "ggperm", precision = "float") {
  if (L0 >= nrow(a) || L0 <= 0) L0 = -1
  if (!(L0_method %in% c("ggperm", "eperm", "exact", "convex"))) stop("L0_method was not valid!")
  if(L0_method == "exact" && nrow(a) > 15) stop("number of variables in system is greater than 15 and L0 method = exact. This is not considered to be computational tractable.")
  if (is.null(x)) {
    if (precision == "float") {
        return(solve_Rcpp_float(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
    } else {
        return(solve_Rcpp_double(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
    }
  } else {
    if (precision == "float") {
        return(solve_cd_Rcpp_float(a, b, x_init, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
    } else {
        return(solve_cd_Rcpp_double(a, b, x_init, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
    }
  }
  return(Rcpp_cls(as.vector(x), as.vector(b), as.matrix(a), as.integer(maxit), as.double(tol), as.double(min), as.double(max), sapply(values, as.double)), as.integer(L0))
}