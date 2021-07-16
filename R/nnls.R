#' @title 
#' Non-negative least squares
#'
#' @description 
#' Solves the equation \eqn{Ax = b} for \eqn{x} given symmetric positive definite \eqn{A} and vector or matrix \eqn{b}, subject to non-negativity constraints.
#'
#' @details 
#' \code{RcppML::solve} implements a new and fastest-in-class algorithm for non-negative least squares:
#'
#' 1. **initialization** is done by solving for the unconstrained least squares solution.
#' 2. Next, **forward active set tuning** (FAST) provides a near-exact solution (often 
#' exact for well-conditioned systems) by setting all negative values in the unconstrained
#' solution to zero, re-solving the system for only positive values, and repeating the process
#' until the solution gives only positive values. Set \code{cd_maxit = 0} to use only the FAST solver.
#' 3. **coordinate descent** refines the FAST solution and finds the best solution discoverable
#' by gradient descent. The coordinate descent solution is only used if it gives a better error
#' than the FAST solution. Generally, coordinate descent re-introduces variables constrained to zero by
#' FAST back into the feasible set, but does not dramatically change the solution.
#' 
#' An exceptionally fast exact solver is used for 2-variable solutions.
#'
#' @param a symmetric positive definite matrix giving coefficients of linear system
#' @param b vector or matrix giving right-hand side(s) of linear system
#' @param x initial value for \eqn{x} if using coordinate descent, and not using an unconstrained least squares solution as initialization.
#' @param fast_maxit maximum number of iterations of pre-conditioning in NNLS solver prior to coordinate descent, see \code{\link{nnls}}
#' @param cd_maxit maximum number of coordinate descent iterations for solution refinement after initial FAST step. Only used as stopping criterion if \code{tol} is not met.
#' @param tol stopping criterion for coordinate descent iterations given by the maximum change in a coefficient between consecutive solutions
#' @param precision either "float" or "double"
#' @param nonneg apply non-negativity constraints
#' @export
#' @md
#'
nnls <- function(a, b, x = NULL, fast_maxit = 10, cd_maxit = 100, tol = 1e-8, precision = "double", nonneg = TRUE) {
  b <- as.matrix(b)
  if(nrow(a) != ncol(a)) stop("a is not symmetric")
  if(nrow(b) != nrow(a)) stop("number of rows in 'b' (or length of 'b') was not equal to the edge length of 'a'")
  if(!("matrix" %in% class(b))) b <- as.matrix(b)
  # use coordinate descent solver if "a" is not positive definite
  if(is.null(x) && any(a < 0)) x <- matrix(0, nrow(b), ncol(b))
  
  if(!is.null(x)){
    if(!("matrix" %in% class(x))) x <- as.matrix(x)
    if(nrow(x) != nrow(b) && ncol(x) != ncol(b)) stop("'x' and 'b' are not of equal dimensions")
    if (precision == "float") {
      return(Rcpp_cdnnls_float(a, b, x, cd_maxit, tol, nonneg))
    } else {
      return(Rcpp_cdnnls_double(a, b, x, cd_maxit, tol, nonneg))
    }
  } else {
    if (precision == "float") {
      return(Rcpp_nnls_float(a, b, fast_maxit, cd_maxit, tol, nonneg))
    } else {
      return(Rcpp_nnls_double(a, b, fast_maxit, cd_maxit, tol, nonneg))
    }
  }
}