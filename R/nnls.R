#' @title 
#' Non-negative least squares
#'
#' @description 
#' Solves the equation \code{a %*% x = b} for \code{x}. \code{b} can be either a vector or a matrix. Non-negativity constraints are optional.
#'
#' @details 
#' This is a very fast implementation of non-negative least squares (NNLS).
#'
#' When \code{nonneg = TRUE}, non-negative solutions are found by a two-step process:
#'  1. **Approximation with FAST**: Forward active set tuning (FAST) uses Cholesky decomposition to find an approximate active set.
#'  2. **Refinement by Coordinate Descent**: Sequential coordinate descent (CD) iteratively refines the solution to specified tolerances. 
#'
#' When \code{nonneg = FALSE}, no non-negativity constraints are enforced and the Eigen Cholesky solver is used to quickly find the unconstrained solution.
#' 
#' Default parameters will generally suffice. However, if coordinate descent should be initialized with a zero-filled vector (and not the FAST approximation), set \code{fast_maxit = 0}, and Cholesky decomposition will not be run. 
#' Similarly, if the system is well-conditioned and small (i.e. < 25 variables) there may be no need for coordinate descent, and one may set \code{cd_maxit = 0} to use only the FAST approximation.
#'
#' \code{a} must be symmetric positive definite if \code{fast_maxit != 0}, but this is not checked.
#'
#' See our BioRXiv manuscript (references) for benchmarking against Lawson-Hanson NNLS and for a more technical introduction to these methods.
#' 
#' @section Coordinate Descent NNLS:
#' Least squares by **sequential coordinate descent** is used to ensure the solution returned is exact. This algorithm was 
#' introduced by Franc et al. (2005), and our implementation is inspired by the NNLM R package by Lin and Boutros (2020). 
#' 
#' @section FAST NNLS:
#' **Forward active set tuning (FAST)** is an exact or near-exact NNLS approximation initialized by an unconstrained 
#' least squares solution. Negative values in this unconstrained solution are set to zero (the "active set"), and all 
#' other values are added  to a "feasible set". An unconstrained least squares solution is then solved for the 
#' "feasible set", any negative values in the resulting solution are set to zero, and the process is repeated until 
#' the feasible set solution is strictly positive. 
#' 
#' The FAST algorithm has a definite convergence guarantee because the 
#' feasible set will either converge or become smaller with each iteration. The result is generally exact or nearly 
#' exact for small well-conditioned systems (< 50 variables) within 2 iterations and thus sets up coordinate
#' descent for very rapid convergence. The FAST method is similar to the first phase of the so-called "TNT-NN" algorithm (Myre et al., 2017), 
#' but the latter half of that method relies heavily on heuristics to refine the approximate active set, which we avoid by using 
#' coordinate descent instead.
#'
#' @param a symmetric positive definite matrix giving coefficients of the linear system
#' @param b vector or matrix giving the right-hand side(s) of the linear system
#' @param nonneg enforce non-negativity
#' @param L1 L1/LASSO penalty to be subtracted from \code{b}. Penalty should be between 0 and 1. It will be multiplied by \code{max(b)} to ensure proper scaling.
#' @param fast_maxit maximum number of FAST iterations
#' @param cd_maxit maximum number of coordinate descent iterations
#' @param cd_tol maximum relative change for any coefficient between solutions across consecutive coordinate descent iterations at convergence
#' @return vector or matrix giving solution for \code{x}
#' @export
#' @author Zach DeBruine
#' @seealso \code{\link{nmf}}, \code{\link{project}}
#' @md
#' 
#' @references
#' 
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' 
#' Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem. Proc. Int'l Conf. Computer Analysis of Images and Patterns."
#'
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#' 
#' Myre, JM, Frahm, E, Lilja DJ, and Saar, MO. (2017) "TNT-NN: A Fast Active Set Method for Solving Large Non-Negative Least Squares Problems". Proc. Computer Science.
#'
#' @examples 
#' # compare solution to base::solve for a random system
#' X <- matrix(runif(100), 10, 10)
#' a <- crossprod(X)
#' b <- crossprod(X, runif(10))
#' unconstrained_soln <- solve(a, b)
#' nonneg_soln <- nnls(a, b)
#' unconstrained_err <- mean((a %*% unconstrained_soln - b)^2)
#' nonnegative_err <- mean((a %*% nonneg_soln - b)^2)
#' unconstrained_err
#' nonnegative_err
#' all.equal(solve(a, b), nnls(a, b, nonneg = FALSE))
#' 
#' # example adapted from multiway::fnnls example 1
#' X <- matrix(1:100,50,2)
#' y <- matrix(101:150,50,1)
#' beta <- solve(crossprod(X)) %*% crossprod(X, y)
#' beta
#' beta <- nnls(crossprod(X), crossprod(X, y))
#' 
nnls <- function(a, b, nonneg = TRUE, L1 = 0, fast_maxit = 10, cd_maxit = 1000, cd_tol = 1e-8) {
  Rcpp_nnls(as.matrix(a), as.matrix(b), fast_maxit, cd_maxit, cd_tol, nonneg, L1)
}