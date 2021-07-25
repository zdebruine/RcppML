#' @title 
#' Non-negative least squares
#'
#' @description 
#' Solves the equation \code{a %*% x = b} for \code{x}, where \code{b} can be either a vector or a matrix, subject to 
#' non-negativity constraints (if specified).
#'
#' @details 
#' This is a very fast implementation of non-negative least squares that outperforms Lawson-Hanson NNLS in our benchmarks 
#' for most applications, particularly for systems >25 variables in size.
#' 
#' If \code{nonneg = FALSE}, the Eigen Cholesky module is used to solve the system of equations, and is usually
#' faster than \code{base::solve}
#'
#' NNLS solutions will be found quickly using default parameters. If \code{x} is provided, only sequential coordinate descent
#'  least squares will be run. If \code{x = NULL}, solutions will be initialized with "FAST" if \code{nonneg = TRUE} or solved
#'   using the Eigen Cholesky module if \code{nonneg = FALSE}. The \code{FAST} method also makes use of the Eigen Cholesky module.
#'
#' If \eqn{a} is not positive, coordinate descent least squares will always be used, and with zero-filled initialization
#'  if \code{x = NULL}.
#' 
#' @section Coordinate Descent NNLS:
#' Least squares by **sequential coordinate descent** is used to ensure the solution returned is exact. This algorithm was 
#' introduced by Franc et al. (2005), and our implementation is adapted from the NNLM R package by Lin and Boutros (2020). 
#' There are two ways to initialize the coordinate descent solver:
#' 1. Specify a value for `x`, either a random or zero-filled vector, or an approximate solution.
#' 2. Leave \code{x = NULL} and use FAST initialization.
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
#' exact for small well-conditioned systems (< 50 variables) within 2 iterations and thus quickly sets up coordinate
#'  descent very well. The FAST method is similar to the first phase of the so-called "TNT-NN" algorithm (Myre et al., 2017), 
#'  but the latter half of that method relies heavily on heuristics to find the true active set, which we avoid by using 
#'  coordinate descent instead.
#'
#' See our BioRXiv manuscript for benchmarking against Lawson-Hanson NNLS and for a more technical introduction to these methods.
#' 
#' @param a symmetric positive definite matrix giving coefficients of linear system
#' @param b vector or matrix giving right-hand side(s) of linear system
#' @param x initial value for \eqn{x} if using only coordinate descent least squares
#' @param fast_maxit maximum number of FAST iterations for finding an approximate solution to initialize coordinate descent
#' @param cd_maxit maximum number of coordinate descent iterations for solution refinement after initialization (with either \code{x} or FAST if \code{x = NULL}). Only used as stopping criterion if \code{cd_tol} is not satisfied previously.
#' @param cd_tol stopping criterion for coordinate descent iterations given by the maximum relative change in any coefficient between consecutive solutions
#' @param nonneg enforce non-negativity of the solution
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
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#' 
#' Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem. Proc. Int'l Conf. Computer Analysis of Images and Patterns."
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
#' # example adapted from multiway::fnnls example 4
#' X <- matrix(rnorm(2000),100,20)
#' btrue <- runif(20)
#' y <- X%*%btrue + rnorm(100)
#' beta <- nnls(crossprod(X),crossprod(X,y))
#' crossprod(btrue-beta)/20
#' 
nnls <- function(a, b, x = NULL, nonneg = TRUE, fast_maxit = 10, cd_maxit = 1000, cd_tol = 1e-8) {
  b <- as.matrix(b)
  if(nrow(a) != ncol(a)) stop("a is not symmetric")
  if(nrow(b) != nrow(a)) stop("number of rows in 'b' (or length of 'b') was not equal to the edge length of 'a'")
  if(is.null(x) && any(a < 0)) x <- matrix(0, nrow(b), ncol(b))
  b <- as.matrix(b)

  if(!is.null(x)){
    x <- as.matrix(x)
    if(nrow(x) != nrow(b) && ncol(x) != ncol(b)) stop("'x' and 'b' are not of equal dimensions")
      return(Rcpp_cdnnls(a, b, x, cd_maxit, cd_tol, nonneg))
  } else {
    return(Rcpp_nnls(a, b, fast_maxit, cd_maxit, cd_tol, nonneg))
  }
}