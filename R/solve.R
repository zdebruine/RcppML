#' Solve a system of equations
#'
#' Least squares solutions with refinement by coordinate descent with support for non-negativity constraints, L0 regularization, and common convex regularizations.
#'
#' Solve the equation \eqn{ax = b} for \eqn{x} when \eqn{a} is a symmetric positive definite matrix and \eqn{b} is a vector or matrix
#' 
#' @section Non-negativity constraints:
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
#' @section L0 regularization:
#' \code{RcppML::solve} provides near-exact L0 regularization using a new and relatively
#' efficient algorithm:
#' 
#' 1. The **unregularized solution** is first found, with non-negativity constraints if applicable.
#' 2. **convex trimming** of one variable at a time is repeated until the cardinality of the
#' solution (number of non-zero values) is equal to, or less than, L0. With each iteration, the 
#' variable is set to zero which least increases the error, and any negative variables are set to 
#' zero (if \code{nonneg = TRUE}).
#' 3. **systematic permutation** may be applied with each trimming iteration (if \code{L0_path = "appx"}),
#' in which every non-zero/zero variable pair (where the non-zero variable is set to zero and the zero-bound variable is unbound) is
#' performed to find the optimal solution discoverable from the initial state. Since this is not always
#' a convex solution path across different cardinalities, the exact solution may not always be found.
#'
#' For well-conditioned systems, L0 regularization without permutation is exact or near-exact,
#' and provides a fast and useful approximation for any application.
#'
#' Systematic permutation is exponentially slower with larger systems, and quickly becomes 
#' intractable. It is useful only for small systems where exact solutions are critical.
#'
#' For ill-conditioned systems, good solutions will still be discovered, but may be significantly
#' different in composition to their exact counterparts even if the relative difference in error
#' is relatively small.
#'
#' @section Exact L0 regularization:
#' It is important to recognize that exact L0 solutions are not always the most informative.
#' Exact solutions may overlap very poorly with the full solution, and thus provide very different
#' information. This is why a convex L0 solution path (\code{L0_path = "appx"}) is attractive,
#' not only in practice, but also in application.
#'
#' However, to benchmark the accuracy of approximate L0 methods, \code{RcppML::solve} supports exact L0. 
#'
#' L0-truncated least squares is an NP-hard non-convex problem, thus to assess the quality
#' of approximate solutions, or to directly solve small problems where exact solutions are
#' crucial, exact least squares may be used. Exact NNLS finds the true solution by solving
#' the solution for every possible feasible set. Note that the number of possible solutions
#' increases exponentially with the number of variables in the system (NP-hard). Exact NNLS
#' is only reasonable for problems of up to ~15 variables or for correspondingly small L0 
#' cardinalities.
#'
#' Number of solutions per variable:
#' 1 (1), 2 (3), 3 (7), 4 (15), 5 (31), 6 (63), 7 (127), 8 (255), 9 (511), 10 (1023), 
#' 11 (2047), 12 (4095), 13 (8191), 14 (16383), 15 (32767).
#' 
#' @param a square numeric matrix containing the coefficients of the linear system
#' @param b numeric vector or matrix giving the right-hand side(s) of the linear system
#' @param x optional initial numeric vector for x (if using exclusively coordinate descent)
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
#' @returns if \eqn{b} is a vector, a single solution will be returned. If \eqn{b} is a matrix, the Cholesky decomposition of \eqn{a} will be computed only once and a matrix will be returned.
#' @export
#' @md
#'
solve <- function(a, b, x = NULL, nonneg = FALSE, L0 = 0, L1 = 0, L2 = 0, RL2 = 0, maxit = 100, tol = 1e-9, L0_method = "eperm", precision = "double", threads = 0) {
  if (L0 >= nrow(a) || L0 <= 0) L0 = -1
  if (!is.matrix(a) || nrow(a) != ncol(a)) stop("'a' is not a square matrix")
  ifelse(is.matrix(b), size_b <- nrow(b), size_b <- length(b))
  if(size_b != nrow(a)) stop("dimensions of 'a' and 'b' are incompatible")
  if (!(L0_method %in% c("ggperm", "eperm", "exact", "convex"))) stop("L0_method was not valid!")
  if(L0_method == "exact" && nrow(a) > 15) stop("number of variables in system is greater than 15 and L0 method = exact. This is not considered to be computational tractable.")
  if (is.null(x)) {
    if (precision == "float") {
      if(is.matrix(b)){
        return(Rcpp_solve_matrix_float(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
      } else {
        return(Rcpp_solve_vector_float(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
      }
    } else {
      if(is.matrix(b)){
        return(Rcpp_solve_matrix_double(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
      } else {
        return(Rcpp_solve_vector_double(a, b, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
      }
    }
  } else {
    if (precision == "float") {
      if(is.matrix(b)){
        return(Rcpp_solve_cd_matrix_float(a, b, x, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
      } else {
        return(Rcpp_solve_cd_vector_float(a, b, x, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
      }
    } else {
      if(is.matrix(b)){
        return(Rcpp_solve_cd_matrix_double(a, b, x, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method, threads))
      } else {
        return(Rcpp_solve_cd_vector_double(a, b, x, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method))
      }
    }
  }
}