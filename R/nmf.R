#' @title Non-negative matrix factorization
#'
#' @description High-performance matrix factorization with optional non-negativity constraints and L1 regularization.
#'
#' @details
#' This fast non-negative matrix factorization (NMF) implementation decomposes a sparse matrix \eqn{A} into orthogonal lower-rank
#'  non-negative matrices \eqn{w} and \eqn{h}, with factors scaled to sum to 1 by a diagonal, \eqn{d}:
#' \deqn{A = wdh}
#' 
#' For theoretical details, please see our manuscript: "DeBruine ZJ, Melcher K, Triche TJ (2021) 
#' High-performance non-negative matrix factorization for large single cell data." on BioRXiv.
#' 
#' @section Stopping criteria:
#' Use the \code{tol} parameter to control the stopping criteria for alternating updates
#' * \code{tol = 1e-2} is appropriate for approximate mean squared error determination and coarse cross-validation, for example in rank determination
#' * \code{tol = 1e-3} to \code{1e-4} are suitable for rapid expermentation, cross-validation, and preliminary analysis
#' * \code{tol = 1e-5} and smaller for publication-quality runs
#' 
#' The \code{maxit} parameter is a secondary stopping criterion that takes effect only if \code{tol} is not satisfied 
#' by the maximum number of specified iterations.
#' 
#' @section L1 regularization:
#' L1 penalization introduces sparsity into all factors. Because RcppML NMF models are diagonalized, sparsity is enforced
#'  equally across all factors regardless of initialization and iteration. Use the \code{L1} parameter to set any desired
#'  L1/LASSO penalty on the right-hand side of systems of equations during updates of \eqn{w} or \eqn{h}. Typical values 
#'  range between 0 and 1, where 1 is extremely or entirely sparse (this may lead to numerical issues).
#'
#' @section Reproducibility:
#' The optional \code{seed} parameter may be specified to guarantee absolute reproducibility between restarts. 
#' The seed is set on the C++ end and will not affect the RNG in the R environment. Note that only random initialization 
#' is supported, as other initializations do not show better performance and can trap the updates into local minima.
#' 
#' Because random initializations are used, the resulting model may be slightly different, especially for factors with lower 
#' diagonal weights. The same information is explained by all models, but in slightly different splits across less robust factors.
#' Use of cross-validation can help optimize rank and L1 regularization to improve robustness if needed.
#' 
#' @section Rank determination:
#' Like any clustering algorithm or dimensional reduction, finding the optimal rank can be a subjective process. An easy way to 
#' estimate rank that works for well-conditioned matrices uses the "elbow method", where the inflection point on a plot of Mean Squared Error loss (MSE) vs. rank 
#' gives a good idea of the rank at which most of the signal has been captured in the model. Unfortunately, this inflection point
#' is not often as obvious for NMF as it is for SVD or PCA.
#' 
#' Better methods include cross-validation against robustness objectives, such as k-fold test-training splits. Missing value of 
#' imputation has previously been proposed, but is arguably no less subjective than test-training splits and requires computationally
#' slower factorization updates.
#' 
#' @section Advanced parameters:
#' Several parameters hidden in the \code{...} argument may be adjusted (although defaults should entirely satisfy) in addition to those documented explicitly:
#' * \code{cd_maxit}, default 1000. Maximum number of coordinate descent iterations for solution refinement after initialization with solution from previous iteration. Only used as stopping criterion if \code{cd_tol} is not satisfied previously. See \code{\link{nnls}}.
#' * \code{fast_maxit}, default 10. Maximum number of FAST iterations for finding an approximate solution to initialize coordinate descent. See \code{\link{nnls}}.
#' * \code{cd_tol}, default 1e-8. Stopping criterion for coordinate descent iterations given by the maximum relative change in any coefficient between consecutive solutions. See \code{\link{nnls}}.
#' * \code{diag}, default TRUE. Enable model diagonalization to normalize factors to sum to 1, guarantee symmetry for symmetric factorizations, and distribute L1 penalties consistently and equally across all factors. 
#'
#' @param A sparse matrix of features x samples, of or coercible to class \code{Matrix::dgCMatrix}
#' @param k rank
#' @param nonneg apply non-negativity constraints
#' @param tol correlation distance between \eqn{w} across consecutive iterations at which to stop factorization
#' @param L1 L1/LASSO penalty, generally between 0 and 1, array of length two for \code{c(w, h)}
#' @param seed random seed for initializing \eqn{w} with C++ \code{srand} RNG
#' @param maxit maximum number of alternating updates of \eqn{w} and \eqn{h}
#' @param threads number of CPU threads for parallelization, default \code{0} for all available threads
#' @param verbose print tolerances after each iteration to the console
#' @param ... advanced parameters, see details
#' @return
#' A list giving the factorization model:
#' 	\itemize{
#'    \item w   : feature factor matrix
#'    \item d   : scaling diagonal vector
#'    \item h   : sample factor matrix
#'  }
#' @references
#' 
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' 
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#' 
#' Lee, D, and Seung, HS (1999). "Learning the parts of objects by non-negative matrix factorization." Nature.
#'
#' Franc, VC, Hlavac, VC, Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem". Proc. Int'l Conf. Computer Analysis of Images and Patterns. Lecture Notes in Computer Science.
#'
#' @author Zach DeBruine
#' 
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{project}}, \code{\link{mse}}
#' @md
#' @examples 
#' \dontrun{
#' # basic NMF
#' model <- nmf(rsparsematrix(1000, 100, 0.1), k = 10)
#' 
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf(A, 10, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' # see package vignette for more examples
#' }
#'
nmf <- function(A, k, tol = 1e-3, maxit = 100, verbose = TRUE, nonneg = TRUE, 
                L1 = c(0, 0), seed = NULL, threads = 0, ...) {
  
  diag <- TRUE
  fast_maxit <- 10
  cd_maxit <- 1000
  cd_tol <- 1e-8
  params <- list(...)
  if(!is.null(params$diag)) diag <- params$diag
  if(!is.null(params$fast_maxit)) fast_maxit <- params$fast_maxit
  if(!is.null(params$cd_maxit)) fast_maxit <- params$cd_maxit
  if(!is.null(params$cd_tol)) cd_tol <- params$cd_tol

  if (length(L1) == 1) L1 <- rep(L1, 2)
  if (is.null(seed) || is.na(seed) || !is.finite(seed)) seed <- 0
  A <- as(A, "dgCMatrix")

  Rcpp_nmf(A, t(A), k, tol, nonneg, L1[1], L1[2], maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, seed, threads)
  
}
