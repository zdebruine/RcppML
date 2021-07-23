#' @title Matrix factorization by alternating least squares
#'
#' @description High-performance matrix factorization with support for non-negativity constraints (i.e. NMF), diagonalization, and L1 regularization.
#'
#' NMF decomposes sparse matrix \eqn{A} into orthogonal matrices \eqn{w} and \eqn{h} such that the crossproduct \eqn{wh} approximates \eqn{A}:
#' 
#' \deqn{A = wh}
#' 
#' We might also consider a diagonalized form, \eqn{A = wdh} where a scaling diagonal \eqn{d} normalizes all factors to sum to 1 and ranks factors in order of decreasing signal explained. Diagonalization enables symmetry between \eqn{w} and \eqn{h} in symmetric factorizations, and enables convex L1 regularization.
#' 
#' For theoretical details, please read (and cite) our manuscript: \eqn{DeBruine ZJ, Melcher K, Triche TJ (2021) "High-performance non-negative matrix factorization for large single cell data". BioRXiv}.
#' 
#' \code{RcppML::nmf} is a very fast and optimized method for NMF, OSNMF, and unconstrained matrix factorizations. It is optimized for sparse matrices, specifically class "dgCMatrix".
#'
#' Use the \code{tol} parameter to control the stopping criteria for alternating updates. \code{tol = 1e-2} is appropriate for crude mean squared error determination, for example in rank determination. \code{tol = 1e-3} is suitable for rapid expermentation, \code{tol = 1e-4} for detailed preliminary analysis, and \code{tol = 1e-5} and smaller for publication-quality runs. Note that for symmetric factorizations where \code{symmetric = TRUE}, correlation between \eqn{w} and \eqn{h} is used which is generally more sensitive than the default measure (correlation between \eqn{w} across consecutive iterations).
#'
#' Use the \code{L1} parameter to set any desired L1/LASSO penalty on the right-hand side of systems of equations during updates of \eqn{w} or \eqn{h}. Typical values range between 0 and 1, where 1 is extremely or entirely sparse (this often leads to numerical issues). Diagonalization with \code{diag = TRUE} is highly recommended when applying L1 regularization, as otherwise factors will not be equally penalized, and their distributions are not known due to random initialization and thus the relative magnitude of the penalty is uncontrollable.
#'
#' The optional \code{seed} parameter may be specified to guarantee absolute reproducibility between restarts. The seed is set on the C++ end and will not affect the RNG in the R environment. Note that only random initialization is supported, as other initializations do not show better performance.
#'
#' The \code{maxit} parameter is a secondary stopping criterion that takes effect only if \code{tol} is not satisfied by the maximum number of specified iterations.
#' 
#' \code{threads} is useful for taking advantage of parallel computing 
#' 
#' @param A sparse matrix of features x samples, of or coercible to class \code{Matrix::dgCMatrix}
#' @param k rank
#' @param nonneg enforce non-negativity on \eqn{w} and \eqn{h} (\code{TRUE}), on \eqn{w} alone (\code{c(TRUE, FALSE)}), on \eqn{h} alone (\code{c(FALSE, TRUE)}), or not at all (\code{FALSE}).
#' @param tol correlation distance between \eqn{w} across consecutive iterations at which to stop factorization. If \code{symmetric = TRUE}, correlation between \eqn{w} and \eqn{h} for the same iteration will be used.
#' @param L1 LASSO penalty on \code{c(w, h)}, generally between 0 and 1.
#' @param seed random seed for initialization of \eqn{w}. Sets C++ \code{srand} RNG seed.
#' @param maxit maximum number of alternating least squares iterations (updates of \eqn{w} and \eqn{h})
#' @param threads number of CPU threads for parallelization. Default is \code{0} to use all available threads.
#' @param verbose print tolerances for each iteration to the console.
#' @param diag Use a scaling diagonal that normalizes rows in \eqn{h} and columns in \eqn{w} to sum to 1. Factors in the final model are sorted by diagonal value.
#' @param symmetric If \eqn{A} is symmetric, set to \code{TURE}. When TRUE, \eqn{A} will not be transposed for alternating least squares updates and tolerance will measure correlation between \eqn{w} and \eqn{h} within each iteration rather than correlation of \eqn{w} across consecutive iterations. Diagonalization is required to guarantee convergence of \eqn{w} and \eqn{h}.
#' @param fast_maxit maximum number of iterations of pre-conditioning in NNLS solver prior to coordinate descent, see \code{\link{nnls}}
#' @param cd_maxit maximum number of iterations of coordinate descent NNLS solver, see \code{\link{nnls}}
#' @param cd_tol tolerance of NNLS coordinate descent solver, see \code{\link{nnls}}
#' @param calc_mse calculate mean squared error loss of the factorization at each iteration.
#' @param precision "double" or "float". Float will limit the tolerance of most models at convergence to around 1e-6 and only sometimes be faster.
#' @return
#' A list of:
#' 	\itemize{
#'    \item w   : feature factor matrix model
#'    \item d   : scaling diagonal vector (if \code{diag = TRUE})
#'    \item h   : sample factor matrix model
#'    \item tol : tolerance of factor models at each iteration
#'    \item mse : mean squared error of the model at each iteration (if \code{calc_mse = TRUE})
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
#' @author Zach DeBruine, \email{zach.debruine@@vai.org}
#' 
#' @export
#' @seealso \code{\link{nnls}}
#' #' @examples 
#' \dontrun{
#' data(moca7k)
#' model <- nmf(moca7k, k = 25)
#' heatmap(model$h[, 1:2500])
#' 
#' # rank determination by inflection point in MSE vs. rank
#' ranks <- c(2:20, seq(22, 40, 2), seq(50, 100, 10))
#' mse <- list()
#' for(rank in ranks){
#'    cat(rank, " ")
#'    model <- nmf(moca7k, rank, tol = 1e-2, verbose = F)
#'    mse[[rank]] <- mse(moca7k, model$w, model$d, model$h)
#' }
#' plot(ranks, unlist(mse), ylab = "MSE", xlab = "rank")
#' abline(v = 25) # estimated rank
#' 
#' # symmetric NMF
#' sim <- as(cosSparse(moca7k[,1:500]), "dgCMatrix")
#' model <- nmf(sim, 10, symmetric = TRUE, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' 
#' # try symmetric NMF without diagonalization (it isn't symmetric)
#' model <- nmf(sim, 10, diag = FALSE, tol = 1e-6, maxit = 1000)
#' plot(model$w, t(model$h))
#' 
#' # see package vignette for more examples
#' }
#'
nmf <- function(A, k, nonneg = TRUE, tol = 1e-3, L1 = c(0, 0), seed = NULL, maxit = 100,
                threads = 0, verbose = TRUE, diag = TRUE, symmetric = FALSE, 
                fast_maxit = 0, cd_maxit = 1000, cd_tol = 1e-8, calc_mse = FALSE, precision = "double") {
  
  if(symmetric && nrow(A) != ncol(A)) stop("'symmetric = TRUE' but 'A' is not square")
  if (length(L1) == 1) L1 <- rep(L1, 2)
  if (!is.logical(verbose)) ifelse(as.numeric(verbose) > 0, verbose <- TRUE, verbose <- FALSE)
  if (is.null(seed) || is.na(seed) || !is.finite(seed)) seed <- 0
  if(class(A) != "dgCMatrix") as(A, "dgCMatrix")
  
  if (precision == "float") {
    return(Rcpp_nmf_float(A, k, seed, tol, nonneg, nonneg, L1[1], L1[2], maxit,
                          threads, verbose, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol))
  } else {
    return(Rcpp_nmf_double(A, k, seed, tol, nonneg, nonneg, L1[1], L1[2], maxit,
                           threads, verbose, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol))
  }
}
