#' @title Non-negative matrix factorization
#'
#' @description High-performance NMF of the form \eqn{A = wdh} for large dense or sparse matrices, returns an object of class \code{nmf}.
#'
#' @details
#' This fast NMF implementation decomposes a matrix \eqn{A} into lower-rank non-negative matrices \eqn{w} and \eqn{h}, 
#' with columns of \eqn{w} and rows of \eqn{h} scaled to sum to 1 via multiplication by a diagonal, \eqn{d}: \deqn{A = wdh}
#'
#' The scaling diagonal ensures convex L1 regularization, consistent factor scalings regardless of random initialization, and model symmetry in factorizations of symmetric matrices.
#' 
#' The factorization model is randomly initialized.  \eqn{w} and \eqn{h} are updated by alternating least squares.
#' 
#' RcppML achieves high performance using the Eigen C++ linear algebra library, OpenMP parallelization, a dedicated Rcpp sparse matrix class, and fast sequential coordinate descent non-negative least squares initialized by Cholesky least squares solutions.
#'
#' Parallelization is applied with OpenMP using the number of threads set by \code{\link{setRcppMLthreads}}. When \code{k < 3} or \code{setRcppMLthreads(1)}, a backend compiled without OpenMP is used for optimal performance. NMF with \code{mask_zeros = TRUE} is always single-threaded for efficiency.
#'
#' Sparse optimization is automatically applied if the input matrix \code{A} is a sparse matrix (i.e. \code{Matrix::dgCMatrix}). There are also specialized back-ends for symmetric, rank-1, and rank-2 factorizations.
#' 
#' L1 penalization can be used for increasing the sparsity of factors and assisting interpretability. Penalty values should range from 0 to 1, where 1 gives complete sparsity.
#'
# @section Development Parameters:
# Several parameters may be specified in the \code{...} argument:
#
# * \code{diag = TRUE}: scale factors in \eqn{w} and \eqn{h} to sum to 1 by introducing a diagonal, \eqn{d}. This should generally never be set to \code{FALSE}. Diagonalization enables symmetry of models in factorization of symmetric matrices, convex L1 regularization, and consistent factor scalings.
# * \code{update_in_place = FALSE}: avoid transposition of 'A' (and associated memory usage) by using a slower in-place algorithm for alternating least squares updates of 'w'. \code{update_in_place} is always \code{FALSE} when \code{data} is symmetric.
# * \code{samples = 1:ncol(data)}: samples to include in factorization, numbered between 1 and \code{ncol(data)}. Default is all samples. This can make the factorization significantly slower. Generally, prefer subsetting prior to factorization.
# * \code{features = 1:nrow(data)}: features to include in factorization, numbered between 1 and \code{nrow(data)}. Default is all features. This can make the factorization significantly slower. Generally, prefer subsetting prior to factorization.
#
#' @section Methods:
#' The \code{nmf} S3 class has several methods:
#' * \code{predict(object, newdata, L1 = 0)}: equivalent to \code{\link{project}}, project the model onto new samples.
#' * \code{mse(object, data)}: calculates mean squared error loss of the model, \code{data} must be the same data used to fit the model.
#' * \code{prod}: compute the product of the model (multiply it out). Returns a dense matrix
#' * \code{summary(object, group_by, stat = "fractional")}: \code{data.frame} giving \code{fractional}, \code{total}, or \code{mean} representation of factors in samples or features grouped by some criteria
#' * \code{sparsity(object)}: returns a \code{data.frame} giving sparsity of each factor in \eqn{w} and \eqn{h}
#' * \code{match(x, y)}: find an ordering of factors in \code{nmf} model \code{y$w} that best matches those in \code{nmf} model \code{x$w} based on cost of \code{\link{bipartiteMatch}} on a \code{\link{cosine}} similarity matrix.
#' * \code{`[`}: subset, reorder, select, or extract factors
#' * generics such as \code{dim}, \code{t} (transpose), \code{print}, \code{head}, \code{dimnames} 
#' 
#' @param data matrix of features-by-samples in dense or sparse format, coercible to \code{matrix} or \code{Matrix::dgCMatrix}, respectively.
#' @param k rank
#' @param tol stopping criteria based on the correlation distance between \eqn{w} across consecutive iterations. Prefer \code{tol <= 1e-5} for publication runs, \code{1e-5 < tol <= 1e-3} for rapid experimentation.
#' @param maxit maximum permitted update iterations
#' @param L1 LASSO penalties in the range (0, 1], array of length two for \code{c(w, h)}
#' @param seed random seed for initializing \code{w}, or initial \code{w} matrix. If an array of seeds (or list of \code{w} matrices) is provided, \code{nmf} will be run for each seed and the fitted model with the lowest Mean Squared Error will be returned.
#' @param verbose print tolerance after each iteration
#' @param nonneg enforce non-negativity
#' @param mask_zeros handle zeros as missing values
#' @param ... development parameters
#' @return
#' `nmf` returns a list with class "`nmf`" containing the following components:
#' 	\itemize{
#'    \item w    : feature factor matrix
#'    \item d    : scaling diagonal vector
#'    \item h    : sample factor matrix
#'    \item tol  : tolerance between models at final update
#'    \item iter : number of alternating updates
#'  }
#' @importFrom methods is
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
#' @seealso \code{\link{predict.nmf}}, \code{\link{mse.nmf}}, \code{\link{nnls}}
#' @md
#' @examples
#' \dontrun{
#' library(Matrix)
#' # basic NMF
#' model <- nmf(rsparsematrix(1000, 100, 0.1), k = 10)
#' 
#' # compare rank-2 NMF to second left vector in an SVD
#' data(iris)
#' A <- as(as.matrix(iris[,1:4]), "dgCMatrix")
#' nmf_model <- nmf(A, 2, tol = 1e-5)
#' bipartitioning_vector <- apply(nmf_model$w, 1, diff)
#' second_left_svd_vector <- base::svd(A, 2)$u[,2]
#' abs(cor(bipartitioning_vector, second_left_svd_vector))
#' 
#' # compare rank-1 NMF with first singular vector in an SVD
#' abs(cor(nmf(A, 1)$w[,1], base::svd(A, 2)$u[,1]))
#'
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf(A, 10, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' # see package vignette for more examples
#' }
nmf <- function(data, k, tol = 1e-4, maxit = 100, verbose = TRUE, L1 = c(0, 0), seed = NULL, nonneg = TRUE, mask_zeros = FALSE, ...) {

  start_time <- Sys.time()

  # apply defaults to advanced parameters
  p <- list(...)
  defaults <- list("diag" = TRUE)
  for(i in 1:length(defaults))
    if(is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]

  if (length(L1) == 1) {
    L1 <- rep(L1, 2)
  } else if (length(L1) != 2) stop("'L1' must be an array of two values, the first for the penalty on 'w', the second for the penalty on 'h'")
  if (max(L1) >= 1 || min(L1) < 0) stop("L1 penalties must be strictly in the range [0,1)")

  threads <- getRcppMLthreads()

  # get 'data' in either sparse or dense matrix format
  if (is(data, "sparseMatrix")) {
    data <- as(data, "dgCMatrix")
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
  } else stop("'data' was not coercible to a matrix")

  # randomly initialize "w", or check dimensions of provided initialization
  w_init <- list()
  if(is.matrix(seed)) seed <- list(seed)
  if (!is.null(seed)){
    if(is.matrix(seed)[[1]]){
      if(!is.list(seed)) stop("if specifying initial 'w' matrices for 'seed', specify a list of matrices.")
      for(i in 1:length(seed)){
        if (ncol(seed[[i]]) == nrow(data) && nrow(seed[[i]]) == k) {
          w_init[[i]] <- seed[[i]]
        } else if (nrow(seed[[i]]) == nrow(data) && ncol(seed[[i]]) == k) {
          w_init[[i]] <- t(seed[[i]])
        } else stop("dimensions of provided initial 'w' matrices in 'seed' were incompatible with dimensions of 'data' and/or 'k'")
      }
    } else if(is.numeric(seed[[1]])){
      for(i in 1:length(seed)){
        set.seed(seed[[i]])
        w_init[[i]] <- matrix(runif(k * nrow(data)), k, nrow(data))
      }
    }
  } else {
    w_init[[1]] <- matrix(runif(k * nrow(data)), k, nrow(data))
  }
  
  # call C++ routines
  if (class(data)[[1]] == "dgCMatrix") {
    model <- Rcpp_nmf_sparse(data, tol, maxit, verbose, nonneg, L1, p$diag, threads, w_init, mask_zeros)
  } else {
    model <- Rcpp_nmf_dense(data, tol, maxit, verbose, nonneg, L1, p$diag, threads, w_init)
  }

  # add back dimnames
  if(!is.null(rownames(data))) rownames(model$w) <- rownames(data)
  if(!is.null(colnames(data))) colnames(model$h) <- colnames(data)
  rownames(model$h) <- colnames(model$w) <- paste0("nmf", 1:ncol(model$w))
  model$runtime <- difftime(Sys.time(), start_time, units = "secs")
  class(model) <- "nmf"
  return(model)
}

#' Mean squared error of model fit
#' 
#' \code{mse} is a generic function for evaluating the fit of models using Mean Squared Error (MSE).
#' 
#' @param object fitted model
#' @param data data to which the model was fit
#' @param ... additional arguments affecting the fit of the model
#'  
#' @seealso \code{\link{mse.nmf}}
#' @export
mse <- function(object, data, ...) {
  UseMethod("mse")
}

#' @param object fitted model, class \code{nmf}, generally the result of calling \code{nmf}
#' @param data input data with the same number of rows as \code{object$w}
#' @param mask_zeros handle zeros as missing values
#' @importFrom methods is
#' @rdname nmf
#' @export
mse.nmf <- function(object, data, mask_zeros = FALSE, ...) {
  validate_nmf(object)
  threads <- getRcppMLthreads()

  if (is(data, "sparseMatrix")) {
    data <- as(data, "dgCMatrix")
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
    if(mask_zeros) data <- as(data, "dgCMatrix")
  } else stop("'data' was not coercible to a matrix")
  
  if (nrow(object$w) != nrow(data)) stop("dimensions of 'w' and 'A' are incompatible")
  if (ncol(object$h) != ncol(data)) stop("dimensions of 'h' and 'A' are incompatible")
  
  if (class(data)[[1]] == "dgCMatrix") {
    Rcpp_mse_sparse(data, t(object$w), object$d, object$h, threads, mask_zeros)
  } else {
    Rcpp_mse_dense(data, t(object$w), object$d, object$h, threads)
  }
}

#' @importFrom stats predict
#' @param object model of class \code{nmf}, generally the result of calling \code{nmf}
#' @param newdata matrix of features-by-samples in dense or sparse format, coercible to \code{matrix} or \code{Matrix::dgCMatrix}, must contain the same number of rows as \code{object$w}.
#' @param mask_zeros handle zeros as missing values
#' @export
#' @method predict nmf
#' @rdname nmf
predict.nmf <- function(object, newdata, L1 = 0, nonneg = TRUE, mask_zeros = FALSE, ...) {
  validate_nmf(object)
  
  # get 'A' in either sparse or dense matrix format
  if (is(newdata, "sparseMatrix")) {
    newdata <- as(newdata, "dgCMatrix")
  } else if (canCoerce(newdata, "matrix")) {
    newdata <- as.matrix(newdata)
    if(mask_zeros) newdata <- as(newdata, "dgCMatrix")
  } else stop("'newdata' was not coercible to a matrix")
  
  # check that dimensions of w or h are compatible with A
  if(nrow(object$w) != nrow(newdata)) stop("dimensions of 'newdata' and 'object$w' are incompatible!")
  
  threads <- getRcppMLthreads()
  
  # select backend based on whether 'A' is dense or sparse
  if (class(newdata)[[1]] == "dgCMatrix") {
    h <- Rcpp_projectW_sparse(newdata, t(object$w), nonneg, L1, threads, mask_zeros)
  } else {
    h <- Rcpp_projectW_dense(newdata, t(object$w), nonneg, L1, threads)
  }
  result <- new_nmf(w = object$w, d = rowSums(h), h = t(apply(h, 1, function(x) x / sum(x))))
  if(!is.null(colnames(newdata))) colnames(result$h) <- colnames(newdata)
  rownames(result$h) <- paste0("nmf", 1:nrow(result$h))
  result
}