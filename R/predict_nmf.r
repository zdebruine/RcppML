#' Project an NMF model onto new samples
#' 
#' Given an NMF model in the form \eqn{A = wdh}, \code{project} projects \code{w} onto \code{A} to solve for \code{h}.
#' 
#' @details
#' TO DO: add documentation
#' For the alternating least squares matrix factorization update problem \eqn{A = wh}, the updates 
#' (or projection) of \eqn{h} is given by the equation: \deqn{w^Twh = wA_j} 
#' which is in the form \eqn{ax = b} where \eqn{a = w^Tw} \eqn{x = h} and \eqn{b = wA_j} for all columns \eqn{j} in \eqn{A}.
#'
#' Any L1 penalty is subtracted from \eqn{b} and should generally be scaled to \code{max(b)}, where \eqn{b = WA_j} for all columns \eqn{j} in \eqn{A}. An easy way to properly scale an L1 penalty is to normalize all columns in \eqn{w} to sum to the same value (e.g. 1). No scaling is applied in this function. Such scaling guarantees that \code{L1 = 1} gives a completely sparse solution.
#'
#' There are specializations for dense and sparse input matrices, symmetric input matrices, and for rank-1 and rank-2 projections. See documentation for \code{\link{nmf}} for theoretical details and guidance.
#' 
#' @importFrom stats predict
#' @inheritParams nmf
#' @param object fitted model, class \code{nmf}, generally the result of calling \code{nmf}, with models of equal dimensions as \code{data}
#' @param L1 a single LASSO penalty in the range (0, 1]
#' @param L2 a single Ridge penalty greater than zero
#' @param ... arguments passed to or from other methods
#' @export
#' @rdname nmf-class-methods
#' @returns object of class \code{\link{nmf}}
#' @references
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' @author Zach DeBruine
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @examples \dontrun{
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' # A is the crossproduct of "w" and "h" with 10% signal dropout
#' A <- (w %*% h_true) * (r_sparsematrix(1000, 100, 10) > 0)
#' h <- project(w, A)
#' cor(as.vector(h_true), as.vector(h))
#' 
#' # alternating projections refine solution (like NMF)
#' mse_bad <- mse(w, rep(1, ncol(w)), h, A) # mse before alternating updates
#' h <- project(w, A)
#' w <- t(project(h, t(A)))
#' h <- project(w, A)
#' w <- t(project(h, t(A)))
#' h <- project(w, A)
#' w <- t(project(h, t(A)))
#' mse_better <- mse(w, rep(1, ncol(w)), h, A) # mse after alternating updates
#' mse_better < mse_bad
#' }
setMethod("predict", signature = "nmf", function(object, data, L1 = 0, L2 = 0, mask = NULL, ...) {
  validObject(object)

  if (length(L1) != 1) stop("'L1' must be a single value giving the penalty on 'h'")
  if (L1 >= 1 || L1 < 0) stop("L1 penalty must be strictly in the range [0,1)")
  if (length(L2) != 1) stop("'L1' must be a single value giving the penalty on 'h'")
  if (L2 < 0) stop("L2 penalty must be strictly >= 0")

  if (is(data, "sparseMatrix")) {
    if (class(data)[[1]] != "dgCMatrix") data <- as(data, "dgCMatrix")
    if (any(is.na(data@x))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      warning("NA values were detected in the data. Setting \"mask = 'NA'\"")
      mask <- is.na(data)
    }
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
    if (!is.numeric(data)) data <- as.numeric(data)
    if (any(is.na(data))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      warning("NA values were detected in the data. Setting \"mask = 'NA'\"")
      mask <- is.na(as(data, "dgCMatrix"))
    }
  } else stop("'data' was not coercible to a matrix")

  if (is.null(mask)) {
    mask_matrix <- new("dgCMatrix")
    mask_zeros <- FALSE
  } else if (class(mask) == "character" && mask == "zeros") {
    mask_matrix <- new("dgCMatrix")
    mask_zeros <- TRUE
  } else {
    mask_zeros <- FALSE
    if (!canCoerce(mask, "dgCMatrix")) {
      if (canCoerce(mask, "matrix")) {
        mask <- as.matrix(matrix)
      } else stop("could not coerce the value of 'mask' to a sparse pattern matrix (dgCMatrix)")
    }
    mask_matrix <- as(mask, "dgCMatrix")
  }

  if (nrow(object@w) == nrow(data) && ncol(object@w) != nrow(data)) {
    w <- t(object@w)
  } else if(ncol(object@w) == nrow(data)) {
    w <- object@w
  }
  if (ncol(w) != nrow(data)) stop("dimensions of 'object@w' and 'A' are not compatible")

  if (class(data)[[1]] == "dgCMatrix") {
    h <- Rcpp_predict_sparse(data, mask_matrix, w, L1[1], L2[1], getOption("RcppML.threads"), mask_zeros)
  } else {
    h <- Rcpp_predict_dense(data, mask_matrix, w, L1[1], L2[1], getOption("RcppML.threads"), mask_zeros)
  }
  if (!is.null(colnames(data))) colnames(h) <- colnames(data)
  rownames(h) <- paste0("nmf", 1:nrow(h))
  h
})

#' Project a model onto new data
#' 
#' Equivalent to \code{predict} method for NMF, but requires only the \code{w} matrix to be supplied and not the entire NMF model. Use NNLS to project a basis factor model onto new samples.
#'
#' @details 
#' See \code{\link{nmf}} for more info, as well as the \code{predict} method for NMF.
#' 
#' @rdname project
#' @param w matrix of features (rows) by factors (columns), corresponding to rows in \code{data}
#' @param data a dense or sparse matrix
#' @param L1 L1/LASSO penalty
#' @param L2 L2/Ridge penalty
#' @param mask masking on data values
#' @export
project <- function(w, data, L1 = 0, L2 = 0, mask = NULL, ...){
  m <- new("nmf", w = w, d = rep(1:ncol(w)), h = matrix(0, nrow = ncol(w), 1))
  predict(m, data, L1 = L1, L2 = L2, mask = mask, ...)
}