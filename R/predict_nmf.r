#' Project an NMF model onto new samples
#' 
#' Given an NMF model in the form \eqn{A = wdh}, \code{predict.nmf} projects \code{w} onto \code{A} to solve for \code{h}.
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
#' ... arguments passed to or from other methods
#' @export
#' @method predict nmf
#' @rdname predict.nmf
#' @returns object of class \code{\link{nmf}}
#' @references
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' @author Zach DeBruine
#' @export
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @examples
#' \dontrun{
#' TO DO
#' library(Matrix)
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' # A is the crossproduct of "w" and "h" with 10% signal dropout
#' A <- (w %*% h_true) * (rsparsematrix(1000, 100, 0.9) > 0)
#' h <- project(A, w)
#' cor(as.vector(h_true), as.vector(h))
#' 
#' # alternating projections refine solution (like NMF)
#' mse_bad <- mse(A, w, rep(1, ncol(w)), h) # mse before alternating updates
#' h <- project(A, w = w)
#' w <- project(A, h = h)
#' h <- project(A, w)
#' w <- project(A, h = h)
#' h <- project(A, w)
#' w <- t(project(A, h = h))
#' mse_better <- mse(A, w, rep(1, ncol(w)), h) # mse after alternating updates
#' mse_better < mse_bad
#'
#' # two ways to solve for "w" that give the same solution
#' w <- project(A, h = h)
#' w2 <- project(t(A), w = t(h))
#' all.equal(w, w2)
#' }
setMethod("predict", signature = "nmf", function(object, data, L1 = 0, nonneg = TRUE, mask = NULL, ...) {
  validate_nmf(object)

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
    mask_matrix <- new("ngCMatrix")
    mask_zeros <- FALSE
  } else if (class(mask) == "character" && mask == "zeros") {
    mask_matrix <- new("ngCMatrix")
    mask_zeros <- TRUE
  } else {
    mask_zeros <- FALSE
    if (!canCoerce(mask, "ngCMatrix")) {
      if (canCoerce(mask, "matrix")) {
        mask <- as.matrix(matrix)
      } else stop("could not coerce the value of 'mask' to a sparse pattern matrix (ngCMatrix)")
    }
    mask_matrix <- as(mask, "ngCMatrix")
  }

  w <- t(object@w)
  if (ncol(w) != nrow(data)) stop("dimensions of 'object@w' and 'A' are not compatible")

  if (class(data)[[1]] == "dgCMatrix") {
    h <- Rcpp_predict_sparse(newdata, mask_matrix, seed[[1]], nonneg, L1[1], getOption("RcppML.threads"), mask_zeros)
  } else {
    h <- Rcpp_predict_dense(newdata, mask_matrix, seed[[1]], nonneg, L1[1], getOption("RcppML.threads"), mask_zeros)
  }
  if (!is.null(colnames(data))) colnames(h) <- colnames(data)
  rownames(h) <- paste0("nmf", 1:nrow(h))
  h
})

#' @rdname predict.nmf
#' @param w matrix of features (rows) by factors (columns), corresponding to rows in \code{data}
predict.nmf <- function(w, data, L1 = 0, nonneg = TRUE, mask = NULL, ...){
  m <- new("nmf", w = w, d = rep(1:ncol(w)), h = matrix(0, nrow = ncol(w), 1))
  m.predict(data)
}