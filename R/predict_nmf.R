#' Project an NMF model onto new samples
#'
#' Given an NMF model in the form \eqn{A = wdh}, \code{project} projects \code{w} onto \code{A} to solve for \code{h}.
#'
#' @details
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
#' @param threads number of threads for OpenMP parallelization (default 0 = all available)
#' @param verbose print progress information (default FALSE)
#' @param ... arguments passed to or from other methods
#' @param upper_bound maximum value permitted in least squares solutions, essentially a bounded-variable least squares problem between 0 and \code{upper_bound}
#' @export
#' @rdname nmf-class-methods
#' @returns An \code{\link{nmf}} object containing the projected model (with updated \code{h} matrix).
#' @references
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' @author Zach DeBruine
#' @seealso \code{\link{nnls}}, \code{\link{nmf}}
#' @examples \donttest{
#' library(Matrix)
#' w <- matrix(runif(1000 * 10), 1000, 10)
#' h_true <- matrix(runif(10 * 100), 10, 100)
#' # A is the crossproduct of "w" and "h" with 10% signal dropout
#' mask <- rsparsematrix(1000, 100, density = 0.1)
#' A <- (w %*% h_true) * (mask != 0)
#' h <- nnls(w = w, A = A)
#' cor(as.vector(h_true), as.vector(h))
#'
#' # alternating projections refine solution (like NMF)
#' h <- nnls(w = w, A = A)
#' w <- nnls(h = h, A = A)
#' h <- nnls(w = w, A = A)
#' w <- nnls(h = h, A = A)
#' h <- nnls(w = w, A = A)
#' w <- nnls(h = h, A = A)
#' }
setMethod("predict", signature = "nmf", function(object, data, L1 = NULL, L2 = NULL, mask = NULL, upper_bound = NULL, threads = 0, verbose = FALSE, ...) {
  validObject(object)

  # Use config from model when not explicitly provided
  if (is.null(L1)) L1 <- if (!is.null(object@misc$L1)) object@misc$L1[2] else 0
  if (is.null(L2)) L2 <- if (!is.null(object@misc$L2)) object@misc$L2[2] else 0
  if (is.null(upper_bound)) upper_bound <- if (!is.null(object@misc$upper_bound)) object@misc$upper_bound[2] else 0

  if (length(L1) != 1) stop("'L1' must be a single value giving the penalty on 'h'")
  if (L1 >= 1 || L1 < 0) stop("L1 penalty must be strictly in the range [0,1)")
  if (length(L2) != 1) stop("'L2' must be a single value giving the penalty on 'h'")
  if (L2 < 0) stop("L2 penalty must be strictly >= 0")

  # Unified input validation (supports file paths, sparse, dense)
  data_info <- validate_data(data)
  data <- data_info$data
  
  # NA detection
  has_na <- FALSE
  if (is(data, "dgCMatrix")) {
    has_na <- any(is.na(data@x))
  } else if (is.matrix(data)) {
    has_na <- any(is.na(data))
  }
  if (has_na && !is.null(mask) && !identical(mask, "NA"))
    stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
  
  # Validate mask (supports NULL, "zeros", "NA", matrix, list("zeros", <matrix>))
  mask_result <- validate_mask(mask, has_na = has_na)
  mask_matrix <- mask_result$mask_matrix
  mask_zeros  <- mask_result$mask_zeros

  if (nrow(object@w) == nrow(data) && ncol(object@w) != nrow(data)) {
    w <- t(object@w)
  } else if (ncol(object@w) == nrow(data)) {
    w <- object@w
  }
  if (ncol(w) != nrow(data)) stop("dimensions of 'object@w' and 'A' are not compatible")

  h <- Rcpp_predict(data, mask_matrix, w, L1[1], L2[1], as.integer(threads), mask_zeros, upper_bound)
  if (!is.null(colnames(data))) colnames(h) <- colnames(data)
  rownames(h) <- paste0("nmf", 1:nrow(h))

  # Return a new nmf object with the projected h
  new("nmf",
      w = if (nrow(object@w) == ncol(w)) object@w else t(w),
      d = object@d,
      h = h,
      misc = list(projected = TRUE))
})
