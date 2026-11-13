#' Deprecated functions in RcppML
#'
#' These functions were part of earlier RcppML releases and are kept as thin
#' wrappers so that existing code keeps working. Each one issues a deprecation
#' warning and calls its replacement.
#'
#' \describe{
#'   \item{\code{project(A, w = NULL, h = NULL, ...)}}{Use
#'     \code{\link{nnls}(w = w, A = A)} to solve for \code{h}, or
#'     \code{nnls(h = h, A = A)} to solve for \code{w}. For a fitted model use
#'     \code{\link[=predict,nmf-method]{predict}}.}
#'   \item{\code{crossValidate(A, k, ...)}}{Use \code{\link{nmf}} with a vector
#'     of ranks and \code{test_fraction}.}
#' }
#'
#' @param A data matrix (dense or sparse), features in rows and samples in columns.
#' @param w,h factor matrices; supply exactly one. See \code{\link{nnls}}.
#' @param nonneg enforce non-negativity of the solution.
#' @param L1 L1/LASSO penalty in \code{[0, 1)}.
#' @param mask_zeros not supported by \code{project()} any more; use
#'   \code{predict(model, data, mask = "zeros")}.
#' @param k vector of ranks to evaluate.
#' @param reps number of cross-validation replicates (independent holdout sets).
#' @param n fraction of entries held out as the test set.
#' @param ... further arguments passed to the replacement function.
#' @return \code{project()} returns the solved factor matrix, as
#'   \code{\link{nnls}} does. \code{crossValidate()} returns what
#'   \code{\link{nmf}} returns for the same arguments (a data frame of class
#'   \code{nmfCrossValidate} when \code{k} is a vector).
#' @name RcppML-deprecated
#' @keywords internal
#' @examples
#' \donttest{
#' A <- abs(Matrix::rsparsematrix(50, 20, 0.5))
#' model <- nmf(A, 3, seed = 1, threads = 1)
#' h <- suppressWarnings(project(A, w = model@w))
#' }
NULL

#' @rdname RcppML-deprecated
#' @export
project <- function(A, w = NULL, h = NULL, nonneg = TRUE, L1 = 0,
                    mask_zeros = FALSE, ...) {
  .Deprecated(msg = paste0(
    "project() is deprecated. Use nnls(w = w, A = A) or nnls(h = h, A = A), ",
    "or predict() for a fitted nmf model."))
  if (isTRUE(mask_zeros))
    stop("project(mask_zeros = TRUE) is no longer supported. ",
         "Use predict(model, data, mask = \"zeros\").", call. = FALSE)
  if (is.null(w) == is.null(h))
    stop("project(): supply exactly one of 'w' or 'h'", call. = FALSE)
  if (!is.null(w)) {
    nnls(w = w, A = A, L1 = L1, nonneg = nonneg, ...)
  } else {
    nnls(h = h, A = A, L1 = L1, nonneg = nonneg, ...)
  }
}

#' @rdname RcppML-deprecated
#' @export
crossValidate <- function(A, k, reps = 3, n = 0.05, ...) {
  .Deprecated(msg = paste0(
    "crossValidate() is deprecated. Use nmf(data, k = <vector of ranks>, ",
    "test_fraction = ..., cv_seed = ...)."))
  nmf(A, k = k, test_fraction = n, cv_seed = seq_len(reps), ...)
}
