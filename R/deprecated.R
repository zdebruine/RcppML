#' Deprecated functions from RcppML 0.3.7
#'
#' These functions are provided for backward compatibility with the CRAN 0.3.7
#' API but are deprecated and will be removed in a future version.
#'
#' @name RcppML-deprecated
#' @keywords internal
NULL

#' @title Project model onto data (DEPRECATED)
#'
#' @description
#' \strong{Deprecated}. Use \code{\link[stats]{predict}} instead.
#'
#' \code{project} has been replaced by \code{\link[stats]{predict}} (the S4
#' method for class \code{\link{nmf}}).
#'
#' @param w feature factor matrix, or an \code{nmf} model object
#' @param data dense or sparse input matrix (features x samples)
#' @param ... additional arguments passed to \code{\link[stats]{predict}} or
#'   \code{\link{nnls}}
#' @return matrix \code{h} of dimension \code{(k, ncol(data))}
#' @seealso \code{\link{nmf}}, \code{\link{nnls}}
#' @export
#' @keywords internal
project <- function(w, data, ...) {
  .Deprecated("predict", package = "RcppML",
              msg = paste0("'project()' is deprecated.\n",
                           "Use 'predict(model, data)' for nmf objects, or ",
                           "'nnls(w = ..., A = ...)' for raw matrix projection."))
  if (is(w, "nmf")) {
    return(predict(w, data, ...))
  }
  # Old API: w is a raw matrix, project onto data
  nnls(w = w, A = data, ...)
}

#' @title Cross-validate NMF (DEPRECATED)
#'
#' @description
#' \strong{Deprecated}. Use \code{\link{nmf}} with \code{test_fraction} instead.
#'
#' \code{crossValidate} has been replaced by passing a vector of ranks and a
#' \code{test_fraction} to \code{\link{nmf}}.
#'
#' @param A input data matrix (features x samples)
#' @param k integer vector of ranks to test
#' @param ... additional arguments passed to \code{\link{nmf}}
#' @return An \code{nmfCrossValidate} data frame
#' @seealso \code{\link{nmf}}
#' @export
#' @keywords internal
crossValidate <- function(A, k, ...) {
  .Deprecated("nmf", package = "RcppML",
              msg = paste0("'crossValidate()' is deprecated.\n",
                           "Use 'nmf(A, k = c(...), test_fraction = ...)' instead."))
  nmf(A, k = k, ...)
}
