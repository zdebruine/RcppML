#' @rdname summary-nmf-method
#' @export
#' @param x \code{nmfSummary} object, the result of calling \code{summary} on an \code{nmf} object
#' @param ... Additional arguments (unused).
#' @return A \code{ggplot2} object showing factor representation by group.
#' @seealso \code{\link{summary,nmf-method}}, \code{\link{nmf}}
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
#' groups <- factor(sample(c("A", "B"), 50, replace = TRUE))
#' s <- summary(model, group_by = groups)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(s)
#' }
#' }
#'
#' @method plot nmfSummary
plot.nmfSummary <- function(x, ...){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Please install it.", call. = FALSE)
  }
  ggplot2::ggplot(x, ggplot2::aes(x = factor(factor, levels = unique(factor)), y = stat, fill = group)) + 
    ggplot2::geom_bar(position = "fill", stat = "identity") +
    ggplot2::theme_classic() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, vjust = 1, hjust = 1)) + 
    ggplot2::labs(x = "NMF factor", y = "Representation in group") + 
    ggplot2::scale_y_continuous(expand = c(0, 0))
}

#' Biplot for NMF factors
#' 
#' Produces a biplot from the output of \code{\link{nmf}}
#' 
#' @param x an object of class "\code{nmf}"
#' @param factors length 2 vector specifying factors to plot.
#' @param matrix either \code{w} or \code{h}
#' @param group_by a discrete factor giving groupings for samples or features. Must be of the same length as number of samples in \code{object$h} or number of features in \code{object$w}.
#' @param ... for consistency with \code{biplot} generic
#' @export
#' @return ggplot2 object
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   biplot(model, factors = c(1, 2))
#' }
#' }
#'
#' @method biplot nmf
#' @seealso \code{\link{nmf}}
setMethod("biplot", signature = "nmf", function(x, factors = c(1,2), matrix = "w", group_by = NULL, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Please install it.", call. = FALSE)
  }
  validObject(x)
  if(length(factors) != 2 || max(factors) > ncol(x$w)) stop("'factors' must be of length 2 containing values less than the rank of the nmf model")
  if(!(matrix %in% c("w", "h"))) stop("'matrix' must be either 'w' or 'h'")
  if(!is.null(group_by)){
    if(matrix == "w" && length(group_by) != nrow(x$w)) stop("length of 'group_by' is not equal to nrow(object$w)")
    if(matrix == "h" && length(group_by) != ncol(x$h)) stop("length of 'group_by' is not equal to ncol(object$h)")
  }
  if(matrix == "w") model <- x$w[, factors]
  if(matrix == "h") model <- t(x$h[factors, ])
  if(is.null(rownames(model))) rownames(model) <- paste0("item", 1:nrow(model))
  if(!is.null(group_by)) {
    df <- data.frame("group" = group_by, "factor1" = model[,1], "factor2" = model[,2], "labels" = rownames(model))
    ggplot2::ggplot(df, ggplot2::aes(x = factor1, y = factor2, color = group, label = labels)) + 
      ggplot2::geom_point() + 
      ggplot2::theme_classic() + 
      ggplot2::labs(x = paste0("NMF factor ", factors[1]), y = paste0("NMF factor ", factors[2]))
  } else {
    df <- data.frame("factor1" = model[,1], "factor2" = model[,2], "labels" = rownames(model))
    ggplot2::ggplot(df, ggplot2::aes(x = factor1, y = factor2, label = labels)) + 
      ggplot2::geom_point() + 
      ggplot2::theme_classic() + 
      ggplot2::labs(x = paste0("NMF factor ", factors[1]), y = paste0("NMF factor ", factors[2]))
  }
})