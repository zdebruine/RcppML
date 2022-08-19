#' @rdname summary-nmf-method
#' @export
#' @param x \code{nmfSummary} object, the result of calling \code{summary} on an \code{nmf} object
#' @method plot nmfSummary
plot.nmfSummary <- function(x, ...){
  ggplot(x, aes(x = factor(factor, levels = unique(factor)), y = stat, fill = group)) + 
    geom_bar(position = "fill", stat = "identity") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + 
    labs(x = "NMF factor", y = "Representation in group") + 
    scale_y_continuous(expand = c(0, 0))
}

#' @rdname crossValidate
#' @export
#' @param x \code{nmfCrossValidate} object, the result of \code{crossValidate}
#' @method plot nmfCrossValidate
plot.nmfCrossValidate <- function(x, ...){
  ggplot(x, aes(x = k, y = value, color = rep)) + 
    geom_point() + 
    geom_line() + 
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(aspect.ratio = 1) +
    scale_y_continuous(trans = "log10")
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
#' @method biplot nmf
#' @seealso \code{\link{nmf}}
setMethod("biplot", signature = "nmf", function(x, factors = c(1,2), matrix = "w", group_by = NULL, ...) {
  validObject(x)
  if(length(factors) != 2 || max(factors) > ncol(x$w)) stop("'factors' must be of length 2 containing values less than the rank of the nmf model")
  if(!(matrix %in% c("w", "h"))) stop("'matrix' must be either 'w' or 'h'")
  if(!is.null(group_by)){
    if(matrix == "w" && length(group_by) != nrow(x$w)) stop("length of 'group_by' is not equal to nrow(object$w)")
    if(matrix == "h" && length(group_by) != ncol(x$h)) stop("length of 'group_by' is not equal to ncol(object$h)")
  }
  if(matrix == "w") model <- x$w[, factors]
  if(matrix == "h") model <- t(x$h[factors, ])
  if(!is.null(group_by)) {
    if(is.null(rownames(model))) rownames(model) <- paste0("item", 1:nrow(model))
    df <- data.frame("group" = group_by, "factor1" = model[,1], "factor2" = model[,2], "labels" = rownames(model))
    ggplot(df, aes(x = factor1, y = factor2, color = group, label = labels)) + 
      geom_point() + 
      theme_classic() + 
      labs(x = paste0("NMF factor ", factors[1]), y = paste0("NMF factor ", factors[2]))
  } else {
    df <- data.frame("factor1" = model[,1], "factor2" = model[,2], "labels" = rownames(model))
    ggplot(df, aes(x = factor1, y = factor2, label = labels)) + 
      geom_point() + 
      theme_classic() + 
      labs(x = paste0("NMF factor ", factors[1]), y = paste0("NMF factor ", factors[2]))
  }
})