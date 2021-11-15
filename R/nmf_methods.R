#' @importFrom methods new validObject
#' @slot w feature factor matrix
#' @slot d scaling diagonal vector
#' @slot h sample factor matrix
#' @slot misc list often containing components:
#'  \itemize{
#'    \item tol     : tolerance of fit
#'    \item iter    : number of fitting updates
#'    \item runtime : runtime in seconds
#'    \item mse     : mean squared error of model (calculated for multiple starts only)
#'    \item w_init  : initial w matrix used for model fitting
#'  }
#' @name nmf
#' @aliases nmf, nmf-class
#' @exportClass nmf
#'
setClass("nmf",
  representation(w = "matrix", d = "numeric", h = "matrix", misc = "list"),
  prototype(w = matrix(), d = NA_real_, h = matrix(), misc = list()),
  validity = function(object) {
    msg <- NULL
    if (ncol(object@w) != nrow(object@h))
      msg <- c(msg, "ranks of 'w' and 'h' are not equal")
    if (ncol(object@w) != length(object@d))
      msg <- c(msg, "ranks of 'd' and 'w' are not equal")
    if (nrow(object@h) != length(object@d))
      msg <- c(msg, "ranks of 'd' and 'h' are not equal")
    if (is.null(msg)) TRUE else msg
  })

#' @method subset nmf
#' @rdname nmf-methods
#' @export
setMethod("subset", signature("nmf"), function(x, i, ...) {
  validObject(x)
  x@w <- x@w[, i]
  x@h <- x@h[i,]
  x@d <- x@d[i]
  x
})

#' 
#' @export
#' @rdname nmf-methods
setMethod("[", signature("nmf"), function(x, i) {
  validObject(x)
  x@w <- x@w[, i]
  x@h <- x@h[i,]
  x@d <- x@d[i]
  x
})

#' nmf class methods
#' 
#' @method head nmf
#' @importFrom utils str
#' @export
#' @docType methods
#' @rdname nmf-methods
#' 
setMethod("head", signature("nmf"), function(x, n = getOption("digits"), ...) {
  validObject(x)
  cat(nrow(x@w), "x", ncol(x@h), "x", length(x@d), "factor model of class \"nmf\"\n")
  n <- n_h <- n_w <- n
  if (nrow(x@w) < n) n_w <- nrow(x@w)
  if (ncol(x@h) < n) n_h <- ncol(x@h)
  if (length(x@d) < n) n <- length(x@d)
  cat("@ w\n")
  print.table(x@w[1:n_w, 1:n], zero.print = ".")
  if (nrow(x@w) > n_w && ncol(x@w) > n) {
    cat("...suppressing", nrow(x@w) - n_w, "rows and", ncol(x@w) - n, "columns\n")
  } else if (nrow(x@w) > n_w) {
    cat("...suppressing", nrow(x@w) - n_w, "rows\n")
  } else if (ncol(x@w) > n) {
    cat("...suppressing", ncol(x@w) - n, "columns\n")
  }
  cat("\n@ d\n")
  d_ <- x@d[1:n]
  for (i in 1:length(d_))
    cat(d_[i], " ")
  cat("\n")
  if (length(x@d) > n) cat("...suppressing", length(x@d) - n, "values\n")
  cat("\n@ h\n")
  print.table(x@h[1:n, 1:n_h], zero.print = ".")
  if (ncol(x@h) > n_h && nrow(x@h) > n) {
    cat("...suppressing", nrow(x@h) - n, "rows and", ncol(x@h) - n_h, "columns\n")
  } else if (ncol(x@h) > n_h) {
    cat("...suppressing", ncol(x@h) - n_h, "columns\n")
  } else if (nrow(x@h) > n) {
    cat("...suppressing", nrow(x@h), "rows\n")
  }
  cat("\n@ misc\n")
  cat(str(x@misc))

  invisible(x)
})

#' @export
#' @method show nmf
#' @rdname nmf-methods
#' 
setMethod("show", signature("nmf"), function(object) {
  head(object)
  invisible(object)
})

#' @export
#' @method dimnames nmf
#' @rdname nmf-methods
#' 
setMethod("dimnames", signature = "nmf", function(x) {
  validObject(x)
  print(list(rownames(x@w), colnames(x@h)))
})

#' @export
#' @method dim nmf
#' @rdname nmf-methods
#' 
setMethod("dim", signature = "nmf", function(x) {
  validObject(x)
  c(nrow(x@w), ncol(x@h), ncol(x@w))
})

#' @export
#' @method t nmf
#' @rdname nmf-methods
#' 
setMethod("t", signature = "nmf", function(x) {
  validObject(x)
  tx <- new("nmf", w = t(x@h), d = x@d, h = t(x@w), misc = x$misc)
  tx
})

#' @export
#' @method sort nmf
#' @rdname nmf-methods
#' 
setMethod("sort", signature = "nmf", function(x, decreasing = TRUE, ...) {
  validObject(x)
  x <- x[order(x@d, decreasing = decreasing)]
  x
})

#' @export
#' @method prod nmf
#' @rdname nmf-methods
#' 
setMethod("prod", signature = "nmf", function(x, ...) {
  validObject(x)
  return(crossprod(t(as.matrix(x@w %*% Diagonal(x = x@d))), x@h))
})

#' @importFrom methods slot
#' @rdname nmf-methods
#' @export
setMethod("$", signature = "nmf", function(x, name) {
  validObject(x)
  name <- tolower(name)
  if (!(name %in% c("w", "d", "h", "misc"))) {
    if (!(name %in% names(x@misc))) {
      stop("'name' not a slot in 'object' or component in 'object@misc'")
    } else x@misc[[name]]
  } else slot(x, name)
})

#' @export
#' @rdname nmf-methods
setMethod("coerce", signature(from = "nmf", to = "list"), function(from, to) {
  list("w" = slot(from, "w"), "d" = slot(from, "d"), "h" = slot(from, "h"), "misc" = slot(from, "misc"))
})

setGeneric("sparsity", function(object, ...) standardGeneric("sparsity"))

#' Compute the sparsity of each NMF factor
#'
#' @details
#' For \code{\link{nmf}} models, the sparsity of each factor is computed and summarized
#' or \eqn{w} and \eqn{h} matrices. A long \code{data.frame} with columns \code{factor}, \code{sparsity}, and \code{model} is returned.
#' @export
#' @rdname sparsity
#' @method sparsity nmf
setMethod("sparsity", signature = "nmf", function(object, ...) {
  validObject(object)
  w <- colSums(object$w == 0) / nrow(object$w)
  h <- rowSums(object$h == 0) / ncol(object$h)
  w <- data.frame("factor" = names(w), "sparsity" = as.vector(w))
  h <- data.frame("factor" = names(h), "sparsity" = as.vector(h))
  w$model <- "w"
  h$model <- "h"
  result <- rbind(w, h)
  result$model <- as.factor(result$model)
  result
})

setGeneric("align", function(object, ...) standardGeneric("align"))

#' Align two NMF models
#'
#' @details
#' For \code{\link{nmf}} models, factors in \code{object} are reordered to minimize the cost of bipartite matching
#' (see \code{\link{bipartiteMatch}}) on a \code{\link{cosine}} or correlation distance matrix. The \eqn{w} matrix is used for matching, and
#' must be equidimensional in \code{object} and \code{ref}.
#'
#' @param object nmf model to be aligned to \code{ref}
#' @param ref reference nmf model to which \code{object} will be aligned
#' @param method either \code{cosine} or \code{cor}
#' @param ... arguments passed to or from other methods
#' @export
#' @importFrom stats cor
#' @rdname align.nmf
#' @method align nmf
#'
setMethod("align", signature = "nmf", function(object, ref, method = "cosine", ...) {
  validObject(object)
  if (all(dim(ref$w) != dim(object$w))) stop("dimensions of object$w and ref$w are not identical")
  if (method == "cosine") {
    cost <- 1 - cosine(object$w, ref$w) + 1e-10
  } else if (method == "cor") {
    cost <- 1 - cor(object$w, ref$w) + 1e-10
  }
  cost[cost < 0] <- 0
  object[bipartiteMatch(cost)$pairs]
})

#' @rdname align.nmf
#' @param w matrix with columns to be aligned to columns in \code{wref}
#' @param wref reference matrix to which columns in \code{w} will be aligned
align.nmf <- function(w, wref, method = "cosine", ...) {
  if (all(dim(wref) != dim(w))) stop("dimensions of 'w' and 'wref' are not identical")
  if (method == "cosine") {
    cost <- 1 - cosine(w, wref) + 1e-10
  } else if (method == "cor") {
    cost <- 1 - cor(w, wref) + 1e-10
  }
  cost[cost < 0] <- 0
  w[bipartiteMatch(cost)$pairs]
}

#' Summarize NMF factors
#'
#' \code{summary} method for class "\code{nmf}". Describes metadata representation in NMF factors. Returns object of class \code{nmfSummary}. Plot results using \code{plot}.
#'
#' @param object an object of class "\code{nmf}", usually, a result of a call to \code{\link{nmf}}
#' @param group_by a discrete factor giving groupings for samples or features. Must be of the same length as number of samples in \code{object$h} or number of features in \code{object$w}.
#' @param stat either \code{sum} (sum of factor weights falling within each group), or \code{mean} (mean factor weight falling within each group).
#' @param ... arguments passed to or from other methods
#' @export
#' @return \code{data.frame} with columns \code{group}, \code{factor}, and \code{stat}
#' @method summary nmf
setMethod("summary", signature = "nmf", function(object, group_by, stat = "sum", ...) {
  validObject(object)
  if (!is.factor(group_by)) group_by <- as.factor(group_by)
  bins <- list()
  if (length(group_by) == ncol(object$h)) {
    for (group in levels(group_by)) {
      if (stat == "sum") {
        bins[[group]] <- rowSums(object$h[, which(group_by == group)])
      } else {
        bins[[group]] <- rowMeans(object$h[, which(group_by == group)])
      }
    }
  } else if (length(group_by) == nrow(object$w)) {
    for (group in levels(group_by)) {
      if (stat == "sum") {
        bins[[group]] <- colSums(object$w[which(group_by == group),])
      } else {
        bins[[group]] <- colMeans(object$w[which(group_by == group),])
      }
    }
  } else stop("'group_by' was not of length equal to rows in 'object$w' (which would correspond to features) or columns in 'object$h' (which would correspond to samples")
  bins <- do.call(cbind, bins)
  bins <- apply(bins, 1, function(x) x)
  result <- list()
  for (i in 1:ncol(bins))
    result[[length(result) + 1]] <- data.frame("group" = rownames(bins), "factor" = colnames(bins)[i], "stat" = as.vector(bins[, i]))
  result <- do.call(rbind, result)
  class(result) <- c("nmfSummary", "data.frame")
  result
})

setGeneric("evaluate", function(x, ...) standardGeneric("evaluate"))

#' Evaluate an NMF model
#'
#' Calculate mean squared error for an NMF model, accounting for any masking schemes requested during fitting.
#'
#' @details
#' TO DO
#'
#' @inheritParams nmf
#' @param x fitted model, class \code{nmf}, generally the result of calling \code{nmf}, with models of equal dimensions as \code{data}
#' @param missing_only calculate mean squared error only for missing values specified as a matrix in \code{mask}
#' @importFrom methods is
#' @export
#' @rdname evaluate.nmf
#' @examples
#' \dontrun{
#' TO DO
#' }
setMethod("evaluate", signature = "nmf", function(x, data, mask = NULL, missing_only = FALSE, ...) {
  validObject(x)

  if (missing_only && is.null(mask)) stop("a mask matrix must be specified to set 'missing_only = TRUE'")

  # get 'data' in either sparse or dense matrix format and look for NA's
  if (is(data, "sparseMatrix")) {
    if (class(data)[[1]] != "dgCMatrix") data <- as(data, "dgCMatrix")
    if (any(is.na(data@x))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      mask <- is.na(data)
    }
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
    if (!is.numeric(data)) data <- as.numeric(data)
    if (any(is.na(data))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
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

  if (class(data)[[1]] == "dgCMatrix") {
    if (missing_only) {
      Rcpp_mse_missing_sparse(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"))
    } else {
      Rcpp_mse_sparse(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"), mask_zeros)
    }
  } else {
    if (missing_only) {
      Rcpp_mse_missing_dense(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"))
    } else {
      Rcpp_mse_dense(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"), mask_zeros)
    }
  }
})

#' @rdname evaluate.nmf
#' @export
#' @param w feature factor matrix (features as rows)
#' @param h sample factor matrix (samples as columns)
#' @param d scaling diagonal vector (if applicable)
evaluate.nmf <- function(w, d = NULL, h, data, mask = NULL, missing_only = FALSE, ...) {
  if (is.null(d)) d <- rep(1, nrow(h))
  m <- new("nmf", w = w, d = d, h = h)
  evaluate(m)
}

#' @slot w feature factor matrix of shared signal across all datasets
#' @slot d_wh scaling diagonal vector for \code{w} and each \code{h} matrix
#' @slot h sample factor matrix mapping shared signal from \code{w} to each dataset
#' @slot u feature factor matrix of unique signal for each dataset
#' @slot d_uv scaling diagonal vector for \code{u} and \code{v} for each dataset
#' @slot v sample factor matrix mapping unique signal from \code{u} to each dataset
#' @slot misc list often containing components:
#'  \itemize{
#'    \item tol     : tolerance of fit
#'    \item iter    : number of fitting updates
#'    \item runtime : runtime in seconds
#'    \item mse     : mean squared error of model (calculated for multiple starts only)
#'    \item w_init  : initial w matrix used for model fitting
#'  }
#' @name lnmf
#' @aliases lnmf, lnmf-class
#' @exportClass lnmf
#'
setClass("lnmf",
         representation(w = "matrix", d_wh = "list", h = "list", u = "list", d_uv = "list", v = "list", misc = "list"),
         prototype(w = matrix(), d_wh = list(), h = list(), u = list(), d_uv = list(), v = list(), misc = list()),
         validity = function(object) {
           msg <- NULL
           # check that ranks of h are equal 
           if (!(all(sapply(object@h, function(x) nrow(x)) == ncol(object@w))))
             msg <- c(msg, "ranks of all 'h' matrices are not equal to rank of 'w'")
           if (!(all(sapply(object@v, function(x) nrow(x)) == sapply(object@u, function(x) ncol(x)))))
             msg <- c(msg, "ranks of all 'u' are not equal to all 'v'")
           if (!(all(sapply(object@d_uv, function(x) length(x)) == sapply(object@u, function(x) ncol(x)))))
             msg <- c(msg, "rank of all 'd_uv' are not equal to all 'u' and 'v'")
           if (!(all(sapply(object@d_wh, function(x) length(x)) == ncol(object@w))))
             msg <- c(msg, "ranks of all 'd_wh' is not always equal to rank of 'w' and 'h'")
           if (is.null(msg)) TRUE else msg
         })

#' @export
#' @rdname nmf-methods
setMethod("coerce", signature(from = "lnmf", to = "nmf"), function(from, to) {
  # multiply each h by scaling diagonal
  h <- slot(from, "h")
  for (i in 1:length(h))
    h[[i]] <- as.matrix(Diagonal(x = from@d_wh[[i]]) %*% h[[i]])
  h <- do.call(cbind, h)
  d <- rowSums(h)
  h <- apply(h, 2, function(x) x / d)
  w <- cbind(slot(from, "w"), do.call(cbind, slot(from, "u")))
  v <- slot(from, "v")
  d <- c(d, do.call(c, slot(from, "d_uv")))
  start_rank <- nrow(h)
  start_sample <- 0
  h <- rbind(h, matrix(0, ncol(w) - nrow(h), ncol(h)))
  for (i in 1:length(v)) {
    stop_rank <- nrow(v[[i]]) + start_rank
    stop_sample <- ncol(v[[i]]) + start_sample
    h[(start_rank + 1):stop_rank, (start_sample + 1):stop_sample] <- v[[i]]
    rownames(h)[(start_rank + 1):stop_rank] <- rownames(v[[i]])
    start_rank <- stop_rank
    start_sample <- stop_sample
  }
  names(d) <- NULL
  new("nmf", w = w, d = d, h = h, misc = slot(from, "misc"))
})

#'
#' @export
#' @rdname nmf-methods
setMethod("[[", signature("lnmf"), function(x, i) {
  validObject(x)
  i <- i[[1]]
  if (i < 1 || i > length(x@u)) stop("specified index was < 1 or > the number of models in the 'lnmf' object")
  w <- cbind(x@w, x@u[[i]])
  h <- rbind(x@h[[i]], x@v[[i]])
  d <- c(x@d_wh[[i]], x@d_uv[[i]])
  d_order <- order(d, decreasing = TRUE)
  new("nmf", w = w[, d_order], h = h[d_order,], d = d[d_order], misc = x@misc)
})

#' @method head lnmf
#' @rdname nmf-methods
#' @export
setMethod("head", signature("lnmf"), function(x, n = getOption("digits"), ...) {
  validObject(x)
  cat(nrow(x@w), "x", sum(sapply(x@h, function(x) ncol(x))), "x", length(x@d_wh[[1]]) + sum(sapply(x@d_uv, function(x) length(x))), "factor model of class \"lnmf\"\n")
  n_w <- n
  if (nrow(x@w) < n) n_w <- nrow(x@w)
  if (ncol(x@w) < n) n <- ncol(x@w)
  cat("@ w (shared model)\n")
  print.table(x@w[1:n_w, 1:n], zero.print = ".")
  if (nrow(x@w) > n_w && ncol(x@w) > n) {
    cat("...suppressing", nrow(x@w) - n_w, "rows and", ncol(x@w) - n, "columns\n")
  } else if (nrow(x@w) > n_w) {
    cat("...suppressing", nrow(x@w) - n_w, "rows\n")
  } else if (ncol(x@w) > n) {
    cat("...suppressing", ncol(x@w) - n, "columns\n")
  }
  cat("\n")
  cat(length(x@u),"unique models for", sum(sapply(x@h, function(x) ncol(x))), "samples of ranks ")
  for(rank in x@d_uv)
    cat(length(rank), " ")
  cat("\n\n@ misc\n")
  cat(str(x@misc))
  
  invisible(x)
})

#' @export
#' @method show nmf
#' @rdname nmf-methods
#' 
setMethod("show", signature("lnmf"), function(object) {
  head(object)
  invisible(object)
})