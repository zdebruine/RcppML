#' Constructor for nmf class
#' 
#' Construct and validate an S3 \code{nmf} object.
#' 
#' @param w matrix of features (rows) by factors (columns)
#' @param d scaling diagonal vector for factors
#' @param h matrix of factors (rows) by samples (columns)
#' @param tol if applicable, tolerance of the fit
#' @param iter if applicable, number of iterations for fitting
#' @param runtime if applicable, runtime of the fit
#' @export
new_nmf <- function(w, d, h, tol = NA_real_, iter = NA_real_, runtime = NA_real_){
  obj <- as.list(environment())
  class(obj) <- "nmf"
  validate_nmf(obj)
  obj
}

# Validate NMF class object
validate_nmf <- function(object){
  if(class(object) != "nmf") stop("object is not of class 'nmf'")
  if(!all((c("w", "d", "h", "tol", "iter", "runtime") %in% names(object)))) stop("object must be a list with items 'w', 'd', 'h', 'tol', 'iter', 'runtime'")
  if(!is.matrix(object$w)) stop("'w' must be a matrix")
  if(!is.vector(object$d)) stop("'d' must be a vector")
  if(!is.matrix(object$h)) stop("'h' must be a matrix")
  if(ncol(object$w) != nrow(object$h)) stop("number of columns in 'w' must be equal to the number of rows in 'h'")
  if(ncol(object$w) != length(object$d)) stop("rank of 'w' and 'h' must be equal to the length of 'd'")
  return(TRUE)
}

#' @export
`[.nmf` <- function(x, i){
  validate_nmf(x)
  x$w <- x$w[, i]
  x$h <- x$h[i, ]
  x$d <- x$d[i]
  x
}

#' @export
#' @method head nmf
head.nmf <- function(x, n = 5L, ...) {
  validate_nmf(x)
  cat(nrow(x$w), "x", ncol(x$h),"x", length(x$d), "factor model of class \"nmf\"\n")
  n <- n_h <- n_w <- n
  if(nrow(x$w) < n) n_w <- nrow(x$w)
  if(ncol(x$h) < n) n_h <- ncol(x$h)
  if(length(x$d) < n) n <- length(x$d)
  cat("$ w\n")
  print(x$w[1:n_w, 1:n])
  if(nrow(x$w) > n_w && ncol(x$w) > n) {
    cat("...suppressing",  nrow(x$w) - n_w, "rows and", ncol(x$w) - n, "columns\n")
  } else if(nrow(x$w) > n_w){
    cat("...suppressing", nrow(x$w) - n_w, "rows\n")
  } else if(ncol(x$w) > n) {
    cat("...suppressing", ncol(x$w) - n, "columns\n")
  }
  cat("\n$ d\n")
  print(x$d[1:n])
  if(length(x$d) > n) cat("...suppressing", length(x$d) - n, "values\n")
  cat("\n$ h\n")
  print(x$h[1:n, 1:n_h])
  if(ncol(x$h) > n_h && nrow(x$h) > n) {
    cat("...suppressing", nrow(x$h) - n, "rows and", ncol(x$h) - n_h, "columns\n")
  } else if(ncol(x$h) > n_h){
    cat("...suppressing", ncol(x$h) - n_h, "columns\n")
  } else if(nrow(x$h) > n){
    cat("...suppressing", nrow(x$h), "rows\n")
  }
  cat("\n$ tol:", x$tol,"\n$ iter:", x$iter,"\n$ runtime:", x$runtime,"sec\n")
  invisible(x)
}

#' @export
#' @method print nmf
print.nmf <- function(x, ...){
  head(x, n = 5L)
}

#' @export
#' @method dimnames nmf
dimnames.nmf <- function(x){
  validate_nmf(x)
  print(list(rownames(x$w), colnames(x$h)))
}

#' @export
#' @method dim nmf
dim.nmf <- function(x) {
  validate_nmf(x)
  c(nrow(x$w), ncol(x$h), ncol(x$w))
}

#' @export
#' @method t nmf
t.nmf <- function(x) {
  validate_nmf(x)
  res <- list("w" = t(x$h), "d" = x$d, "h" = t(x$w), "tol" = x$tol, "iter" = x$iter, "runtime" = x$runtime)
  class(res) <- "nmf"
  return(res)
}

#' @export
#' @method sort nmf
sort.nmf <- function(x, decreasing = TRUE, ...){
  validate_nmf(x)
  x <- x[order(x$d, decreasing = decreasing)]
  x
}

#' @export
#' @method prod nmf
prod.nmf <- function(x, na.rm = FALSE, ...) {
  validate_nmf(x)
  return(crossprod(t(as.matrix(x$w %*% Diagonal(x = x$d))), x$h))
}

#' Calculate sparsity of an object
#' 
#' \code{sparsity} is a generic function for calculating the (marginal) sparsity of a model.
#' 
#' @param object object to be evaluated
#' @param ... arguments passed to or from other methods
#' 
#' @export
sparsity <- function(object, ...) {
  UseMethod("sparsity")
}

#' @details For \code{\link{nmf}} models, the sparsity of each factor is computed and summarized for \eqn{w} and \eqn{h} matrices. A long \code{data.frame} with columns \code{factor}, \code{sparsity}, and \code{model} is returned.
#' @export
#' @rdname sparsity
#' @method sparsity nmf
sparsity.nmf <- function(object, ...){
  validate_nmf(object)
  w <- colSums(object$w == 0) / nrow(object$w)
  h <- rowSums(object$h == 0) / ncol(object$h)
  w <- data.frame("factor" = names(w), "sparsity" = as.vector(w))
  h <- data.frame("factor" = names(h), "sparsity" = as.vector(h))
  w$model <- "w"
  h$model <- "h"
  result <- rbind(w, h)
  result$model <- as.factor(result$model)
  result
}

#' Align two factor models
#' 
#' \code{align} is a generic function for reordering factors in one factor model (\code{object}) to correspond maximally to those in a reference factor model (\code{ref}) based on some criterion.
#' 
#' @param object factor model to be aligned
#' @param ref factor model to align to
#' @param ... arguments passed to or from other methods
#' 
#' @export
align <- function(object, ref, ...){
  UseMethod("align")
}

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
#' @method align nmf
#' @rdname align
align.nmf <- function(object, ref, method = "cosine", ...){
  validate_nmf(object)
  if(dim(ref$w) != dim(object$w)) stop("dimensions of object$w and ref$w are not identical")
  if(method == "cosine"){
    cost <- 1 - cosine(object$w, ref$w) + 1e-10
  } else if(method == "cor"){
    cost <- 1 - cor(object$w, ref$w) + 1e-10
  }
  ref[bipartiteMatch(cost)$pairs]
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
summary.nmf <- function(object, group_by, stat = "sum", ...){
  validate_nmf(object)
  if(!is.factor(group_by)) group_by <- as.factor(group_by)
  bins <- list()
  if(length(group_by) == ncol(object$h)){
    for(group in levels(group_by)){
      if(stat == "sum") {
        bins[[group]] <- rowSums(object$h[, which(group_by == group)])
      } else {
        bins[[group]] <- rowMeans(object$h[, which(group_by == group)])
      }
    }
  } else if(length(group_by) == nrow(object$w)){
    for(group in levels(group_by)){
      if(stat == "sum") {
        bins[[group]] <- colSums(object$w[which(group_by == group), ])
      } else {
        bins[[group]] <- colMeans(object$w[which(group_by == group), ])
      }
    }
  } else stop("'group_by' was not of length equal to rows in 'object$w' (which would correspond to features) or columns in 'object$h' (which would correspond to samples")
  bins <- do.call(cbind, bins)
  bins <- apply(bins, 1, function(x) x)
  result <- list()
  for(i in 1:ncol(bins))
    result[[length(result) + 1]] <- data.frame("group" = rownames(bins), "factor" = colnames(bins)[i], "stat" = as.vector(bins[, i]))
  result <- do.call(rbind, result)
  class(result) <- c("nmfSummary", "data.frame")
  result
}