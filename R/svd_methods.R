#' @importFrom methods new validObject
#' @name svd_pca-class
#' @rdname svd_pca-class
#' @title svd_pca S4 Class
#' @description S4 class for deflation SVD/PCA results.
#' @slot u left singular vectors (scores), m x k matrix
#' @slot d singular values, numeric vector of length k
#' @slot v right singular vectors (loadings), n x k matrix
#' @slot misc list containing metadata: centered, row_means, test_loss,
#'   iters_per_factor, wall_time_ms, auto_rank
#' @aliases svd_pca-class
#' @exportClass svd_pca
#'
setClass("svd_pca",
  representation(u = "matrix", d = "numeric", v = "matrix", misc = "list"),
  prototype(u = matrix(), d = NA_real_, v = matrix(), misc = list()),
  validity = function(object) {
    msg <- NULL
    if (ncol(object@u) != ncol(object@v))
      msg <- c(msg, "ranks of 'u' and 'v' are not equal")
    if (ncol(object@u) != length(object@d))
      msg <- c(msg, "ranks of 'd' and 'u' are not equal")
    if (is.null(msg)) TRUE else msg
  })

#' @rdname svd_pca-class
#' @param x object of class \code{svd_pca}
#' @param i indices for subsetting factors
#' @param ... additional parameters
#' @return A subsetted \code{svd_pca} object containing only the selected factors.
#' @export
setMethod("[", "svd_pca", function(x, i) {
  validObject(x)
  x@u <- x@u[, i, drop = FALSE]
  x@v <- x@v[, i, drop = FALSE]
  x@d <- x@d[i]
  x
})

#' @rdname svd_pca-class
#' @param x object of class \code{svd_pca}
#' @param n number of rows/columns to show
#' @param ... additional parameters
#' @return Invisibly returns the \code{svd_pca} object \code{x}.
#' @export
setMethod("head", "svd_pca", function(x, n = getOption("digits"), ...) {
  validObject(x)
  k <- length(x@d)
  mode <- if (isTRUE(x@misc$centered)) "PCA" else "SVD"
  cat(nrow(x@u), "x", ncol(x@v), " rank-", k, " ", mode,
      " model of class \"svd_pca\"\n", sep = "")

  n_u <- min(nrow(x@u), n)
  n_v <- min(nrow(x@v), n)
  n_k <- min(k, n)

  cat("@ d (singular values)\n")
  print(x@d[1:n_k])

  cat("@ u (scores, first ", n_u, " rows)\n", sep = "")
  print.table(x@u[1:n_u, 1:n_k, drop = FALSE], zero.print = ".")

  cat("@ v (loadings, first ", n_v, " rows)\n", sep = "")
  print.table(x@v[1:n_v, 1:n_k, drop = FALSE], zero.print = ".")

  if (isTRUE(x@misc$centered))
    cat("(centered: row means stored in misc$row_means)\n")

  if (!is.null(x@misc$test_loss))
    cat("Auto-rank: best k = ", k, ", final test MSE = ",
        format(tail(x@misc$test_loss, 1), digits = 4), "\n", sep = "")

  invisible(x)
})

#' @rdname svd_pca-class
#' @return Invisibly returns the \code{svd_pca} object.
#' @export
setMethod("show", signature("svd_pca"), function(object) {
  k <- length(object@d)
  mode <- if (isTRUE(object@misc$centered)) "PCA" else "SVD"
  cat(nrow(object@u), "x", ncol(object@v), " rank-", k, " ", mode,
      " model of class \"svd_pca\"\n", sep = "")
  cat("  sigma range: [", format(min(object@d), digits = 4), ", ",
      format(max(object@d), digits = 4), "]\n", sep = "")
  if (!is.null(object@misc$wall_time_ms))
    cat("  wall time: ", round(object@misc$wall_time_ms, 1), " ms\n", sep = "")
  invisible(object)
})

#' @rdname svd_pca-class
#' @return Integer vector of length 3: \code{c(m, n, k)} where m is the number of
#'   rows, n the number of columns, and k the rank.
#' @export
setMethod("dim", signature("svd_pca"), function(x) {
  c(nrow(x@u), ncol(x@v), length(x@d))
})

#' Reconstruct approximation from svd_pca
#'
#' @param object An \code{svd_pca} object
#' @return Dense matrix: \eqn{U \cdot diag(d) \cdot V'} (plus row means if centered)
#' @examples
#' \donttest{
#' A <- matrix(rnorm(200), 20, 10)
#' s <- RcppML::svd(A, k = 3)
#' Ahat <- reconstruct(s)
#' dim(Ahat)
#' }
#' @export
#' @rdname svd_pca-class
setGeneric("reconstruct", function(object, ...) standardGeneric("reconstruct"))

#' @rdname svd_pca-class
#' @export
setMethod("reconstruct", signature("svd_pca"), function(object, ...) {
  approx <- object@u %*% diag(object@d, nrow = length(object@d)) %*% t(object@v)
  if (isTRUE(object@misc$centered) && !is.null(object@misc$row_means)) {
    approx <- approx + object@misc$row_means  # broadcasts: m×1 + m×n
  }
  approx
})

#' Project new data onto an svd_pca model
#'
#' Computes scores for new samples by projecting \code{newdata} onto the right
#' singular vectors. Equivalent to PCA "out-of-sample prediction".
#'
#' @param object An \code{svd_pca} object (the fitted model).
#' @param newdata A numeric matrix of new samples (rows = samples,
#'   columns = features). Must have the same number of features as the
#'   original data (\code{ncol(newdata) == nrow(object@v)}).
#' @param ... Ignored.
#' @return A numeric matrix of scores with \code{nrow(newdata)} rows and
#'   \code{length(object@d)} columns (i.e., same \eqn{k} as the model).
#'   Each row is the projection of one new sample onto the singular space.
#' @importFrom stats predict
#' @export
#' @rdname svd_pca-class
setMethod("predict", signature("svd_pca"), function(object, newdata, ...) {
  validObject(object)
  if (missing(newdata))
    stop("'newdata' is required for predict.svd_pca")

  if (!is.matrix(newdata) && !inherits(newdata, "dgCMatrix"))
    newdata <- as.matrix(newdata)

  if (ncol(newdata) != nrow(object@v))
    stop(sprintf(
      "ncol(newdata) (%d) must equal nrow(object@v) (%d) (number of features)",
      ncol(newdata), nrow(object@v)))

  if (isTRUE(object@misc$centered)) {
    if (is.matrix(newdata)) {
      row_m <- rowMeans(newdata)
      newdata <- sweep(newdata, 1L, row_m, FUN = "-")
    } else {
      # For sparse: project centered = project(newdata, v) - outer(1, v' * row_means)
      # i.e. defer centering until after the matrix product to avoid densification
      row_m <- Matrix::rowMeans(newdata)
      scores <- as.matrix(newdata %*% object@v)
      # Subtract contribution of row means: each row i loses sum_j mean_i * v[j, :]
      v_colsums <- colSums(object@v)   # length k: sum_j v[j, f] for each factor f
      scores <- scores - outer(row_m, v_colsums)
      scores <- sweep(scores, 2L, object@d, FUN = "/")
      return(scores)
    }
  }

  # scores = newdata %*% V %*% diag(1/d)
  scores <- as.matrix(newdata %*% object@v)
  scores <- sweep(scores, 2L, object@d, FUN = "/")
  scores
})

#' Variance explained by each factor
#'
#' @param object An \code{svd_pca} object
#' @return Numeric vector of proportion of variance explained by each factor
#' @examples
#' \donttest{
#' A <- matrix(rnorm(200), 20, 10)
#' s <- RcppML::svd(A, k = 3)
#' variance_explained(s)
#' }
#' @export
#' @rdname svd_pca-class
setGeneric("variance_explained", function(object, ...) standardGeneric("variance_explained"))

#' @rdname svd_pca-class
#' @export
setMethod("variance_explained", signature("svd_pca"), function(object, ...) {
  # Use ||A_c||^2_F as denominator when available (correct for truncated SVD)
  total <- object@misc$frobenius_norm_sq
  if (is.null(total) || total == 0) {
    total <- sum(object@d^2)  # Fallback: only accounts for extracted factors
  }
  object@d^2 / total
})
