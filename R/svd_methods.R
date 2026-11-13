#' @importFrom methods new validObject
#' @name svd-class
#' @rdname svd-class
#' @title svd S4 Class
#' @description S4 class for deflation SVD/PCA results.
#' @slot u left singular vectors (scores), m x k matrix
#' @slot d singular values, numeric vector of length k
#' @slot v right singular vectors (loadings), n x k matrix
#' @slot misc list containing metadata: centered, row_means, test_loss,
#'   iters_per_factor, wall_time_ms, auto_rank
#' @aliases svd-class
#' @seealso \code{\link{svd}}, \code{\link{pca}}, \code{\link{reconstruct}}, \code{\link{variance_explained}}
#' @exportClass svd
#'
setClass("svd",
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

#' @rdname svd-class
#' @param x object of class \code{svd}
#' @param i indices for subsetting factors
#' @param j not used
#' @param drop not used; the result is always an \code{svd} object
#' @param ... additional parameters
#' @return A subsetted \code{svd} object containing only the selected factors.
#' @export
setMethod("[", "svd", function(x, i, j, ..., drop = TRUE) {
  validObject(x)
  x@u <- x@u[, i, drop = FALSE]
  x@v <- x@v[, i, drop = FALSE]
  x@d <- x@d[i]
  if (!is.null(x@misc$variance_explained))
    x@misc$variance_explained <- x@misc$variance_explained[i]
  x
})

# Print the first (or last) rows of u and v and the leading singular values
.svd_peek <- function(x, n = getOption("digits"), last = FALSE) {
  validObject(x)
  k <- length(x@d)
  mode <- if (isTRUE(x@misc$centered)) "PCA" else "SVD"
  cat(nrow(x@u), "x", nrow(x@v), " rank-", k, " ", mode,
      " model of class \"svd\"\n", sep = "")

  n <- max(0L, as.integer(n))
  pick <- function(total) {
    cnt <- min(total, n)
    if (cnt == 0L) integer(0) else if (last) seq.int(total - cnt + 1L, total) else seq_len(cnt)
  }
  i_u <- pick(nrow(x@u))
  i_v <- pick(nrow(x@v))
  i_k <- seq_len(min(k, n))
  which_rows <- if (last) "last" else "first"

  cat("@ d (singular values)\n")
  print(x@d[i_k])

  cat("@ u (", which_rows, " ", length(i_u), " rows)\n", sep = "")
  print.table(x@u[i_u, i_k, drop = FALSE], zero.print = ".")

  cat("@ v (", which_rows, " ", length(i_v), " rows)\n", sep = "")
  print.table(x@v[i_v, i_k, drop = FALSE], zero.print = ".")

  if (isTRUE(x@misc$centered))
    cat("(centered: row means stored in misc$row_means)\n")

  if (!is.null(x@misc$test_loss))
    cat("Auto-rank: best k = ", k, ", final test MSE = ",
        format(x@misc$test_loss[length(x@misc$test_loss)], digits = 4), "\n", sep = "")

  invisible(x)
}

#' @rdname svd-class
#' @param x object of class \code{svd}
#' @param n number of rows/columns to show
#' @param ... additional parameters
#' @return \code{head} and \code{tail} print the first or last \code{n} rows of
#'   \code{u} and \code{v} and invisibly return the \code{svd} object \code{x}.
#' @method head svd
#' @export
head.svd <- function(x, n = getOption("digits"), ...) {
  .svd_peek(x, n = n, last = FALSE)
}

#' @rdname svd-class
#' @method tail svd
#' @export
tail.svd <- function(x, n = getOption("digits"), ...) {
  .svd_peek(x, n = n, last = TRUE)
}

#' @rdname svd-class
#' @export
setMethod("head", signature("svd"), function(x, ...) head.svd(x, ...))

#' @rdname svd-class
#' @export
setMethod("tail", signature("svd"), function(x, ...) tail.svd(x, ...))

#' @rdname svd-class
#' @return Invisibly returns the \code{svd} object.
#' @export
setMethod("show", signature("svd"), function(object) {
  k <- length(object@d)
  mode <- if (isTRUE(object@misc$centered)) "PCA" else "SVD"
  cat(nrow(object@u), "x", nrow(object@v), " rank-", k, " ", mode,
      " model of class \"svd\"\n", sep = "")
  cat("  sigma range: [", format(min(object@d), digits = 4), ", ",
      format(max(object@d), digits = 4), "]\n", sep = "")
  if (!is.null(object@misc$wall_time_ms))
    cat("  wall time: ", round(object@misc$wall_time_ms, 1), " ms\n", sep = "")
  invisible(object)
})

#' @rdname svd-class
#' @return Integer vector of length 3: \code{c(m, n, k)} where m is the number of
#'   rows, n the number of columns, and k the rank.
#' @export
setMethod("dim", signature("svd"), function(x) {
  c(nrow(x@u), nrow(x@v), length(x@d))
})

#' Reconstruct approximation from svd
#'
#' @param object An \code{svd} object
#' @return Dense matrix on the scale of the original data:
#'   \eqn{U \cdot diag(d) \cdot V'}, multiplied row-wise by the row standard
#'   deviations if the model was scaled, plus the row means if it was centered.
#' @examples
#' \donttest{
#' A <- matrix(rnorm(200), 20, 10)
#' s <- RcppML::svd(A, k = 3)
#' Ahat <- reconstruct(s)
#' dim(Ahat)
#' }
#' @export
#' @rdname svd-class
setGeneric("reconstruct", function(object, ...) standardGeneric("reconstruct"))

#' @rdname svd-class
#' @export
setMethod("reconstruct", signature("svd"), function(object, ...) {
  approx <- object@u %*% diag(object@d, nrow = length(object@d)) %*% t(object@v)
  # Undo the row transformation X = D^{-1}(A - mu 1'): A = D X + mu 1'
  if (isTRUE(object@misc$scaled) && !is.null(object@misc$row_sds)) {
    approx <- approx * object@misc$row_sds    # recycles down rows: m-vector * m×n
  }
  if (isTRUE(object@misc$centered) && !is.null(object@misc$row_means)) {
    approx <- approx + object@misc$row_means  # broadcasts: m×1 + m×n
  }
  approx
})

#' Project new data onto an svd model
#'
#' Computes \code{u}-type coordinates for new rows by projecting them onto the
#' right singular vectors: \code{newdata \%*\% v \%*\% diag(1/d)}.
#'
#' The model treats rows as variables and columns as samples (see
#' \code{\link{svd}}). \code{newdata} therefore holds \strong{new rows measured
#' on the same columns (samples) as the training data}. Each new row receives
#' the same transformation the model applied to its own rows: it is centered by
#' its own mean if the model was centered and divided by its own sample
#' standard deviation if the model was scaled. For orthonormal \code{v} this is
#' the least-squares projection; for constrained (non-orthogonal) factors it is
#' an approximation.
#'
#' To place new \emph{samples} (new columns \code{x}, one value per variable)
#' in the space of \code{v}, use
#' \code{crossprod(object@u, (x - object@misc$row_means) / object@misc$row_sds) / object@d}
#' (dropping the terms that do not apply).
#'
#' @param object An \code{svd} object (the fitted model).
#' @param newdata A numeric matrix or \code{dgCMatrix} of new rows, with the
#'   same columns as the training data
#'   (\code{ncol(newdata) == nrow(object@v)}).
#' @param ... Ignored.
#' @return A numeric matrix with \code{nrow(newdata)} rows and
#'   \code{length(object@d)} columns, comparable to rows of \code{object@u}.
#' @importFrom stats predict
#' @export
#' @rdname svd-class
setMethod("predict", signature("svd"), function(object, newdata, ...) {
  validObject(object)
  if (missing(newdata))
    stop("'newdata' is required for predict.svd")

  if (!is.matrix(newdata) && !inherits(newdata, "dgCMatrix"))
    newdata <- as.matrix(newdata)

  if (ncol(newdata) != nrow(object@v))
    stop(sprintf(
      "ncol(newdata) (%d) must equal nrow(object@v) (%d) (columns of the training data)",
      ncol(newdata), nrow(object@v)))

  centered <- isTRUE(object@misc$centered)
  scaled <- isTRUE(object@misc$scaled)
  n <- ncol(newdata)

  # Row statistics of the new rows (same transformation as the fit)
  if (inherits(newdata, "dgCMatrix")) {
    row_m <- Matrix::rowMeans(newdata)
    row_ss <- Matrix::rowSums(newdata^2)
  } else {
    row_m <- rowMeans(newdata)
    row_ss <- rowSums(newdata^2)
  }
  row_sd <- rep(1, nrow(newdata))
  if (scaled && n > 1) {
    v_row <- pmax(row_ss - n * row_m^2, 0) / (n - 1)
    ok <- v_row > 1e-12 * (row_ss / n) & v_row > 0
    row_sd[ok] <- sqrt(v_row[ok])
  }

  # (newdata - m 1') V = newdata V - m (1' V): centering is applied after the
  # product so sparse input is never densified
  scores <- as.matrix(newdata %*% object@v)
  if (centered) {
    scores <- scores - outer(row_m, colSums(object@v))
  }
  if (scaled) {
    scores <- scores / row_sd
  }
  sweep(scores, 2L, object@d, FUN = "/")
})

#' Variance explained by each factor
#'
#' @details
#' For orthogonal factors this is \eqn{d_i^2 / \|A_c\|_F^2}, where \eqn{A_c}
#' is the centered (and scaled) matrix. Constrained factors (\code{nonneg},
#' \code{L1} with \code{method = "deflation"}) are not orthogonal, so
#' \eqn{d_i^2} does not partition the variance; for those models the value is
#' the drop in residual variance when component \code{i} is added and the
#' scales are refit by least squares, which sums to the variance explained by
#' \code{\link{reconstruct}}.
#'
#' @param object An \code{svd} object
#' @return Numeric vector of proportion of variance explained by each factor
#' @examples
#' \donttest{
#' A <- matrix(rnorm(200), 20, 10)
#' s <- RcppML::svd(A, k = 3)
#' variance_explained(s)
#' }
#' @export
#' @rdname svd-class
setGeneric("variance_explained", function(object, ...) standardGeneric("variance_explained"))

#' @rdname svd-class
#' @export
setMethod("variance_explained", signature("svd"), function(object, ...) {
  # Constrained (non-orthogonal) factors: incremental explained variance
  # computed at fit time, consistent with reconstruct()
  if (!is.null(object@misc$variance_explained) &&
      length(object@misc$variance_explained) == length(object@d)) {
    return(as.numeric(object@misc$variance_explained))
  }
  # Use ||A_c||^2_F as denominator when available (correct for truncated SVD)
  total <- object@misc$frobenius_norm_sq
  if (is.null(total) || total == 0) {
    total <- sum(object@d^2)  # Fallback: only accounts for extracted factors
  }
  object@d^2 / total
})
