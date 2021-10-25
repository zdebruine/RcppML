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
    if (!(all(sapply(object@u, function(x) ncol(x)) == ncol(object@u[[1]]))))
      msg <- c(msg, "ranks of all 'u' matrices are not equal")
    if (!(all(sapply(object@v, function(x) nrow(x)) == nrow(object@v[[1]]))))
      msg <- c(msg, "ranks of all 'v' matrices are not equal")
    if (!(all(sapply(object@v, function(x) nrow(x)) == sapply(object@u, function(x) ncol(x)))))
      msg <- c(msg, "ranks of all 'u' are not equal to all 'v'")
    if (!(all(sapply(object@d_uv, function(x) length(x)) == sapply(object@u, function(x) ncol(x)))))
      msg <- c(msg, "rank of all 'd_uv' are not equal to all 'u' and 'v'")
    if (!(all(sapply(object@d_wh, function(x) length(x)) == ncol(object@w))))
      msg <- c(msg, "ranks of all 'd_wh' is not always equal to rank of 'w' and 'h'")
    if (is.null(msg)) TRUE else msg
  })

#' @export
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
  diag_order <- order(d, decreasing = TRUE)
  names(d) <- NULL
  new("nmf", w = w[, diag_order], d = d[diag_order], h = h[diag_order,], misc = slot(from, "misc"))
})

#'
#' @export
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
#'@export
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
#' 
setMethod("show", signature("lnmf"), function(object) {
  head(object)
  invisible(object)
})