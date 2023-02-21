setClass("svd",
  representation(u = "matrix", v = "matrix", misc = "list"),
  prototype(u = matrix(), v = matrix(), misc = list()),
  validity = function(object) {
    msg <- NULL
    if (ncol(object@u) != nrow(object@v))
      msg <- c(msg, "ranks of 'u' and 'v' are not equal")
    if (is.null(msg)) TRUE else msg
  })



setMethod("evaluate", signature = "svd", function(x, data, mask = NULL, missing_only = FALSE, ...) {
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
  } else if (class(mask)[[1]] == "character" && mask == "zeros") {
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
      Rcpp_mse_sparse(data, mask_matrix, t(x@u), x@h, getOption("RcppML.threads"), mask_zeros)
    }
  } else {
    if (missing_only) {
      Rcpp_mse_missing_dense(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"))
    } else {
      Rcpp_mse_dense(data, mask_matrix, t(x@w), x@d, x@h, getOption("RcppML.threads"), mask_zeros)
    }
  }
})
