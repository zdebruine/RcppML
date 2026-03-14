#' @importFrom methods new validObject
#' @name nmf-class
#' @rdname nmf-class
#' @title nmf S4 Class
#' @description The S4 class for NMF model results.
#' @slot w feature factor matrix
#' @slot d scaling diagonal vector
#' @slot h sample factor matrix
#' @slot misc list containing optional components including tol (tolerance), iter (iterations), 
#'   loss (final loss value), loss_type (loss function used), runtime (in seconds), 
#'   w_init (initial w matrix), test_mask (CV test set), test_seed (CV seed), 
#'   test_fraction (CV holdout fraction), train_loss (CV training loss), 
#'   test_loss (CV test loss), and best_iter (CV best iteration)
#' @aliases nmf-class
#' @seealso \code{\link{nmf}}, \code{\link{evaluate}}, \code{\link{align}}, \code{\link{predict,nmf-method}}
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
#' @rdname nmf-class-methods
#' @param x object of class \code{nmf}
#' @param i indices
#' @param ... additional parameters
#' @param na.rm remove na values
#' @return An \code{nmf} object subsetted to the specified factors.
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
#' @rdname nmf-class-methods
#' @param x object of class \code{nmf}
#' @param i indices
#' @return An \code{nmf} object subsetted to the specified factors.
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
#' @rdname nmf-class-methods
#' @param n number of rows/columns to show
#' @param ... additional parameters
#' @return Invisibly returns the \code{nmf} object.
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
#' @param object object of class \code{nmf}
#' @rdname nmf-class-methods
#' @return Invisibly returns the \code{nmf} object.
#' 
setMethod("show", signature("nmf"), function(object) {
  head(object)
  invisible(object)
})

#' @export
#' @method dimnames nmf
#' @param x object of class \code{nmf}
#' @rdname nmf-class-methods
#' @return A list of length two: row names of \code{w} and column names of \code{h}.
#' 
setMethod("dimnames", signature = "nmf", function(x) {
  validObject(x)
  list(rownames(x@w), colnames(x@h))
})

#' @export
#' @method dim nmf
#' @rdname nmf-class-methods
#' @return An integer vector of length 3: number of features, number of samples, and rank.
#' 
setMethod("dim", signature = "nmf", function(x) {
  validObject(x)
  c(nrow(x@w), ncol(x@h), ncol(x@w))
})

#' @export
#' @method t nmf
#' @rdname nmf-class-methods
#' @return An \code{nmf} object with transposed factorization (\code{w} and \code{h} swapped and transposed).
#' 
setMethod("t", signature = "nmf", function(x) {
  validObject(x)
  tx <- new("nmf", w = t(x@h), d = x@d, h = t(x@w), misc = x@misc)
  tx
})

#' @export
#' @method sort nmf
#' @rdname nmf-class-methods
#' @param decreasing logical. Should the sort be increasing or decreasing?
#' @return An \code{nmf} object with factors reordered by decreasing (or increasing) \code{d}.
#' 
setMethod("sort", signature = "nmf", function(x, decreasing = TRUE, ...) {
  validObject(x)
  x <- x[order(x@d, decreasing = decreasing)]
  x
})

#' @export
#' @method prod nmf
#' @rdname nmf-class-methods
#' @param x object of class \code{nmf}.
#' @param ... additional parameters
#' @return A dense matrix equal to \code{w \%*\% diag(d) \%*\% h}.
#' 
setMethod("prod", signature = "nmf", function(x, ...) {
  validObject(x)
  return(crossprod(t(as.matrix(x@w %*% Diagonal(x = x@d))), x@h))
})

#' @importFrom methods slot
#' @rdname nmf-class-methods
#' @param x object of class \code{nmf}.
#' @param name name of nmf class slot
#' @return Contents of the named slot (\code{w}, \code{d}, \code{h}, or \code{misc}), or the named element from \code{misc}.
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
#' @param from class which the coerce method should perform coercion from
#' @param to class which the coerce method should perform coercion to
#' @rdname nmf-class-methods
#' @return A named list with elements \code{w}, \code{d}, \code{h}, and \code{misc}.
setMethod("coerce", signature(from = "nmf", to = "list"), function(from, to) {
  list("w" = slot(from, "w"), "d" = slot(from, "d"), "h" = slot(from, "h"), "misc" = slot(from, "misc"))
})

#' Compute the sparsity of each NMF factor
#'
#' @details
#' For \code{\link{nmf}} models, the sparsity of each factor is computed and summarized
#' or \eqn{w} and \eqn{h} matrices. A long \code{data.frame} with columns \code{factor}, \code{sparsity}, and \code{model} is returned.
#' @export
#' @param object object of class \code{nmf}.
#' @param ... additional parameters
#' @return A \code{data.frame} with columns \code{factor}, \code{sparsity}, and \code{model} ("w" or "h").
#' @seealso \code{\link{nmf}}, \code{\link{summary,nmf-method}}
#' @examples
#' \donttest{
#' data <- simulateNMF(50, 30, k = 3, seed = 1)
#' model <- nmf(data$A, 3, seed = 1, maxit = 50)
#' sparsity(model)
#' }
setGeneric("sparsity", function(object, ...) standardGeneric("sparsity"))

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
#' @return An \code{nmf} object with factors reordered to best match \code{ref}.
#' @seealso \code{\link{bipartiteMatch}}, \code{\link{cosine}}, \code{\link{nmf}}
#' @export
#' @importFrom stats cor
#' @examples
#' \donttest{
#' data <- simulateNMF(50, 30, k = 3, seed = 1)
#' m1 <- nmf(data$A, 3, seed = 1, maxit = 50)
#' m2 <- nmf(data$A, 3, seed = 2, maxit = 50)
#' aligned <- align(m2, m1)  # reorder m2 factors to match m1
#' }
setGeneric("align", function(object, ...) standardGeneric("align"))

#' @rdname align
#' @method align nmf
setMethod("align", signature = "nmf", function(object, ref, method = "cosine", ...) {
  validObject(object)
  if (any(dim(ref$w) != dim(object$w))) stop("dimensions of object$w and ref$w are not identical")
  if (method == "cosine") {
    cost <- 1 - cosine(object$w, ref$w) + 1e-10
  } else if (method == "cor") {
    cost <- 1 - cor(object$w, ref$w) + 1e-10
  }
  cost[cost < 0] <- 0
  object[bipartiteMatch(cost)$assignment + 1L]
})

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
#' @seealso \code{\link{nmf}}, \code{\link{sparsity}}
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
#' groups <- factor(sample(c("A", "B"), 50, replace = TRUE))
#' s <- summary(model, group_by = groups)
#' }
#'
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

#' Evaluate an NMF model
#'
#' Calculate loss for an NMF model using the specified loss function, accounting for any masking schemes requested during fitting.
#'
#' @inheritParams nmf
#' @param x fitted model, class \code{nmf}, generally the result of calling \code{nmf}, with models of equal dimensions as \code{data}
#' @param loss loss function to use: "mse" (Mean Squared Error, default) or
#'   "gp" (Generalized Poisson / KL divergence)
#' @param missing_only calculate loss only for missing values specified as a matrix in \code{mask}
#' @param test_fraction fraction of entries to hold out as test set (default 0 = disabled). When > 0, creates a random mask for test/train split.
#' @param test_seed seed for test set generation. If NULL, attempts to use test mask from model's @misc$test_mask if available.
#' @param eval_set which set to evaluate: "all" (default), "test" (held-out entries only), or "train" (non-held-out entries only). 
#'   Only used when test_fraction > 0 or test mask exists in model.
#' @param threads number of threads for OpenMP parallelization (default 0 = all available)
#' @param verbose print progress information (default FALSE)
#' @importFrom methods is
#' @return A single numeric value: the loss (MSE or GP/KL divergence) of the model on the data.
#' @seealso \code{\link{nmf}}
#' @export
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4)
#' evaluate(model, A)  # MSE
#' }
setGeneric("evaluate", function(x, ...) standardGeneric("evaluate"))

#' @rdname evaluate
#' @method evaluate nmf
setMethod("evaluate", signature = "nmf", function(x, data, mask = NULL, missing_only = FALSE, 
                                                   loss = c("mse", "gp"),
                                                   test_fraction = 0,
                                                   test_seed = NULL,
                                                   eval_set = c("all", "test", "train"),
                                                   threads = 0, verbose = FALSE, ...) {
  validObject(x)
  
  eval_set <- match.arg(eval_set)

  if (missing_only && is.null(mask)) stop("a mask matrix must be specified to set 'missing_only = TRUE'")

  # Validate and convert loss parameter
  loss <- match.arg(loss)
  loss_type <- switch(loss,
    "mse" = 0L,
    "gp" = 3L  # KL divergence formula (LossType::KL internally)
  )

  # Unified input validation (supports file paths, sparse, dense)
  data_info <- validate_data(data)
  data <- data_info$data
  # NA detection
  if (is(data, "dgCMatrix")) {
    if (any(is.na(data@x))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      mask <- is.na(data)
    }
  } else if (is.matrix(data)) {
    if (any(is.na(data))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      mask <- is.na(.to_dgCMatrix(data))
    }
  }
  
  # Handle test set mask generation or retrieval
  test_mask <- NULL
  if (test_fraction > 0 || !is.null(test_seed) || eval_set != "all") {
    # Try to get test mask from model if available
    if (!is.null(x@misc$test_mask) && is.null(test_seed)) {
      # Reuse existing test mask from model
      test_mask <- x@misc$test_mask
      if (verbose) {
        cat("Using test mask from model (", round(sum(test_mask@x) / (nrow(data) * ncol(data)) * 100, 2), 
            "% held out)\n", sep = "")
      }
    } else if (test_fraction > 0) {
      # Generate new test mask
      if (is.null(test_seed)) {
        test_seed <- sample.int(.Machine$integer.max, 1)
      }
      
      # Create holdout mask using rsparsematrix
      set.seed(test_seed)
      test_mask <- rsparsematrix(nrow(data), ncol(data), test_fraction, rand.x = NULL)
      
      if (verbose) {
        cat("Generated test mask with seed", test_seed, "(", 
            round(test_fraction * 100, 2), "% target holdout)\n", sep = " ")
      }
    } else if (eval_set != "all") {
      stop("eval_set = '", eval_set, "' requires test_fraction > 0 or a test mask in the model")
    }
  }

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
        mask <- as.matrix(mask)
      } else stop("could not coerce the value of 'mask' to a sparse pattern matrix (dgCMatrix)")
    }
    # Use dMatrix intermediate to avoid ngCMatrix -> dgCMatrix deprecation warning
    mask_matrix <- .to_dgCMatrix(as(mask, "dMatrix"))
  }
  
  # Apply test set logic
  if (!is.null(test_mask)) {
    if (eval_set == "test") {
      # Evaluate only on test set (held-out entries)
      mask_matrix <- test_mask
      missing_only <- TRUE
    } else if (eval_set == "train") {
      # Evaluate only on training set (non-held-out entries)
      # Invert the test mask: mask out the test entries
      test_mask_inv <- .to_dgCMatrix(test_mask != 0)
      if (is.null(mask) || (is.character(mask) && mask == "zeros")) {
        mask_matrix <- test_mask_inv
        missing_only <- FALSE
      } else {
        # Combine with existing mask
        mask_matrix <- .to_dgCMatrix((mask_matrix | test_mask_inv) != 0)
      }
    }
    # For eval_set == "all", don't modify mask - evaluate on everything
  }

  # Unified loss evaluation — always use general loss functions
  # (loss_type=0 corresponds to MSE, handled by compute_loss_general)
  huber_delta <- 0.0  # kept for C++ interface compatibility
  if (missing_only) {
    Rcpp_evaluate_loss_missing(data, mask_matrix, x@w, x@d, x@h, 
                               loss_type, huber_delta, as.integer(threads))
  } else {
    Rcpp_evaluate_loss(data, mask_matrix, x@w, x@d, x@h, 
                       loss_type, huber_delta, as.integer(threads), mask_zeros)
  }
})

#' Mean squared error of factor model
#' 
#' Same as the \code{evaluate} S4 method for the \code{nmf} class, but allows one to input the `w`, `d`, `h`, and `data` independently.
#' @keywords internal
#' @param w feature factor matrix (features as rows)
#' @param h sample factor matrix (samples as columns)
#' @param d scaling diagonal vector (if applicable)
#' @inheritParams nmf
#' @param missing_only only calculate mean squared error at masked values
#' @param ... additional arguments
#' @return A single numeric value: the mean squared error of the factorization.
#' @examples
#' \dontrun{
#' data <- simulateNMF(50, 30, k = 3, seed = 1)
#' model <- nmf(data$A, 3, seed = 1, maxit = 50)
#' RcppML:::mse(model$w, model$d, model$h, data$A)
#' }
mse <- function(w, d = NULL, h, data, mask = NULL, missing_only = FALSE, ...) {
  if (is.null(d)) d <- rep(1, nrow(h))
  m <- new("nmf", w = w, d = d, h = h)
  evaluate(m, data, mask = mask, missing_only = missing_only, loss = "mse", ...)
}

#'
#' @export
#' @rdname nmf-class-methods
#' @return The element from the \code{misc} list at the given name or index.
setMethod("[[", signature("nmf"), function(x, i) {
  validObject(x)
  # Access misc list elements by name or index
  if (is.character(i)) {
    if (!(i %in% names(x@misc))) stop(sprintf("no element '%s' in misc", i))
    return(x@misc[[i]])
  }
  if (is.numeric(i)) {
    if (i < 1 || i > length(x@misc)) stop("index out of bounds for misc list")
    return(x@misc[[i]])
  }
  stop("[[ requires a character or numeric index")
})