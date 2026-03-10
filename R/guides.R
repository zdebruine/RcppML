# Guide constructors for nmf()
#
# Guides steer NMF factors toward target structure by adding
# a soft penalty to the NNLS objective:
#   min ||Gh - b||² + λ||h - t||²
#
# This modifies the Gram diagonal (G += |λ|) and RHS (B += λ·t).
# Positive lambda attracts toward target; negative repels.

#' Create a classifier guide
#'
#' Steers a factor toward class-consistent structure using per-class
#' centroid targets computed from the current factor at each iteration.
#'
#' @param labels Integer vector of class labels (0-indexed). Use -1 for
#'   unlabeled samples. Length must match the number of columns (H guide)
#'   or rows (W guide) of the data matrix.
#' @param lambda Guide strength. Positive attracts toward class centroids;
#'   negative repels. Default 1.0.
#' @param side Which factor to guide: "H" (default, sample embeddings) or
#'   "W" (feature loadings).
#' @return An \code{nmf_guide} object to pass to \code{nmf(guides = ...)}.
#' @examples
#' # Create a classifier guide for 3 classes
#' labels <- rep(0:2, each = 10)
#' g <- guide_classifier(labels, lambda = 0.5)
#' g
#' @seealso \code{\link{guide_external}}, \code{\link{guide_callback}}, \code{\link{nmf}}
#' @export
guide_classifier <- function(labels, lambda = 1.0, side = c("H", "W")) {
  side <- match.arg(side)
  labels <- as.integer(labels)
  if (any(is.na(labels))) stop("NA labels not allowed; use -1 for unlabeled samples")
  if (all(labels < 0)) stop("All samples are unlabeled; at least one label >= 0 required")
  structure(list(
    type = "classifier",
    labels = labels,
    lambda = as.double(lambda),
    side = side
  ), class = "nmf_guide")
}

#' Create an external target guide
#'
#' Steers a factor toward a fixed target matrix. Useful for transfer
#' learning, prior knowledge, or cross-layer coupling.
#'
#' @param target Target matrix with dimensions matching the guided factor
#'   (k × n for H, k × m for W).
#' @param lambda Guide strength. Positive attracts; negative repels. Default 1.0.
#' @param side Which factor to guide: "H" (default) or "W".
#' @return An \code{nmf_guide} object to pass to \code{nmf(guides = ...)}.
#' @examples
#' # Create an external target guide
#' target <- matrix(runif(30), nrow = 5, ncol = 6)
#' g <- guide_external(target, lambda = 2.0)
#' g
#' @seealso \code{\link{guide_classifier}}, \code{\link{guide_callback}}, \code{\link{nmf}}
#' @export
guide_external <- function(target, lambda = 1.0, side = c("H", "W")) {
  side <- match.arg(side)
  target <- as.matrix(target)
  if (!is.numeric(target)) stop("target must be a numeric matrix")
  structure(list(
    type = "external",
    target = target,
    lambda = as.double(lambda),
    side = side
  ), class = "nmf_guide")
}

#' Print an nmf_guide
#' @param x An \code{nmf_guide} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{guide_classifier}}, \code{\link{guide_external}}, \code{\link{nmf}}
#' @method print nmf_guide
#' @export
print.nmf_guide <- function(x, ...) {
  cat(sprintf("nmf_guide: %s (lambda=%.2f, side=%s)\n",
              x$type, x$lambda, x$side))
  if (x$type == "classifier") {
    n_labeled <- sum(x$labels >= 0)
    n_classes <- length(unique(x$labels[x$labels >= 0]))
    cat(sprintf("  %d labeled samples, %d classes\n", n_labeled, n_classes))
  } else if (x$type == "external") {
    cat(sprintf("  target: %d x %d\n", nrow(x$target), ncol(x$target)))
  } else if (x$type == "callback") {
    cat("  custom callback function\n")
  } else if (x$type == "reference") {
    cat(sprintf("  reference to layer '%s'\n", x$layer_name))
  }
  invisible(x)
}

#' Create a callback guide
#'
#' Steers a factor toward a target computed by a user-supplied R function.
#' The function is called each ALS iteration with the current factor matrix
#' and iteration number, and must return a target matrix of matching
#' dimensions.
#'
#' @param fn A function with signature \code{fn(factor, iter)} that returns
#'   a target matrix (k x n for H, k x m for W).
#' @param lambda Guide strength. Positive attracts; negative repels. Default 1.0.
#' @param side Which factor to guide: "H" (default) or "W".
#' @return An \code{nmf_guide} object.
#' @examples
#' # Guide that steers factors toward a decaying target
#' g <- guide_callback(
#'   fn = function(factor, iter) factor * exp(-0.01 * iter),
#'   lambda = 1.0
#' )
#' @seealso \code{\link{guide_classifier}}, \code{\link{guide_external}}, \code{\link{nmf}}
#' @export
guide_callback <- function(fn, lambda = 1.0, side = c("H", "W")) {
  side <- match.arg(side)
  if (!is.function(fn)) stop("'fn' must be a function(factor, iter)")
  structure(list(
    type = "callback",
    fn = fn,
    lambda = as.double(lambda),
    side = side
  ), class = "nmf_guide")
}

#' Create a reference guide
#'
#' Steers a factor toward another layer's output matrix. The target
#' changes each ALS iteration as the referenced layer updates. Used
#' for cross-layer consistency or hierarchical alignment.
#'
#' @param layer_name Name of the referenced layer in the factor_net.
#' @param lambda Guide strength. Positive attracts; negative repels. Default 1.0.
#' @param side Which factor to guide: "H" (default) or "W".
#' @return An \code{nmf_guide} object.
#' @examples
#' # Reference guide coupling to another layer
#' g <- guide_reference("L1", lambda = 0.5, side = "H")
#' g
#' @seealso \code{\link{guide_classifier}}, \code{\link{guide_external}}, \code{\link{nmf}}
#' @export
guide_reference <- function(layer_name, lambda = 1.0, side = c("H", "W")) {
  side <- match.arg(side)
  if (!is.character(layer_name) || length(layer_name) != 1)
    stop("'layer_name' must be a single character string")
  structure(list(
    type = "reference",
    layer_name = layer_name,
    lambda = as.double(lambda),
    side = side
  ), class = "nmf_guide")
}
