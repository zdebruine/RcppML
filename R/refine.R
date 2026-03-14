#' Refine an NMF Model Using Label-Guided Correction
#'
#' Post-hoc correction of an NMF embedding to improve class separation
#' or remove batch effects, optionally followed by W-refit cycles that
#' propagate the corrected H back through the model.
#'
#' @param x An NMF model (S4 object of class \code{nmf}) or a k x n
#'   embedding matrix.
#' @param data Original data matrix (required when \code{cycles > 0}
#'   or \code{batch} is supplied with \code{cycles > 0}).
#'   Must be the same matrix used to fit \code{x}.
#' @param labels Factor or character/integer vector of class labels (length n).
#' @param batch Optional factor of batch labels (length n) for batch removal.
#'   When supplied, batch-correlated structure is suppressed using the
#'   PROJ_ADV method (eigenvalue-projected adversarial removal).
#'   See \code{vignette("guided-nmf")} for details.
#' @param lambda Correction strength in \code{[0, 1]}. Default 0.8.
#'   Controls both label enrichment (positive direction) and batch
#'   removal strength (when \code{batch} is supplied).
#' @param cycles Number of W-refit cycles. 0 = post-hoc only (default).
#'   Each cycle: refit W from corrected H, refit H from new W, re-correct H.
#' @param nonneg Enforce non-negativity on refitted factors. Default \code{TRUE}.
#' @param whiten Apply OAS-ZCA whitening to class centroids. Default \code{TRUE}.
#'
#' @return If \code{x} is an \code{nmf} object, returns an updated
#'   \code{nmf} object.  If \code{x} is a matrix, returns a corrected
#'   k x n matrix.
#'
#' @details
#' The correction proceeds in two stages:
#'
#' \strong{Stage 1: Post-hoc centroid correction} (always performed)
#'
#' Computes a target matrix from \code{labels} via \code{\link{compute_target}},
#' then shifts each sample's embedding toward its class centroid:
#' \deqn{H_{corrected} = H + \lambda \cdot T}
#' where T is the target matrix (class centroid shifts, optionally whitened).
#'
#' \strong{Stage 1b: Batch removal} (when \code{batch} is supplied)
#'
#' Uses PROJ_ADV (Projected Adversarial) method: computes a batch target
#' from batch labels with negative \code{target_lambda}, which subtracts
#' the batch Gram matrix from the NNLS Gram matrix, eigendecomposes,
#' and clips negative eigenvalues.  This suppresses batch-correlated
#' directions while preserving all other structure.
#'
#' \strong{Stage 2: W-refit cycles} (when \code{cycles > 0})
#'
#' Iteratively refits the model:
#' \enumerate{
#'   \item Solve for W given corrected H: \eqn{W = \arg\min \|A - W d H_c\|_F^2}
#'   \item Solve for H given new W: \eqn{H = \arg\min \|A - W_{new} d H\|_F^2}
#'   \item Re-apply centroid correction to the new H
#' }
#'
#' @examples
#' data(hawaiibirds)
#' model <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50)
#' meta <- attr(hawaiibirds, "metadata_h")
#'
#' # Post-hoc correction only (cycles = 0)
#' corrected <- refine(model, labels = meta$island, lambda = 0.8)
#'
#' # W-refit cycles: propagate correction back through the model
#' refined <- refine(model, data = hawaiibirds, labels = meta$island,
#'                   lambda = 0.8, cycles = 3)
#'
#' @seealso \code{\link{compute_target}}, \code{\link{nmf}}
#' @export
refine <- function(x, data = NULL, labels, batch = NULL,
                   lambda = 0.8, cycles = 0L, nonneg = TRUE,
                   whiten = TRUE) {

  # --- Extract H and model info ---
  is_nmf <- is(x, "nmf")
  if (is_nmf) {
    H <- x@h
    W <- x@w   # m x k
    d <- x@d
    k <- length(d)
    n <- ncol(H)
  } else if (is.matrix(x)) {
    H <- x
    k <- nrow(H)
    n <- ncol(H)
    W <- NULL
    d <- NULL
  } else {
    stop("'x' must be an nmf object or a k x n matrix")
  }

  # --- Validate ---
  labels <- as.factor(labels)
  if (length(labels) != n)
    stop("length(labels) must equal ncol(H) [= ", n, "]")
  if (lambda < 0 || lambda > 1)
    stop("'lambda' must be in [0, 1]")
  cycles <- as.integer(cycles)
  if (cycles > 0 && is.null(data))
    stop("'data' is required when cycles > 0")
  has_batch <- !is.null(batch)
  if (has_batch) {
    batch <- as.factor(batch)
    if (length(batch) != n)
      stop("length(batch) must equal ncol(H) [= ", n, "]")
  }

  # --- Stage 1: Compute target and apply correction ---
  target <- compute_target(H, labels, whiten = whiten)

  # Scale target to match H energy — whitened targets can be orders of
  # magnitude larger than H (which is unit-norm-scaled after NMF).
  # Without this, even small lambda destroys reconstruction.
  fro_H <- sqrt(sum(H^2))
  fro_T <- sqrt(sum(target^2))
  if (fro_T > 1e-10)
    target <- target * (fro_H / fro_T)

  H_corr <- H + lambda * target

  # Enforce non-negativity if requested
  if (nonneg) H_corr[H_corr < 0] <- 0

  # --- Stage 2: W-refit cycles ---
  if (cycles > 0 && is_nmf) {
    if (is.null(data))
      stop("'data' is required for W-refit cycles")

    # Pre-compute batch target if needed (for PROJ_ADV in-NMF removal)
    batch_target <- NULL
    if (has_batch) {
      batch_target <- compute_target(H, labels = batch, whiten = FALSE)
    }

    for (cyc in seq_len(cycles)) {
      # Refit W from corrected H
      dH <- diag(d) %*% H_corr  # k x n
      G <- tcrossprod(dH)        # k x k
      B <- if (inherits(data, "dgCMatrix")) {
        as.matrix(data %*% Matrix::t(dH))
      } else {
        data %*% t(dH)
      }
      W_new <- t(solve(G + 1e-8 * diag(k), t(B)))  # m x k
      if (nonneg) W_new[W_new < 0] <- 0

      # Refit H from new W, with batch removal via PROJ_ADV
      if (has_batch && !is.null(batch_target)) {
        # Use in-NMF PROJ_ADV: negative target_lambda suppresses batch directions
        H_model <- nmf(data, k = k,
                       seed = W_new,
                       maxit = 1L, nonneg = nonneg,
                       target_H = batch_target,
                       target_lambda = c(0, -lambda))
        H_new <- H_model@h
        d_new <- H_model@d
        W_new <- H_model@w
      } else {
        WtW <- crossprod(W_new)   # k x k
        WtA <- if (inherits(data, "dgCMatrix")) {
          as.matrix(Matrix::crossprod(data, W_new))
        } else {
          crossprod(data, W_new)
        }
        H_new <- solve(WtW + 1e-8 * diag(k), t(WtA))  # k x n
        if (nonneg) H_new[H_new < 0] <- 0

        # Rescale: extract d as column norms of H
        d_new <- sqrt(rowSums(H_new^2))
        d_new[d_new < 1e-10] <- 1e-10
        H_new <- H_new / d_new
        W_new <- sweep(W_new, 2, d_new, "*")
      }

      # Update for next cycle
      W <- W_new
      d <- d_new
      H <- H_new

      # Re-apply label centroid correction (with Frobenius scaling)
      target <- compute_target(H, labels, whiten = whiten)
      fro_H <- sqrt(sum(H^2))
      fro_T <- sqrt(sum(target^2))
      if (fro_T > 1e-10)
        target <- target * (fro_H / fro_T)
      H_corr <- H + lambda * target
      if (nonneg) H_corr[H_corr < 0] <- 0
    }
  }

  # --- Return ---
  if (is_nmf) {
    new("nmf",
        w = if (!is.null(W) && cycles > 0) W else x@w,
        d = if (!is.null(d) && cycles > 0) d else x@d,
        h = H_corr,
        misc = x@misc)
  } else {
    H_corr
  }
}
