#' Compute a Target Matrix for Guided NMF
#'
#' Builds a k x n target matrix from class labels, suitable for passing
#' as \code{target_H} to \code{\link{nmf}}.  Each sample column is set
#' to its class centroid (optionally ZCA-whitened), so that target
#' regularization steers H toward class-discriminative structure.
#'
#' With positive \code{target_lambda}, NMF attracts H toward the target
#' (label enrichment).  With negative \code{target_lambda}, NMF uses
#' PROJ_ADV eigenvalue-projected adversarial removal to suppress
#' target-correlated structure (batch removal).  See
#' \code{vignette("guided-nmf")} for mathematical details.
#'
#' @param H  k x n embedding matrix (e.g. from an initial NMF fit).
#' @param labels Factor or character/integer vector of class labels
#'   (length n).  \code{NA} entries are left unguided (zero column).
#' @param whiten  Logical; apply OAS-shrinkage ZCA whitening to class
#'   centroids before broadcasting.  Default \code{TRUE}.
#'
#' @return A numeric k x n matrix.  Pass it to \code{nmf(..., target_H = T,
#'   target_lambda = 0.5)} for enrichment or \code{target_lambda = -0.5}
#'   for batch removal.
#'
#' @details
#' \strong{Algorithm:}
#' \enumerate{
#'   \item Compute per-class centroids from rows/columns of \code{H}.
#'   \item (Optional) Apply Oracle-Approximating Shrinkage (OAS) covariance
#'     estimation followed by ZCA whitening to the centroids, ensuring
#'     isotropic class structure.
#'   \item Broadcast each sample's class centroid back to a k x n matrix.
#' }
#'
#' @examples
#' data(hawaiibirds)
#' model <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50)
#' meta <- attr(hawaiibirds, "metadata_h")
#'
#' # Build target from class labels
#' target <- compute_target(model@h, labels = meta$island)
#'
#' # Enrichment: attract H toward island structure
#' guided <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50,
#'               target_H = target, target_lambda = 0.5)
#'
#' # Batch removal: suppress island-correlated structure
#' removed <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50,
#'                target_H = target, target_lambda = -0.5)
#'
#' @seealso \code{\link{nmf}}, \code{\link{refine}}
#' @export
compute_target <- function(H, labels, whiten = TRUE) {

  if (is.null(dim(H)) || length(dim(H)) != 2)
    stop("'H' must be a k x n matrix")
  k <- nrow(H)
  n <- ncol(H)

  labels <- as.factor(labels)
  if (length(labels) != n)
    stop("length(labels) must equal ncol(H)")
  lvls <- levels(labels)
  C <- length(lvls)

  # --- 1. Class centroids (k x C) ---
  centroids <- matrix(0, nrow = k, ncol = C)
  counts <- integer(C)
  label_idx <- as.integer(labels)  # 1-based
  for (j in seq_len(n)) {
    ci <- label_idx[j]
    if (!is.na(ci)) {
      centroids[, ci] <- centroids[, ci] + H[, j]
      counts[ci] <- counts[ci] + 1L
    }
  }
  for (ci in seq_len(C)) {
    if (counts[ci] > 0)
      centroids[, ci] <- centroids[, ci] / counts[ci]
  }

  grand_mean <- rowMeans(centroids[, counts > 0, drop = FALSE])

  # --- 2. OAS-ZCA whitening (optional) ---
  if (whiten && C > 1) {
    # Centered centroids (each weighted by sqrt(count) for proper covariance)
    wts <- sqrt(pmax(counts, 1))
    X <- sweep(centroids, 1, grand_mean) * rep(wts, each = k)  # k x C
    S <- tcrossprod(X) / sum(counts)  # k x k sample covariance

    # OAS shrinkage: alpha = ((1-2/k)*tr(S^2) + tr(S)^2) / ((n_eff+1-2/k)*(tr(S^2) - tr(S)^2/k))
    trS <- sum(diag(S))
    trS2 <- sum(S * S)
    n_eff <- sum(counts)
    rho_num <- (1 - 2/k) * trS2 + trS^2
    rho_den <- (n_eff + 1 - 2/k) * (trS2 - trS^2 / k)
    rho <- if (abs(rho_den) < 1e-12) 1 else min(1, max(0, rho_num / rho_den))

    # Shrink toward scaled identity
    S_shrunk <- (1 - rho) * S + rho * (trS / k) * diag(k)

    # Eigen-decompose for ZCA: V diag(1/sqrt(d)) V^T
    eig <- eigen(S_shrunk, symmetric = TRUE)
    vals <- pmax(eig$values, 1e-10)
    V <- eig$vectors
    # ZCA whitening matrix
    W_zca <- V %*% diag(1 / sqrt(vals)) %*% t(V)

    centroids <- W_zca %*% centroids
    grand_mean <- W_zca %*% grand_mean
  }

  # --- 3. Broadcast to k x n target ---
  # Each sample gets: centroid[label] - grand_mean (centered shift)
  target <- matrix(0, nrow = k, ncol = n)
  shift <- sweep(centroids, 1, grand_mean)
  for (j in seq_len(n)) {
    ci <- label_idx[j]
    if (!is.na(ci)) {
      target[, j] <- shift[, ci]
    }
  }

  target
}
