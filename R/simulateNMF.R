#' Simulate an NMF dataset
#' 
#' @description Generate a random nonnegative matrix with known factor structure for benchmarking NMF recovery.
#' Uses block-diagonal construction: each factor owns a disjoint subset of features (rows) and dominates
#' a disjoint subset of samples (columns), with small cross-talk for realism. This produces clearly
#' recoverable factors even at moderate noise levels. Inspired by \code{NMF::syntheticNMF}.
#'
#' @param nrow number of rows (features)
#' @param ncol number of columns (samples)
#' @param k true rank (number of factors)
#' @param noise noise level as a multiplier on the mean signal. A value of 1.0 means the noise
#'   standard deviation equals the mean signal value. Default: 0.5.
#' @param dropout fraction of entries to set to zero (0 = no dropout). Default: 0.
#' @param seed seed for random number generation
#' @seealso \code{\link{simulateSwimmer}}, \code{\link{nmf}}
#' @export
#' @importFrom stats rnorm rbinom
#' @return list of dense matrix \code{A} and true \code{w} and \code{h} models
#' @examples
#' data <- simulateNMF(50, 30, k = 3, noise = 0.1, seed = 42)
#' dim(data$A)  # 50 x 3
#' dim(data$w)  # 50 x 3
#' dim(data$h)  # 3 x 30
#'
simulateNMF <- function(nrow, ncol, k, noise = 0.5, dropout = 0, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)

  # ── Feature blocks: each factor owns a disjoint set of rows ──
  block_w <- nrow %/% k
  w <- matrix(0, nrow, k)
  for (i in seq_len(k)) {
    start <- (i - 1) * block_w + 1
    end <- if (i == k) nrow else i * block_w
    w[start:end, i] <- abs(rnorm(end - start + 1, mean = 1, sd = 0.3))
  }
  # Small cross-talk between blocks for realism
  w <- w + matrix(abs(rnorm(nrow * k, 0, 0.05)), nrow, k)

  # ── Sample blocks: each factor dominates a subset of columns ──
  block_h <- ncol %/% k
  h <- matrix(0, k, ncol)
  for (i in seq_len(k)) {
    start <- (i - 1) * block_h + 1
    end <- if (i == k) ncol else i * block_h
    h[i, start:end] <- abs(rnorm(end - start + 1, mean = 1, sd = 0.3))
  }
  # Background activation
  h <- h + matrix(abs(rnorm(k * ncol, 0, 0.05)), k, ncol)

  # Normalize
  w <- sweep(w, 2, colSums(w), "/")
  h <- sweep(h, 1, rowSums(h), "/")

  # Build clean signal
  res <- w %*% h

  # Add noise (scaled to mean signal)
  if (noise > 0) {
    res <- res + matrix(rnorm(nrow * ncol, 0, sd = noise * mean(res)), nrow, ncol)
    res[res < 0] <- 0
  }

  # Dropout
  if (dropout > 0) {
    mask <- matrix(rbinom(nrow * ncol, 1, 1 - dropout), nrow, ncol)
    res <- res * mask
  }

  list(A = res, w = w, h = h)
}