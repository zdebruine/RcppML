#' Simulate an NMF dataset
#' 
#' @description Generate a random matrix that follows some defined NMF model to test NMF factorizations. Adapts methods from \code{NMF::syntheticNMF}.
#'
#' @param nrow number of rows
#' @param ncol number of columns
#' @param k true rank of simulated model
#' @param noise standard deviation of Gaussian noise centered at 0 to add to input matrix. Any negative values after noise addition are set to 0.
#' @param dropout density of dropout events
#' @param seed seed for random number generation
#' @export
#' @importFrom stats cor rmultinom rnorm runif
#' @return list of dense matrix \code{A} and true \code{w} and \code{h} models
#'
simulateNMF <- function(nrow, ncol, k, noise = 0.5, dropout = 0.5, seed = NULL) {

  if (!is.null(seed)) set.seed(seed)
  num_nzh <- round(rnorm(k, ncol / 2, sd = ncol / 4))
  num_nzw <- round(rnorm(k, nrow / 2, sd = nrow / 4))
  num_nzw[num_nzw < 2] <- 2
  num_nzw[num_nzw > nrow] <- nrow
  num_nzh[num_nzh < 2] <- 2
  num_nzh[num_nzh > ncol] <- ncol

  h <- matrix(0, k, ncol)
  w <- matrix(0, nrow, k)
  
  # decide which indices are going to be non-zero

  for (i in 1:k){
    h[i, sample(1:ncol, num_nzh[i])] <- abs(rnorm(num_nzh[i], 1, 1))
    w[sample(1:nrow, num_nzw[i]), i] <- abs(rnorm(num_nzw[i], 1, 1))
  }
  w <- as.matrix(w %*% Diagonal(x = 1 / colSums(w)))
  h <- as.matrix(Diagonal(x = 1 / rowSums(h)) %*% h)

  # build the input matrix
  res <- w %*% h

  # add noise
  if (noise > 0) {
    res <- res + matrix(rnorm(nrow * ncol, mean = 0, sd = noise), nrow, ncol)
    res[res < 0] <- 0
  }

  # introduce dropout
  if (dropout > 0) {
    d <- as.matrix(as(Matrix::rsparsematrix(nrow, ncol, 1 - dropout, rand.x = NULL), "dgCMatrix"))
    res <- res * d
  }

  list(A = res, w = W, h = H)
}