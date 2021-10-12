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
#' @return dense matrix
#'
simulateNMF <- function(nrow, ncol, k, noise = 0.5, dropout = 0.5, seed = NULL){
  
  if(!is.null(seed)) set.seed(seed)

  g <- rmultinom(1, ncol, rep(1, k))
  b <- rmultinom(1, nrow, rep(1, k))
  
  # generate H
  H <- matrix(0, k, ncol)
  tmp <- 0
  for(i in 1:k){
    H[i, (tmp+1):(tmp+g[i])] <- 1
    tmp <- tmp+g[i]
  } 	
  
  # generate W
  W <- matrix(0, nrow, k)
  tmp <- 0
  for(i in 1:k){		
    W[(tmp+1):(tmp+b[i]),i] <- abs(rnorm(b[i], 1, 1))
    tmp <- tmp + b[i]
  }
  
  # build the input matrix
  res <- W %*% H

  # add noise
  if(noise > 0){
    res <- res + matrix(rnorm(nrow * ncol, mean = 0, sd = noise), nrow, ncol)
    res[res < 0] <- 0
  }
  
  # introduce dropout
  if(dropout > 0){
    d <- as.matrix(as(Matrix::rsparsematrix(nrow, ncol, 1 - dropout, rand.x = NULL), "dgCMatrix"))
    res <- res * d
  }
  
  res
}