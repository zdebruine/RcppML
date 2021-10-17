#' Bipartite graph matching
#'
#' @description Hungarian algorithm for matching samples in a bipartite graph from a distance ("cost") matrix
#' 
#' @details
#' This implementation was adapted from RcppHungarian, an Rcpp wrapper for the original C++ implementation by Cong Ma (2016).
#'
#' @param x symmetric matrix giving the cost of every possible pairing
#' @return List of "cost" and "pairs"
#' @export
#'
bipartiteMatch <- function(x) {
  x <- as.matrix(x)
  if (min(x) < 0) {
    if(getOption("RcppML.verbose")) warning("Negative values in 'x' were replaced by zero. Negative values are not permitted!")
    x[x < 0] <- 0
  }
  return(Rcpp_bipartite_match(x))
}