#' Bipartite graph matching
#'
#' @description Hungarian algorithm for matching samples in a bipartite graph from a distance ("cost") matrix
#' 
#' @details
#' This implementation was adapted from RcppHungarian, an Rcpp wrapper for the original C++ implementation by Cong Ma (2016).
#'
#' @param x symmetric matrix giving the cost of every possible pairing
#' @return List with elements \code{cost} (total matching cost) and \code{assignment} (0-indexed column assignment for each row).
#' @seealso \code{\link{align}}, \code{\link{consensus_nmf}}
#' @export
#' @examples
#' cost <- matrix(c(1, 0.5, 0.2,
#'                   0.5, 1, 0.3,
#'                   0.2, 0.3, 1), nrow = 3, byrow = TRUE)
#' result <- bipartiteMatch(cost)
#' result$cost
#' result$assignment
#'
bipartiteMatch <- function(x) {
  x <- as.matrix(x)
  if (min(x) < 0) {
    if(getOption("RcppML.verbose")) warning("Negative values in 'x' were replaced by zero. Negative values are not permitted!")
    x[x < 0] <- 0
  }
  return(Rcpp_bipartite_match(x))
}