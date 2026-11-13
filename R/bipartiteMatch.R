#' Bipartite graph matching
#'
#' @description Hungarian algorithm for matching samples in a bipartite graph from a distance ("cost") matrix
#' 
#' @details
#' This implementation was adapted from RcppHungarian, an Rcpp wrapper for the original C++ implementation by Cong Ma (2016).
#'
#' Negative costs are permitted. The solver requires non-negative costs, so a matrix
#' with negative entries is shifted by its minimum before solving; a constant shift does
#' not change the optimal assignment, and the returned \code{cost} is computed from the
#' original matrix.
#'
#' @param x matrix giving the cost of every possible pairing (rows are matched to columns)
#' @return List with elements \code{cost} (total matching cost) and \code{assignment} (0-indexed column assignment for each row, \code{-1} for a row left unmatched).
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
  if (!is.numeric(x) || length(x) == 0) stop("'x' must be a non-empty numeric matrix")
  if (anyNA(x)) stop("'x' must not contain NA values")
  storage.mode(x) <- "double"
  x_min <- min(x)
  if (x_min >= 0) return(Rcpp_bipartite_match(x))

  # Shift to non-negative costs (the assignment is invariant to a constant
  # shift), then report the cost on the original scale.
  res <- Rcpp_bipartite_match(x - x_min)
  rows <- which(res$assignment >= 0)
  res$cost <- sum(x[cbind(rows, res$assignment[rows] + 1L)])
  res
}