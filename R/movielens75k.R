#' 75k movie ratings
#' 
#' @description Ratings of 9730 movies across 19 genres by 610 users on a scale of 1 to 5 stars. Zeros indicate no rating.
#' 
#' @details 
#' This dataset was derived from https://grouplens.org/datasets/movielens/ (ml-latest-small.zip), and movies without a genre assignment were removed. Half-star ratings were rounded up.
#'
#' @md
#' @export
#' @docType data
#' @usage data(movielens75k)
#' @format list of two components: \code{ratings} matrix in \code{Matrix::dgCMatrix} format of movies (rows) vs. users (columns), and \code{genres} as a sparse logical matrix in \code{Matrix::lgCMatrix} format, giving genres (rows) to which each movie (columns) is assigned.
#'
"movielens75k"