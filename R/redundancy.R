#' Calculate redundancy of data per column
#' 
#' @param A sparse matrix of class \code{dgCMatrix}
#' @export
#' @return fractional redundancy by column
#'
calc_redundancy <- function(A){
  redundancy <- c()
  for(i in 1:ncol(A)){
    redundancy <- c(redundancy, length(unique(A[,1])) / (A@p[[i + 1]] - A@p[[i]]))
  }
  1 - redundancy
}