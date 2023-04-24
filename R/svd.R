#' @title Single Value Decomposition
#'
#' @description Performs SVD via alternating-least-squares, returns a list
#'
#' @details
#' Parallelization is applied with OpenMP using the number of threads in \code{getOption("RcppML.threads")} and set by \code{option(RcppML.threads = 0)}, for example. \code{0} corresponds to all threads, let OpenMP decide.
#
#' @param data dense or sparse matrix of features in rows and samples in columns. Prefer \code{matrix} or \code{Matrix::dgCMatrix}, respectively
#' @param k rank

#' @author Wyatt Barber
#'
#' @export
#' @md
svd_rcppml <- function(data, k, tol=1e-9, maxit=100){
  if (class(data)[[1]] == "dgCMatrix") {
    model <- Rcpp_svd_sparse(data, k, tol, maxit, getOption("RcppML.threads"), getOption("RcppML.verbose")) 
  } else {
    model <- Rcpp_svd_dense(data, k, tol, maxit, getOption("RcppML.threads"), getOption("RcppML.verbose"))  
  }
  return(model)
}