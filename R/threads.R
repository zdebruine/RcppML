#' Get and set the number of threads RcppML should use
#'
#' @description Get or set the number of threads that will be used by RcppML functions supporting parallelization with OpenMP. 
#' 
#' @details
#' Note: \code{0} corresponds to all available threads as determined by OpenMP.
#' 
#' The number of threads set affects OpenMP parallelization only for functions in the RcppML package.
#' It does not affect other R packages that use OpenMP. 
#' 
#' Parallelization is used for projection of linear factor models with rank > 2, 
#' calculation of mean squared error for linear factor models, and for divisive clustering.
#'
#' @export
#' @author Zach DeBruine
#' @rdname threads
#' @examples
#' \dontrun{
#' # set serial configuration
#' setRcppMLthreads(1)
#' getRcppMLthreads()
#'
#' # restore default parallel configuration, 
#' # letting OpenMP decide how many threads to use
#' setRcppMLthreads(0)
#' getRcppMLthreads()
#' }
getRcppMLthreads <- function() {
  threads <- Sys.getenv("RcppMLthreads")
  ifelse(threads == "", 0L, as.integer(threads))
}

#' @param threads number of threads to be used in RcppML functions that are parallelized with OpenMP.
#' @export
#' @rdname threads
setRcppMLthreads <- function(threads) {
  Sys.setenv("RcppMLthreads" = threads)
}
