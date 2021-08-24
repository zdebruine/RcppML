#' Set the number of threads RcppML should use
#'
#' @description The number of threads is 0 by default (corresponding to all available threads), but can be set manually using this function. If you clear environment variables or affect the "RcppMLthreads" environment variable specifically, you will need to set your number of preferred threads again.
#' 
#' @details
#' The number of threads set affects OpenMP parallelization only for functions in the RcppML package. It does not affect other R packages that use OpenMP. Parallelization is used for projection of linear factor models with rank > 2, calculation of mean squared error for linear factor models, and for divisive clustering.
#'
#' @param threads number of threads to be used in RcppML functions that are parallelized with OpenMP.
#' @export
#' @seealso \code{\link{getRcppMLthreads}}
#' @author Zach DeBruine
#' @examples
#' # set parallel configuration, 
#' # letting OpenMP decide how many threads to use
#' setRcppMLthreads()
#' getRcppMLthreads()
#'
#' # set serial configuration
#' setRcppMLthreads(1)
#' getRcppMLthreads()
#'
setRcppMLthreads <- function(threads = 0){
    Sys.setenv("RcppMLthreads" = threads)
}

#' Get the number of threads RcppML should use
#'
#' @description Get the number of threads that will be used by RcppML functions supporting parallelization with OpenMP. Use \code{\link{setRcppMLthreads}} to set the number of threads to be used.
#' 
#' @returns integer giving number of threads to be used by RcppML functions. \code{0} corresponds to all available threads, as determined by OpenMP.
#' @export
#' @seealso \code{\link{setRcppMLthreads}}
#' @author Zach DeBruine
#' @examples
#' # set parallel configuration, 
#' # letting OpenMP decide how many threads to use
#' setRcppMLthreads()
#' getRcppMLthreads()
#'
#' # set serial configuration
#' setRcppMLthreads(1)
#' getRcppMLthreads()
#'
getRcppMLthreads <- function(){
    num_threads <- as.integer(Sys.getenv("RcppMLthreads"))
    ifelse(is.na(num_threads), 0, num_threads)
}