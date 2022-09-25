#' View internal structure of sparse encodings
#' 
#' @param A object of class \code{dgCMatrix}
#' @param encoding sparse encoding to be used
#' @importFrom utils str
#' @export
#' @return debug level output to console
#' @examples 
#' library(Matrix)
#' A <- rsparsematrix(20, 5, density = 0.2,
#'   rand.x = function(n) { sample(1:5, n, replace = TRUE)})
#' inspect(A, "TRCSC_NP")
inspect <- function(A, encoding = "TRCSC_NP"){
  str(c_inspect(A, encoding)[-1])
}