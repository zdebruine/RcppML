svd_rcppml <- function(data, k){
  if (class(data) == "dgCMatrix") {
    model <- Rcpp_svd_sparse(data, k) 
  } else {
    model <- Rcpp_svd_dense(data, k) 
  }
  return(model)
}