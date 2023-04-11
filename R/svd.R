svd <- function(data, k){
  start_time <- Sys.time()
  
  model <- Rcpp_svd_dense(data, k)
}