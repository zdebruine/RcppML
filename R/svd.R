svd <- function(data, k){
    model <- Rcpp_svd_sparse(data,k)
    new("svd", w = model$w, h = model$h, misc = misc)
}