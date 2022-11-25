#' Cross-validation for NMF
#' 
#' @description Find an "optimal" rank for a Non-Negative Matrix Factorization using cross-validation. Returns a \code{data.frame} with class \code{nmfCrossValidate}. Plot results using the \code{plot} class method.
#' 
#' @details 
#' A random speckled pattern of values is masked off during model fitting, and the mean squared error of prediction is evaluated after the model has reached the desired tolerance. The rank at which the model achieves the lowest error (best prediction accuracy) is the optimal rank.
#' 
#' @inheritParams nmf
#' @param k array of factorization ranks to test
#' @param reps number of independent replicates to run
#' @param n fraction of values to handle as missing (default is 5%, or \code{0.05})
#' @param verbose should updates be displayed when each factorization is completed
#' @param ... parameters to \code{RcppML::nmf}, not including \code{data} or \code{k}
#' @return \code{data.frame} with class \code{nmfCrossValidate} with columns \code{rep}, \code{k}, and \code{value}
#' @md
#' @seealso \code{\link{nmf}}
#' @export
crossValidate <- function(data, k, reps = 3, n = 0.05, verbose = FALSE, ...) {
  verbose <- getOption("RcppML.verbose")
  options("RcppML.verbose" = FALSE)
  p <- list(...)
  defaults <- list("mask" = NULL)
  for (i in 1:length(defaults))
    if (is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]
  
  if(!is.null(p$mask)){
    if(!is.character(p$mask)){
      if(canCoerce(p$mask, "ngCMatrix")){
        p$mask <- as(p$mask, "ngCMatrix")
      } else if(canCoerce(p$mask, "matrix")){
        p$mask <- as(as.matrix(p$mask), "ngCMatrix")
      } else stop("supplied masking value was invalid or not coercible to a matrix")
    }
  }
  
  results <- list()
  for(rep in 1:reps){
    if(verbose) cat("\nReplicate ", rep, ", rank: ", sep = "")
    
    # get samples, features, or missing values for test/training/cross-validation
    if(!is.null(p$seed)) set.seed(p$seed + rep)
    mask_matrix <- rsparsematrix(as.numeric(nrow(data)), as.numeric(ncol(data)), n, rand.x = NULL)

    for(rank in k){
      if(verbose) cat(rank, " ")
      
      mask_22 <- mask_21 <- mask_11 <- mask_w1 <- mask_w2 <- p$mask
      m <- nmf(data, rank, mask = mask_matrix, ...)
      value <- evaluate(m, data, mask = mask_matrix, missing_only = TRUE)
      results[[length(results) + 1]] <- c(rep, rank, value)
    }
  }
  results <- data.frame(do.call(rbind, results))
  colnames(results) <- c("rep", "k", "value")
  class(results) <- c("nmfCrossValidate", "data.frame")
  results$rep <- as.factor(results$rep)
  options("RcppML.verbose" = verbose)
  return(results)
}
