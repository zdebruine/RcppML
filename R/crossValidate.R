#' Cross-validation for NMF
#' 
#' @description Find an "optimal" rank for a Non-Negative Matrix Factorization using cross-validation. Returns a \code{data.frame} with class \code{nmfCrossValidate}. Plot results using the \code{plot} S3 method.
#' 
#' @details 
#' Two algorithms are available for cross-validation and can be selected using the \code{method} parameter:
#'
#' 1. **Bi-cross-validation**: Data is split into an equally-sized 2x2 grid (A1, A2, B1, and B2). Data is trained on A1 and projected to A2 to B2 where the mean squared error of the model on B2 is measured.
#' 2. **Cost of bipartite matching** between independent replicates. Samples are partitioned into two sets and independently factorized, then correlation between factors in both \code{w} models is computed, factors are matched using bipartite matching on a cosine similarity cost matrix, and the mean cost of matched factors is returned.
#' 
#' @inheritParams nmf
#' @param k array of factorization ranks to test
#' @param method algorithm to use for cross-validation, see "details", either \code{1} or \code{2}.
#' @param reps number of independent replicates to run
#' @param verbose level of verbosity, \code{c(0, 1, 2)}
#' @param ... parameters to \code{RcppML::nmf}, not including \code{data} or \code{k}
#' @return \code{data.frame} with columns \code{rep}, \code{k}, and \code{value}, where \code{value} depends on the \code{method} selected (i.e. MSE, cost of bipartite matching)
#' @md
#' @seealso \code{\link{nmf}}
#' @export
crossValidate <- function(data, k, method = 1, reps = 5, verbose = 1, ...) {
  results <- list()
  for(rep in 1:reps){
    if(verbose > 0) cat("\nREPLICATE", rep, "\n")
    
    # get samples and features for test and training sets
    set.seed(rep)
    samples <- sample(1:ncol(data), floor(ncol(data) * 0.5))
    set.seed(rep)
    features <- sample(1:nrow(data), floor(nrow(data) * 0.5))
    ifelse(verbose == 2, verbose2 <- TRUE, verbose2 <- FALSE)
    
    for(rank in k){
      if(verbose > 0) cat("Rank:", rank, "\n")
      
      if(method == 1){
        m22 <- nmf(data[-features, -samples], rank, verbose = verbose2, ...)
        m21 <- project(m22$w, data[-features, samples])
        m11 <- project(m21, t(data[features, samples]))
        m <- new_nmf("w" = t(m11), "d" = rep(1, rank), "h" = m21)
        class(m) <- "nmf"
        value <- mse(m, data[features, samples])
      } else {
        w1 <- nmf(data[, samples], rank, verbose = verbose2, ...)$w
        w2 <- nmf(data[, -samples], rank, verbose = verbose2, ...)$w
        value <- bipartiteMatch(1 - cosine(w1, w2))$cost / rank
      }
      results[[length(results) + 1]] <- c(rep, rank, value)
    }
  }
  results <- data.frame(do.call(rbind, results))
  colnames(results) <- c("rep", "k", "value")
  class(results) <- c("nmfCrossValidate", "data.frame")
  results$rep <- as.factor(results$rep)
  return(results)
}