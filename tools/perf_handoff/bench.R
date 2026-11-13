suppressPackageStartupMessages(library(Matrix))
Rcpp::sourceCpp("parallel_transpose.cpp", cacheDir = "cache")
set.seed(1)
mk <- function(m, n, d, skew = FALSE) {
  A <- rsparsematrix(m, n, d, rand.x = function(k) rpois(k, 2) + 1)
  if (skew) { w <- rexp(m)^2; A <- A[order(-w), ]; A <- A[rbinom(m, 1, pmin(1, w / max(w) * 5 + 0.05)) == 1, ] }
  A
}
for (cfg in list(list(20000, 50000, 0.03, FALSE), list(20000, 50000, 0.03, TRUE), list(200, 3000, 0.2, FALSE), list(5, 7, 0.5, FALSE))) {
  A <- mk(cfg[[1]], cfg[[2]], cfg[[3]], cfg[[4]])
  for (th in c(1, 4, 8)) {
    r <- bench_setup(A, th); r <- bench_setup(A, th)
    cat(sprintf("%6dx%-6d nnz=%-9d th=%d  eigen cast %.3f transpose %.3f | parallel cast %.3f transpose %.3f | identical=%s\n",
      nrow(A), ncol(A), length(A@x), th, r$eigen_cast, r$eigen_transpose, r$par_cast, r$par_transpose, r$identical))
  }
}
