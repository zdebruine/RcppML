suppressPackageStartupMessages(library(RcppML))
cat("R:", R.Version()$version.string, "\n")

data(movielens)
cat("movielens:", nrow(movielens), "x", ncol(movielens), "\n")

f <- tempfile(fileext = ".spz")
suppressWarnings(sp_write(movielens, f, include_transpose = TRUE))
cat("SPZ file:", file.info(f)$size, "bytes\n")

cat("Calling Rcpp_nmf_streaming_spz directly (forces chunked path)...\n")
flush.console()

# Call C++ directly to bypass auto-dispatch
result <- RcppML:::Rcpp_nmf_streaming_spz(
  path = f, k = 3L, tol = 1e-4, maxit = 3L,
  L1 = 0, L2 = 0, L21 = 0, angular = 0, upper_bound = 0,
  graph_lambda = c(0, 0),
  nonneg_w = TRUE, nonneg_h = TRUE,
  cd_maxit = 100L, cd_tol = 1e-8,
  threads = 4L, seed = 42L,
  verbose = 3L,
  loss_str = "mse", huber_delta = 1.0,
  holdout_fraction = 0, mask_zeros = FALSE,
  cv_seed = 42L
)
cat("streaming loss:", result$loss, "\n")
unlink(f)
cat("ALL DONE\n")
