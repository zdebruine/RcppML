suppressPackageStartupMessages(library(RcppML))
cat("R:", R.Version()$version.string, "\n")

# Test 1: Non-streaming NMF
data(movielens)
cat("movielens:", nrow(movielens), "x", ncol(movielens), "\n")
res <- nmf(movielens, k = 3, maxit = 5, seed = 42, verbose = FALSE)
cat("non-streaming loss:", res@misc$loss, "\n")

# Test 2: Streaming NMF with verbose
f <- tempfile(fileext = ".spz")
suppressWarnings(sp_write(movielens, f, include_transpose = TRUE))
cat("SPZ file:", file.info(f)$size, "bytes\n")
cat("Starting streaming NMF...\n")
flush.console()
res2 <- nmf(f, k = 3, maxit = 3, streaming = TRUE, seed = 42, verbose = TRUE)
cat("streaming loss:", res2@misc$loss, "\n")
unlink(f)
cat("ALL DONE\n")
