suppressPackageStartupMessages(library(RcppML))
cat("R:", R.Version()$version.string, "\n")

data(movielens)
cat("movielens:", nrow(movielens), "x", ncol(movielens), "\n")

f <- tempfile(fileext = ".spz")
suppressWarnings(sp_write(movielens, f, include_transpose = TRUE))
cat("SPZ file:", file.info(f)$size, "bytes\n")

cat("Starting streaming NMF with verbose=3...\n")
flush.console()

# Use verbose = 3 to get DEBUG-level output
res <- nmf(f, k = 3, maxit = 3, streaming = TRUE, seed = 42, verbose = 3)

cat("streaming loss:", res@misc$loss, "\n")
unlink(f)
cat("ALL DONE\n")
