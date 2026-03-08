suppressPackageStartupMessages(library(RcppML))
cat("RcppML loaded\n")

# Quick streaming smoke test
data(movielens)
f <- tempfile(fileext = ".spz")
suppressWarnings(sp_write(movielens, f, include_transpose = TRUE))
cat("SPZ written\n")

res <- nmf(f, k = 3, maxit = 5, streaming = TRUE, seed = 42, verbose = FALSE)
cat("loss:", res@misc$loss, "\n")
cat("solver_mode:", res@misc$solver_mode, "\n")
cat("dim W:", paste(dim(res@w), collapse = "x"), "\n")
cat("dim H:", paste(dim(res@h), collapse = "x"), "\n")
unlink(f)
cat("DONE\n")
