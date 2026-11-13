library(RcppML); library(Matrix)
data(pbmc3k); tmp <- tempfile(fileext = ".spz"); writeBin(pbmc3k, tmp); A <- st_read(tmp)
A <- as(A, "CsparseMatrix"); cs <- colSums(A); A@x <- log1p(A@x / rep(cs, diff(A@p)) * 1e4)
A <- A[rowSums(A) > 0, ]
saveRDS(A, file.path(Sys.getenv("RW"), "out/pbmc3k_lognorm.rds")); cat(dim(A), length(A@x), "\n")
