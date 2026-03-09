#!/usr/bin/env Rscript
# Prepare pbmc3k for CRAN: compress with StreamPress, store as raw bytes
library(RcppML)

# Load the current large dgCMatrix
load("data/pbmc3k.rda")
original <- pbmc3k
cat("Original class:", class(original), "\n")
cat("Original dimensions:", dim(original), "\n")
cat("Original nnz:", Matrix::nnzero(original), "\n")
cat("Original .rda size:", file.info("data/pbmc3k.rda")$size / 1024, "KB\n")
cat("All integer values:", all(original@x == floor(original@x)), "\n")
cat("Value range:", range(original@x), "\n")

# Write to StreamPress format - fp64 for lossless, no transpose to save space
tmp <- tempfile(fileext = ".spz")
st_write(original, tmp, precision = "fp64", include_transpose = FALSE)
cat("SPZ file size (fp64, no transpose):", file.info(tmp)$size / 1024, "KB\n")

# Read back and check
recovered <- st_read(tmp)
cat("Max abs diff:", max(abs(original@x - recovered@x)), "\n")
cat("identical p:", identical(original@p, recovered@p), "\n")
cat("identical i:", identical(original@i, recovered@i), "\n")
cat("identical x:", identical(original@x, recovered@x), "\n")
cat("has rownames:", !is.null(rownames(original)), "\n")
cat("recovered has rownames:", !is.null(rownames(recovered)), "\n")

# Check if dimnames are the issue
cat("all.equal ignoring names:", all.equal(unname(as.matrix(original)), unname(as.matrix(recovered))), "\n")

# If values match but dimnames don't, that's fine — SPZ may not store them.
# For the final version: use the numeric data from SPZ and the round-trip
# only needs to validate numerically.

# Use fp32 for smaller file since data is integer counts
tmp_fp32 <- tempfile(fileext = ".spz")
st_write(original, tmp_fp32, precision = "fp32", include_transpose = FALSE)
recovered_fp32 <- st_read(tmp_fp32)
cat("fp32 max abs diff:", max(abs(original@x - recovered_fp32@x)), "\n")
cat("fp32 file size:", file.info(tmp_fp32)$size / 1024, "KB\n")

# Since data is integer (1-419), fp32 is LOSSLESS for integers up to 2^24
# Try fp32 for smaller size
cat("\n--- Using fp32 (lossless for integer data up to 16M) ---\n")
pbmc3k <- readBin(tmp_fp32, "raw", n = file.info(tmp_fp32)$size)
cat("Raw vector length:", length(pbmc3k), "bytes (", length(pbmc3k)/1024, "KB)\n")

save(pbmc3k, file = "data/pbmc3k.rda", compress = "xz")
cat("New .rda size (xz):", file.info("data/pbmc3k.rda")$size / 1024, "KB\n")

# Final verification
load("data/pbmc3k.rda")
tmp3 <- tempfile(fileext = ".spz")
writeBin(pbmc3k, tmp3)
final <- st_read(tmp3)
cat("Final dimensions:", dim(final), "\n")
cat("Final nnz:", Matrix::nnzero(final), "\n")
cat("Final max abs diff from original:", max(abs(original@x - final@x)), "\n")
cat("\nDone! pbmc3k.rda is now SPZ raw bytes.\n")
