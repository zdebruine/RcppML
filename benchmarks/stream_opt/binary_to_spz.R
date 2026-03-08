#!/usr/bin/env Rscript
# Convert binary CSC arrays (from convert_npz_to_binary.py) to SPZ format.
#
# Usage: Rscript binary_to_spz.R <binary_dir> <output.spz> [chunk_cols]
#
# binary_dir must contain: data.bin, indices.bin, indptr.bin, shape.txt
# chunk_cols controls SPZ internal chunking (default: 256)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript binary_to_spz.R <binary_dir> <output.spz> [chunk_cols]")
}
binary_dir <- args[1]
output_spz <- args[2]
chunk_cols <- if (length(args) >= 3) as.integer(args[3]) else 256L

library(Matrix)

# Locate the package — try installed first, fall back to devtools
tryCatch(library(RcppML), error = function(e) {
  message("RcppML not in default lib, trying /tmp/rcppml_agent7")
  library(RcppML, lib.loc = "/tmp/rcppml_agent7")
})

cat("Reading shape...\n")
shape_line <- readLines(file.path(binary_dir, "shape.txt"))
parts <- as.numeric(strsplit(shape_line, " ")[[1]])
nrow <- as.integer(parts[1])
ncol <- as.integer(parts[2])
nnz  <- parts[3]  # keep as numeric (could exceed int range)
cat(sprintf("  Matrix: %d x %d, %.0f nnz\n", nrow, ncol, nnz))

cat("Reading data.bin...\n")
t0 <- proc.time()
con <- file(file.path(binary_dir, "data.bin"), "rb")
x <- readBin(con, "double", n = nnz)
close(con)
cat(sprintf("  Read %.0f values (%.1f GB) in %.1fs\n",
            length(x), length(x) * 8 / 1e9, (proc.time() - t0)[3]))

cat("Reading indices.bin...\n")
t0 <- proc.time()
con <- file(file.path(binary_dir, "indices.bin"), "rb")
i <- readBin(con, "integer", n = nnz)
close(con)
cat(sprintf("  Read %.0f indices in %.1fs\n", length(i), (proc.time() - t0)[3]))

cat("Reading indptr.bin...\n")
t0 <- proc.time()
con <- file(file.path(binary_dir, "indptr.bin"), "rb")
p <- readBin(con, "integer", n = ncol + 1L)
close(con)
cat(sprintf("  Read %d pointers in %.1fs\n", length(p), (proc.time() - t0)[3]))

cat("Constructing dgCMatrix...\n")
t0 <- proc.time()
mat <- new("dgCMatrix", i = i, p = p, x = x, Dim = c(nrow, ncol))
cat(sprintf("  dgCMatrix constructed in %.1fs\n", (proc.time() - t0)[3]))

# Free raw arrays
rm(x, i, p)
gc(verbose = FALSE)

cat(sprintf("Writing SPZ: %s (include_transpose=TRUE)...\n", output_spz))
t0 <- proc.time()
stats <- sp_write(mat, output_spz,
                  include_transpose = TRUE,
                  verbose = TRUE)
elapsed <- (proc.time() - t0)[3]
cat(sprintf("  SPZ write completed in %.1fs\n", elapsed))
cat(sprintf("  Compressed: %s -> %s (%.1fx)\n",
            format(object.size(mat), units = "auto"),
            format(file.size(output_spz), units = "auto"),
            stats$ratio))

cat("Done.\n")
