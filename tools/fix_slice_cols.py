#!/usr/bin/env python3
"""Fix st_slice_cols to do post-hoc column subsetting in R."""
path = "/mnt/home/debruinz/RcppML-2/R/streampress.R"
with open(path, "r") as f:
    src = f.read()

# Fix st_slice_cols to subset after reading
old = """st_slice_cols <- function(path, cols, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read(path, cols = as.integer(cols), reorder = TRUE)
}"""
new = """st_slice_cols <- function(path, cols, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  cols <- as.integer(cols)
  A <- Rcpp_sp_read(path, reorder = TRUE)
  A[, cols, drop = FALSE]
}"""
if old in src:
    src = src.replace(old, new, 1)
    print("Fixed st_slice_cols to do post-hoc column subsetting")
else:
    print("WARNING: Could not find st_slice_cols pattern")

with open(path, "w") as f:
    f.write(src)
print("Done.")
