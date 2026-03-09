#!/usr/bin/env Rscript
# StreamPress performance benchmark
# Usage: Rscript bench_streampress.R [path.spz]
# If no path given, generates synthetic data.

library(RcppML)
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && file.exists(args[1])) {
  message("Loading real file: ", args[1])
  A <- st_read(args[1])
} else {
  message("No valid file given -- using synthetic 50k x 10k, density=1%")
  set.seed(42)
  A <- rsparsematrix(50000, 10000, density = 0.01)
}
message(sprintf("Matrix: %d x %d, nnz=%d (%.2f%%)",
        nrow(A), ncol(A), Matrix::nnzero(A),
        100 * Matrix::nnzero(A) / (as.double(nrow(A)) * ncol(A))))

f <- tempfile(fileext = ".spz")
on.exit(unlink(f))

# --- Write benchmarks -------------------------------------------------------
message("\n=== Write Benchmarks ===")
for (thr in c(1L, 4L)) {
  times <- numeric(3)
  for (r in 1:3) {
    t0 <- proc.time()["elapsed"]
    st_write(A, f, threads = thr, include_transpose = TRUE, verbose = FALSE)
    times[r] <- proc.time()["elapsed"] - t0
  }
  message(sprintf("  write (threads=%d): %.2f / %.2f / %.2f s  (median %.2f s)",
                  thr, times[1], times[2], times[3], median(times)))
}

# Ensure file written for read benchmarks
st_write(A, f, threads = 0L, include_transpose = TRUE, verbose = FALSE)
fsize <- file.info(f)$size
message(sprintf("  File size: %.1f MB (ratio: %.1fx)",
                fsize / 1e6,
                (as.double(nrow(A)) * ncol(A) * 4) / fsize))

# --- Read benchmarks ---------------------------------------------------------
message("\n=== Read Benchmarks ===")
for (thr in c(1L, 4L)) {
  times <- numeric(3)
  for (r in 1:3) {
    t0 <- proc.time()["elapsed"]
    B <- st_read(f, threads = thr)
    times[r] <- proc.time()["elapsed"] - t0
    rm(B)
  }
  message(sprintf("  read (threads=%d): %.2f / %.2f / %.2f s  (median %.2f s)",
                  thr, times[1], times[2], times[3], median(times)))
}

# --- Slice benchmarks --------------------------------------------------------
message("\n=== Slice Benchmarks ===")
n_cols <- ncol(A)
n_rows <- nrow(A)
col_range <- 1:min(1000, n_cols)
row_range <- 1:min(1000, n_rows)

for (thr in c(1L, 4L)) {
  t0 <- proc.time()["elapsed"]
  sub <- st_slice_cols(f, col_range, threads = thr)
  t_cols <- proc.time()["elapsed"] - t0
  message(sprintf("  slice_cols(%d cols, threads=%d): %.3f s",
                  length(col_range), thr, t_cols))
}

for (thr in c(1L, 4L)) {
  t0 <- proc.time()["elapsed"]
  sub <- st_slice_rows(f, row_range, threads = thr)
  t_rows <- proc.time()["elapsed"] - t0
  message(sprintf("  slice_rows(%d rows, threads=%d): %.3f s",
                  length(row_range), thr, t_rows))
}

# --- Chunk iteration ---------------------------------------------------------
message("\n=== Chunk Iteration ===")
t0 <- proc.time()["elapsed"]
st_map_chunks(f, function(chunk, cs, ce) NULL, threads = 4L)
t_map <- proc.time()["elapsed"] - t0
cr <- st_chunk_ranges(f)
message(sprintf("  map_chunks (%d chunks, threads=4): %.2f s", nrow(cr), t_map))

# --- Correctness check -------------------------------------------------------
message("\n=== Correctness Check ===")
A2 <- st_read(f)
ok <- all.equal(as.matrix(A), as.matrix(A2), tolerance = 1e-6)
if (isTRUE(ok)) {
  message("  Round-trip correctness: PASS")
} else {
  message("  Round-trip correctness: ", ok)
}

message("\nDone.")
