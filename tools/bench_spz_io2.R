#!/usr/bin/env Rscript
# bench_spz_io2.R — Focused SPZ vs raw-binary I/O benchmark
# ----------------------------------------------------------------
# Strategy:
#   Phase A: Measure true NFS read bandwidth with a fresh large file
#   Phase B: Warm-cache decompress timing (many reps, stable)
#   Phase C: Theoretical crossover analysis
# ----------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(RcppML)
})

NFS_BASE <- "/mnt/home/debruinz/RcppML-2/.bench_tmp"
dir.create(NFS_BASE, showWarnings = FALSE)

cat("==========================================================\n")
cat("SPZ Compression: Value Proposition Benchmark\n")
cat("==========================================================\n\n")

# ---- locate best SPZ file ---------------------------------------------
spz_candidates <- c(
  "/mnt/home/debruinz/R/x86_64-pc-linux-gnu-library/4.5/00LOCK-RcppML-2/RcppML/extdata/geo_sketch.spz",
  system.file("extdata", "geo_sketch.spz", package = "RcppML"),
  system.file("extdata", "pbmc3k.spz",     package = "RcppML")
)
spz_path <- Filter(file.exists, spz_candidates)[1]
cat("Using SPZ:", spz_path, "\n")
spz_bytes <- file.size(spz_path)
cat(sprintf("SPZ size: %.2f MB\n\n", spz_bytes / 1e6))

# ---- load matrix once to know dimensions --------------------------------
cat("Loading matrix from SPZ...\n")
mat <- st_read(spz_path)
nnz      <- nnzero(mat)
nr       <- nrow(mat); nc <- ncol(mat)
raw_bytes  <- (nc + 1L) * 4 + nnz * 4 + nnz * 8   # CSC: p(int32) + i(int32) + x(float64)
cat(sprintf("Dims: %d × %d,  NNZ: %d (%.2f%%),  density: %.4f\n",
            nr, nc, nnz,
            100 * nnz / (nr * nc),
            nnz / (nr * nc)))
cat(sprintf("Compression: %.2f MB → %.2f MB (%.1fx ratio)\n\n",
            raw_bytes / 1e6, spz_bytes / 1e6, raw_bytes / spz_bytes))

# ---- write comparison files to NFS (never cached before) ----------------
# Raw CSC binary (minimal format: 4-byte header ints then raw arrays)
write_raw_csc <- function(mat, path) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(c(nrow(mat), ncol(mat), nnzero(mat)), con, size = 4L)
  writeBin(mat@p, con, size = 4L)
  writeBin(mat@i, con, size = 4L)
  writeBin(mat@x, con, size = 8L)
}
read_raw_csc <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  hdr <- readBin(con, integer(), n = 3L, size = 4L)
  m <- hdr[1]; n2 <- hdr[2]; nnz2 <- hdr[3]
  p <- readBin(con, integer(), n = n2 + 1L, size = 4L)
  i <- readBin(con, integer(), n = nnz2,    size = 4L)
  x <- readBin(con, double(),  n = nnz2,    size = 8L)
  new("dgCMatrix", Dim = c(m, n2), p = p, i = i, x = x)
}

nfs_rds_path <- file.path(NFS_BASE, "mat_unc.rds")
nfs_bin_path <- file.path(NFS_BASE, "mat_csc.bin")
nfs_spz_path <- file.path(NFS_BASE, "mat.spz")       # SPZ copy on NFS too

cat("Writing test files to NFS...\n")
saveRDS(mat, nfs_rds_path, compress = FALSE)
write_raw_csc(mat, nfs_bin_path)
file.copy(spz_path, nfs_spz_path, overwrite = TRUE)

rds_bytes <- file.size(nfs_rds_path)
bin_bytes <- file.size(nfs_bin_path)
cat(sprintf("  NFS raw CSC:  %.2f MB\n", bin_bytes / 1e6))
cat(sprintf("  NFS RDS:      %.2f MB\n", rds_bytes / 1e6))
cat(sprintf("  NFS SPZ copy: %.2f MB\n\n", file.size(nfs_spz_path) / 1e6))

# ================================================================
# PHASE A: NFS bandwidth — cold-read a freshly written probe file
# ================================================================
cat("=== PHASE A: True NFS Read Bandwidth ===\n")

# Write probe file AFTER timing setup to ensure it's not cached  
probe_mb   <- 128L      # 128 MB probe is large enough to stabilize NFS throughput
probe_path <- file.path(NFS_BASE, "probe_bw.bin")
cat(sprintf("Writing %d MB probe file...\n", probe_mb))
probe_data <- raw(probe_mb * 1024L * 1024L)  # zero-filled raw bytes
t_write_start <- proc.time()[["elapsed"]]
writeBin(probe_data, probe_path)
t_write <- proc.time()[["elapsed"]] - t_write_start
rm(probe_data)
cat(sprintf("  Write time: %.2f s (%.0f MB/s)\n", t_write, probe_mb / t_write))

# Stat the file to force NFS attribute sync
file.size(probe_path)
Sys.sleep(0.1)

# Read back (first read = cold NFS, not in local RAM / tmpfs)
gc(verbose = FALSE)
t_read_start <- proc.time()[["elapsed"]]
probe_in <- readBin(probe_path, raw(), n = probe_mb * 1024L * 1024L)
t_read <- proc.time()[["elapsed"]] - t_read_start
rm(probe_in)
bw_MBs <- probe_mb / t_read
cat(sprintf("  Cold NFS read: %.2f s → bandwidth %.0f MB/s\n\n", t_read, bw_MBs))

# Second read (warm / cached by NFS or VFS)
gc(verbose = FALSE)
t_warm_start <- proc.time()[["elapsed"]]
probe_in2 <- readBin(probe_path, raw(), n = probe_mb * 1024L * 1024L)
t_warm <- proc.time()[["elapsed"]] - t_warm_start
rm(probe_in2)
cat(sprintf("  Warm cache read: %.3f s → %.0f MB/s\n\n", t_warm, probe_mb / t_warm))

# ================================================================
# PHASE B: Warm-cache decompress / load timing (10 reps, stable)
# ================================================================
cat("=== PHASE B: Warm-Cache Load Timing (10 reps) ===\n")
cat("(Files read from NFS; NFS page cache may be warm — measures CPU overhead)\n\n")

bench <- function(expr, reps = 10L) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    gc(verbose = FALSE)
    t0 <- proc.time()[["elapsed"]]
    eval(expr, envir = parent.frame())
    times[[i]] <- proc.time()[["elapsed"]] - t0
  }
  times
}

# --- decompress: vary threads ---
cat("SPZ decompress (NFS path):\n")
for (thr in c(1L, 2L, 4L, 8L)) {
  t <- bench(quote({ m <- st_read(nfs_spz_path, threads = thr); rm(m) }))
  cat(sprintf("  %d thread(s): median=%.3f s  min=%.3f  max=%.3f\n",
              thr, median(t), min(t), max(t)))
}

# --- decompress: same-node SPZ from installed package ---
cat("SPZ decompress (local tmpfs copy at /tmp):\n")
spz_tmp <- "/tmp/mat_bench.spz"
file.copy(spz_path, spz_tmp, overwrite = TRUE)
for (thr in c(1L, 4L)) {
  t <- bench(quote({ m <- st_read(spz_tmp, threads = thr); rm(m) }))
  cat(sprintf("  %d thread(s): median=%.3f s  min=%.3f  max=%.3f\n",
              thr, median(t), min(t), max(t)))
}

# --- RDS uncompressed ---
cat("\nreadRDS uncompressed (NFS path):\n")
t_rds <- bench(quote({ m <- readRDS(nfs_rds_path); rm(m) }))
cat(sprintf("  median=%.3f s  min=%.3f  max=%.3f\n",
            median(t_rds), min(t_rds), max(t_rds)))

# --- raw CSC binary ---
cat("readBin raw CSC (NFS path):\n")
t_bin <- bench(quote({ m <- read_raw_csc(nfs_bin_path); rm(m) }))
cat(sprintf("  median=%.3f s  min=%.3f  max=%.3f\n\n",
            median(t_bin), min(t_bin), max(t_bin)))

# --- Decompress-only (measure from tmpfs to exclude I/O) ---
cat("Decompress-only (tmpfs SPZ, I/O ~= 0 for small file):\n")
t_decomp_1 <- bench(quote({ m <- st_read(spz_tmp, threads = 1L); rm(m) }))
t_decomp_4 <- bench(quote({ m <- st_read(spz_tmp, threads = 4L); rm(m) }))
t_decomp_8 <- bench(quote({ m <- st_read(spz_tmp, threads = 8L); rm(m) }))
cat(sprintf("  1 thread:    %.3f s median (decompress dominates)\n", median(t_decomp_1)))
cat(sprintf("  4 threads:   %.3f s median\n", median(t_decomp_4)))
cat(sprintf("  8 threads:   %.3f s median\n\n", median(t_decomp_8)))

# ================================================================
# PHASE C: Crossover analysis
# ================================================================
cat("=== PHASE C: Crossover Analysis ===\n\n")

t_decomp_cpu <- median(t_decomp_1)   # single-thread CPU decompress, no I/O
compress_ratio <- raw_bytes / spz_bytes

# At disk bandwidth B (MB/s):
#   time(SPZ)  = spz_bytes/(B*1e6) + t_decomp_cpu
#   time(raw)  = raw_bytes/(B*1e6)
# SPZ wins when time(SPZ) < time(raw):
#   → t_decomp_cpu < (raw_bytes - spz_bytes)/(B*1e6)
#   → B < (raw_bytes - spz_bytes)/(t_decomp_cpu * 1e6)
crossover_MBs <- (raw_bytes - spz_bytes) / (t_decomp_cpu * 1e6)

cat(sprintf("  Compression ratio:       %.1fx\n", compress_ratio))
cat(sprintf("  Savings per load:        %.1f MB\n", (raw_bytes - spz_bytes) / 1e6))
cat(sprintf("  CPU decompress time:     %.3f s (1 thread, warm cache)\n", t_decomp_cpu))
cat(sprintf("  Crossover disk BW:       %.0f MB/s\n", crossover_MBs))
cat(sprintf("  Observed cold NFS BW:    %.0f MB/s\n\n", bw_MBs))

if (bw_MBs < crossover_MBs) {
  cat(sprintf(
    ">>> SPZ wins at this NFS bandwidth (%.0f MB/s < crossover %.0f MB/s)\n",
    bw_MBs, crossover_MBs))
  spz_total  <- spz_bytes / 1e6 / bw_MBs + t_decomp_cpu
  raw_total  <- raw_bytes / 1e6 / bw_MBs
  cat(sprintf("    SPZ: %.3f s  vs  raw: %.3f s  → %.1fx speedup\n",
              spz_total, raw_total, raw_total / spz_total))
} else {
  cat(sprintf(
    ">>> Raw wins at this NFS bandwidth (%.0f MB/s > crossover %.0f MB/s)\n",
    bw_MBs, crossover_MBs))
  spz_total  <- spz_bytes / 1e6 / bw_MBs + t_decomp_cpu
  raw_total  <- raw_bytes / 1e6 / bw_MBs
  cat(sprintf("    SPZ: %.3f s  vs  raw: %.3f s  → %.1fx speedup for raw\n",
              spz_total, raw_total, spz_total / raw_total))
}

# Show crossover table at multiple bandwidths
cat(sprintf("\n  Break-even table (matrix: %dx%d, %dM NNZ, compression %.1fx):\n",
            nr, nc, round(nnz / 1e6), compress_ratio))
cat(sprintf("  %-15s  %-12s  %-12s  %-10s\n",
            "Disk BW (MB/s)", "SPZ total(s)", "Raw total(s)", "Winner"))
cat("  ", strrep("-", 55), "\n", sep = "")
for (bw in c(50, 100, 200, 500, 1000, 2000, 5000, 10000)) {
  t_spz_at <- spz_bytes / 1e6 / bw + t_decomp_cpu
  t_raw_at <- raw_bytes / 1e6 / bw
  winner   <- if (t_spz_at < t_raw_at) "SPZ" else "raw"
  marker   <- if (abs(bw - bw_MBs) < 50) " <-- measured" else ""
  cat(sprintf("  %-15d  %-12.3f  %-12.3f  %s%s\n",
              bw, t_spz_at, t_raw_at, winner, marker))
}

cat("\n=== PARALLEL DECOMPRESS VERDICT ===\n")
cat(sprintf("1 thread = %.3f s,  4 threads = %.3f s,  8 threads = %.3f s\n",
            median(t_decomp_1), median(t_decomp_4), median(t_decomp_8)))
speedup_4 <- median(t_decomp_1) / median(t_decomp_4)
speedup_8 <- median(t_decomp_1) / median(t_decomp_8)
cat(sprintf("Parallel speedup: 4T=%.1fx, 8T=%.1fx\n", speedup_4, speedup_8))
num_chunks <- ceiling(nc / 256)   # default 256 cols/chunk
cat(sprintf("Number of chunks (256 cols each): %d → limits parallelism\n\n", num_chunks))

cat("NOTE: pread() in FileReader is thread-safe (POSIX pread, no mutex).\n")
cat("      Parallel IO at the OS level requires multiple concurrent pread() calls\n")
cat("      from different threads, which SPZ IS designed to support.\n")
cat("      However, for < 2 GB files, the entire file is read in ONE pread() call\n")
cat("      into RAM first (SpzLoader::in_core_ path), then decoded in parallel.\n")
cat("      So IO is serial but decode is parallel.\n\n")

# cleanup
unlink(c(probe_path, nfs_rds_path, nfs_bin_path, nfs_spz_path, spz_tmp))
unlink(NFS_BASE, recursive = TRUE)
cat("Done.\n")
