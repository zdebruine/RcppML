#!/usr/bin/env Rscript
# bench_spz_io.R
# --------------------------------------------------------------------
# Benchmark: SPZ (compressed) vs raw binary disk I/O + decompress
#
# Questions answered:
#  1. How long does it take to read + decompress an SPZ file?
#  2. How does that break down: disk read vs CPU decompress?
#  3. How does it compare to reading a plain uncompressed binary (RDS)?
#  4. Does parallel decompression help (1 vs 2 vs 4 vs 8 threads)?
#  5. What is the compression ratio?
#  6. At what disk bandwidth does the crossover favor compressed vs uncompressed?
# --------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Matrix)
  library(RcppML)
})

# ---- helper: drop OS page cache if possible ----------------------------
# On Linux, writing and re-reading a dummy file to a different path
# helps evict previously cached pages.
# Strategy: read large amounts of /dev/urandom into /dev/null to evict NFS page cache.
# Then sync. We also burn through ~4x the RAM to displace cached file pages.
drop_page_cache <- function(file_bytes = 0) {
  # Try sudo drop_caches first
  if (file.exists("/proc/sys/vm/drop_caches")) {
    ret <- system("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null",
                  ignore.stdout = TRUE, ignore.stderr = TRUE)
    if (ret == 0) return(invisible(NULL))
  }
  # Fallback: read a large file to displace old cached pages.
  # Read at least 2x the file size worth of /dev/urandom to defeat cache.
  flush_bytes <- max(file_bytes * 4, 512 * 1024 * 1024)  # at least 512 MB
  flush_mb    <- ceiling(flush_bytes / (1024 * 1024))
  system(sprintf("dd if=/dev/urandom bs=1M count=%d | md5sum 2>/dev/null", flush_mb),
         ignore.stdout = TRUE, ignore.stderr = TRUE)
  invisible(NULL)
}

# ---- helper: precise wall-clock timer ----------------------------------
time_expr <- function(expr, reps = 3L, file_bytes = 0) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    drop_page_cache(file_bytes)
    gc(verbose = FALSE)
    t0 <- proc.time()[["elapsed"]]
    force(expr)
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  times
}

# ---- locate the geo_sketch SPZ file ------------------------------------
search_paths <- c(
  "/mnt/home/debruinz/R/x86_64-pc-linux-gnu-library/4.5/00LOCK-RcppML-2/RcppML/extdata/geo_sketch.spz",
  system.file("extdata", "geo_sketch.spz", package = "RcppML"),
  "/mnt/home/debruinz/R/tmplib/RcppML/extdata/geo_sketch.spz"
)
spz_path <- Filter(file.exists, search_paths)[1]
if (length(spz_path) == 0 || is.na(spz_path) || !nzchar(spz_path)) {
  # Fall back to pbmc3k which is always in package
  spz_path <- Filter(file.exists, c(
    system.file("extdata", "pbmc3k.spz", package = "RcppML"),
    "/mnt/home/debruinz/R/x86_64-pc-linux-gnu-library/4.5/00LOCK-RcppML-2/RcppML/extdata/pbmc3k.spz"
  ))[1]
  cat("NOTE: geo_sketch.spz not found, using pbmc3k.spz\n")
}
# Copy SPZ to /tmp so it's on local (faster) disk if available, and less cached
local_spz <- file.path("/tmp", basename(spz_path))
file.copy(spz_path, local_spz, overwrite = TRUE)
spz_path <- local_spz
cat("SPZ file:", spz_path, "\n")

spz_bytes <- file.size(spz_path)
cat(sprintf("SPZ file size: %.2f MB\n\n", spz_bytes / 1e6))

# ---- read SPZ once to learn the uncompressed size ----------------------
cat("Reading SPZ (reference decompress) ...\n")
mat <- st_read(spz_path)
stopifnot(inherits(mat, "dgCMatrix"))

cat(sprintf("Matrix dims : %d x %d\n", nrow(mat), ncol(mat)))
cat(sprintf("NNZ         : %d  (density %.3f%%)\n",
            nnzero(mat), 100 * nnzero(mat) / (nrow(mat) * ncol(mat))))

# Estimate uncompressed CSC size (in bytes):
#   p: (ncol+1) * 4 bytes (int32)
#   i: nnz * 4 bytes (int32)
#   x: nnz * 8 bytes (float64 for dgCMatrix)
nnz <- nnzero(mat)
raw_csc_bytes <- (ncol(mat) + 1L) * 4 + nnz * 4 + nnz * 8
cat(sprintf("Raw CSC size: %.2f MB  (theoretical)\n", raw_csc_bytes / 1e6))
cat(sprintf("Compression ratio: %.2fx\n\n", raw_csc_bytes / spz_bytes))

# ---- write temporary comparison files ---------------------------------
tmp_dir   <- "/tmp"
rds_path  <- file.path(tmp_dir, "mat_uncompressed.rds")
bin_path  <- file.path(tmp_dir, "mat_raw.bin")

# Uncompressed RDS (R native, no gzip)
cat("Writing uncompressed RDS ...\n")
saveRDS(mat, rds_path, compress = FALSE)
rds_bytes <- file.size(rds_path)
cat(sprintf("  RDS size: %.2f MB\n", rds_bytes / 1e6))

# Raw binary dump: write p, i, x arrays directly
# Format: [int32 nrow][int32 ncol][int64 nnz][p: (ncol+1)*int32][i: nnz*int32][x: nnz*float64]
write_raw_csc <- function(mat, path) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.integer(nrow(mat)), con, size = 4L)
  writeBin(as.integer(ncol(mat)), con, size = 4L)
  writeBin(as.double(nnzero(mat)), con, size = 8L)  # nnz as float64
  writeBin(mat@p,          con, size = 4L)
  writeBin(mat@i,          con, size = 4L)
  writeBin(mat@x,          con, size = 8L)
}

read_raw_csc <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  m   <- readBin(con, integer(), n = 1L, size = 4L)
  n   <- readBin(con, integer(), n = 1L, size = 4L)
  nnz <- as.integer(readBin(con, double(), n = 1L, size = 8L))
  p   <- readBin(con, integer(), n = n + 1L, size = 4L)
  i   <- readBin(con, integer(), n = nnz,    size = 4L)
  x   <- readBin(con, double(),  n = nnz,    size = 8L)
  new("dgCMatrix", Dim = c(m, n), p = p, i = i, x = x)
}

cat("Writing raw binary CSC ...\n")
write_raw_csc(mat, bin_path)
bin_bytes <- file.size(bin_path)
cat(sprintf("  Raw bin size: %.2f MB\n\n", bin_bytes / 1e6))

# ---- benchmark disk read only (no decompress) -------------------------
cat("=== PHASE 1: Pure disk read speed ===\n")

# Time reading raw bytes of SPZ (just disk, no decompress)
t_disk_spz <- time_expr({
  raw <- readBin(spz_path, what = "raw", n = spz_bytes)
  rm(raw)
}, reps = 3L, file_bytes = spz_bytes)

# Time reading raw bytes of uncompressed RDS
t_disk_rds <- time_expr({
  raw <- readBin(rds_path, what = "raw", n = rds_bytes)
  rm(raw)
}, reps = 3L, file_bytes = rds_bytes)

# Time reading raw bytes of raw CSC binary
t_disk_bin <- time_expr({
  raw <- readBin(bin_path, what = "raw", n = bin_bytes)
  rm(raw)
}, reps = 3L, file_bytes = bin_bytes)

cat(sprintf("Disk read SPZ (%.1f MB):     %.3f ± %.3f s  [%.0f MB/s]\n",
            spz_bytes / 1e6,
            mean(t_disk_spz), sd(t_disk_spz),
            spz_bytes / 1e6 / mean(t_disk_spz)))

cat(sprintf("Disk read RDS (%.1f MB):     %.3f ± %.3f s  [%.0f MB/s]\n",
            rds_bytes / 1e6,
            mean(t_disk_rds), sd(t_disk_rds),
            rds_bytes / 1e6 / mean(t_disk_rds)))

cat(sprintf("Disk read raw (%.1f MB):     %.3f ± %.3f s  [%.0f MB/s]\n",
            bin_bytes / 1e6,
            mean(t_disk_bin), sd(t_disk_bin),
            bin_bytes / 1e6 / mean(t_disk_bin)))

# ---- benchmark: full read + decompress (SPZ) --------------------------
cat("\n=== PHASE 2: Full read + decompress ===\n")

# SPZ: full read + decompress (single-threaded)
t_spz_1 <- time_expr({
  m <- st_read(spz_path, threads = 1L)
  rm(m)
}, reps = 3L, file_bytes = spz_bytes)
cat(sprintf("st_read  1 thread:  %.3f ± %.3f s\n", mean(t_spz_1), sd(t_spz_1)))

# SPZ: full read + decompress (2 threads)
t_spz_2 <- time_expr({
  m <- st_read(spz_path, threads = 2L)
  rm(m)
}, reps = 3L, file_bytes = spz_bytes)
cat(sprintf("st_read  2 threads: %.3f ± %.3f s\n", mean(t_spz_2), sd(t_spz_2)))

# SPZ: full read + decompress (4 threads)
t_spz_4 <- time_expr({
  m <- st_read(spz_path, threads = 4L)
  rm(m)
}, reps = 3L, file_bytes = spz_bytes)
cat(sprintf("st_read  4 threads: %.3f ± %.3f s\n", mean(t_spz_4), sd(t_spz_4)))

# SPZ: full read + decompress (8 threads)
t_spz_8 <- time_expr({
  m <- st_read(spz_path, threads = 8L)
  rm(m)
}, reps = 3L, file_bytes = spz_bytes)
cat(sprintf("st_read  8 threads: %.3f ± %.3f s\n\n", mean(t_spz_8), sd(t_spz_8)))

# RDS: full load (no decompress overhead)
t_rds <- time_expr({
  m <- readRDS(rds_path)
  rm(m)
}, reps = 3L, file_bytes = rds_bytes)
cat(sprintf("readRDS (uncompressed): %.3f ± %.3f s\n", mean(t_rds), sd(t_rds)))

# Raw binary: full load + reconstruct Matrix object
t_bin <- time_expr({
  m <- read_raw_csc(bin_path)
  rm(m)
}, reps = 3L, file_bytes = bin_bytes)
cat(sprintf("read raw CSC binary:    %.3f ± %.3f s\n\n", mean(t_bin), sd(t_bin)))

# ---- decompress-only time estimate ------------------------------------
cat("=== PHASE 3: Decompress-only estimate ===\n")
# Disk read time at observed bandwidth for SPZ-sized file
t_disk_only_spz  <- mean(t_disk_spz)
t_decomp_est_1   <- mean(t_spz_1) - t_disk_only_spz
t_decomp_est_4   <- mean(t_spz_4) - t_disk_only_spz
cat(sprintf("  Disk read SPZ:           %.3f s\n", t_disk_only_spz))
cat(sprintf("  Decompress est (1 thrd): %.3f s  (total: %.3f s)\n",
            max(0, t_decomp_est_1), mean(t_spz_1)))
cat(sprintf("  Decompress est (4 thrd): %.3f s  (total: %.3f s)\n",
            max(0, t_decomp_est_4), mean(t_spz_4)))

# ---- summary table ----------------------------------------------------
cat("\n=== SUMMARY TABLE ===\n")
cat(sprintf("%-35s %10s %10s %10s\n",
            "Method", "File MB", "Time (s)", "eff. MB/s"))
cat(strrep("-", 70), "\n")

fmt <- function(label, size_b, t_s) {
  cat(sprintf("%-35s %10.2f %10.3f %10.0f\n",
              label, size_b / 1e6, t_s, size_b / 1e6 / t_s))
}

fmt("SPZ disk-read only",       spz_bytes,  mean(t_disk_spz))
fmt("SPZ read+decomp (1 thrd)", spz_bytes,  mean(t_spz_1))
fmt("SPZ read+decomp (2 thrd)", spz_bytes,  mean(t_spz_2))
fmt("SPZ read+decomp (4 thrd)", spz_bytes,  mean(t_spz_4))
fmt("SPZ read+decomp (8 thrd)", spz_bytes,  mean(t_spz_8))
fmt("RDS uncompressed read",    rds_bytes,  mean(t_rds))
fmt("Raw CSC binary read",      bin_bytes,  mean(t_bin))

cat("\n--- Verdict ---\n")
best_spz <- min(mean(t_spz_1), mean(t_spz_2), mean(t_spz_4), mean(t_spz_8))
best_raw <- min(mean(t_rds), mean(t_bin))
if (best_spz < best_raw) {
  cat(sprintf("SPZ wins: %.3f s vs %.3f s (%.1fx faster)\n",
              best_spz, best_raw, best_raw / best_spz))
} else {
  cat(sprintf("Raw wins: %.3f s vs %.3f s (%.1fx faster)\n",
              best_raw, best_spz, best_spz / best_raw))
}

disk_bw_MBs <- spz_bytes / 1e6 / t_disk_only_spz
ratio        <- raw_csc_bytes / spz_bytes
# Crossover: compressed wins when decompress time < (ratio-1) * disk_read_time
# i.e. when disk_bw * (ratio-1) > decompress_throughput
# simplified: t_spz_total < t_raw_total
# → t_disk(SPZ) + t_decomp < t_disk(raw)
# → t_decomp < (raw_MB - spz_MB) / disk_bw
saved_MB   <- (raw_csc_bytes - spz_bytes) / 1e6
crossover_s <- saved_MB / disk_bw_MBs
cat(sprintf(
  "\nDisk bandwidth  : %.0f MB/s\n",
  disk_bw_MBs))
cat(sprintf(
  "Compression ratio: %.2fx  (saves %.1f MB per load)\n",
  ratio, saved_MB))
cat(sprintf(
  "Crossover (SPZ better if decompress < %.3f s): \n  1-thread decomp est = %.3f s -> %s\n  4-thread decomp est = %.3f s -> %s\n",
  crossover_s,
  max(0, t_decomp_est_1),
  if (max(0, t_decomp_est_1) < crossover_s) "SPZ WINS (1 thread)" else "raw wins vs 1-thread",
  max(0, t_decomp_est_4),
  if (max(0, t_decomp_est_4) < crossover_s) "SPZ WINS (4 threads)" else "raw wins vs 4-threads"
))

cat("\nDone.\n")
