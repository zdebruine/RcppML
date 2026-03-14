#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(RcppML); library(Matrix) })

base  <- "/mnt/projects/debruinz_project/cellarium/harmonized/data/homo_sapiens"
gses  <- list.dirs(base, recursive = FALSE)
files <- character(0)
for (g in gses) {
  hits  <- list.files(g, "^counts\\.spz$", recursive = TRUE, full.names = TRUE)
  files <- c(files, hits)
  if (length(files) >= 20) break
}
files   <- sort(files)[1:20]
tot_mb  <- sum(file.size(files)) / 1e6
tot_mb_100 <- 825.9   # from 100-file benchmark
disk_bw    <- 1732    # MB/s measured
ratio      <- 9.5     # compression ratio (representative)

# -- time per thread count ------------------------------------------------
bench_n <- function(n) {
  t <- proc.time()[3]
  for (f in files) { m <- st_read(f, threads = n); rm(m) }
  as.numeric(proc.time()[3] - t)
}

cat("Measuring decompression speed (20 files, warm cache)...\n")
t1  <- bench_n(1L)
t4  <- bench_n(4L)
t8  <- bench_n(8L)
t16 <- bench_n(16L)

cat(sprintf("\n20 files, %.1f MB total compressed\n\n", tot_mb))
cat(sprintf("%-6s  %-8s  %-12s  %-8s\n", "Thrds", "Time(s)", "MB/s(comp)", "Speedup"))
cat(strrep("-", 42), "\n")
for (row in list(c(1,t1), c(4,t4), c(8,t8), c(16,t16))) {
  cat(sprintf("%-6d  %-8.3f  %-12.0f  %.1fx\n",
              as.integer(row[1]), row[2], tot_mb/row[2], t1/row[2]))
}

# -- chunk count for median file ------------------------------------------
f_rep  <- files[[10]]
m_rep  <- st_read(f_rep, threads = 1L)
nc     <- ncol(m_rep); rm(m_rep)
n_chunks_256  <- ceiling(nc / 256L)
n_chunks_32   <- ceiling(nc / 32L)
cat(sprintf("\nRep file: %d cols\n", nc))
cat(sprintf("  chunk_cols=256: %d chunks  -> max %.1fx speedup\n",
            n_chunks_256, as.numeric(n_chunks_256)))
cat(sprintf("  chunk_cols=32:  %d chunks  -> max %.1fx speedup\n",
            n_chunks_32,  as.numeric(n_chunks_32)))

# -- theoretical crossover analysis ---------------------------------------
cat("\n\n=== THEORETICAL BREAK-EVEN ANALYSIS (100 files, 825.9 MB) ===\n\n")

# actual 1-thread decomp rate:  tot_mb / t1 MB/s
decomp_rate_1t <- tot_mb / t1          # MB/s compressed, 1 thread
decomp_t100_1t <- tot_mb_100 / decomp_rate_1t  # seconds to decompress 100 files at 1 thread
disk_read_t100 <- tot_mb_100 / disk_bw
raw_read_t100  <- tot_mb_100 * ratio / disk_bw

cat(sprintf("Single-thread decompress rate:      %.1f MB/s (compressed)\n", decomp_rate_1t))
cat(sprintf("1-thread decompress time (100 files): %.2f s\n", decomp_t100_1t))
cat(sprintf("Disk read time, SPZ (100 files):    %.2f s\n", disk_read_t100))
cat(sprintf("Disk read time, raw (100 files):    %.2f s  (%.0f MB @ %d MB/s)\n",
            raw_read_t100, tot_mb_100 * ratio, disk_bw))
cat(sprintf("Compression ratio:                  %.1fx\n\n", ratio))

# For SPZ to win at N cores with perfect parallel scaling:
# t_spz  = disk_read_t100 + decomp_t100_1t / N
# t_raw  = raw_read_t100
# SPZ wins when t_spz < t_raw:
#   disk_read_t100 + decomp_t100_1t / N < raw_read_t100
#   decomp_t100_1t / N < raw_read_t100 - disk_read_t100
#   N > decomp_t100_1t / (raw_read_t100 - disk_read_t100)
savings_s <- raw_read_t100 - disk_read_t100
N_breakeven <- decomp_t100_1t / savings_s
cat(sprintf("IO savings SPZ vs raw:              %.2f s\n", savings_s))
cat(sprintf("Minimum cores needed (perfect):     %.1f cores\n", N_breakeven))
cat(sprintf("  (i.e. need >%.0f cores with PERFECT scaling to break even)\n\n", ceiling(N_breakeven)))

# table: total time vs cores, perfect efficiency
cat(sprintf("%-8s  %-12s  %-12s  %-8s\n", "Cores", "SPZ total", "Raw total", "Winner"))
cat(strrep("-", 46), "\n")
for (N in c(1, 2, 4, 8, 16, 32, 64, 128, 256)) {
  t_spz_n <- disk_read_t100 + decomp_t100_1t / N
  winner  <- if (t_spz_n < raw_read_t100) "SPZ" else "raw"
  marker  <- if (abs(N - N_breakeven) < 3) " <-- breakeven" else ""
  cat(sprintf("%-8d  %-12.2f  %-12.2f  %s%s\n",
              N, t_spz_n, raw_read_t100, winner, marker))
}

# Now check: at the chunk-count ceiling (not infinite cores)
cat(sprintf("\n--- Constrained by chunk count (files processed serially) ---\n"))
cat(sprintf("Max speedup per file at chunk_cols=256: %dx  (cap from %d chunks)\n",
            n_chunks_256, n_chunks_256))
cat(sprintf("Max speedup per file at chunk_cols=32:  %dx  (cap from %d chunks)\n",
            n_chunks_32, n_chunks_32))
for (label in c("chunk_cols=256", "chunk_cols=32")) {
  N_eff <- if (label == "chunk_cols=256") n_chunks_256 else n_chunks_32
  t_spz_eff <- disk_read_t100 + decomp_t100_1t / N_eff
  cat(sprintf("  %s: max SPZ = %.2f s  vs raw = %.2f s  -> %s\n",
              label, t_spz_eff, raw_read_t100,
              if (t_spz_eff < raw_read_t100) "SPZ wins" else "raw wins"))
}

# -- Streaming NMF use case (single large concatenated file) ---------------
cat("\n=== STREAMING NMF USE CASE (large single file, cold NFS) ===\n")
cat("(This is the intended use case for StreamPress)\n\n")
# Assume: 1000 samples concatenated into one ~8 GB SPZ file
# Cold NFS: ~500 MB/s (typical for uncached NFS on spinning-disk or network-limited)
large_spz_mb   <- 8000
cold_bw        <- 500    # MB/s cold NFS
large_raw_mb   <- large_spz_mb * ratio
n_chunks_large <- ceiling(100000L / 256L)  # ~100k columns -> 391 chunks
cat(sprintf("Scenario: %d MB compressed SPZ  (~%.0f MB raw CSC)\n",
            large_spz_mb, large_raw_mb))
cat(sprintf("Cold NFS bandwidth: %d MB/s\n", cold_bw))
cat(sprintf("Chunks (100k cols @ 256): %d -> max parallel speedup: %dx\n\n",
            n_chunks_large, n_chunks_large))
for (N in c(1, 4, 8, 16, 32)) {
  t_spz_cold <- large_spz_mb / cold_bw + large_spz_mb / decomp_rate_1t / N
  t_raw_cold <- large_raw_mb / cold_bw
  winner     <- if (t_spz_cold < t_raw_cold) "SPZ" else "raw"
  cat(sprintf("  %2d cores: SPZ = %5.1f s  raw = %5.1f s  -> %s\n",
              N, t_spz_cold, t_raw_cold, winner))
}

cat("\nDone.\n")
