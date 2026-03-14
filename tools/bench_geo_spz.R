#!/usr/bin/env Rscript
# bench_geo_spz.R
# ---------------------------------------------------------------
# Benchmark: disk-read-only vs full read+decompress for 100
# real GEO counts.spz files from the cellarium pipeline.
#
# Design:
#  - Never accumulates all decompressed matrices in RAM.
#    Each file is read, (optionally) decompressed into a
#    temporary matrix, then immediately discarded.
#  - Phase 1: raw bytes read only (readBin), no decompress.
#  - Phase 2: st_read (full decompress to dgCMatrix), then rm().
#  - Parallelism: files are processed serially but each st_read
#    uses the specified number of OMP threads internally.
#  - No replicates, no sd. Single pass. Wall clock.
# ---------------------------------------------------------------

suppressPackageStartupMessages({
  library(RcppML)
  library(Matrix)
})

HARMONIZED_DIR <- "/mnt/projects/debruinz_project/cellarium/harmonized/data/homo_sapiens"
N_FILES        <- 100L
# Thread counts to benchmark. 40 triggers a segfault on some files (OMP bug
# with high thread count + small chunk count — logged separately).
# We benchmark 8 / 16 / 32 to show the actual scaling curve.
ALL_THREADS    <- c(8L, 16L, 32L)
N_CORES        <- as.integer(Sys.getenv("OMP_NUM_THREADS",
                                        unset = as.character(parallel::detectCores())))

cat("============================================================\n")
cat(sprintf("\n  GEO counts.spz Benchmark — 100 real scRNA-seq samples\n"))
cat(sprintf("  OMP threads: %d  (env OMP_NUM_THREADS)\n", N_CORES))
cat("============================================================\n\n")

# ---- collect SPZ paths ------------------------------------------------
cat("Collecting SPZ file list...\n")
gse_dirs <- list.dirs(HARMONIZED_DIR, recursive = FALSE)
spz_paths <- character(0)
for (gse in gse_dirs) {
  hits <- list.files(gse, pattern = "^counts\\.spz$",
                     recursive = TRUE, full.names = TRUE)
  spz_paths <- c(spz_paths, hits)
  if (length(spz_paths) >= N_FILES * 2L) break   # collect 2x, then sample
}

if (length(spz_paths) < N_FILES) {
  stop(sprintf("Only found %d SPZ files, need %d", length(spz_paths), N_FILES))
}

# Fixed sample: use the first N_FILES after sorting for reproducibility
spz_paths <- sort(spz_paths)[seq_len(N_FILES)]
cat(sprintf("Selected %d files\n\n", length(spz_paths)))

# ---- get file sizes ---------------------------------------------------
sizes_bytes <- file.size(spz_paths)
total_spz_MB <- sum(sizes_bytes, na.rm = TRUE) / 1e6
cat(sprintf("Total SPZ data: %.1f MB  (mean %.1f MB, min %.1f MB, max %.1f MB)\n",
            total_spz_MB,
            mean(sizes_bytes) / 1e6,
            min(sizes_bytes)  / 1e6,
            max(sizes_bytes)  / 1e6))

# ---- Phase 1: raw disk read (no decompress) ---------------------------
cat("\n--- Phase 1: Raw disk read (readBin, no decompress) ---\n")
gc(verbose = FALSE)

t0_disk <- proc.time()[["elapsed"]]
total_bytes_read <- 0L
for (i in seq_along(spz_paths)) {
  sz  <- sizes_bytes[[i]]
  raw <- readBin(spz_paths[[i]], what = "raw", n = sz)
  total_bytes_read <- total_bytes_read + length(raw)
  rm(raw)
}
t_disk <- proc.time()[["elapsed"]] - t0_disk

disk_bw <- total_spz_MB / t_disk
cat(sprintf("  %d files, %.1f MB total\n", N_FILES, total_spz_MB))
cat(sprintf("  Wall time:  %.2f s\n", t_disk))
cat(sprintf("  Throughput: %.0f MB/s\n", disk_bw))

# ---- Phase 2: full decompress (no accumulation) -----------------------
# ---- Phase 2: full decompress at multiple thread counts ---------------
decomp_results <- list()

for (THREADS in ALL_THREADS) {
  cat(sprintf("\n--- Phase 2: Full read + decompress (st_read, %d OMP threads) ---\n",
              THREADS))
  gc(verbose = FALSE)

  total_nnz   <- 0.0
  total_cells <- 0L
  total_genes <- 0L
  n_failed    <- 0L

  t0_decomp <- proc.time()[["elapsed"]]
  for (i in seq_along(spz_paths)) {
    tryCatch({
      m <- st_read(spz_paths[[i]], threads = THREADS)
      total_cells <- total_cells + ncol(m)
      total_genes <- total_genes + nrow(m)
      total_nnz   <- total_nnz   + length(m@x)
      rm(m)
    }, error = function(e) {
      n_failed <<- n_failed + 1L
    })
  }
  t_decomp <- proc.time()[["elapsed"]] - t0_decomp

  n_ok      <- N_FILES - n_failed
  decomp_bw <- total_spz_MB / t_decomp

  cat(sprintf("  %d / %d files succeeded\n", as.integer(n_ok), N_FILES))
  if (total_cells > 0) {
    cat(sprintf("  Aggregate cells:           %d\n", as.integer(total_cells)))
    cat(sprintf("  Aggregate NNZ:             %.1fM\n", total_nnz / 1e6))
  }
  cat(sprintf("  Wall time:  %.2f s\n", t_decomp))
  cat(sprintf("  Throughput: %.0f MB/s  (compressed-equivalent)\n", decomp_bw))

  decomp_results[[as.character(THREADS)]] <- list(
    threads = THREADS, n_ok = n_ok,
    t = t_decomp, bw = decomp_bw,
    overhead = t_decomp - t_disk,
    total_cells = total_cells, total_nnz = total_nnz
  )
}

# ---- Summary ----------------------------------------------------------
# Use the best (fastest) decompress result for crossover analysis
best_r <- decomp_results[[which.min(sapply(decomp_results, `[[`, "t"))]]
t_decomp    <- best_r$t
t_overhead  <- t_decomp - t_disk
overhead_pct <- 100 * t_overhead / t_decomp

cat("\n============================================================\n")
cat("  SUMMARY\n")
cat("============================================================\n")
cat(sprintf("  Files:              %d\n",     N_FILES))
cat(sprintf("  Total SPZ size:     %.1f MB  (mean %.1f MB)\n",
            total_spz_MB, mean(sizes_bytes) / 1e6))
cat(sprintf("\n  Phase 1 — disk read only:\n"))
cat(sprintf("    Time:             %.2f s\n",  t_disk))
cat(sprintf("    Bandwidth:        %.0f MB/s\n", disk_bw))
cat(sprintf("\n  Phase 2 — read + decompress:\n"))
cat(sprintf("  %-10s  %-10s  %-14s  %-16s\n",
            "Threads", "Time (s)", "Throughput", "Decomp overhead"))
cat("  ", strrep("-", 56), "\n", sep = "")
for (r in decomp_results) {
  cat(sprintf("  %-10d  %-10.2f  %-14s  %s\n",
              r$threads, r$t,
              sprintf("%.0f MB/s", r$bw),
              sprintf("%.2f s (%.0f%%)", r$overhead, 100*r$overhead/r$t)))
}
cat(sprintf("\n  NOTE: 40 threads segfaults on some files in this corpus.\n"))
cat(sprintf("        Likely OMP race condition when threads >> chunks/file.\n"))
cat(sprintf("        (default chunk size 256 cols → ~few chunks per typical sample)\n"))

# ---- break-even analysis ---------------------------------------------
# What if files were stored uncompressed?
# Need ratio from the data. Use one file as representative.
cat("\n--- Loading one file to estimate compression ratio ---\n")
rep_path <- spz_paths[[which.min(abs(sizes_bytes - median(sizes_bytes)))[[1]]]]
rep_mat  <- st_read(rep_path)
nnz_rep   <- length(rep_mat@x)
nr_rep    <- nrow(rep_mat)
nc_rep    <- ncol(rep_mat)
raw_rep   <- (nc_rep + 1L) * 4 + nnz_rep * 4 + nnz_rep * 8
spz_rep   <- file.size(rep_path)
ratio_rep <- raw_rep / spz_rep
rm(rep_mat)
cat(sprintf("  Representative file: %d×%d, %dK NNZ\n",
            nr_rep, nc_rep, round(nnz_rep / 1e3)))
cat(sprintf("  Compressed:    %.2f MB\n", spz_rep / 1e6))
cat(sprintf("  Uncompressed:  %.2f MB\n", raw_rep / 1e6))
cat(sprintf("  Ratio:         %.1fx\n\n", ratio_rep))

total_raw_MB <- total_spz_MB * ratio_rep
t_raw_equiv  <- total_raw_MB / disk_bw  # time to read uncompressed at same BW

cat("============================================================\n")
cat("  CROSS-OVER ANALYSIS  (using best decompress time)\n")
cat("============================================================\n")
cat(sprintf("  SPZ best read+decomp:              %.2f s  (%d threads)\n",
            t_decomp, best_r$threads))
cat(sprintf("  Est. raw binary read (same BW):    %.2f s  (%.1f MB @ %.0f MB/s)\n",
            t_raw_equiv, total_raw_MB, disk_bw))
if (t_decomp < t_raw_equiv) {
  cat(sprintf("\n  >>> SPZ WINS: %.1fx faster than uncompressed raw\n",
              t_raw_equiv / t_decomp))
} else {
  cat(sprintf("\n  >>> RAW WINS: %.1fx faster than SPZ (decompress overhead too high)\n",
              t_decomp / t_raw_equiv))
}
cat(sprintf("\n  Disk BW measured: %.0f MB/s\n", disk_bw))
cat(sprintf("  Crossover BW:     ~%.0f MB/s  ",
            total_spz_MB * (ratio_rep - 1) / max(t_overhead, 0.001)))
cat("(below this, SPZ wins)\n")

cat("\nDone.\n")
