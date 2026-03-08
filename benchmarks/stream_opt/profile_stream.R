#!/usr/bin/env Rscript
# Phase 2C: Streaming baseline benchmarks
# Compares in-memory vs streaming (SPZ) NMF at various sizes.
# Run on compute node (not login): Rscript benchmarks/stream_opt/profile_stream.R

cat("=== Phase 2C: Streaming Baseline Profile ===\n\n")

.libPaths(c("/tmp/rcppml_agent7", .libPaths()))
library(RcppML)
library(Matrix)

setwd("/mnt/home/debruinz/RcppML-2")
source("benchmarks/harness/bench_runner.R")
source("benchmarks/harness/generators.R")

# Configuration
RANKS   <- c(8L, 16L, 32L)
MAXIT   <- 20L
TOL     <- 1e-10  # Force all iterations
NREPS   <- 3L
SEED    <- 42L

# Sizes: (m, n, density)
CONFIGS <- list(
  small  = list(m = 5000L,  n = 2000L,  density = 0.05),
  medium = list(m = 20000L, n = 10000L, density = 0.02),
  large  = list(m = 50000L, n = 20000L, density = 0.01)
)

results <- list()

for (sz_name in names(CONFIGS)) {
  cfg <- CONFIGS[[sz_name]]
  m <- cfg$m; n <- cfg$n; dens <- cfg$density

  cat(sprintf("--- Size: %s (%d x %d, density=%.2f) ---\n", sz_name, m, n, dens))

  # Generate sparse matrix
  A <- gen_sparse(m, n, density = dens, seed = SEED)
  cat(sprintf("  Generated: %d x %d, nnz = %d\n", nrow(A), ncol(A), nnzero(A)))

  # Write SPZ file
  spz_path <- file.path(tempdir(), sprintf("bench_%s.spz", sz_name))
  sp_write(A, spz_path, include_transpose = TRUE)
  cat(sprintf("  SPZ written: %s (%.1f MB)\n", spz_path,
              file.info(spz_path)$size / 1024^2))

  for (k in RANKS) {
    cat(sprintf("  k = %d:\n", k))

    # --- In-memory NMF ---
    res_mem <- run_bench(
      label = sprintf("inmem_%s_k%d", sz_name, k),
      expr = quote(nmf(A, k = k, tol = TOL, maxit = MAXIT, seed = SEED,
                        verbose = FALSE)),
      n_reps = NREPS,
      extra = list(k = k, m = m, n = n, nnz = nnzero(A),
                   backend = "cpu_inmem", loss = "mse")
    )
    cat(sprintf("    in-memory:  median %.0f ms\n", median(res_mem$time_ms)))
    results <- c(results, list(res_mem))

    # --- Streaming SPZ NMF ---
    # Use option to force streaming path
    res_stream <- run_bench(
      label = sprintf("stream_%s_k%d", sz_name, k),
      expr = quote({
        old_opt <- getOption("RcppML.streaming", FALSE)
        options(RcppML.streaming = TRUE)
        on.exit(options(RcppML.streaming = old_opt))
        nmf(spz_path, k = k, tol = TOL, maxit = MAXIT, seed = SEED,
            verbose = FALSE)
      }),
      n_reps = NREPS,
      extra = list(k = k, m = m, n = n, nnz = nnzero(A),
                   backend = "cpu_stream", loss = "mse")
    )
    cat(sprintf("    streaming:  median %.0f ms\n", median(res_stream$time_ms)))
    results <- c(results, list(res_stream))
  }

  # Clean up to free RAM before next size
  rm(A)
  gc(verbose = FALSE)
}

# Combine and report
all_results <- do.call(rbind, results)
report_bench(all_results, save_csv = TRUE, tag = "stream_baseline")

# Compute speedup/slowdown summary
cat("\n=== Streaming vs In-Memory Comparison ===\n")
cat(sprintf("%-30s %10s %10s %10s\n", "Config", "InMem(ms)", "Stream(ms)", "Ratio"))
cat(paste(rep("-", 65), collapse = ""), "\n")

for (sz_name in names(CONFIGS)) {
  for (k in RANKS) {
    mem_label <- sprintf("inmem_%s_k%d", sz_name, k)
    str_label <- sprintf("stream_%s_k%d", sz_name, k)
    mem_med <- median(all_results$time_ms[all_results$label == mem_label])
    str_med <- median(all_results$time_ms[all_results$label == str_label])
    ratio <- str_med / mem_med
    cat(sprintf("%-30s %10.0f %10.0f %10.2fx\n",
                sprintf("%s k=%d", sz_name, k), mem_med, str_med, ratio))
  }
}

cat("\n=== Phase 2C Complete ===\n")
