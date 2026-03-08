#!/usr/bin/env Rscript
# suite_stream.R — Streaming NMF benchmark suite using bench_runner API
#
# Compares streaming (SPZ) vs in-memory NMF. Writes SPZ files of different sizes.
# Produces CSV: benchmarks/harness/results/<timestamp>_stream_suite.csv
#
# Usage:
#   Rscript benchmarks/harness/suite_stream.R

cat("=== Streaming NMF Benchmark Suite ===\n")

harness_dir <- normalizePath(file.path(dirname(
  sys.frame(1)$ofile %||% "benchmarks/harness/suite_stream.R")), mustWork = FALSE)
if (!dir.exists(harness_dir))
  harness_dir <- file.path(getwd(), "benchmarks", "harness")

source(file.path(harness_dir, "bench_runner.R"))
source(file.path(harness_dir, "generators.R"))

suppressPackageStartupMessages(library(RcppML))
suppressPackageStartupMessages(library(Matrix))

meta <- get_bench_metadata()
cat(sprintf("Node: %s | Threads: %d\n\n", meta$node, meta$omp_threads))

has_gpu <- tryCatch(RcppML:::gpu_available(), error = function(e) FALSE)
backends <- if (has_gpu) c("cpu", "gpu") else "cpu"
cat(sprintf("Backends: %s\n\n", paste(backends, collapse = ", ")))

# ============================================================================
# Configuration
# ============================================================================
k <- 16L
maxit <- 20L
n_reps <- 5L

# Dataset configs for streaming
stream_configs <- list(
  list(name = "small_1Kx500",   m = 1000L,  n = 500L,   density = 0.1),
  list(name = "medium_10Kx5K",  m = 10000L, n = 5000L,  density = 0.1)
)

# ============================================================================
# Run benchmarks
# ============================================================================
all_results <- data.frame()
spz_files <- character()

for (sc in stream_configs) {
  cat(sprintf("Dataset: %s (%dx%d, density=%.2f)\n", sc$name, sc$m, sc$n, sc$density))

  # Generate sparse matrix
  A <- gen_sparse(sc$m, sc$n, density = sc$density, seed = 42)
  nnz_val <- nnzero(A)

  # Write SPZ file
  spz_path <- tempfile(fileext = ".spz")
  sp_write(A, spz_path, include_transpose = TRUE)
  spz_files <- c(spz_files, spz_path)
  cat(sprintf("  SPZ file: %s (%.1f MB)\n", spz_path,
              file.size(spz_path) / 1024 / 1024))

  for (backend in backends) {
    # In-memory benchmark
    label_mem <- sprintf("stream_%s_%s_inmem_k%d", backend, sc$name, k)
    cat(sprintf("  %s ... ", label_mem))

    res_mem <- run_bench(
      label = label_mem,
      expr = quote(nmf(A, k = k, tol = 1e-10, maxit = maxit,
                       seed = 42, verbose = FALSE, resource = backend)),
      n_reps = n_reps,
      extra = list(k = k, m = sc$m, n = sc$n, nnz = nnz_val,
                   backend = backend, loss = "mse")
    )
    cat(sprintf("%.1f ms\n", median(res_mem$time_ms)))
    all_results <- rbind(all_results, res_mem)

    # Streaming benchmark
    label_str <- sprintf("stream_%s_%s_spz_k%d", backend, sc$name, k)
    cat(sprintf("  %s ... ", label_str))

    res_str <- run_bench(
      label = label_str,
      expr = quote(nmf(spz_path, k = k, tol = 1e-10, maxit = maxit,
                       seed = 42, verbose = FALSE, resource = backend)),
      n_reps = n_reps,
      extra = list(k = k, m = sc$m, n = sc$n, nnz = nnz_val,
                   backend = backend, loss = "mse")
    )
    cat(sprintf("%.1f ms\n", median(res_str$time_ms)))
    all_results <- rbind(all_results, res_str)
  }
}

# Cleanup temp SPZ files
for (f in spz_files) unlink(f)

# ============================================================================
# Report
# ============================================================================
report_bench(all_results, save_csv = TRUE, tag = "stream_suite")
cat("Streaming suite complete.\n")
