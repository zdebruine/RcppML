#!/usr/bin/env Rscript
# suite_cpu.R — CPU NMF benchmark suite using bench_runner API
#
# Produces CSV: benchmarks/harness/results/<timestamp>_cpu_suite.csv
#
# Usage:
#   Rscript benchmarks/harness/suite_cpu.R
#   # Or with custom library:
#   R --no-save -e '.libPaths(c("/tmp/rcppml_agent7", .libPaths())); source("benchmarks/harness/suite_cpu.R")'

cat("=== CPU NMF Benchmark Suite ===\n")

# Setup paths
harness_dir <- normalizePath(file.path(dirname(
  sys.frame(1)$ofile %||% "benchmarks/harness/suite_cpu.R")), mustWork = FALSE)
if (!dir.exists(harness_dir))
  harness_dir <- file.path(getwd(), "benchmarks", "harness")

source(file.path(harness_dir, "bench_runner.R"))
source(file.path(harness_dir, "generators.R"))

suppressPackageStartupMessages(library(RcppML))
suppressPackageStartupMessages(library(Matrix))

meta <- get_bench_metadata()
cat(sprintf("Node: %s | Threads: %d\n\n", meta$node, meta$omp_threads))

# ============================================================================
# Configuration
# ============================================================================
ranks <- c(4L, 8L, 16L, 32L, 64L)
losses <- c("mse", "gp", "nb")
maxit <- 20L
n_reps <- 5L

# Generate datasets (small + medium sparse for tractable timing)
cat("Generating datasets...\n")
A_small  <- gen_sparse(1000, 500, density = 0.1, seed = 42)
A_medium <- gen_sparse(10000, 5000, density = 0.1, seed = 42)

datasets <- list(
  list(name = "small_1Kx500",  A = A_small,
       m = 1000L, n = 500L, nnz = nnzero(A_small)),
  list(name = "medium_10Kx5K", A = A_medium,
       m = 10000L, n = 5000L, nnz = nnzero(A_medium))
)

# ============================================================================
# Run benchmarks
# ============================================================================
all_results <- data.frame()

for (ds in datasets) {
  for (k in ranks) {
    for (loss in losses) {
      label <- sprintf("cpu_%s_%s_k%d", ds$name, loss, k)
      cat(sprintf("  %s ... ", label))

      A <- ds$A
      res <- run_bench(
        label = label,
        expr = quote(nmf(A, k = k, tol = 1e-10, maxit = maxit,
                         loss = loss, seed = 42, verbose = FALSE,
                         resource = "cpu")),
        n_reps = n_reps,
        extra = list(k = k, m = ds$m, n = ds$n, nnz = ds$nnz,
                     backend = "cpu", loss = loss)
      )
      cat(sprintf("%.1f ms (median)\n", median(res$time_ms)))
      all_results <- rbind(all_results, res)
    }
  }
}

# ============================================================================
# Report
# ============================================================================
report_bench(all_results, save_csv = TRUE, tag = "cpu_suite")
cat("CPU suite complete.\n")
