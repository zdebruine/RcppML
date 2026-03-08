#!/usr/bin/env Rscript
# CPU Benchmark for RcppML NMF on bigmem nodes

library(RcppML)
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/results"

RANKS <- c(2, 4, 8, 16, 32, 64)
THREADS <- c(1, 4, 8, 16, 24, 48)
MAXIT <- 100
TOL <- 1e-10
REPLICATES <- 5

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data(pbmc3k)
cat("Dataset: pbmc3k\n")
cat("Dimensions:", dim(pbmc3k)[1], "x", dim(pbmc3k)[2], "\n")
cat("Non-zeros:", length(pbmc3k@x), "\n\n")

benchmark_nmf <- function(A, k, threads, maxit = MAXIT, tol = TOL) {
  options(RcppML.threads = threads)
  invisible(nmf(A, k = k, maxit = 5, tol = 1e-4, verbose = FALSE))
  gc()
  start_time <- Sys.time()
  result <- nmf(A, k = k, maxit = maxit, tol = tol, verbose = FALSE)
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  list(elapsed = elapsed, iterations = length(result@loss),
       time_per_iter = elapsed / length(result@loss))
}

results <- data.frame()
total_runs <- length(RANKS) * length(THREADS) * REPLICATES
run_count <- 0

for (k in RANKS) {
  for (threads in THREADS) {
    for (rep in seq_len(REPLICATES)) {
      run_count <- run_count + 1
      cat(sprintf("[%d/%d] k=%d, threads=%d, rep=%d ... ", 
                  run_count, total_runs, k, threads, rep))
      timing <- benchmark_nmf(pbmc3k, k = k, threads = threads)
      results <- rbind(results, data.frame(
        rank = k, threads = threads, replicate = rep,
        elapsed_sec = timing$elapsed,
        time_per_iter_ms = timing$time_per_iter * 1000))
      cat(sprintf("%.2f sec\n", timing$elapsed))
    }
  }
}

output_file <- file.path(output_dir, sprintf("cpu_benchmark_%s.csv",
                          format(Sys.time(), "%Y%m%d_%H%M%S")))
write.csv(results, output_file, row.names = FALSE)
cat("\nResults saved to:", output_file, "\n")
