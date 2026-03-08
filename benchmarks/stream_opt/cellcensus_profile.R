#!/usr/bin/env Rscript
# cellcensus_profile.R — Profile streaming NMF on real CellCensus 500K/900K data
#
# Tests streaming performance with real-world scRNA-seq data from CellCensus.
# 60,664 genes x 500K-900K cells, ~3% density.
#
# Usage: Rscript cellcensus_profile.R [500k|900k]

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "500k"

suppressPackageStartupMessages({
  library(RcppML)
  library(Matrix)
})

cat("=== CellCensus Streaming NMF Profiling ===\n")
cat("Date:", format(Sys.time()), "\n")
cat("R:", R.version.string, "\n")
cat("OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "unset"), "\n")
cat("Dataset:", dataset, "\n\n")

spz_path <- sprintf("benchmarks/stream_opt/cellcensus_%s.spz", dataset)
if (!file.exists(spz_path)) stop("SPZ file not found: ", spz_path)
cat(sprintf("SPZ file: %s (%.2f GB)\n", spz_path, file.size(spz_path) / 1e9))

# Configure streaming and profiling
options(RcppML.streaming = TRUE)
options(RcppML.verbose = TRUE)
options(RcppML.threads = as.integer(Sys.getenv("OMP_NUM_THREADS", "4")))

# Test with different ranks
ranks <- c(8L, 16L, 32L)
maxit <- 5L  # Few iterations to profile I/O vs compute ratio

results <- list()

for (k in ranks) {
  cat(sprintf("\n--- Rank k=%d, maxit=%d ---\n", k, maxit))

  gc(verbose = FALSE)
  t0 <- proc.time()

  # Streaming NMF from SPZ file
  res <- nmf(spz_path, k = k, maxit = maxit, tol = 1e-10,
             seed = 42L, verbose = TRUE)

  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Total wallclock: %.2fs\n", elapsed))
  cat(sprintf("  Final loss: %.6e\n", res@misc$loss))

  results[[paste0("k", k)]] <- list(
    k = k, elapsed = elapsed,
    loss = res@misc$loss,
    iter = res@misc$iter
  )
}

cat("\n\n=== SUMMARY ===\n")
cat(sprintf("%-6s %12s %12s %8s\n", "Rank", "Time(s)", "Loss", "Iters"))
for (nm in names(results)) {
  r <- results[[nm]]
  cat(sprintf("k=%-4d %12.2f %12.4e %8d\n", r$k, r$elapsed, r$loss, r$iter))
}

# Now test with more iterations for k=16 to get steady-state timing
cat("\n\n--- Steady-state profiling: k=16, maxit=20 ---\n")
gc(verbose = FALSE)
t0 <- proc.time()
res_long <- nmf(spz_path, k = 16L, maxit = 20L, tol = 1e-10,
                seed = 42L, verbose = TRUE)
elapsed_long <- (proc.time() - t0)[3]
cat(sprintf("  Total wallclock: %.2fs (%.2fs/iter)\n",
            elapsed_long, elapsed_long / 20))

cat("\nDone.\n")
