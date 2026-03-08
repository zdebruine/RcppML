#!/usr/bin/env Rscript
# profile_cellcensus_large.R — Profile streaming NMF with CellCensus 500K data
# Focus: I/O vs compute breakdown at real scale

.libPaths("/tmp/rcppml_stream")
suppressPackageStartupMessages(library(RcppML))

cat("=== CellCensus 500K Streaming NMF Profile ===\n")
cat("Date:", format(Sys.time()), "\n")
cat("OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "unset"), "\n")
cat("RcppML version:", as.character(packageVersion("RcppML")), "\n\n")

spz_path <- "benchmarks/stream_opt/cellcensus_500k.spz"
stopifnot(file.exists(spz_path))
cat(sprintf("SPZ: %s (%.2f GB)\n\n", spz_path, file.size(spz_path) / 1e9))

# Profile at k=8, 16, 32 with 10 iterations each
for (k in c(8L, 16L, 32L)) {
  cat(sprintf("--- k=%d, maxit=10 ---\n", k))
  gc(verbose = FALSE)
  
  t0 <- proc.time()
  res <- nmf(spz_path, k = k, maxit = 10L, tol = 1e-10,
             seed = 42L, verbose = TRUE, profile = TRUE,
             streaming = TRUE)
  elapsed <- (proc.time() - t0)[3]
  
  cat(sprintf("  Wallclock: %.2fs (%.2fs/iter)\n", elapsed, elapsed / 10))
  
  # Check for profiling data in misc
  if (!is.null(res@misc$profile)) {
    cat("  Profile data:\n")
    prof <- res@misc$profile
    for (nm in sort(names(prof))) {
      cat(sprintf("    %-20s: %8.1f ms\n", nm, prof[[nm]]))
    }
  }
  
  if (!is.null(res@misc$loss)) {
    cat(sprintf("  Final loss: %.6e\n", res@misc$loss))
  }
  cat("\n")
}

cat("Done.\n")
