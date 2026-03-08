#!/usr/bin/env Rscript
# Flagship benchmark: GEO single-cell dataset, k=64, NB loss, no manual configuration.
# Run on a compute node with: Rscript benchmarks/harness/streaming_spz/bench_geo_k64.R <path.spz>
#
# Records: wall time, per-iteration breakdown, peak RAM, dispatch mode.

library(RcppML)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript bench_geo_k64.R <path.spz>")
spz_path <- args[1]

stopifnot(file.exists(spz_path))

cat("System info:\n")
cat("  Node:", Sys.info()["nodename"], "\n")
cat("  R version:", R.version$version.string, "\n")
cat("  OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "not set"), "\n")
cat("  GPU available:", gpu_available(), "\n")
cat("  Available RAM (MB):", round(RcppML:::Rcpp_get_available_ram_mb(), 1), "\n\n")

info <- st_info(spz_path)
cat(sprintf("Dataset: %d genes x %d cells, nnz=%d (%.1f%% density)\n",
            info$m, info$n, info$nnz,
            100 * info$nnz / (as.numeric(info$m) * info$n)))
cat("File size:", round(file.info(spz_path)$size / 1e9, 2), "GB\n\n")

cat("Running nmf(k=64, loss='nb', maxit=20) ...\n")
t0 <- proc.time()
res <- nmf(
  spz_path,
  k     = 64,
  loss  = "nb",
  maxit = 20,
  tol   = 1e-4,
  verbose = TRUE
)
elapsed <- proc.time() - t0

cat(sprintf("\nTotal wall time: %.1f seconds (%.1f min)\n",
            elapsed["elapsed"], elapsed["elapsed"] / 60))
cat(sprintf("W: %d x %d, H: %d x %d\n",
            nrow(res@w), ncol(res@w), nrow(res@h), ncol(res@h)))

# Save results
results <- list(
  spz_path    = spz_path,
  info        = info,
  k           = 64L,
  loss        = "nb",
  maxit       = 20L,
  elapsed_sec = as.numeric(elapsed["elapsed"]),
  sys_info    = as.list(Sys.info()),
  timestamp   = Sys.time()
)
dir.create("benchmarks/results", showWarnings = FALSE, recursive = TRUE)
out_rds <- file.path("benchmarks/results",
                     paste0("geo_k64_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds"))
saveRDS(results, out_rds)
cat("Results saved to:", out_rds, "\n")
