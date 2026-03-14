#!/usr/bin/env Rscript
# Real GPU benchmarks for vignette data — Run on GPU node (g051)
# Must set: R_LIBS_USER=/mnt/home/debruinz/R/tmplib

library(RcppML)
library(Matrix)

cat("=== REAL GPU BENCHMARKS FOR VIGNETTE ===\n")
cat("GPU info:\n")
print(gpu_info())
cat("\n")

bench <- function(expr, reps = 3) {
  times <- sapply(1:reps, function(r) system.time(eval(expr))[3])
  median(times)
}

results <- list()

# 1. PBMC3K k=10
cat("=== PBMC3K k=10 ===\n")
data(pbmc3k)
cpu_t <- bench(quote(nmf(pbmc3k, k = 10, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(pbmc3k, k = 10, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[1]] <- c("PBMC3K (scRNA-seq)", "8K x 500", "~5%", 10, cpu_t, gpu_t)

# 2. PBMC3K k=20
cat("=== PBMC3K k=20 ===\n")
cpu_t <- bench(quote(nmf(pbmc3k, k = 20, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(pbmc3k, k = 20, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[2]] <- c("PBMC3K (scRNA-seq)", "8K x 500", "~5%", 20, cpu_t, gpu_t)

# 3. Sparse synthetic 10K x 5K, k=20
cat("=== Sparse synthetic 10Kx5K k=20 ===\n")
set.seed(42)
A_sp <- rsparsematrix(10000, 5000, density = 0.03)
A_sp@x <- abs(A_sp@x)
cpu_t <- bench(quote(nmf(A_sp, k = 20, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(A_sp, k = 20, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[3]] <- c("Sparse synthetic", "10K x 5K", "~3%", 20, cpu_t, gpu_t)

# 4. Large sparse 20K x 10K, k=30
cat("=== Large sparse 20Kx10K k=30 ===\n")
A_lg <- rsparsematrix(20000, 10000, density = 0.03)
A_lg@x <- abs(A_lg@x)
cpu_t <- bench(quote(nmf(A_lg, k = 30, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(A_lg, k = 30, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[4]] <- c("Large sparse synthetic", "20K x 10K", "~3%", 30, cpu_t, gpu_t)

# 5. Dense 5K x 2K, k=20
cat("=== Dense 5Kx2K k=20 ===\n")
A_de <- matrix(abs(rnorm(5000 * 2000)), 5000, 2000)
cpu_t <- bench(quote(nmf(A_de, k = 20, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(A_de, k = 20, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[5]] <- c("Dense synthetic", "5K x 2K", "100%", 20, cpu_t, gpu_t)

# 6. Dense 5K x 2K, k=50
cat("=== Dense 5Kx2K k=50 ===\n")
cpu_t <- bench(quote(nmf(A_de, k = 50, seed = 42, maxit = 100, tol = 0, resource = "cpu")))
gpu_t <- bench(quote(nmf(A_de, k = 50, seed = 42, maxit = 100, tol = 0, resource = "gpu")))
cat(sprintf("  CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))
results[[6]] <- c("Dense synthetic", "5K x 2K", "100%", 50, cpu_t, gpu_t)

# SVD benchmarks
cat("\n=== SVD BENCHMARKS ===\n")
cat("=== PBMC3K SVD k=10 ===\n")
cpu_t <- bench(quote(RcppML::svd(pbmc3k, k = 10, seed = 42, resource = "cpu")))
gpu_t <- bench(quote(RcppML::svd(pbmc3k, k = 10, seed = 42, resource = "gpu")))
cat(sprintf("  SVD PBMC3K k=10: CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))

cat("=== Sparse 10Kx5K SVD k=20 ===\n")
cpu_t <- bench(quote(RcppML::svd(A_sp, k = 20, seed = 42, resource = "cpu")))
gpu_t <- bench(quote(RcppML::svd(A_sp, k = 20, seed = 42, resource = "gpu")))
cat(sprintf("  SVD sparse 10Kx5K k=20: CPU: %.3fs  GPU: %.3fs  Speedup: %.1fx\n", cpu_t, gpu_t, cpu_t/gpu_t))

cat("\n=== ALL BENCHMARKS COMPLETE ===\n")

# Print results table
cat("\nSummary table:\n")
for (r in results) {
  cat(sprintf("%-25s %-12s %-6s k=%-3s CPU=%.3fs GPU=%.3fs Speedup=%.1fx\n",
              r[1], r[2], r[3], r[4], as.numeric(r[5]), as.numeric(r[6]),
              as.numeric(r[5]) / as.numeric(r[6])))
}
