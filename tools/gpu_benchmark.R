#!/usr/bin/env Rscript
# GPU Benchmark Script — Run on g051 with R_LIBS_USER=/mnt/home/debruinz/R/tmplib
# Tests GPU vs CPU performance on real datasets (H100 NVL)

library(RcppML)
library(Matrix)

cat("=== GPU BENCHMARK SUITE — H100 NVL ===\n")
gi <- gpu_info()
cat("GPU devices:\n")
print(gi)
cat("\n")

output_dir <- "/tmp/sandbox3"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

benchmark_nmf <- function(A, k, label, maxit = 50, reps = 3) {
  cat(paste0("Benchmarking: ", label, " (", nrow(A), " x ", ncol(A), ", k=", k, ")\n"))
  
  # CPU timing
  cpu_times <- sapply(1:reps, function(r) {
    t0 <- proc.time()
    nmf(A, k = k, seed = 42 + r, maxit = maxit, tol = 1e-10, resource = "cpu")
    (proc.time() - t0)[3]
  })
  
  # GPU timing
  gpu_times <- sapply(1:reps, function(r) {
    t0 <- proc.time()
    nmf(A, k = k, seed = 42 + r, maxit = maxit, tol = 1e-10, resource = "gpu")
    (proc.time() - t0)[3]
  })
  
  cpu_med <- median(cpu_times)
  gpu_med <- median(gpu_times)
  speedup <- cpu_med / gpu_med
  
  cat(paste0("  CPU: ", round(cpu_med, 3), "s (median of ", reps, ")\n"))
  cat(paste0("  GPU: ", round(gpu_med, 3), "s (median of ", reps, ")\n"))
  cat(paste0("  Speedup: ", round(speedup, 1), "x\n\n"))
  
  data.frame(
    Dataset = label,
    Rows = nrow(A),
    Cols = ncol(A),
    k = k,
    Density = paste0(round(100 * nnzero(A) / (nrow(A) * ncol(A)), 1), "%"),
    CPU_s = round(cpu_med, 3),
    GPU_s = round(gpu_med, 3),
    Speedup = paste0(round(speedup, 1), "x"),
    stringsAsFactors = FALSE
  )
}

results <- list()

# 1. PBMC3K (real scRNA-seq data)
cat("--- Loading pbmc3k ---\n")
data(pbmc3k)
cat("pbmc3k dims:", dim(pbmc3k), "\n")
results[[1]] <- benchmark_nmf(pbmc3k, k = 10, "PBMC3K (k=10)")
results[[2]] <- benchmark_nmf(pbmc3k, k = 20, "PBMC3K (k=20)")

# 2. AML (methylation data)
cat("--- Loading aml ---\n")
data(aml)
results[[3]] <- benchmark_nmf(aml, k = 6, "AML methylation (k=6)")

# 3. Golub (gene expression)
cat("--- Loading golub ---\n")
data(golub)
results[[4]] <- benchmark_nmf(golub, k = 5, "Golub leukemia (k=5)")

# 4. Hawaiian birds
cat("--- Loading hawaiibirds ---\n")
data(hawaiibirds)
results[[5]] <- benchmark_nmf(hawaiibirds, k = 8, "Hawaii birds (k=8)")

# 5. MovieLens (recommendation)
cat("--- Loading movielens ---\n")
data(movielens)
results[[6]] <- benchmark_nmf(movielens, k = 20, "MovieLens (k=20)")

# 6. Synthetic dense (larger)
cat("--- Creating synthetic dense ---\n")
set.seed(42)
A_dense <- matrix(abs(rnorm(5000 * 2000)), 5000, 2000)
results[[7]] <- benchmark_nmf(A_dense, k = 20, "Dense synthetic (5Kx2K, k=20)")
results[[8]] <- benchmark_nmf(A_dense, k = 50, "Dense synthetic (5Kx2K, k=50)")

# 7. Synthetic sparse (larger)
cat("--- Creating synthetic sparse ---\n")
A_sparse <- rsparsematrix(10000, 5000, density = 0.05)
A_sparse@x <- abs(A_sparse@x)
results[[9]] <- benchmark_nmf(A_sparse, k = 20, "Sparse synthetic (10Kx5K, k=20)")

# 8. SVD/PCA benchmarks
cat("=== SVD BENCHMARKS ===\n")
svd_results <- list()

benchmark_svd <- function(A, k, label) {
  cat(paste0("SVD benchmark: ", label, "\n"))
  cpu_t <- system.time(RcppML::svd(A, k = k, seed = 42, resource = "cpu"))[3]
  gpu_t <- system.time(RcppML::svd(A, k = k, seed = 42, resource = "gpu"))[3]
  cat(paste0("  CPU: ", round(cpu_t, 3), "s, GPU: ", round(gpu_t, 3), "s, Speedup: ", round(cpu_t/gpu_t, 1), "x\n"))
  data.frame(Dataset = label, k = k, CPU_s = round(cpu_t, 3), GPU_s = round(gpu_t, 3),
             Speedup = paste0(round(cpu_t/gpu_t, 1), "x"), stringsAsFactors = FALSE)
}

svd_results[[1]] <- benchmark_svd(pbmc3k, 10, "PBMC3K SVD (k=10)")
svd_results[[2]] <- benchmark_svd(A_dense, 20, "Dense 5Kx2K SVD (k=20)")
svd_results[[3]] <- benchmark_svd(A_sparse, 20, "Sparse 10Kx5K SVD (k=20)")

# 9. Cross-validation benchmarks
cat("\n=== CROSS-VALIDATION BENCHMARKS ===\n")
cv_results <- list()

benchmark_cv <- function(A, label) {
  cat(paste0("CV benchmark: ", label, "\n"))
  cpu_t <- system.time(
    cross_validate(A, k = c(3, 5, 8, 10), reps = 3, seed = 42, resource = "cpu", verbose = FALSE)
  )[3]
  gpu_t <- system.time(
    cross_validate(A, k = c(3, 5, 8, 10), reps = 3, seed = 42, resource = "gpu", verbose = FALSE)
  )[3]
  cat(paste0("  CPU: ", round(cpu_t, 3), "s, GPU: ", round(gpu_t, 3), "s, Speedup: ", round(cpu_t/gpu_t, 1), "x\n"))
  data.frame(Dataset = label, CPU_s = round(cpu_t, 3), GPU_s = round(gpu_t, 3),
             Speedup = paste0(round(cpu_t/gpu_t, 1), "x"), stringsAsFactors = FALSE)
}

cv_results[[1]] <- benchmark_cv(pbmc3k, "PBMC3K CV (k=3,5,8,10)")

# Combine and save results
cat("\n=== FINAL NMF BENCHMARK TABLE ===\n")
bench_df <- do.call(rbind, results)
print(bench_df)

cat("\n=== SVD BENCHMARK TABLE ===\n")
svd_df <- do.call(rbind, svd_results)
print(svd_df)

cat("\n=== CV BENCHMARK TABLE ===\n")
cv_df <- do.call(rbind, cv_results)
print(cv_df)

save(bench_df, svd_df, cv_df, file = file.path(output_dir, "gpu_benchmarks.RData"))
cat("\nResults saved to:", file.path(output_dir, "gpu_benchmarks.RData"), "\n")
